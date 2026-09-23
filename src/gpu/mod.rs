/*! gpu/mod.rs: Rust wrapper around the CUDA recovery-enumerate kernel. */

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

const PBKDF2_ITERATIONS: i32 = 2048;

const KERNEL_SRC: &str = include_str!("cuda/kernel.cu");

pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    enum_kernel: CudaFunction,
    d_wordlist: CudaSlice<u8>,
    d_g_table: CudaSlice<u8>,
}

/* Precompute 64 windows x 15 multiples of G for windowed scalar multiplication.
 * Layout: 64 windows, each 15 entries (j = 1..=15), each entry = (X_be 32 bytes, Y_be 32 bytes).
 * Total size = 64 * 15 * 64 = 61440 bytes. Stored in global memory; the GPU loads each
 * entry once per scalar mult (64 ldg-style reads, L2-cached). */
fn build_g_table_bytes() -> Vec<u8> {
    use secp256k1::{Secp256k1, SecretKey, PublicKey};
    let secp = Secp256k1::new();
    let mut table = vec![0u8; 64 * 15 * 64];
    for i in 0..64usize {
        for j in 1..=15u32 {
            let mut sk_bytes = [0u8; 32];
            let byte_idx = 31 - i / 2;
            let shift = ((i as u32) % 2) * 4;
            sk_bytes[byte_idx] |= (j as u8) << shift;
            let sk = SecretKey::from_slice(&sk_bytes).expect("g_table scalar should be < N");
            let pk = PublicKey::from_secret_key(&secp, &sk);
            let unc = pk.serialize_uncompressed();
            let off = (i * 15 + (j as usize - 1)) * 64;
            table[off..off + 32].copy_from_slice(&unc[1..33]);
            table[off + 32..off + 64].copy_from_slice(&unc[33..65]);
        }
    }
    table
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        let opts = CompileOptions {
            // This fork targets RTX 3070 / GA104 (SM86). Explicit virtual architecture
            // lets NVRTC generate Ampere-aware PTX instead of relying on its toolkit default.
            arch: Some("compute_86"),
            // The hot path is integer SHA/secp256k1; fast-math does not help it.
            use_fast_math: Some(false),
            ..Default::default()
        };

        // Production default is __launch_bounds__(256, 2). Expose only the min-blocks
        // term as an explicit experiment knob so local benchmarking does not need to
        // patch kernel.cu in-place. Invalid values fail loudly rather than silently
        // changing the generated code.
        let lb_min_blocks = match std::env::var("SEEDPHRASE_LB_MIN_BLOCKS") {
            Ok(raw) => {
                let v: u32 = raw
                    .parse()
                    .map_err(|_| format!("invalid SEEDPHRASE_LB_MIN_BLOCKS={raw:?}: expected integer 1..=4"))?;
                if !(1..=4).contains(&v) {
                    return Err(format!(
                        "invalid SEEDPHRASE_LB_MIN_BLOCKS={v}: expected 1..=4"
                    ));
                }
                v
            }
            Err(std::env::VarError::NotPresent) => 2,
            Err(e) => return Err(format!("read SEEDPHRASE_LB_MIN_BLOCKS: {e}")),
        };
        let parse_probe_flag = |name: &str| -> Result<bool, String> {
            match std::env::var(name) {
                Ok(raw) => match raw.as_str() {
                    "0" | "false" | "FALSE" => Ok(false),
                    "1" | "true" | "TRUE" => Ok(true),
                    _ => Err(format!("invalid {name}={raw:?}: expected 0/1 or false/true")),
                },
                Err(std::env::VarError::NotPresent) => Ok(false),
                Err(e) => Err(format!("read {name}: {e}")),
            }
        };
        let noinline_sha512_words = parse_probe_flag("SEEDPHRASE_NOINLINE_SHA512_WORDS")?;
        let noinline_fixed64_hmac = parse_probe_flag("SEEDPHRASE_NOINLINE_FIXED64_HMAC")?;
        let noinline_pbkdf2 = parse_probe_flag("SEEDPHRASE_NOINLINE_PBKDF2")?;
        let pbkdf2_scalar_ut = parse_probe_flag("SEEDPHRASE_PBKDF2_SCALAR_UT")?;

        // Hardware validation on the reference RTX 3070 found exactly one probe
        // combination that reproducibly produces wrong BIP84 results under NVRTC:
        // SHA words noinline + PBKDF2 noinline + fixed64 HMAC inline.
        // Reject it before compiling so a throughput-only experiment can never promote
        // a silently incorrect kernel.
        if noinline_sha512_words && noinline_pbkdf2 && !noinline_fixed64_hmac {
            return Err(
                "unsafe SM86 probe combination disabled: SEEDPHRASE_NOINLINE_SHA512_WORDS=1 + SEEDPHRASE_NOINLINE_PBKDF2=1 requires SEEDPHRASE_NOINLINE_FIXED64_HMAC=1; the hmac=0 combination failed the 3-vector BIP84 GPU self-test reproducibly"
                    .to_string(),
            );
        }

        let kernel_src = format!(
            "#define RECOVERY_LAUNCH_MIN_BLOCKS {}\n#define RECOVERY_NOINLINE_SHA512_WORDS {}\n#define RECOVERY_NOINLINE_FIXED64_HMAC {}\n#define RECOVERY_NOINLINE_PBKDF2 {}\n#define RECOVERY_PBKDF2_SCALAR_UT {}\n{}",
            lb_min_blocks,
            if noinline_sha512_words { 1 } else { 0 },
            if noinline_fixed64_hmac { 1 } else { 0 },
            if noinline_pbkdf2 { 1 } else { 0 },
            if pbkdf2_scalar_ut { 1 } else { 0 },
            KERNEL_SRC
        );
        let ptx = compile_ptx_with_opts(&kernel_src, opts)
            .map_err(|e| format!("nvrtc compile failed: {e}"))?;
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda init failed: {e}"))?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module failed: {e}"))?;
        let enum_kernel = module
            .load_function("recovery_enumerate")
            .map_err(|e| format!("load recovery_enumerate: {e}"))?;

        /* Upload BIP39 wordlist to device memory in [length || up-to-8-chars || 3 pad] format. */
        let wordlist = bip39::Language::English.word_list();
        let mut packed = vec![0u8; 2048 * 12];
        for (i, word) in wordlist.iter().enumerate() {
            let bytes = word.as_bytes();
            assert!(bytes.len() <= 8, "BIP39 word longer than 8 bytes");
            let off = i * 12;
            packed[off] = bytes.len() as u8;
            packed[off + 1..off + 1 + bytes.len()].copy_from_slice(bytes);
        }
        let d_wordlist = stream
            .clone_htod(&packed)
            .map_err(|e| format!("htod wordlist: {e}"))?;

        let g_table = build_g_table_bytes();
        let d_g_table = stream
            .clone_htod(&g_table)
            .map_err(|e| format!("htod g_table: {e}"))?;

        Ok(Self {
            ctx,
            stream,
            enum_kernel,
            d_wordlist,
            d_g_table,
        })
    }

    /** Launch the enumeration kernel for a contiguous chunk of the search space.
     *
     * Each thread tid in [0, chunk_size) processes candidate index (chunk_offset + tid).
     * On match, the absolute candidate index is written to d_match_idx (i64) via atomicCAS.
     *
     * Returns Some(absolute_candidate_index) on match, else None.
     */
    pub fn run_enumeration(
        &self,
        known_indices: &[u16; 24],
        mnemonic_length: usize,
        checksum_bits: usize,
        missing_positions: &[u8],
        last_is_missing: bool,
        chunk_offset: u64,
        chunk_size: u64,
        salt: &[u8],
        path: &[u32],
        target_hash160: &[u8; 20],
    ) -> Result<Option<u64>, String> {
        if chunk_size == 0 {
            return Ok(None);
        }
        if missing_positions.len() > 3 {
            return Err(format!(
                "missing_count > 3 not supported (got {})",
                missing_positions.len()
            ));
        }

        let d_known = self
            .stream
            .clone_htod(&known_indices[..])
            .map_err(|e| format!("htod known: {e}"))?;
        let mut mp_padded = [0u8; 3];
        for (i, &p) in missing_positions.iter().enumerate() {
            mp_padded[i] = p;
        }
        let d_mp = self
            .stream
            .clone_htod(&mp_padded[..])
            .map_err(|e| format!("htod mp: {e}"))?;
        let d_salt = self
            .stream
            .clone_htod(salt)
            .map_err(|e| format!("htod salt: {e}"))?;
        let d_path = self
            .stream
            .clone_htod(path)
            .map_err(|e| format!("htod path: {e}"))?;
        let d_target = self
            .stream
            .clone_htod(&target_hash160[..])
            .map_err(|e| format!("htod target: {e}"))?;

        let initial_match: [i64; 1] = [-1];
        let mut d_match = self
            .stream
            .clone_htod(&initial_match)
            .map_err(|e| format!("htod match: {e}"))?;

        let mnemonic_length_i32 = mnemonic_length as i32;
        let checksum_bits_i32 = checksum_bits as i32;
        let missing_count_i32 = missing_positions.len() as i32;
        let last_is_missing_i32 = if last_is_missing { 1i32 } else { 0i32 };
        let salt_len_i32 = salt.len() as i32;
        let iterations = PBKDF2_ITERATIONS;
        let path_len_i32 = path.len() as i32;

        /* Block size is overridable for benchmarking via the SEEDPHRASE_BLOCK env var. The SM86
         * fork defaults to 256: on the reference RTX 3070 that was the fastest of 64/128/256 for
         * every branch tested, and it matches the __launch_bounds__ ceiling declared on the
         * kernel. Values outside 32..=256 or not divisible by 32 are rejected explicitly;
         * never silently fall back during benchmarking. */
        let block: u32 = match std::env::var("SEEDPHRASE_BLOCK") {
            Ok(raw) => {
                let b: u32 = raw
                    .parse()
                    .map_err(|_| format!("invalid SEEDPHRASE_BLOCK={raw:?}: expected a warp multiple in 32..=256"))?;
                if !(32..=256).contains(&b) || b % 32 != 0 {
                    return Err(format!(
                        "invalid SEEDPHRASE_BLOCK={b}: expected a warp multiple in 32..=256"
                    ));
                }
                b
            }
            Err(std::env::VarError::NotPresent) => 256,
            Err(e) => return Err(format!("read SEEDPHRASE_BLOCK: {e}")),
        };
        let grid: u32 = (chunk_size.div_ceil(block as u64) as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launcher = self.stream.launch_builder(&self.enum_kernel);
        launcher.arg(&d_known);
        launcher.arg(&mnemonic_length_i32);
        launcher.arg(&checksum_bits_i32);
        launcher.arg(&d_mp);
        launcher.arg(&missing_count_i32);
        launcher.arg(&last_is_missing_i32);
        launcher.arg(&chunk_offset);
        launcher.arg(&chunk_size);
        launcher.arg(&d_salt);
        launcher.arg(&salt_len_i32);
        launcher.arg(&iterations);
        launcher.arg(&d_path);
        launcher.arg(&path_len_i32);
        launcher.arg(&d_target);
        launcher.arg(&self.d_wordlist);
        launcher.arg(&self.d_g_table);
        launcher.arg(&mut d_match);
        unsafe { launcher.launch(cfg) }.map_err(|e| format!("kernel launch: {e}"))?;

        let host_match: Vec<i64> = self
            .stream
            .clone_dtoh(&d_match)
            .map_err(|e| format!("dtoh match: {e}"))?;
        let idx = host_match[0];
        if idx < 0 {
            Ok(None)
        } else {
            Ok(Some(idx as u64))
        }
    }

    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "unknown".to_string())
    }

    /// Return CUDA driver-reported resource usage for the loaded enumeration kernel.
    /// This uses cuFuncGetAttribute through cudarc and works even when Nsight Compute
    /// performance counters are unavailable (for example under restricted WSL setups).
    pub fn kernel_resource_summary(&self) -> Result<String, String> {
        let regs = self
            .enum_kernel
            .num_regs()
            .map_err(|e| format!("query kernel registers: {e}"))?;
        let local = self
            .enum_kernel
            .local_size_bytes()
            .map_err(|e| format!("query kernel local memory: {e}"))?;
        let shared = self
            .enum_kernel
            .shared_size_bytes()
            .map_err(|e| format!("query kernel shared memory: {e}"))?;
        let max_threads = self
            .enum_kernel
            .max_threads_per_block()
            .map_err(|e| format!("query kernel max threads/block: {e}"))?;
        Ok(format!(
            "regs/thread={regs} local/thread={local}B shared/block={shared}B max_threads/block={max_threads}"
        ))
    }

    /** Verify the production enumeration kernel against BIP84 reference vectors.
     *
     * For each vector we drop the last word of the mnemonic and let the kernel enumerate
     * all 128 valid completions (the BIP39 checksum constrains the last 4 bits, so there
     * are 2^7 candidate last words for a 12-word phrase). A correct kernel must find the
     * canonical match. The known_indices for the missing slot are filler and overwritten
     * inside the kernel. */
    pub fn self_test(&self) -> Result<(), String> {
        let vectors: &[(&str, &str)] = &[
            (
                "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
                "bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu",
            ),
            (
                "legal winner thank year wave sausage worth useful legal winner thank yellow",
                "bc1qgkju4yvvtuz0s8vqn837q396jezu2h8ex7gk98",
            ),
            (
                "letter advice cage absurd amount doctor acoustic avoid letter advice cage above",
                "bc1q4f2e4sm7w094l7t6w6vtgynjw7q3987lgx9ssj",
            ),
        ];
        let salt: &[u8] = b"mnemonic\x00\x00\x00\x01";
        let path: [u32; 5] = [0x80000000 | 84, 0x80000000, 0x80000000, 0, 0];
        let wordlist = bip39::Language::English.word_list();
        let word_to_idx: std::collections::HashMap<&str, u16> = wordlist
            .iter()
            .enumerate()
            .map(|(i, w)| (*w, i as u16))
            .collect();

        for (i, (mnemonic, addr)) in vectors.iter().enumerate() {
            let target = decode_bech32_p2wpkh_hash160(addr)?;
            let words: Vec<&str> = mnemonic.split_whitespace().collect();
            let mut known = [0u16; 24];
            for (k, w) in words.iter().enumerate().take(11) {
                known[k] = *word_to_idx.get(*w).ok_or_else(|| format!("vector {i}: bad word {w}"))?;
            }
            let result = self.run_enumeration(&known, 12, 4, &[11u8], true, 0, 128, salt, &path, &target)?;
            if result.is_none() {
                return Err(format!("self-test vector {i} FAILED: no match for {addr}"));
            }
        }
        Ok(())
    }
}

/** Decode a bc1q... P2WPKH address and return the 20-byte witness program (hash160). */
pub fn decode_bech32_p2wpkh_hash160(addr: &str) -> Result<[u8; 20], String> {
    use bitcoin::address::Address;
    use std::str::FromStr;
    let parsed = Address::from_str(addr).map_err(|e| format!("invalid address: {e}"))?;
    let net_checked = parsed
        .require_network(bitcoin::Network::Bitcoin)
        .map_err(|e| format!("address not on mainnet: {e}"))?;
    use bitcoin::address::Payload;
    match &net_checked.payload {
        Payload::WitnessProgram(wp) => {
            let prog = wp.program().as_bytes();
            if prog.len() != 20 {
                return Err(format!(
                    "expected 20-byte witness program (P2WPKH), got {} bytes",
                    prog.len()
                ));
            }
            let mut out = [0u8; 20];
            out.copy_from_slice(prog);
            Ok(out)
        }
        _ => Err("address is not a witness program (not bc1...)".to_string()),
    }
}
