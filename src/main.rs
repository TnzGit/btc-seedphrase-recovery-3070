/*! main.rs: CLI for the CUDA-accelerated BTC (BIP84) seed phrase recovery tool. */

mod utils;
mod gpu;
mod network {
    pub mod btc;
}

use colored::*;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::utils::print_header;
use crate::gpu::{Gpu, decode_bech32_p2wpkh_hash160};
use bip39::Language;
use regex::Regex;

#[derive(Clone)]
struct TestWordInfo {
    pos: usize,
    word_idx: u16,
}

const CHUNK_SIZE: u64 = 1 << 22;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--bench" {
        run_bench();
        return;
    }
    if args.len() >= 2 && args[1] == "--self-test" {
        run_self_test_cli();
        return;
    }

    print_header();

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted_c = interrupted.clone();
        if let Err(e) = ctrlc::set_handler(move || {
            if interrupted_c.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
            eprintln!("\n  Interrupt received - press Ctrl-C again to force exit.");
        }) {
            eprintln!("  Could not install Ctrl-C handler: {e}");
        }
    }

    println!("{}", "  Wallet Recovery Setup".white().bold());
    println!("{}", "  ---------------------".dimmed());
    println!("{}", "  Network: Bitcoin (native SegWit, BIP84)".cyan());
    println!("{}", "  Mode: full GPU pipeline (PBKDF2 + BIP32 + secp256k1 + hash160)".cyan());

    print!("\n  Initializing CUDA... ");
    io::stdout().flush().unwrap();
    let gpu_inst = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            println!("{}", "FAIL".red().bold());
            eprintln!("  {}", e.red());
            eprintln!("  This tool requires an NVIDIA GPU with CUDA driver installed.");
            return;
        }
    };
    println!("{}", "OK".green().bold());
    println!("  Device: {}", gpu_inst.device_name().green());

    print!("  GPU self-test (3 BIP84 reference vectors)... ");
    io::stdout().flush().unwrap();
    if let Err(e) = gpu_inst.self_test() {
        println!("{}", "FAIL".red().bold());
        eprintln!("  {}", e.red());
        eprintln!("  Aborting: GPU pipeline does not produce correct output.");
        return;
    }
    println!("{}", "PASS".green().bold());

    let gpu = Arc::new(Mutex::new(gpu_inst));

    println!("{}", "\n  Select Your Wallet Seed Phrase Length:".cyan());
    println!("  1. 12 words");
    println!("  2. 15 words");
    println!("  3. 18 words");
    println!("  4. 21 words");
    println!("  5. 24 words");

    let length_choice = prompt(&format!("\n{}", "  > Enter Choice: ".magenta()));
    let mnemonic_length = match length_choice.trim() {
        "1" => 12,
        "2" => 15,
        "3" => 18,
        "4" => 21,
        "5" => 24,
        _ => {
            println!("{}", "Invalid choice. Exiting.".red());
            return;
        }
    };

    let missing_count_input = prompt(&format!("\n{}", "  How many words are missing? (1-3): ".magenta()));
    let missing_count: usize = match missing_count_input.trim().parse() {
        Ok(n) if (1..=3).contains(&n) => n,
        _ => {
            println!("{}", "Currently only 1, 2 or 3 missing words are supported.".red());
            return;
        }
    };

    println!("{}", format!("\n  NOTE: Please enter the {} known words in sequence.", mnemonic_length - missing_count).yellow());
    println!("{}", "  Even if you are missing words, enter the known words in their correct relative order.".dimmed());
    println!();

    let known_words: Vec<String>;
    loop {
        let known_words_input = prompt(&"  Known words: ".magenta().to_string());
        let words: Vec<String> = known_words_input.split_whitespace().map(|s| s.to_string()).collect();

        if words.len() != mnemonic_length - missing_count {
            println!("{}", format!("  Error: Expected {} words, received {}. Please try again.", mnemonic_length - missing_count, words.len()).red());
            continue;
        }

        let wordlist = Language::English.word_list();
        let invalid_words: Vec<String> = words.iter()
            .filter(|w| !wordlist.contains(&w.as_str()))
            .cloned()
            .collect();

        if !invalid_words.is_empty() {
            println!("{}", format!("  Error: Invalid BIP39 words found: {}. Please try again.", invalid_words.join(", ")).red());
            continue;
        }

        known_words = words;
        break;
    }

    let remember_position = prompt(&format!("{}", format!("  Are the positions of the {} missing word(s) known? (y/n): ", missing_count).magenta()));
    let mut known_positions = Vec::new();

    if is_yes(&remember_position) {
        if missing_count == 1 {
            let pos_input = prompt(&format!("{}", format!("  Enter position number (1-{}): ", mnemonic_length).magenta()));
            let p: usize = match pos_input.trim().parse() {
                Ok(p) if p >= 1 && p <= mnemonic_length => p,
                _ => { println!("{}", format!("  Invalid position. Expected 1-{}.", mnemonic_length).red()); return; }
            };
            known_positions.push(p - 1);
        } else if missing_count == 2 {
            let p1_input = prompt(&format!("{}", format!("  Enter the position number of the missing word that comes first in the sequence (1-{}): ", mnemonic_length).magenta()));
            let p2_input = prompt(&format!("{}", format!("  Enter the position number of the missing word that comes second in the sequence (1-{}): ", mnemonic_length).magenta()));
            let p1: usize = match p1_input.trim().parse() {
                Ok(p) if p >= 1 && p <= mnemonic_length => p,
                _ => { println!("{}", format!("  Invalid position. Expected 1-{}.", mnemonic_length).red()); return; }
            };
            let p2: usize = match p2_input.trim().parse() {
                Ok(p) if p >= 1 && p <= mnemonic_length => p,
                _ => { println!("{}", format!("  Invalid position. Expected 1-{}.", mnemonic_length).red()); return; }
            };
            if p1 == p2 {
                println!("{}", "  Positions must be distinct".red());
                return;
            }
            known_positions.push(p1 - 1);
            known_positions.push(p2 - 1);
            known_positions.sort();
        } else {
            let p1_input = prompt(&format!("{}", format!("  Enter the position number of the missing word that comes first in the sequence (1-{}): ", mnemonic_length).magenta()));
            let p2_input = prompt(&format!("{}", format!("  Enter the position number of the missing word that comes second in the sequence (1-{}): ", mnemonic_length).magenta()));
            let p3_input = prompt(&format!("{}", format!("  Enter the position number of the missing word that comes third in the sequence (1-{}): ", mnemonic_length).magenta()));
            let p1: usize = match p1_input.trim().parse() {
                Ok(p) if p >= 1 && p <= mnemonic_length => p,
                _ => { println!("{}", format!("  Invalid position. Expected 1-{}.", mnemonic_length).red()); return; }
            };
            let p2: usize = match p2_input.trim().parse() {
                Ok(p) if p >= 1 && p <= mnemonic_length => p,
                _ => { println!("{}", format!("  Invalid position. Expected 1-{}.", mnemonic_length).red()); return; }
            };
            let p3: usize = match p3_input.trim().parse() {
                Ok(p) if p >= 1 && p <= mnemonic_length => p,
                _ => { println!("{}", format!("  Invalid position. Expected 1-{}.", mnemonic_length).red()); return; }
            };
            let mut unique = vec![p1, p2, p3];
            unique.sort();
            unique.dedup();
            if unique.len() != 3 {
                println!("{}", "  Positions must be distinct".red());
                return;
            }
            known_positions = vec![p1 - 1, p2 - 1, p3 - 1];
            known_positions.sort();
        }
    }

    let addr_input = prompt(&"  Enter the BTC wallet address (bc1q...): ".magenta().to_string());
    let addr = addr_input.trim().to_string();
    let btc_re = Regex::new(r"^bc1q[a-z0-9]{38}$").unwrap();
    if !btc_re.is_match(&addr) {
        println!("{}", "  Warning: Address does not look like a standard BTC native SegWit P2WPKH (bc1q + 38 chars).".red());
        let proceed = prompt(&"  Continue anyway? (y/n): ".yellow().to_string());
        if !is_yes(&proceed) {
            return;
        }
    }
    let target_h160 = match decode_bech32_p2wpkh_hash160(&addr) {
        Ok(h) => h,
        Err(e) => {
            println!("{}", format!("  Could not decode address: {e}").red());
            return;
        }
    };

    let salt_bytes: Vec<u8> = b"mnemonic\x00\x00\x00\x01".to_vec();

    let mut positions_to_test: Vec<Vec<usize>> = Vec::new();
    if !known_positions.is_empty() {
        positions_to_test.push(known_positions.clone());
    } else if missing_count == 1 {
        for i in 0..mnemonic_length {
            positions_to_test.push(vec![i]);
        }
    } else if missing_count == 2 {
        for i in 0..mnemonic_length {
            for j in i + 1..mnemonic_length {
                positions_to_test.push(vec![i, j]);
            }
        }
    } else {
        for i in 0..mnemonic_length {
            for j in i + 1..mnemonic_length {
                for k in j + 1..mnemonic_length {
                    positions_to_test.push(vec![i, j, k]);
                }
            }
        }
    }

    let mut paths_to_test: Vec<&str> = vec![network::btc::DEFAULT_PATH];
    let alternative_paths: Vec<&str> = network::btc::ALTERNATIVE_PATHS.to_vec();

    let found = Arc::new(AtomicBool::new(false));
    let found_data: Arc<Mutex<Option<(String, Vec<TestWordInfo>)>>> = Arc::new(Mutex::new(None));
    let mut trying_alternatives = false;
    let mut overall_found = false;

    loop {
        let checksum_bits = mnemonic_length / 3;
        let reduction_factor = 2u64.pow(checksum_bits as u32);
        let wordlist_len = 2048u64;

        let mut total_to_test = 0u64;
        let last_word_pos = mnemonic_length - 1;
        for positions in &positions_to_test {
            let count = wordlist_len.pow(missing_count as u32);
            if positions.contains(&last_word_pos) {
                total_to_test += count / reduction_factor;
            } else {
                total_to_test += count;
            }
        }
        total_to_test *= paths_to_test.len() as u64;

        println!("{}", "\n\n  Configuration".white().bold());
        println!("{}", "  -------------".dimmed());
        println!("  Network:           {}", "BTC (native SegWit, BIP84)".blue());
        println!("  Acceleration:      {}", "CUDA (full pipeline on GPU)".blue());
        println!("  Chunk size:        {}", format!("{}", CHUNK_SIZE).blue());
        println!("  Mnemonic Type:     {}", format!("{}-word", mnemonic_length).blue());
        println!("  Known Words:       {}", format!("{}/{}", known_words.len(), mnemonic_length).blue());
        println!("  Missing Words:     {}", format!("{}", missing_count).blue());
        println!("  Positions to test: {}", if !known_positions.is_empty() {
            format!("Known positions {}", known_positions.iter().map(|p| p + 1).map(|p| p.to_string()).collect::<Vec<_>>().join(", "))
        } else {
            "All combinations".to_string()
        }.blue());
        println!("  Derivation Paths:  {}", format!("{} path(s) to scan", paths_to_test.len()).blue());
        println!("  Total candidates:  {}", format!("{}", total_to_test).blue());
        println!("  Target Address:    {}\n", addr.blue());

        'outer: for (path_index, current_path) in paths_to_test.iter().enumerate() {
            println!("{}", format!("\n  [Path {}/{}] {}", path_index + 1, paths_to_test.len(), current_path).blue());

            let path_indices = match parse_bip32_path(current_path) {
                Some(p) => p,
                None => {
                    eprintln!("  Could not parse path {current_path}, skipping.");
                    continue;
                }
            };

            for positions in &positions_to_test {
                println!("{}", format!("  Scanning positions {}...", positions.iter().map(|p| p + 1).map(|p| p.to_string()).collect::<Vec<_>>().join(", ")).dimmed());

                /* Build the known_indices array (BIP39 word indices for the fixed positions). */
                let wordlist = Language::English.word_list();
                let word_to_idx: std::collections::HashMap<&str, u16> = wordlist
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (*w, i as u16))
                    .collect();
                let mut known_indices = [0u16; 24];
                let mut known_idx_cursor = 0usize;
                for i in 0..mnemonic_length {
                    if !positions.contains(&i) {
                        known_indices[i] = *word_to_idx.get(known_words[known_idx_cursor].as_str()).unwrap();
                        known_idx_cursor += 1;
                    }
                }

                let last_is_missing = positions.contains(&(mnemonic_length - 1));
                let free_count = if last_is_missing { missing_count - 1 } else { missing_count };
                let missing_entropy_bits = 11 - checksum_bits as u64;
                let total_for_this: u64 = if last_is_missing {
                    (1u64 << missing_entropy_bits) * (2048u64.pow(free_count as u32))
                } else {
                    2048u64.pow(missing_count as u32)
                };

                let positions_bytes: Vec<u8> = positions.iter().map(|&p| p as u8).collect();

                /* Choose a chunk size that fills the GPU. 2^22 = 4M candidates per launch keeps
                 * launch overhead negligible while keeping device memory pressure low. */
                let chunk_size: u64 = CHUNK_SIZE;
                let mut offset = 0u64;
                let pos_set_start_time = std::time::Instant::now();
                let mut found_for_pos: Option<u64> = None;

                while offset < total_for_this {
                    if interrupted.load(Ordering::Relaxed) { break; }
                    if found.load(Ordering::Relaxed) { break; }

                    let this_chunk = std::cmp::min(chunk_size, total_for_this - offset);
                    let result = {
                        let g = gpu.lock().unwrap();
                        g.run_enumeration(
                            &known_indices,
                            mnemonic_length,
                            checksum_bits as usize,
                            &positions_bytes,
                            last_is_missing,
                            offset,
                            this_chunk,
                            &salt_bytes,
                            &path_indices,
                            &target_h160,
                        )
                    };

                    match result {
                        Ok(Some(cand_idx)) => {
                            found_for_pos = Some(cand_idx);
                            break;
                        }
                        Ok(None) => {}
                        Err(e) => { eprintln!("\n  GPU error: {e}"); break; }
                    }

                    offset += this_chunk;

                    /* Print progress for this position set. */
                    let elapsed = pos_set_start_time.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 { offset as f64 / elapsed } else { 0.0 };
                    let remaining = total_for_this.saturating_sub(offset);
                    let progress = (offset as f64 / total_for_this as f64) * 100.0;
                    let eta = if rate > 0.0 { remaining as f64 / rate } else { 0.0 };
                    print!(
                        "\r  >  Progress: {:>6.2}% | Tested: {:>11} | Rate: {:>8.2} M c/s | Remaining: {:>11} | Elapsed: {:>7.1}s | ETA: {:>7.1}s   ",
                        progress, offset, rate / 1_000_000.0, remaining, elapsed, eta
                    );
                    io::stdout().flush().ok();
                }

                if let Some(cand_idx) = found_for_pos {
                    /* Decode the candidate index to recover the full mnemonic. */
                    let mut indices = known_indices;
                    let mut free_positions: Vec<usize> = Vec::new();
                    for &p in positions {
                        if last_is_missing && p == (mnemonic_length - 1) { continue; }
                        free_positions.push(p);
                    }
                    let mut remaining = cand_idx;
                    let mut last_entropy: u16 = 0;
                    if last_is_missing {
                        last_entropy = (remaining & ((1u64 << missing_entropy_bits) - 1)) as u16;
                        remaining >>= missing_entropy_bits;
                    }
                    for &p in &free_positions {
                        indices[p] = (remaining & 0x7FF) as u16;
                        remaining >>= 11;
                    }
                    if last_is_missing {
                        /* Recompute the last word from the checksum to match what the GPU did. */
                        let last_word = compute_last_word_checksum(
                            &indices,
                            mnemonic_length,
                            checksum_bits,
                            last_entropy,
                        );
                        indices[mnemonic_length - 1] = last_word;
                    }
                    let mut full_mn = String::new();
                    for k in 0..mnemonic_length {
                        if k > 0 { full_mn.push(' '); }
                        full_mn.push_str(wordlist[indices[k] as usize]);
                    }
                    let mut info = Vec::with_capacity(missing_count);
                    for &p in positions {
                        info.push(TestWordInfo { pos: p + 1, word_idx: indices[p] });
                    }
                    *found_data.lock().unwrap() = Some((full_mn, info));
                    found.store(true, Ordering::Release);
                }

                println!();

                if found.load(Ordering::Acquire) {
                    if let Some((mn, info)) = found_data.lock().unwrap().clone() {
                        print_success(&mn, &addr, current_path, &info);
                        overall_found = true;
                        break 'outer;
                    }
                }

                if interrupted.load(Ordering::Relaxed) {
                    println!("{}", "\n  Aborted by user.".yellow());
                    return;
                }
            }
        }

        if overall_found { break; }
        if interrupted.load(Ordering::Relaxed) { return; }

        if !trying_alternatives && !alternative_paths.is_empty() {
            println!("{}", "\n\n  No exact match found with the default path.".yellow());
            let try_alt = prompt(&format!("{}", format!("  Try {} alternative paths? (y/n): ", alternative_paths.len()).cyan()));
            if is_yes(&try_alt) {
                paths_to_test = alternative_paths.clone();
                trying_alternatives = true;
                continue;
            }
        }
        break;
    }

    if !overall_found {
        println!("{}", "\n\n  Recovery Complete: No matching wallet found within the search parameters.".red());
    }
}

/** Recompute the BIP39 last word given the other 11 (or 23) word indices and the explicit
 * leading entropy bits of the last word. Mirrors what the GPU enumeration kernel does so the
 * host-side decoder yields the exact same final word. */
fn compute_last_word_checksum(
    indices: &[u16; 24],
    mnemonic_length: usize,
    checksum_bits: usize,
    last_entropy: u16,
) -> u16 {
    use sha2::{Sha256, Digest};
    let missing_entropy_bits = 11usize - checksum_bits;
    let total_entropy_bits = mnemonic_length * 11 - checksum_bits;
    let entropy_bytes_len = total_entropy_bits / 8;
    let mut entropy = [0u8; 32];
    let mut bit_ptr = 0usize;
    for i in 0..mnemonic_length - 1 {
        let idx = indices[i];
        for b in (0..11).rev() {
            if (idx >> b) & 1 == 1 {
                entropy[bit_ptr / 8] |= 1 << (7 - (bit_ptr % 8));
            }
            bit_ptr += 1;
        }
    }
    for b in (0..missing_entropy_bits as i32).rev() {
        if (last_entropy >> b) & 1 == 1 {
            entropy[bit_ptr / 8] |= 1 << (7 - (bit_ptr % 8));
        }
        bit_ptr += 1;
    }
    let mut hasher = Sha256::new();
    hasher.update(&entropy[..entropy_bytes_len]);
    let hash = hasher.finalize();
    let cs = hash[0] >> (8 - checksum_bits);
    (last_entropy << checksum_bits) | (cs as u16)
}

fn run_self_test_cli() {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gpu init FAILED: {e}");
            std::process::exit(1);
        }
    };
    println!("Device: {}", gpu.device_name());
    match gpu.kernel_resource_summary() {
        Ok(summary) => println!("Kernel resources: {summary}"),
        Err(e) => eprintln!("Kernel resource query warning: {e}"),
    }
    match gpu.self_test() {
        Ok(()) => println!("GPU self-test: PASS (3/3 BIP84 reference vectors)"),
        Err(e) => {
            eprintln!("GPU self-test: FAIL: {e}");
            std::process::exit(1);
        }
    }
}

fn run_bench() {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gpu init FAILED: {e}");
            std::process::exit(1);
        }
    };
    println!("Device: {}", gpu.device_name());
    match gpu.kernel_resource_summary() {
        Ok(summary) => println!("Kernel resources: {summary}"),
        Err(e) => eprintln!("Kernel resource query warning: {e}"),
    }
    println!(
        "Bench config: block={}  lb_min_blocks={}  noinline_sha512_words={}  noinline_fixed64_hmac={}  noinline_pbkdf2={}  pbkdf2_scalar_ut={}  pbkdf2_t_shared={}",
        std::env::var("SEEDPHRASE_BLOCK").unwrap_or_else(|_| "256(default)".to_string()),
        std::env::var("SEEDPHRASE_LB_MIN_BLOCKS").unwrap_or_else(|_| "2(default)".to_string()),
        std::env::var("SEEDPHRASE_NOINLINE_SHA512_WORDS").unwrap_or_else(|_| "0(default)".to_string()),
        std::env::var("SEEDPHRASE_NOINLINE_FIXED64_HMAC").unwrap_or_else(|_| "0(default)".to_string()),
        std::env::var("SEEDPHRASE_NOINLINE_PBKDF2").unwrap_or_else(|_| "0(default)".to_string()),
        std::env::var("SEEDPHRASE_PBKDF2_SCALAR_UT").unwrap_or_else(|_| "0(default)".to_string()),
        std::env::var("SEEDPHRASE_PBKDF2_T_SHARED").unwrap_or_else(|_| "0(default)".to_string()),
    );
    let wordlist = Language::English.word_list();
    let abandon_idx = wordlist.iter().position(|w| *w == "abandon").unwrap() as u16;
    let mut known = [0u16; 24];
    for slot in &mut known[..11] { *slot = abandon_idx; }
    let missing = [9u8, 10u8, 11u8];
    let salt: &[u8] = b"mnemonic\x00\x00\x00\x01";
    let path: [u32; 5] = [0x80000000 | 84, 0x80000000, 0x80000000, 0, 0];
    let target = decode_bech32_p2wpkh_hash160("bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu").unwrap();
    /* Run a single 4M-candidate chunk at an offset known to miss, twice to warm up the cache. */
    let offset: u64 = 1u64 << 28;
    let chunk: u64 = 1u64 << 22;
    /* Time chunks back-to-back with NO warm-up - the cold first chunk reveals GPU clock
     * ramp-up plus first-launch JIT cost. Subsequent chunks are steady state.
     *
     * The enumeration result MUST NOT be discarded: a failed kernel launch returns Err, and
     * silently ignoring it makes a broken configuration look like an impossibly fast one
     * (a zero-second chunk printed as tens of thousands of M c/s). Abort instead, and never
     * divide by a zero elapsed time. */
    let mut total_secs = 0.0f64;
    let mut measured = 0u32;
    let mut steady_secs = 0.0f64;
    let mut steady_measured = 0u32;
    for i in 0..5 {
        let start = std::time::Instant::now();
        match gpu.run_enumeration(&known, 12, 4, &missing, true, offset, chunk, salt, &path, &target) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("chunk #{i} FAILED: {e}");
                eprintln!("benchmark aborted - refusing to report a rate for a kernel that did not run.");
                std::process::exit(1);
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            eprintln!("chunk #{i} reported a non-positive elapsed time ({elapsed}); aborting.");
            std::process::exit(1);
        }
        total_secs += elapsed;
        measured += 1;
        if i > 0 {
            steady_secs += elapsed;
            steady_measured += 1;
        }
        println!(
            "chunk #{}  elapsed = {:.3}s  rate = {:.2} M c/s",
            i,
            elapsed,
            chunk as f64 / elapsed / 1_000_000.0
        );
    }
    if measured > 0 {
        let candidates = chunk * measured as u64;
        println!(
            "total {} candidates in {:.3}s  weighted_all = {:.0} c/s",
            candidates,
            total_secs,
            candidates as f64 / total_secs
        );
    }
    if steady_measured > 0 {
        let steady_candidates = chunk * steady_measured as u64;
        println!(
            "steady(chunks 1..4) {} candidates in {:.3}s  weighted_steady = {:.0} c/s",
            steady_candidates,
            steady_secs,
            steady_candidates as f64 / steady_secs
        );
    }
}

fn parse_bip32_path(path: &str) -> Option<Vec<u32>> {
    if !path.starts_with("m/") {
        return None;
    }
    let mut out = Vec::new();
    for part in path[2..].split('/') {
        if part.is_empty() {
            continue;
        }
        let (num_str, hardened) = if let Some(stripped) = part.strip_suffix('\'') {
            (stripped, true)
        } else {
            (part, false)
        };
        let num: u32 = num_str.parse().ok()?;
        if num >= 0x80000000 {
            return None;
        }
        out.push(if hardened { 0x80000000u32 | num } else { num });
    }
    Some(out)
}

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn is_yes(input: &str) -> bool {
    let s = input.trim().to_lowercase();
    s == "y" || s == "yes" || s == "ye" || s == "yeah" || s == "yep"
}

fn print_success(mnemonic: &str, address: &str, path: &str, info: &[TestWordInfo]) {
    let wordlist = Language::English.word_list();
    println!("\n");
    println!("{}", "  RECOVERY SUCCESSFUL".green().bold());
    println!("{}", "  ===================\n".green());
    println!("  Derivation Path: {}", path.yellow());
    for item in info {
        let word = wordlist[item.word_idx as usize];
        println!("  Missing Word:    \"{}\" at position {}", word.yellow(), item.pos.to_string().yellow());
    }
    println!("  Address:         {}", address.green());
    println!("{}", "  ----------------------------------------------------------------".dimmed());
    println!("  Complete Seed Phrase: {}", mnemonic.green());
    println!("{}", "  ----------------------------------------------------------------".dimmed());
}
