/*
 * kernel.cu - CUDA pipeline: PBKDF2 -> BIP32 -> secp256k1 -> hash160 -> compare
 *
 * Portions adapted from BitCrack (https://github.com/brichard19/BitCrack), MIT License,
 * Copyright (c) 2018 Ben Richard. Specifically: field arithmetic over secp256k1 P,
 * SHA-256 compression schedule, and RIPEMD-160 compression rounds.
 *
 * All BIP32 derivation, scalar-mult-by-G, batch wiring, and pipeline glue is original.
 */

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

/* =========================================================================
 * PTX inline-assembly helpers for 32-bit multiword add/sub/mul-with-carry.
 * Adapted from BitCrack/cudaMath/ptx.cuh (MIT).
 * ========================================================================= */

#define add_cc(d,a,b)   asm volatile("add.cc.u32 %0, %1, %2;\n\t"   : "=r"(d) : "r"(a), "r"(b))
#define addc_cc(d,a,b)  asm volatile("addc.cc.u32 %0, %1, %2;\n\t"  : "=r"(d) : "r"(a), "r"(b))
#define addc(d,a,b)     asm volatile("addc.u32 %0, %1, %2;\n\t"     : "=r"(d) : "r"(a), "r"(b))
#define sub_cc(d,a,b)   asm volatile("sub.cc.u32 %0, %1, %2;\n\t"   : "=r"(d) : "r"(a), "r"(b))
#define subc_cc(d,a,b)  asm volatile("subc.cc.u32 %0, %1, %2;\n\t"  : "=r"(d) : "r"(a), "r"(b))
#define subc(d,a,b)     asm volatile("subc.u32 %0, %1, %2;\n\t"     : "=r"(d) : "r"(a), "r"(b))
#define mad_lo_cc(d,a,x,b)  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t"  : "=r"(d) : "r"(a), "r"(x), "r"(b))
#define madc_lo_cc(d,a,x,b) asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(d) : "r"(a), "r"(x), "r"(b))
#define mad_hi_cc(d,a,x,b)  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t"  : "=r"(d) : "r"(a), "r"(x), "r"(b))
#define madc_hi_cc(d,a,x,b) asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(d) : "r"(a), "r"(x), "r"(b))
#define madc_hi(d,a,x,b)    asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t"    : "=r"(d) : "r"(a), "r"(x), "r"(b))
#define madc_lo(d,a,x,b)    asm volatile("madc.lo.u32 %0, %1, %2, %3;\n\t"    : "=r"(d) : "r"(a), "r"(x), "r"(b))

/* =========================================================================
 * SHA-512 (64-bit). Used for HMAC-SHA512 and PBKDF2.
 * ========================================================================= */

__device__ __constant__ uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ __constant__ uint64_t H0_512[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

#define ROTR64(x,n) (((x) >> (n)) | ((x) << (64-(n))))
#define Ch64(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define Maj64(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define S0_64(x) (ROTR64(x,28) ^ ROTR64(x,34) ^ ROTR64(x,39))
#define S1_64(x) (ROTR64(x,14) ^ ROTR64(x,18) ^ ROTR64(x,41))
#define s0_64(x) (ROTR64(x, 1) ^ ROTR64(x, 8) ^ ((x) >> 7))
#define s1_64(x) (ROTR64(x,19) ^ ROTR64(x,61) ^ ((x) >> 6))

#define SHA512_ROUND(a,b,c,d,e,f,g,h,w,k) do { \
    uint64_t _t1 = (h) + S1_64(e) + Ch64((e),(f),(g)) + (k) + (w); \
    uint64_t _t2 = S0_64(a) + Maj64((a),(b),(c)); \
    (d) += _t1; \
    (h) = _t1 + _t2; \
} while (0)

__device__ __forceinline__ uint64_t load_be64(const uint8_t* p) {
    return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] <<  8) | ((uint64_t)p[7]);
}

/* SHA-512 compression with a 16-word rolling schedule.
 * Keeping W in 16 scalar u64s avoids the original W[80] per-thread local-memory pressure.
 * The round calls are explicitly unrolled so NVRTC can keep the schedule in registers on SM86. */
__device__ __forceinline__ void sha512_compress_words(
    uint64_t state[8],
    uint64_t w0,
    uint64_t w1,
    uint64_t w2,
    uint64_t w3,
    uint64_t w4,
    uint64_t w5,
    uint64_t w6,
    uint64_t w7,
    uint64_t w8,
    uint64_t w9,
    uint64_t wa,
    uint64_t wb,
    uint64_t wc,
    uint64_t wd,
    uint64_t we,
    uint64_t wf
) {
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    SHA512_ROUND(a, b, c, d, e, f, g, h, w0, K512[0]);
    SHA512_ROUND(h, a, b, c, d, e, f, g, w1, K512[1]);
    SHA512_ROUND(g, h, a, b, c, d, e, f, w2, K512[2]);
    SHA512_ROUND(f, g, h, a, b, c, d, e, w3, K512[3]);
    SHA512_ROUND(e, f, g, h, a, b, c, d, w4, K512[4]);
    SHA512_ROUND(d, e, f, g, h, a, b, c, w5, K512[5]);
    SHA512_ROUND(c, d, e, f, g, h, a, b, w6, K512[6]);
    SHA512_ROUND(b, c, d, e, f, g, h, a, w7, K512[7]);
    SHA512_ROUND(a, b, c, d, e, f, g, h, w8, K512[8]);
    SHA512_ROUND(h, a, b, c, d, e, f, g, w9, K512[9]);
    SHA512_ROUND(g, h, a, b, c, d, e, f, wa, K512[10]);
    SHA512_ROUND(f, g, h, a, b, c, d, e, wb, K512[11]);
    SHA512_ROUND(e, f, g, h, a, b, c, d, wc, K512[12]);
    SHA512_ROUND(d, e, f, g, h, a, b, c, wd, K512[13]);
    SHA512_ROUND(c, d, e, f, g, h, a, b, we, K512[14]);
    SHA512_ROUND(b, c, d, e, f, g, h, a, wf, K512[15]);
    w0 = s1_64(we) + w9 + s0_64(w1) + w0;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w0, K512[16]);
    w1 = s1_64(wf) + wa + s0_64(w2) + w1;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w1, K512[17]);
    w2 = s1_64(w0) + wb + s0_64(w3) + w2;
    SHA512_ROUND(g, h, a, b, c, d, e, f, w2, K512[18]);
    w3 = s1_64(w1) + wc + s0_64(w4) + w3;
    SHA512_ROUND(f, g, h, a, b, c, d, e, w3, K512[19]);
    w4 = s1_64(w2) + wd + s0_64(w5) + w4;
    SHA512_ROUND(e, f, g, h, a, b, c, d, w4, K512[20]);
    w5 = s1_64(w3) + we + s0_64(w6) + w5;
    SHA512_ROUND(d, e, f, g, h, a, b, c, w5, K512[21]);
    w6 = s1_64(w4) + wf + s0_64(w7) + w6;
    SHA512_ROUND(c, d, e, f, g, h, a, b, w6, K512[22]);
    w7 = s1_64(w5) + w0 + s0_64(w8) + w7;
    SHA512_ROUND(b, c, d, e, f, g, h, a, w7, K512[23]);
    w8 = s1_64(w6) + w1 + s0_64(w9) + w8;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w8, K512[24]);
    w9 = s1_64(w7) + w2 + s0_64(wa) + w9;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w9, K512[25]);
    wa = s1_64(w8) + w3 + s0_64(wb) + wa;
    SHA512_ROUND(g, h, a, b, c, d, e, f, wa, K512[26]);
    wb = s1_64(w9) + w4 + s0_64(wc) + wb;
    SHA512_ROUND(f, g, h, a, b, c, d, e, wb, K512[27]);
    wc = s1_64(wa) + w5 + s0_64(wd) + wc;
    SHA512_ROUND(e, f, g, h, a, b, c, d, wc, K512[28]);
    wd = s1_64(wb) + w6 + s0_64(we) + wd;
    SHA512_ROUND(d, e, f, g, h, a, b, c, wd, K512[29]);
    we = s1_64(wc) + w7 + s0_64(wf) + we;
    SHA512_ROUND(c, d, e, f, g, h, a, b, we, K512[30]);
    wf = s1_64(wd) + w8 + s0_64(w0) + wf;
    SHA512_ROUND(b, c, d, e, f, g, h, a, wf, K512[31]);
    w0 = s1_64(we) + w9 + s0_64(w1) + w0;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w0, K512[32]);
    w1 = s1_64(wf) + wa + s0_64(w2) + w1;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w1, K512[33]);
    w2 = s1_64(w0) + wb + s0_64(w3) + w2;
    SHA512_ROUND(g, h, a, b, c, d, e, f, w2, K512[34]);
    w3 = s1_64(w1) + wc + s0_64(w4) + w3;
    SHA512_ROUND(f, g, h, a, b, c, d, e, w3, K512[35]);
    w4 = s1_64(w2) + wd + s0_64(w5) + w4;
    SHA512_ROUND(e, f, g, h, a, b, c, d, w4, K512[36]);
    w5 = s1_64(w3) + we + s0_64(w6) + w5;
    SHA512_ROUND(d, e, f, g, h, a, b, c, w5, K512[37]);
    w6 = s1_64(w4) + wf + s0_64(w7) + w6;
    SHA512_ROUND(c, d, e, f, g, h, a, b, w6, K512[38]);
    w7 = s1_64(w5) + w0 + s0_64(w8) + w7;
    SHA512_ROUND(b, c, d, e, f, g, h, a, w7, K512[39]);
    w8 = s1_64(w6) + w1 + s0_64(w9) + w8;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w8, K512[40]);
    w9 = s1_64(w7) + w2 + s0_64(wa) + w9;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w9, K512[41]);
    wa = s1_64(w8) + w3 + s0_64(wb) + wa;
    SHA512_ROUND(g, h, a, b, c, d, e, f, wa, K512[42]);
    wb = s1_64(w9) + w4 + s0_64(wc) + wb;
    SHA512_ROUND(f, g, h, a, b, c, d, e, wb, K512[43]);
    wc = s1_64(wa) + w5 + s0_64(wd) + wc;
    SHA512_ROUND(e, f, g, h, a, b, c, d, wc, K512[44]);
    wd = s1_64(wb) + w6 + s0_64(we) + wd;
    SHA512_ROUND(d, e, f, g, h, a, b, c, wd, K512[45]);
    we = s1_64(wc) + w7 + s0_64(wf) + we;
    SHA512_ROUND(c, d, e, f, g, h, a, b, we, K512[46]);
    wf = s1_64(wd) + w8 + s0_64(w0) + wf;
    SHA512_ROUND(b, c, d, e, f, g, h, a, wf, K512[47]);
    w0 = s1_64(we) + w9 + s0_64(w1) + w0;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w0, K512[48]);
    w1 = s1_64(wf) + wa + s0_64(w2) + w1;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w1, K512[49]);
    w2 = s1_64(w0) + wb + s0_64(w3) + w2;
    SHA512_ROUND(g, h, a, b, c, d, e, f, w2, K512[50]);
    w3 = s1_64(w1) + wc + s0_64(w4) + w3;
    SHA512_ROUND(f, g, h, a, b, c, d, e, w3, K512[51]);
    w4 = s1_64(w2) + wd + s0_64(w5) + w4;
    SHA512_ROUND(e, f, g, h, a, b, c, d, w4, K512[52]);
    w5 = s1_64(w3) + we + s0_64(w6) + w5;
    SHA512_ROUND(d, e, f, g, h, a, b, c, w5, K512[53]);
    w6 = s1_64(w4) + wf + s0_64(w7) + w6;
    SHA512_ROUND(c, d, e, f, g, h, a, b, w6, K512[54]);
    w7 = s1_64(w5) + w0 + s0_64(w8) + w7;
    SHA512_ROUND(b, c, d, e, f, g, h, a, w7, K512[55]);
    w8 = s1_64(w6) + w1 + s0_64(w9) + w8;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w8, K512[56]);
    w9 = s1_64(w7) + w2 + s0_64(wa) + w9;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w9, K512[57]);
    wa = s1_64(w8) + w3 + s0_64(wb) + wa;
    SHA512_ROUND(g, h, a, b, c, d, e, f, wa, K512[58]);
    wb = s1_64(w9) + w4 + s0_64(wc) + wb;
    SHA512_ROUND(f, g, h, a, b, c, d, e, wb, K512[59]);
    wc = s1_64(wa) + w5 + s0_64(wd) + wc;
    SHA512_ROUND(e, f, g, h, a, b, c, d, wc, K512[60]);
    wd = s1_64(wb) + w6 + s0_64(we) + wd;
    SHA512_ROUND(d, e, f, g, h, a, b, c, wd, K512[61]);
    we = s1_64(wc) + w7 + s0_64(wf) + we;
    SHA512_ROUND(c, d, e, f, g, h, a, b, we, K512[62]);
    wf = s1_64(wd) + w8 + s0_64(w0) + wf;
    SHA512_ROUND(b, c, d, e, f, g, h, a, wf, K512[63]);
    w0 = s1_64(we) + w9 + s0_64(w1) + w0;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w0, K512[64]);
    w1 = s1_64(wf) + wa + s0_64(w2) + w1;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w1, K512[65]);
    w2 = s1_64(w0) + wb + s0_64(w3) + w2;
    SHA512_ROUND(g, h, a, b, c, d, e, f, w2, K512[66]);
    w3 = s1_64(w1) + wc + s0_64(w4) + w3;
    SHA512_ROUND(f, g, h, a, b, c, d, e, w3, K512[67]);
    w4 = s1_64(w2) + wd + s0_64(w5) + w4;
    SHA512_ROUND(e, f, g, h, a, b, c, d, w4, K512[68]);
    w5 = s1_64(w3) + we + s0_64(w6) + w5;
    SHA512_ROUND(d, e, f, g, h, a, b, c, w5, K512[69]);
    w6 = s1_64(w4) + wf + s0_64(w7) + w6;
    SHA512_ROUND(c, d, e, f, g, h, a, b, w6, K512[70]);
    w7 = s1_64(w5) + w0 + s0_64(w8) + w7;
    SHA512_ROUND(b, c, d, e, f, g, h, a, w7, K512[71]);
    w8 = s1_64(w6) + w1 + s0_64(w9) + w8;
    SHA512_ROUND(a, b, c, d, e, f, g, h, w8, K512[72]);
    w9 = s1_64(w7) + w2 + s0_64(wa) + w9;
    SHA512_ROUND(h, a, b, c, d, e, f, g, w9, K512[73]);
    wa = s1_64(w8) + w3 + s0_64(wb) + wa;
    SHA512_ROUND(g, h, a, b, c, d, e, f, wa, K512[74]);
    wb = s1_64(w9) + w4 + s0_64(wc) + wb;
    SHA512_ROUND(f, g, h, a, b, c, d, e, wb, K512[75]);
    wc = s1_64(wa) + w5 + s0_64(wd) + wc;
    SHA512_ROUND(e, f, g, h, a, b, c, d, wc, K512[76]);
    wd = s1_64(wb) + w6 + s0_64(we) + wd;
    SHA512_ROUND(d, e, f, g, h, a, b, c, wd, K512[77]);
    we = s1_64(wc) + w7 + s0_64(wf) + we;
    SHA512_ROUND(c, d, e, f, g, h, a, b, we, K512[78]);
    wf = s1_64(wd) + w8 + s0_64(w0) + wf;
    SHA512_ROUND(b, c, d, e, f, g, h, a, wf, K512[79]);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ __forceinline__ void sha512_compress(uint64_t state[8], const uint8_t block[128]) {
    uint64_t w0 = load_be64(block + 0);
    uint64_t w1 = load_be64(block + 8);
    uint64_t w2 = load_be64(block + 16);
    uint64_t w3 = load_be64(block + 24);
    uint64_t w4 = load_be64(block + 32);
    uint64_t w5 = load_be64(block + 40);
    uint64_t w6 = load_be64(block + 48);
    uint64_t w7 = load_be64(block + 56);
    uint64_t w8 = load_be64(block + 64);
    uint64_t w9 = load_be64(block + 72);
    uint64_t wa = load_be64(block + 80);
    uint64_t wb = load_be64(block + 88);
    uint64_t wc = load_be64(block + 96);
    uint64_t wd = load_be64(block + 104);
    uint64_t we = load_be64(block + 112);
    uint64_t wf = load_be64(block + 120);
    sha512_compress_words(state, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf);
}

__device__ __forceinline__ void sha512_state_to_bytes(const uint64_t state[8], uint8_t out[64]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t v = state[i];
        out[i*8 + 0] = (uint8_t)(v >> 56);
        out[i*8 + 1] = (uint8_t)(v >> 48);
        out[i*8 + 2] = (uint8_t)(v >> 40);
        out[i*8 + 3] = (uint8_t)(v >> 32);
        out[i*8 + 4] = (uint8_t)(v >> 24);
        out[i*8 + 5] = (uint8_t)(v >> 16);
        out[i*8 + 6] = (uint8_t)(v >>  8);
        out[i*8 + 7] = (uint8_t)(v);
    }
}

/* HMAC-SHA512 ipad/opad precomputation. Key is padded/hashed to a 128-byte block per RFC 2104. */
__device__ void hmac_sha512_precompute(
    const uint8_t* key, int key_len,
    uint64_t ipad_state[8], uint64_t opad_state[8]
) {
    uint8_t kblock[128];
    if (key_len > 128) {
        uint64_t st[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) st[i] = H0_512[i];

        int processed = 0;
        uint8_t blk[128];
        while (key_len - processed >= 128) {
            #pragma unroll
            for (int i = 0; i < 128; i++) blk[i] = key[processed + i];
            sha512_compress(st, blk);
            processed += 128;
        }
        int remaining = key_len - processed;
        for (int i = 0; i < remaining; i++) blk[i] = key[processed + i];
        blk[remaining] = 0x80;
        if (remaining + 1 > 112) {
            for (int i = remaining + 1; i < 128; i++) blk[i] = 0;
            sha512_compress(st, blk);
            for (int i = 0; i < 112; i++) blk[i] = 0;
        } else {
            for (int i = remaining + 1; i < 112; i++) blk[i] = 0;
        }
        uint64_t bit_len = (uint64_t)key_len * 8ULL;
        for (int i = 0; i < 8; i++) blk[112 + i] = 0;
        for (int i = 0; i < 8; i++) blk[120 + i] = (uint8_t)(bit_len >> (56 - i*8));
        sha512_compress(st, blk);

        uint8_t hashed[64];
        sha512_state_to_bytes(st, hashed);
        for (int i = 0; i < 64; i++) kblock[i] = hashed[i];
        for (int i = 64; i < 128; i++) kblock[i] = 0;
    } else {
        for (int i = 0; i < key_len; i++) kblock[i] = key[i];
        for (int i = key_len; i < 128; i++) kblock[i] = 0;
    }

    uint8_t ipad[128], opad[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        ipad[i] = kblock[i] ^ 0x36;
        opad[i] = kblock[i] ^ 0x5c;
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ipad_state[i] = H0_512[i];
        opad_state[i] = H0_512[i];
    }
    sha512_compress(ipad_state, ipad);
    sha512_compress(opad_state, opad);
}

/* Complete HMAC-SHA512 from precomputed ipad/opad states. msg_len <= 127. */
__device__ __forceinline__ void hmac_sha512_finish(
    const uint64_t ipad_state[8], const uint64_t opad_state[8],
    const uint8_t* msg, int msg_len,
    uint8_t out[64]
) {
    uint64_t inner[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) inner[i] = ipad_state[i];

    uint8_t blk[128];
    for (int i = 0; i < msg_len; i++) blk[i] = msg[i];
    blk[msg_len] = 0x80;
    if (msg_len + 1 > 112) {
        for (int i = msg_len + 1; i < 128; i++) blk[i] = 0;
        sha512_compress(inner, blk);
        for (int i = 0; i < 112; i++) blk[i] = 0;
    } else {
        for (int i = msg_len + 1; i < 112; i++) blk[i] = 0;
    }
    uint64_t bit_len = ((uint64_t)128 + (uint64_t)msg_len) * 8ULL;
    for (int i = 0; i < 8; i++) blk[112 + i] = 0;
    for (int i = 0; i < 8; i++) blk[120 + i] = (uint8_t)(bit_len >> (56 - i*8));
    sha512_compress(inner, blk);

    uint8_t inner_digest[64];
    sha512_state_to_bytes(inner, inner_digest);

    uint64_t outer[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) outer[i] = opad_state[i];

    for (int i = 0; i < 64; i++) blk[i] = inner_digest[i];
    blk[64] = 0x80;
    for (int i = 65; i < 112; i++) blk[i] = 0;
    uint64_t bit_len2 = ((uint64_t)128 + 64ULL) * 8ULL;
    for (int i = 0; i < 8; i++) blk[112 + i] = 0;
    for (int i = 0; i < 8; i++) blk[120 + i] = (uint8_t)(bit_len2 >> (56 - i*8));
    sha512_compress(outer, blk);

    sha512_state_to_bytes(outer, out);
}


/* PBKDF2 hot-path HMAC: message is exactly one 64-byte SHA-512 digest.
 * Keep the digest as eight big-endian u64 words across iterations, avoiding
 * byte buffers, state_to_bytes(), and reparsing on every U2..U2048 round. */
__device__ __forceinline__ void hmac_sha512_finish_fixed64_words(
    const uint64_t ipad_state[8], const uint64_t opad_state[8],
    const uint64_t msg_words[8], uint64_t out_words[8]
) {
    uint64_t m0 = msg_words[0], m1 = msg_words[1], m2 = msg_words[2], m3 = msg_words[3];
    uint64_t m4 = msg_words[4], m5 = msg_words[5], m6 = msg_words[6], m7 = msg_words[7];
    uint64_t st[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] = ipad_state[i];

    /* 128-byte ipad already compressed + 64-byte message = 1536 bits total. */
    sha512_compress_words(
        st, m0, m1, m2, m3, m4, m5, m6, m7,
        0x8000000000000000ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 1536ULL
    );

    m0 = st[0]; m1 = st[1]; m2 = st[2]; m3 = st[3];
    m4 = st[4]; m5 = st[5]; m6 = st[6]; m7 = st[7];

    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] = opad_state[i];
    sha512_compress_words(
        st, m0, m1, m2, m3, m4, m5, m6, m7,
        0x8000000000000000ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 1536ULL
    );

    #pragma unroll
    for (int i = 0; i < 8; i++) out_words[i] = st[i];
}

/* HMAC-SHA512 in one shot (recomputes ipad/opad per call). Used for BIP32 master derivation
 * (key = "Bitcoin seed", different per call would defeat caching). */
__device__ void hmac_sha512(
    const uint8_t* key, int key_len,
    const uint8_t* msg, int msg_len,
    uint8_t out[64]
) {
    uint64_t ipad_state[8], opad_state[8];
    hmac_sha512_precompute(key, key_len, ipad_state, opad_state);
    hmac_sha512_finish(ipad_state, opad_state, msg, msg_len, out);
}

/* PBKDF2-HMAC-SHA512 producing exactly one 64-byte block (dkLen = 64).
 * salt should already include the 4-byte big-endian block counter (e.g. "mnemonic\0\0\0\1"). */
__device__ void pbkdf2_hmac_sha512_block(
    const uint8_t* pwd, int pwd_len,
    const uint8_t* salt, int salt_len,
    int iterations,
    uint8_t out[64]
) {
    uint64_t ipad_state[8], opad_state[8];
    hmac_sha512_precompute(pwd, pwd_len, ipad_state, opad_state);

    /* U1 has variable salt length, so use the generic path once. */
    uint8_t u1_bytes[64];
    hmac_sha512_finish(ipad_state, opad_state, salt, salt_len, u1_bytes);

    uint64_t U[8], T[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        U[i] = load_be64(u1_bytes + i * 8);
        T[i] = U[i];
    }

    /* U2..Uiterations are always HMACs of an exact 64-byte digest. */
    for (int it = 1; it < iterations; it++) {
        hmac_sha512_finish_fixed64_words(ipad_state, opad_state, U, U);
        #pragma unroll
        for (int i = 0; i < 8; i++) T[i] ^= U[i];
    }

    sha512_state_to_bytes(T, out);
}

/* =========================================================================
 * SHA-256 (32-bit). Used for hash160 (sha256 -> ripemd160).
 * Adapted from BitCrack/cudaMath/sha256.cuh (MIT).
 * ========================================================================= */

__device__ __constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __constant__ uint32_t H0_256[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#define ROTR32(x,n) (((x) >> (n)) | ((x) << (32-(n))))
#define CH32(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ32(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x) (ROTR32(x,2) ^ ROTR32(x,13) ^ ROTR32(x,22))
#define BSIG1(x) (ROTR32(x,6) ^ ROTR32(x,11) ^ ROTR32(x,25))
#define SSIG0(x) (ROTR32(x,7) ^ ROTR32(x,18) ^ ((x) >> 3))
#define SSIG1(x) (ROTR32(x,17) ^ ROTR32(x,19) ^ ((x) >> 10))

__device__ __forceinline__ void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4 + 0] << 24) | ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] <<  8) | ((uint32_t)block[i*4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = SSIG1(W[i-2]) + W[i-7] + SSIG0(W[i-15]) + W[i-16];
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + BSIG1(e) + CH32(e, f, g) + K256[i] + W[i];
        uint32_t T2 = BSIG0(a) + MAJ32(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* SHA-256 of a 33-byte compressed public key. Output is 8 big-endian u32 words. */
__device__ void sha256_compressed_pubkey(const uint8_t pubkey33[33], uint32_t digest[8]) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0_256[i];

    uint8_t blk[64];
    for (int i = 0; i < 33; i++) blk[i] = pubkey33[i];
    blk[33] = 0x80;
    for (int i = 34; i < 56; i++) blk[i] = 0;
    /* 33 bytes = 264 bits */
    uint64_t bit_len = 264ULL;
    for (int i = 0; i < 8; i++) blk[56 + i] = (uint8_t)(bit_len >> (56 - i*8));
    sha256_compress(state, blk);

    #pragma unroll
    for (int i = 0; i < 8; i++) digest[i] = state[i];
}

/* =========================================================================
 * RIPEMD-160. Hash a 32-byte SHA-256 digest to 20 bytes.
 * Adapted from BitCrack/cudaMath/ripemd160.cuh (MIT).
 * ========================================================================= */

#define ROL(x,n) (((x) << (n)) | ((x) >> (32-(n))))
#define RIPE_F(x,y,z) ((x) ^ (y) ^ (z))
#define RIPE_G(x,y,z) (((x) & (y)) | (~(x) & (z)))
#define RIPE_H(x,y,z) (((x) | ~(y)) ^ (z))
#define RIPE_I(x,y,z) (((x) & (z)) | ((y) & ~(z)))
#define RIPE_J(x,y,z) ((x) ^ ((y) | ~(z)))

#define RIPE_K0 0x00000000u
#define RIPE_K1 0x5a827999u
#define RIPE_K2 0x6ed9eba1u
#define RIPE_K3 0x8f1bbcdcu
#define RIPE_K4 0xa953fd4eu

#define RIPE_KK0 0x50a28be6u
#define RIPE_KK1 0x5c4dd124u
#define RIPE_KK2 0x6d703ef3u
#define RIPE_KK3 0x7a6d76e9u
#define RIPE_KK4 0x00000000u

#define FF(a,b,c,d,e,x,s)  do { (a) += RIPE_F((b),(c),(d)) + (x) + RIPE_K0; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define GG(a,b,c,d,e,x,s)  do { (a) += RIPE_G((b),(c),(d)) + (x) + RIPE_K1; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define HH(a,b,c,d,e,x,s)  do { (a) += RIPE_H((b),(c),(d)) + (x) + RIPE_K2; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define II(a,b,c,d,e,x,s)  do { (a) += RIPE_I((b),(c),(d)) + (x) + RIPE_K3; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define JJ(a,b,c,d,e,x,s)  do { (a) += RIPE_J((b),(c),(d)) + (x) + RIPE_K4; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)

#define FFF(a,b,c,d,e,x,s) do { (a) += RIPE_F((b),(c),(d)) + (x) + RIPE_KK4; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define GGG(a,b,c,d,e,x,s) do { (a) += RIPE_G((b),(c),(d)) + (x) + RIPE_KK3; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define HHH(a,b,c,d,e,x,s) do { (a) += RIPE_H((b),(c),(d)) + (x) + RIPE_KK2; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define III(a,b,c,d,e,x,s) do { (a) += RIPE_I((b),(c),(d)) + (x) + RIPE_KK1; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)
#define JJJ(a,b,c,d,e,x,s) do { (a) += RIPE_J((b),(c),(d)) + (x) + RIPE_KK0; (a) = ROL((a),(s)) + (e); (c) = ROL((c),10); } while(0)

/* RIPEMD-160 of a single 32-byte block (the SHA-256 digest). Output 20 bytes. */
__device__ void ripemd160_of_sha256(const uint32_t sha_digest[8], uint8_t out[20]) {
    /* RIPEMD-160 uses little-endian word ordering, while SHA-256 is big-endian.
     * Convert each u32 from big-endian to little-endian for the message schedule. */
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t v = sha_digest[i];
        X[i] = (v << 24) | ((v << 8) & 0x00ff0000) | ((v >> 8) & 0x0000ff00) | (v >> 24);
    }
    /* Padding: 32 bytes data + 0x80 + zeros + 64-bit length (little-endian).
     * Length = 256 bits. In RIPEMD-160 length is little-endian. */
    X[8]  = 0x00000080;
    X[9]  = 0;
    X[10] = 0;
    X[11] = 0;
    X[12] = 0;
    X[13] = 0;
    X[14] = 256;
    X[15] = 0;

    uint32_t a1 = 0x67452301u, b1 = 0xefcdab89u, c1 = 0x98badcfeu, d1 = 0x10325476u, e1 = 0xc3d2e1f0u;
    uint32_t a2 = 0x67452301u, b2 = 0xefcdab89u, c2 = 0x98badcfeu, d2 = 0x10325476u, e2 = 0xc3d2e1f0u;

    /* Left line - 80 rounds */
    FF(a1,b1,c1,d1,e1,X[ 0],11);
    FF(e1,a1,b1,c1,d1,X[ 1],14);
    FF(d1,e1,a1,b1,c1,X[ 2],15);
    FF(c1,d1,e1,a1,b1,X[ 3],12);
    FF(b1,c1,d1,e1,a1,X[ 4], 5);
    FF(a1,b1,c1,d1,e1,X[ 5], 8);
    FF(e1,a1,b1,c1,d1,X[ 6], 7);
    FF(d1,e1,a1,b1,c1,X[ 7], 9);
    FF(c1,d1,e1,a1,b1,X[ 8],11);
    FF(b1,c1,d1,e1,a1,X[ 9],13);
    FF(a1,b1,c1,d1,e1,X[10],14);
    FF(e1,a1,b1,c1,d1,X[11],15);
    FF(d1,e1,a1,b1,c1,X[12], 6);
    FF(c1,d1,e1,a1,b1,X[13], 7);
    FF(b1,c1,d1,e1,a1,X[14], 9);
    FF(a1,b1,c1,d1,e1,X[15], 8);

    GG(e1,a1,b1,c1,d1,X[ 7], 7);
    GG(d1,e1,a1,b1,c1,X[ 4], 6);
    GG(c1,d1,e1,a1,b1,X[13], 8);
    GG(b1,c1,d1,e1,a1,X[ 1],13);
    GG(a1,b1,c1,d1,e1,X[10],11);
    GG(e1,a1,b1,c1,d1,X[ 6], 9);
    GG(d1,e1,a1,b1,c1,X[15], 7);
    GG(c1,d1,e1,a1,b1,X[ 3],15);
    GG(b1,c1,d1,e1,a1,X[12], 7);
    GG(a1,b1,c1,d1,e1,X[ 0],12);
    GG(e1,a1,b1,c1,d1,X[ 9],15);
    GG(d1,e1,a1,b1,c1,X[ 5], 9);
    GG(c1,d1,e1,a1,b1,X[ 2],11);
    GG(b1,c1,d1,e1,a1,X[14], 7);
    GG(a1,b1,c1,d1,e1,X[11],13);
    GG(e1,a1,b1,c1,d1,X[ 8],12);

    HH(d1,e1,a1,b1,c1,X[ 3],11);
    HH(c1,d1,e1,a1,b1,X[10],13);
    HH(b1,c1,d1,e1,a1,X[14], 6);
    HH(a1,b1,c1,d1,e1,X[ 4], 7);
    HH(e1,a1,b1,c1,d1,X[ 9],14);
    HH(d1,e1,a1,b1,c1,X[15], 9);
    HH(c1,d1,e1,a1,b1,X[ 8],13);
    HH(b1,c1,d1,e1,a1,X[ 1],15);
    HH(a1,b1,c1,d1,e1,X[ 2],14);
    HH(e1,a1,b1,c1,d1,X[ 7], 8);
    HH(d1,e1,a1,b1,c1,X[ 0],13);
    HH(c1,d1,e1,a1,b1,X[ 6], 6);
    HH(b1,c1,d1,e1,a1,X[13], 5);
    HH(a1,b1,c1,d1,e1,X[11],12);
    HH(e1,a1,b1,c1,d1,X[ 5], 7);
    HH(d1,e1,a1,b1,c1,X[12], 5);

    II(c1,d1,e1,a1,b1,X[ 1],11);
    II(b1,c1,d1,e1,a1,X[ 9],12);
    II(a1,b1,c1,d1,e1,X[11],14);
    II(e1,a1,b1,c1,d1,X[10],15);
    II(d1,e1,a1,b1,c1,X[ 0],14);
    II(c1,d1,e1,a1,b1,X[ 8],15);
    II(b1,c1,d1,e1,a1,X[12], 9);
    II(a1,b1,c1,d1,e1,X[ 4], 8);
    II(e1,a1,b1,c1,d1,X[13], 9);
    II(d1,e1,a1,b1,c1,X[ 3],14);
    II(c1,d1,e1,a1,b1,X[ 7], 5);
    II(b1,c1,d1,e1,a1,X[15], 6);
    II(a1,b1,c1,d1,e1,X[14], 8);
    II(e1,a1,b1,c1,d1,X[ 5], 6);
    II(d1,e1,a1,b1,c1,X[ 6], 5);
    II(c1,d1,e1,a1,b1,X[ 2],12);

    JJ(b1,c1,d1,e1,a1,X[ 4], 9);
    JJ(a1,b1,c1,d1,e1,X[ 0],15);
    JJ(e1,a1,b1,c1,d1,X[ 5], 5);
    JJ(d1,e1,a1,b1,c1,X[ 9],11);
    JJ(c1,d1,e1,a1,b1,X[ 7], 6);
    JJ(b1,c1,d1,e1,a1,X[12], 8);
    JJ(a1,b1,c1,d1,e1,X[ 2],13);
    JJ(e1,a1,b1,c1,d1,X[10],12);
    JJ(d1,e1,a1,b1,c1,X[14], 5);
    JJ(c1,d1,e1,a1,b1,X[ 1],12);
    JJ(b1,c1,d1,e1,a1,X[ 3],13);
    JJ(a1,b1,c1,d1,e1,X[ 8],14);
    JJ(e1,a1,b1,c1,d1,X[11],11);
    JJ(d1,e1,a1,b1,c1,X[ 6], 8);
    JJ(c1,d1,e1,a1,b1,X[15], 5);
    JJ(b1,c1,d1,e1,a1,X[13], 6);

    /* Right line - 80 rounds */
    JJJ(a2,b2,c2,d2,e2,X[ 5], 8);
    JJJ(e2,a2,b2,c2,d2,X[14], 9);
    JJJ(d2,e2,a2,b2,c2,X[ 7], 9);
    JJJ(c2,d2,e2,a2,b2,X[ 0],11);
    JJJ(b2,c2,d2,e2,a2,X[ 9],13);
    JJJ(a2,b2,c2,d2,e2,X[ 2],15);
    JJJ(e2,a2,b2,c2,d2,X[11],15);
    JJJ(d2,e2,a2,b2,c2,X[ 4], 5);
    JJJ(c2,d2,e2,a2,b2,X[13], 7);
    JJJ(b2,c2,d2,e2,a2,X[ 6], 7);
    JJJ(a2,b2,c2,d2,e2,X[15], 8);
    JJJ(e2,a2,b2,c2,d2,X[ 8],11);
    JJJ(d2,e2,a2,b2,c2,X[ 1],14);
    JJJ(c2,d2,e2,a2,b2,X[10],14);
    JJJ(b2,c2,d2,e2,a2,X[ 3],12);
    JJJ(a2,b2,c2,d2,e2,X[12], 6);

    III(e2,a2,b2,c2,d2,X[ 6], 9);
    III(d2,e2,a2,b2,c2,X[11],13);
    III(c2,d2,e2,a2,b2,X[ 3],15);
    III(b2,c2,d2,e2,a2,X[ 7], 7);
    III(a2,b2,c2,d2,e2,X[ 0],12);
    III(e2,a2,b2,c2,d2,X[13], 8);
    III(d2,e2,a2,b2,c2,X[ 5], 9);
    III(c2,d2,e2,a2,b2,X[10],11);
    III(b2,c2,d2,e2,a2,X[14], 7);
    III(a2,b2,c2,d2,e2,X[15], 7);
    III(e2,a2,b2,c2,d2,X[ 8],12);
    III(d2,e2,a2,b2,c2,X[12], 7);
    III(c2,d2,e2,a2,b2,X[ 4], 6);
    III(b2,c2,d2,e2,a2,X[ 9],15);
    III(a2,b2,c2,d2,e2,X[ 1],13);
    III(e2,a2,b2,c2,d2,X[ 2],11);

    HHH(d2,e2,a2,b2,c2,X[15], 9);
    HHH(c2,d2,e2,a2,b2,X[ 5], 7);
    HHH(b2,c2,d2,e2,a2,X[ 1],15);
    HHH(a2,b2,c2,d2,e2,X[ 3],11);
    HHH(e2,a2,b2,c2,d2,X[ 7], 8);
    HHH(d2,e2,a2,b2,c2,X[14], 6);
    HHH(c2,d2,e2,a2,b2,X[ 6], 6);
    HHH(b2,c2,d2,e2,a2,X[ 9],14);
    HHH(a2,b2,c2,d2,e2,X[11],12);
    HHH(e2,a2,b2,c2,d2,X[ 8],13);
    HHH(d2,e2,a2,b2,c2,X[12], 5);
    HHH(c2,d2,e2,a2,b2,X[ 2],14);
    HHH(b2,c2,d2,e2,a2,X[10],13);
    HHH(a2,b2,c2,d2,e2,X[ 0],13);
    HHH(e2,a2,b2,c2,d2,X[ 4], 7);
    HHH(d2,e2,a2,b2,c2,X[13], 5);

    GGG(c2,d2,e2,a2,b2,X[ 8],15);
    GGG(b2,c2,d2,e2,a2,X[ 6], 5);
    GGG(a2,b2,c2,d2,e2,X[ 4], 8);
    GGG(e2,a2,b2,c2,d2,X[ 1],11);
    GGG(d2,e2,a2,b2,c2,X[ 3],14);
    GGG(c2,d2,e2,a2,b2,X[11],14);
    GGG(b2,c2,d2,e2,a2,X[15], 6);
    GGG(a2,b2,c2,d2,e2,X[ 0],14);
    GGG(e2,a2,b2,c2,d2,X[ 5], 6);
    GGG(d2,e2,a2,b2,c2,X[12], 9);
    GGG(c2,d2,e2,a2,b2,X[ 2],12);
    GGG(b2,c2,d2,e2,a2,X[13], 9);
    GGG(a2,b2,c2,d2,e2,X[ 9],12);
    GGG(e2,a2,b2,c2,d2,X[ 7], 5);
    GGG(d2,e2,a2,b2,c2,X[10],15);
    GGG(c2,d2,e2,a2,b2,X[14], 8);

    FFF(b2,c2,d2,e2,a2,X[12], 8);
    FFF(a2,b2,c2,d2,e2,X[15], 5);
    FFF(e2,a2,b2,c2,d2,X[10],12);
    FFF(d2,e2,a2,b2,c2,X[ 4], 9);
    FFF(c2,d2,e2,a2,b2,X[ 1],12);
    FFF(b2,c2,d2,e2,a2,X[ 5], 5);
    FFF(a2,b2,c2,d2,e2,X[ 8],14);
    FFF(e2,a2,b2,c2,d2,X[ 7], 6);
    FFF(d2,e2,a2,b2,c2,X[ 6], 8);
    FFF(c2,d2,e2,a2,b2,X[ 2],13);
    FFF(b2,c2,d2,e2,a2,X[13], 6);
    FFF(a2,b2,c2,d2,e2,X[14], 5);
    FFF(e2,a2,b2,c2,d2,X[ 0],15);
    FFF(d2,e2,a2,b2,c2,X[ 3],13);
    FFF(c2,d2,e2,a2,b2,X[ 9],11);
    FFF(b2,c2,d2,e2,a2,X[11],11);

    /* Combine */
    uint32_t H0 = 0x67452301u, H1 = 0xefcdab89u, H2 = 0x98badcfeu, H3 = 0x10325476u, H4 = 0xc3d2e1f0u;
    uint32_t t = H1 + c1 + d2;
    uint32_t newH1 = H2 + d1 + e2;
    uint32_t newH2 = H3 + e1 + a2;
    uint32_t newH3 = H4 + a1 + b2;
    uint32_t newH4 = H0 + b1 + c2;
    uint32_t newH0 = t;

    /* Write little-endian */
    out[ 0] = (uint8_t)(newH0      ); out[ 1] = (uint8_t)(newH0 >>  8); out[ 2] = (uint8_t)(newH0 >> 16); out[ 3] = (uint8_t)(newH0 >> 24);
    out[ 4] = (uint8_t)(newH1      ); out[ 5] = (uint8_t)(newH1 >>  8); out[ 6] = (uint8_t)(newH1 >> 16); out[ 7] = (uint8_t)(newH1 >> 24);
    out[ 8] = (uint8_t)(newH2      ); out[ 9] = (uint8_t)(newH2 >>  8); out[10] = (uint8_t)(newH2 >> 16); out[11] = (uint8_t)(newH2 >> 24);
    out[12] = (uint8_t)(newH3      ); out[13] = (uint8_t)(newH3 >>  8); out[14] = (uint8_t)(newH3 >> 16); out[15] = (uint8_t)(newH3 >> 24);
    out[16] = (uint8_t)(newH4      ); out[17] = (uint8_t)(newH4 >>  8); out[18] = (uint8_t)(newH4 >> 16); out[19] = (uint8_t)(newH4 >> 24);
}

/* =========================================================================
 * secp256k1 field arithmetic modulo P = 2^256 - 2^32 - 977.
 * Big-int representation: u32 array of 8 words, x[0] = most-significant word.
 * Adapted from BitCrack/cudaMath/secp256k1.cuh (MIT).
 * ========================================================================= */

__device__ __constant__ uint32_t SECP_P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

/* Group order N */
__device__ __constant__ uint32_t SECP_N[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
    0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

/* Base point G coordinates (affine) */
__device__ __constant__ uint32_t SECP_GX[8] = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
    0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

__device__ __constant__ uint32_t SECP_GY[8] = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
    0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};

__device__ __forceinline__ void big_copy(uint32_t dst[8], const uint32_t src[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

__device__ __forceinline__ bool big_eq(const uint32_t a[8], const uint32_t b[8]) {
    bool eq = true;
    #pragma unroll
    for (int i = 0; i < 8; i++) eq &= (a[i] == b[i]);
    return eq;
}

__device__ __forceinline__ bool big_is_zero(const uint32_t a[8]) {
    uint32_t r = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) r |= a[i];
    return r == 0;
}

/* Returns nonzero borrow if a < b (i.e. a - b underflows). */
__device__ __forceinline__ uint32_t big_sub(const uint32_t a[8], const uint32_t b[8], uint32_t c[8]) {
    sub_cc(c[7], a[7], b[7]);
    subc_cc(c[6], a[6], b[6]);
    subc_cc(c[5], a[5], b[5]);
    subc_cc(c[4], a[4], b[4]);
    subc_cc(c[3], a[3], b[3]);
    subc_cc(c[2], a[2], b[2]);
    subc_cc(c[1], a[1], b[1]);
    subc_cc(c[0], a[0], b[0]);
    uint32_t borrow = 0;
    subc(borrow, 0, 0);
    return borrow & 0x01;
}

__device__ __forceinline__ uint32_t big_add(const uint32_t a[8], const uint32_t b[8], uint32_t c[8]) {
    add_cc(c[7], a[7], b[7]);
    addc_cc(c[6], a[6], b[6]);
    addc_cc(c[5], a[5], b[5]);
    addc_cc(c[4], a[4], b[4]);
    addc_cc(c[3], a[3], b[3]);
    addc_cc(c[2], a[2], b[2]);
    addc_cc(c[1], a[1], b[1]);
    addc_cc(c[0], a[0], b[0]);
    uint32_t carry = 0;
    addc(carry, 0, 0);
    return carry;
}

/* a >= m ? */
__device__ __forceinline__ bool big_ge(const uint32_t a[8], const uint32_t m[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[i] > m[i]) return true;
        if (a[i] < m[i]) return false;
    }
    return true;
}

__device__ void sub_mod_p(const uint32_t a[8], const uint32_t b[8], uint32_t c[8]) {
    sub_cc(c[7], a[7], b[7]);
    subc_cc(c[6], a[6], b[6]);
    subc_cc(c[5], a[5], b[5]);
    subc_cc(c[4], a[4], b[4]);
    subc_cc(c[3], a[3], b[3]);
    subc_cc(c[2], a[2], b[2]);
    subc_cc(c[1], a[1], b[1]);
    subc_cc(c[0], a[0], b[0]);
    uint32_t borrow = 0;
    subc(borrow, 0, 0);
    if (borrow) {
        add_cc(c[7], c[7], SECP_P[7]);
        addc_cc(c[6], c[6], SECP_P[6]);
        addc_cc(c[5], c[5], SECP_P[5]);
        addc_cc(c[4], c[4], SECP_P[4]);
        addc_cc(c[3], c[3], SECP_P[3]);
        addc_cc(c[2], c[2], SECP_P[2]);
        addc_cc(c[1], c[1], SECP_P[1]);
        addc(c[0], c[0], SECP_P[0]);
    }
}

__device__ void add_mod_p(const uint32_t a[8], const uint32_t b[8], uint32_t c[8]) {
    add_cc(c[7], a[7], b[7]);
    addc_cc(c[6], a[6], b[6]);
    addc_cc(c[5], a[5], b[5]);
    addc_cc(c[4], a[4], b[4]);
    addc_cc(c[3], a[3], b[3]);
    addc_cc(c[2], a[2], b[2]);
    addc_cc(c[1], a[1], b[1]);
    addc_cc(c[0], a[0], b[0]);
    uint32_t carry = 0;
    addc(carry, 0, 0);
    bool gt = false;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (c[i] > SECP_P[i]) { gt = true; break; }
        if (c[i] < SECP_P[i]) break;
    }
    if (carry || gt) {
        sub_cc(c[7], c[7], SECP_P[7]);
        subc_cc(c[6], c[6], SECP_P[6]);
        subc_cc(c[5], c[5], SECP_P[5]);
        subc_cc(c[4], c[4], SECP_P[4]);
        subc_cc(c[3], c[3], SECP_P[3]);
        subc_cc(c[2], c[2], SECP_P[2]);
        subc_cc(c[1], c[1], SECP_P[1]);
        subc(c[0], c[0], SECP_P[0]);
    }
}

/* Schoolbook 256x256 -> 512 multiply, then Crandall reduction mod P = 2^256 - 2^32 - 977.
 * c (out) is 8 words; an internal 16-word product is reduced. */
__device__ void mul_mod_p(const uint32_t a[8], const uint32_t b[8], uint32_t c[8]) {
    uint32_t high[8] = { 0 };
    uint32_t t = a[7];

    /* row a[7] */
    #pragma unroll
    for (int i = 7; i >= 0; i--) c[i] = t * b[i];

    mad_hi_cc(c[6], t, b[7], c[6]);
    madc_hi_cc(c[5], t, b[6], c[5]);
    madc_hi_cc(c[4], t, b[5], c[4]);
    madc_hi_cc(c[3], t, b[4], c[3]);
    madc_hi_cc(c[2], t, b[3], c[2]);
    madc_hi_cc(c[1], t, b[2], c[1]);
    madc_hi_cc(c[0], t, b[1], c[0]);
    madc_hi(high[7], t, b[0], high[7]);

    /* row a[6] */
    t = a[6];
    mad_lo_cc(c[6], t, b[7], c[6]);
    madc_lo_cc(c[5], t, b[6], c[5]);
    madc_lo_cc(c[4], t, b[5], c[4]);
    madc_lo_cc(c[3], t, b[4], c[3]);
    madc_lo_cc(c[2], t, b[3], c[2]);
    madc_lo_cc(c[1], t, b[2], c[1]);
    madc_lo_cc(c[0], t, b[1], c[0]);
    madc_lo_cc(high[7], t, b[0], high[7]);
    addc(high[6], high[6], 0);
    mad_hi_cc(c[5], t, b[7], c[5]);
    madc_hi_cc(c[4], t, b[6], c[4]);
    madc_hi_cc(c[3], t, b[5], c[3]);
    madc_hi_cc(c[2], t, b[4], c[2]);
    madc_hi_cc(c[1], t, b[3], c[1]);
    madc_hi_cc(c[0], t, b[2], c[0]);
    madc_hi_cc(high[7], t, b[1], high[7]);
    madc_hi(high[6], t, b[0], high[6]);

    /* row a[5] */
    t = a[5];
    mad_lo_cc(c[5], t, b[7], c[5]);
    madc_lo_cc(c[4], t, b[6], c[4]);
    madc_lo_cc(c[3], t, b[5], c[3]);
    madc_lo_cc(c[2], t, b[4], c[2]);
    madc_lo_cc(c[1], t, b[3], c[1]);
    madc_lo_cc(c[0], t, b[2], c[0]);
    madc_lo_cc(high[7], t, b[1], high[7]);
    madc_lo_cc(high[6], t, b[0], high[6]);
    addc(high[5], high[5], 0);
    mad_hi_cc(c[4], t, b[7], c[4]);
    madc_hi_cc(c[3], t, b[6], c[3]);
    madc_hi_cc(c[2], t, b[5], c[2]);
    madc_hi_cc(c[1], t, b[4], c[1]);
    madc_hi_cc(c[0], t, b[3], c[0]);
    madc_hi_cc(high[7], t, b[2], high[7]);
    madc_hi_cc(high[6], t, b[1], high[6]);
    madc_hi(high[5], t, b[0], high[5]);

    /* row a[4] */
    t = a[4];
    mad_lo_cc(c[4], t, b[7], c[4]);
    madc_lo_cc(c[3], t, b[6], c[3]);
    madc_lo_cc(c[2], t, b[5], c[2]);
    madc_lo_cc(c[1], t, b[4], c[1]);
    madc_lo_cc(c[0], t, b[3], c[0]);
    madc_lo_cc(high[7], t, b[2], high[7]);
    madc_lo_cc(high[6], t, b[1], high[6]);
    madc_lo_cc(high[5], t, b[0], high[5]);
    addc(high[4], high[4], 0);
    mad_hi_cc(c[3], t, b[7], c[3]);
    madc_hi_cc(c[2], t, b[6], c[2]);
    madc_hi_cc(c[1], t, b[5], c[1]);
    madc_hi_cc(c[0], t, b[4], c[0]);
    madc_hi_cc(high[7], t, b[3], high[7]);
    madc_hi_cc(high[6], t, b[2], high[6]);
    madc_hi_cc(high[5], t, b[1], high[5]);
    madc_hi(high[4], t, b[0], high[4]);

    /* row a[3] */
    t = a[3];
    mad_lo_cc(c[3], t, b[7], c[3]);
    madc_lo_cc(c[2], t, b[6], c[2]);
    madc_lo_cc(c[1], t, b[5], c[1]);
    madc_lo_cc(c[0], t, b[4], c[0]);
    madc_lo_cc(high[7], t, b[3], high[7]);
    madc_lo_cc(high[6], t, b[2], high[6]);
    madc_lo_cc(high[5], t, b[1], high[5]);
    madc_lo_cc(high[4], t, b[0], high[4]);
    addc(high[3], high[3], 0);
    mad_hi_cc(c[2], t, b[7], c[2]);
    madc_hi_cc(c[1], t, b[6], c[1]);
    madc_hi_cc(c[0], t, b[5], c[0]);
    madc_hi_cc(high[7], t, b[4], high[7]);
    madc_hi_cc(high[6], t, b[3], high[6]);
    madc_hi_cc(high[5], t, b[2], high[5]);
    madc_hi_cc(high[4], t, b[1], high[4]);
    madc_hi(high[3], t, b[0], high[3]);

    /* row a[2] */
    t = a[2];
    mad_lo_cc(c[2], t, b[7], c[2]);
    madc_lo_cc(c[1], t, b[6], c[1]);
    madc_lo_cc(c[0], t, b[5], c[0]);
    madc_lo_cc(high[7], t, b[4], high[7]);
    madc_lo_cc(high[6], t, b[3], high[6]);
    madc_lo_cc(high[5], t, b[2], high[5]);
    madc_lo_cc(high[4], t, b[1], high[4]);
    madc_lo_cc(high[3], t, b[0], high[3]);
    addc(high[2], high[2], 0);
    mad_hi_cc(c[1], t, b[7], c[1]);
    madc_hi_cc(c[0], t, b[6], c[0]);
    madc_hi_cc(high[7], t, b[5], high[7]);
    madc_hi_cc(high[6], t, b[4], high[6]);
    madc_hi_cc(high[5], t, b[3], high[5]);
    madc_hi_cc(high[4], t, b[2], high[4]);
    madc_hi_cc(high[3], t, b[1], high[3]);
    madc_hi(high[2], t, b[0], high[2]);

    /* row a[1] */
    t = a[1];
    mad_lo_cc(c[1], t, b[7], c[1]);
    madc_lo_cc(c[0], t, b[6], c[0]);
    madc_lo_cc(high[7], t, b[5], high[7]);
    madc_lo_cc(high[6], t, b[4], high[6]);
    madc_lo_cc(high[5], t, b[3], high[5]);
    madc_lo_cc(high[4], t, b[2], high[4]);
    madc_lo_cc(high[3], t, b[1], high[3]);
    madc_lo_cc(high[2], t, b[0], high[2]);
    addc(high[1], high[1], 0);
    mad_hi_cc(c[0], t, b[7], c[0]);
    madc_hi_cc(high[7], t, b[6], high[7]);
    madc_hi_cc(high[6], t, b[5], high[6]);
    madc_hi_cc(high[5], t, b[4], high[5]);
    madc_hi_cc(high[4], t, b[3], high[4]);
    madc_hi_cc(high[3], t, b[2], high[3]);
    madc_hi_cc(high[2], t, b[1], high[2]);
    madc_hi(high[1], t, b[0], high[1]);

    /* row a[0] */
    t = a[0];
    mad_lo_cc(c[0], t, b[7], c[0]);
    madc_lo_cc(high[7], t, b[6], high[7]);
    madc_lo_cc(high[6], t, b[5], high[6]);
    madc_lo_cc(high[5], t, b[4], high[5]);
    madc_lo_cc(high[4], t, b[3], high[4]);
    madc_lo_cc(high[3], t, b[2], high[3]);
    madc_lo_cc(high[2], t, b[1], high[2]);
    madc_lo_cc(high[1], t, b[0], high[1]);
    addc(high[0], high[0], 0);
    mad_hi_cc(high[7], t, b[7], high[7]);
    madc_hi_cc(high[6], t, b[6], high[6]);
    madc_hi_cc(high[5], t, b[5], high[5]);
    madc_hi_cc(high[4], t, b[4], high[4]);
    madc_hi_cc(high[3], t, b[3], high[3]);
    madc_hi_cc(high[2], t, b[2], high[2]);
    madc_hi_cc(high[1], t, b[1], high[1]);
    madc_hi(high[0], t, b[0], high[0]);

    /* high[0..7] and c[0..7] now hold a 512-bit product (high << 256 + c).
     * Reduce mod P = 2^256 - 2^32 - 977.
     * P = 2^256 - s where s = 2^32 + 977. So 2^256 == s (mod P).
     * high * 2^256 == high * s (mod P).
     * That is: high * (2^32 + 977) added to c.
     * Following BitCrack: shift high left by 32 bits and add, then add high * 977. */
    const uint32_t s = 977;
    uint32_t high7 = high[7];
    uint32_t high6 = high[6];

    add_cc(c[6], high[7], c[6]);
    addc_cc(c[5], high[6], c[5]);
    addc_cc(c[4], high[5], c[4]);
    addc_cc(c[3], high[4], c[3]);
    addc_cc(c[2], high[3], c[2]);
    addc_cc(c[1], high[2], c[1]);
    addc_cc(c[0], high[1], c[0]);
    addc_cc(high[7], high[0], 0);
    addc(high[6], 0, 0);

    mad_lo_cc(c[7], high7, s, c[7]);
    madc_lo_cc(c[6], high6, s, c[6]);
    madc_lo_cc(c[5], high[5], s, c[5]);
    madc_lo_cc(c[4], high[4], s, c[4]);
    madc_lo_cc(c[3], high[3], s, c[3]);
    madc_lo_cc(c[2], high[2], s, c[2]);
    madc_lo_cc(c[1], high[1], s, c[1]);
    madc_lo_cc(c[0], high[0], s, c[0]);
    addc(high[7], high[7], 0);
    /* high[6] does not increase further because high[0..7]*s only spills one extra word into high[7]. */

    mad_hi_cc(c[6], high7, s, c[6]);
    madc_hi_cc(c[5], high6, s, c[5]);
    madc_hi_cc(c[4], high[5], s, c[4]);
    madc_hi_cc(c[3], high[4], s, c[3]);
    madc_hi_cc(c[2], high[3], s, c[2]);
    madc_hi_cc(c[1], high[2], s, c[1]);
    madc_hi_cc(c[0], high[1], s, c[0]);
    madc_hi_cc(high[7], high[0], s, high[7]);
    addc(high[6], high[6], 0);

    /* Now high[6..7] || c[0..7] is at most ~290 bits. Reduce one more round:
     * extra = high[7] (low word) + (high[6] << 32). Multiply by s and add to c. */
    uint32_t hh = high[6];
    uint32_t hl = high[7];

    /* shift hh by 32 -> add hl in upper slot of c */
    add_cc(c[6], hl, c[6]);
    addc_cc(c[5], hh, c[5]);
    addc_cc(c[4], 0, c[4]);
    addc_cc(c[3], 0, c[3]);
    addc_cc(c[2], 0, c[2]);
    addc_cc(c[1], 0, c[1]);
    addc(c[0], c[0], 0);

    /* multiply low by s and add */
    mad_lo_cc(c[7], hl, s, c[7]);
    madc_lo_cc(c[6], hh, s, c[6]);
    madc_lo_cc(c[5], 0, s, c[5]);
    madc_lo_cc(c[4], 0, s, c[4]);
    madc_lo_cc(c[3], 0, s, c[3]);
    madc_lo_cc(c[2], 0, s, c[2]);
    madc_lo_cc(c[1], 0, s, c[1]);
    madc_lo(c[0], 0, s, c[0]);

    mad_hi_cc(c[6], hl, s, c[6]);
    madc_hi_cc(c[5], hh, s, c[5]);
    madc_hi_cc(c[4], 0, s, c[4]);
    madc_hi_cc(c[3], 0, s, c[3]);
    madc_hi_cc(c[2], 0, s, c[2]);
    madc_hi_cc(c[1], 0, s, c[1]);
    madc_hi(c[0], 0, s, c[0]);

    /* Final reduction: c may be >= P. Subtract P if so. */
    bool gt = false;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (c[i] > SECP_P[i]) { gt = true; break; }
        if (c[i] < SECP_P[i]) break;
    }
    if (gt) {
        sub_cc(c[7], c[7], SECP_P[7]);
        subc_cc(c[6], c[6], SECP_P[6]);
        subc_cc(c[5], c[5], SECP_P[5]);
        subc_cc(c[4], c[4], SECP_P[4]);
        subc_cc(c[3], c[3], SECP_P[3]);
        subc_cc(c[2], c[2], SECP_P[2]);
        subc_cc(c[1], c[1], SECP_P[1]);
        subc(c[0], c[0], SECP_P[0]);
    }
}

__device__ __forceinline__ void sqr_mod_p(const uint32_t a[8], uint32_t c[8]) {
    /* mul_mod_p writes to c as it computes, so the output buffer must not alias the input.
     * Use a temporary to permit sqr_mod_p(x, x). */
    uint32_t tmp[8];
    mul_mod_p(a, a, tmp);
    big_copy(c, tmp);
}

__device__ __forceinline__ void mul_mod_p_inplace(uint32_t a[8], const uint32_t b[8]) {
    uint32_t tmp[8];
    mul_mod_p(a, b, tmp);
    big_copy(a, tmp);
}

/* Modular inverse via Fermat: a^(P-2) mod P. Uses addition chain optimized for secp256k1.
 * Sourced conceptually from BitCrack's invModP (MIT). */
/* Modular inverse via Fermat: x^(P-2) mod P. Plain left-to-right square-and-multiply
 * over the 256-bit exponent (P - 2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D).
 * A specialised addition chain would cut ~249 multiplies to ~14, but the extra temporary
 * arrays it needs lowered SM occupancy on Blackwell and made the kernel slower overall. */
__device__ void inv_mod_p(uint32_t value[8]) {
    static const uint32_t e[8] = {
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFEu, 0xFFFFFC2Du
    };
    uint32_t result[8];
    big_copy(result, value);
    bool started = false;
    for (int w = 0; w < 8; w++) {
        uint32_t word = e[w];
        for (int bi = 31; bi >= 0; bi--) {
            if (!started) {
                if ((word >> bi) & 1u) started = true;
                continue;
            }
            sqr_mod_p(result, result);
            if ((word >> bi) & 1u) mul_mod_p_inplace(result, value);
        }
    }
    big_copy(value, result);
}

/* =========================================================================
 * Big-int arithmetic modulo N (group order).
 * Used for BIP32 child priv key: child = (IL + parent) mod N.
 * ========================================================================= */

__device__ void add_mod_n(const uint32_t a[8], const uint32_t b[8], uint32_t c[8]) {
    add_cc(c[7], a[7], b[7]);
    addc_cc(c[6], a[6], b[6]);
    addc_cc(c[5], a[5], b[5]);
    addc_cc(c[4], a[4], b[4]);
    addc_cc(c[3], a[3], b[3]);
    addc_cc(c[2], a[2], b[2]);
    addc_cc(c[1], a[1], b[1]);
    addc_cc(c[0], a[0], b[0]);
    uint32_t carry = 0;
    addc(carry, 0, 0);
    if (carry || big_ge(c, SECP_N)) {
        sub_cc(c[7], c[7], SECP_N[7]);
        subc_cc(c[6], c[6], SECP_N[6]);
        subc_cc(c[5], c[5], SECP_N[5]);
        subc_cc(c[4], c[4], SECP_N[4]);
        subc_cc(c[3], c[3], SECP_N[3]);
        subc_cc(c[2], c[2], SECP_N[2]);
        subc_cc(c[1], c[1], SECP_N[1]);
        subc(c[0], c[0], SECP_N[0]);
    }
}

/* =========================================================================
 * secp256k1 point operations in Jacobian coordinates.
 * Point (X, Y, Z) corresponds to affine (X/Z^2, Y/Z^3). Infinity is Z == 0.
 * ========================================================================= */

__device__ bool jac_is_infinity(const uint32_t Z[8]) {
    return big_is_zero(Z);
}

/* Doubling: P3 = 2 * P1.
 * Standard formula for a=0 curve (secp256k1):
 *   A = X1^2; B = Y1^2; C = B^2
 *   D = 2 * ((X1 + B)^2 - A - C)
 *   E = 3 * A
 *   F = E^2
 *   X3 = F - 2*D
 *   Y3 = E*(D - X3) - 8*C
 *   Z3 = 2*Y1*Z1 */
__device__ void point_double(
    const uint32_t X1[8], const uint32_t Y1[8], const uint32_t Z1[8],
    uint32_t X3[8], uint32_t Y3[8], uint32_t Z3[8]
) {
    if (jac_is_infinity(Z1)) {
        for (int i = 0; i < 8; i++) { X3[i] = 0; Y3[i] = 0; Z3[i] = 0; }
        return;
    }
    if (big_is_zero(Y1)) {
        /* Tangent vertical -> 2P = infinity */
        for (int i = 0; i < 8; i++) { X3[i] = 0; Y3[i] = 0; Z3[i] = 0; }
        return;
    }

    uint32_t A[8], B[8], C[8], D[8], E[8], F[8], T[8];

    sqr_mod_p(X1, A);                 /* A = X1^2 */
    sqr_mod_p(Y1, B);                 /* B = Y1^2 */
    sqr_mod_p(B, C);                  /* C = B^2 */

    add_mod_p(X1, B, T);              /* T = X1 + B */
    sqr_mod_p(T, T);                  /* T = (X1+B)^2 */
    sub_mod_p(T, A, T);
    sub_mod_p(T, C, T);
    add_mod_p(T, T, D);               /* D = 2 * ((X1+B)^2 - A - C) */

    add_mod_p(A, A, E);
    add_mod_p(E, A, E);               /* E = 3*A */
    sqr_mod_p(E, F);                  /* F = E^2 */

    add_mod_p(D, D, T);
    sub_mod_p(F, T, X3);              /* X3 = F - 2D */

    sub_mod_p(D, X3, T);              /* T = D - X3 */
    mul_mod_p_inplace(T, E);          /* T = E*(D - X3) */
    add_mod_p(C, C, C);
    add_mod_p(C, C, C);
    add_mod_p(C, C, C);               /* C = 8*C */
    sub_mod_p(T, C, Y3);              /* Y3 = E*(D-X3) - 8C */

    mul_mod_p(Y1, Z1, T);             /* T = Y1*Z1 */
    add_mod_p(T, T, Z3);              /* Z3 = 2*Y1*Z1 */
}

/* Mixed addition: P3 = P1 + P2, where P1 = (X1,Y1,Z1) Jacobian, P2 = (X2,Y2) affine (Z=1). */
__device__ void point_add_mixed(
    const uint32_t X1[8], const uint32_t Y1[8], const uint32_t Z1[8],
    const uint32_t X2[8], const uint32_t Y2[8],
    uint32_t X3[8], uint32_t Y3[8], uint32_t Z3[8]
) {
    if (jac_is_infinity(Z1)) {
        big_copy(X3, X2);
        big_copy(Y3, Y2);
        /* Z3 = 1 */
        for (int i = 0; i < 7; i++) Z3[i] = 0;
        Z3[7] = 1;
        return;
    }

    uint32_t Z1Z1[8], U2[8], S2[8], H[8], HH[8], HHH[8], r[8], V[8], rr[8];

    sqr_mod_p(Z1, Z1Z1);              /* Z1Z1 = Z1^2 */
    mul_mod_p(X2, Z1Z1, U2);          /* U2 = X2 * Z1^2 */
    uint32_t Z1Z1Z1[8];
    mul_mod_p(Z1, Z1Z1, Z1Z1Z1);      /* Z1^3 */
    mul_mod_p(Y2, Z1Z1Z1, S2);        /* S2 = Y2 * Z1^3 */

    if (big_eq(U2, X1)) {
        if (big_eq(S2, Y1)) {
            /* Same point - double */
            point_double(X1, Y1, Z1, X3, Y3, Z3);
            return;
        }
        /* P1 = -P2 -> infinity */
        for (int i = 0; i < 8; i++) { X3[i] = 0; Y3[i] = 0; Z3[i] = 0; }
        return;
    }

    sub_mod_p(U2, X1, H);             /* H = U2 - X1 */
    sqr_mod_p(H, HH);                 /* HH = H^2 */
    mul_mod_p(H, HH, HHH);            /* HHH = H^3 */
    sub_mod_p(S2, Y1, r);             /* r = S2 - Y1 */
    mul_mod_p(X1, HH, V);             /* V = X1 * HH */

    sqr_mod_p(r, rr);                 /* rr = r^2 */
    uint32_t T[8];
    add_mod_p(V, V, T);               /* T = 2V */
    sub_mod_p(rr, HHH, X3);
    sub_mod_p(X3, T, X3);             /* X3 = r^2 - HHH - 2V */

    sub_mod_p(V, X3, T);              /* T = V - X3 */
    mul_mod_p_inplace(T, r);          /* T = r*(V - X3) */
    uint32_t Y1HHH[8];
    mul_mod_p(Y1, HHH, Y1HHH);        /* Y1 * H^3 */
    sub_mod_p(T, Y1HHH, Y3);          /* Y3 = r*(V - X3) - Y1*HHH */

    mul_mod_p(Z1, H, Z3);             /* Z3 = Z1 * H */
}

/* Convert Jacobian (X, Y, Z) to affine (x, y) by dividing by Z^2 and Z^3.
 * Z must be non-zero (i.e. point != infinity). */
__device__ void jac_to_affine(
    const uint32_t X[8], const uint32_t Y[8], const uint32_t Z[8],
    uint32_t x_aff[8], uint32_t y_aff[8]
) {
    uint32_t Zinv[8], Zinv2[8], Zinv3[8];
    big_copy(Zinv, Z);
    inv_mod_p(Zinv);
    sqr_mod_p(Zinv, Zinv2);
    mul_mod_p(Zinv2, Zinv, Zinv3);
    mul_mod_p(X, Zinv2, x_aff);
    mul_mod_p(Y, Zinv3, y_aff);
}

/* Forward declarations for helpers defined later in this file. */
__device__ __forceinline__ void bytes_to_words(const uint8_t b[32], uint32_t x[8]);
__device__ __forceinline__ void words_to_bytes(const uint32_t x[8], uint8_t b[32]);

/* Windowed scalar multiplication k * G using a precomputed table.
 *
 * The host pre-builds a 64*15*16-word table where entry (i, j-1) holds the affine (X, Y)
 * of (j * 16^i) * G for window index i in [0, 64) and window value j in [1, 15]. Each
 * entry is 16 aligned u32 words: 8 X words followed by 8 Y words.
 *
 * For a 256-bit scalar k, the kernel scans 64 4-bit windows from LSB to MSB and adds
 * the matching precomputed point to a running Jacobian accumulator. This replaces 256
 * doublings + ~64 additions with 64 lookups + 64 additions per scalar mult.
 */
__device__ void scalar_mul_g_table(
    const uint32_t k[8],
    const uint32_t* d_g_table,
    uint32_t x_aff[8],
    uint32_t y_aff[8]
) {
    uint32_t X[8] = {0}, Y[8] = {0}, Z[8] = {0};
    for (int i = 0; i < 64; i++) {
        /* word_idx maps window index (LSB first) into the big-endian-word storage of k. */
        int word_idx = 7 - (i >> 3);
        int bit_offset = (i & 7) * 4;
        uint32_t window = (k[word_idx] >> bit_offset) & 0xFu;
        if (window == 0) continue;

        int entry_offset = (i * 15 + (int)(window - 1)) * 16;
        const uint32_t* entry = d_g_table + entry_offset;
        uint32_t tx[8], ty[8];
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            tx[w] = entry[w];
            ty[w] = entry[8 + w];
        }

        uint32_t Xn[8], Yn[8], Zn[8];
        point_add_mixed(X, Y, Z, tx, ty, Xn, Yn, Zn);
        big_copy(X, Xn); big_copy(Y, Yn); big_copy(Z, Zn);
    }
    jac_to_affine(X, Y, Z, x_aff, y_aff);
}

/* =========================================================================
 * Byte <-> 8-word big-int conversion. 32 bytes big-endian <-> 8 u32 (x[0] MSW).
 * ========================================================================= */

__device__ __forceinline__ void bytes_to_words(const uint8_t b[32], uint32_t x[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x[i] = ((uint32_t)b[i*4 + 0] << 24) |
               ((uint32_t)b[i*4 + 1] << 16) |
               ((uint32_t)b[i*4 + 2] <<  8) |
               ((uint32_t)b[i*4 + 3]);
    }
}

__device__ __forceinline__ void words_to_bytes(const uint32_t x[8], uint8_t b[32]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        b[i*4 + 0] = (uint8_t)(x[i] >> 24);
        b[i*4 + 1] = (uint8_t)(x[i] >> 16);
        b[i*4 + 2] = (uint8_t)(x[i] >>  8);
        b[i*4 + 3] = (uint8_t)(x[i]);
    }
}

/* =========================================================================
 * BIP32 derivation.
 * One step: priv, chain (in/out) updated by a single child index.
 * index >= 0x80000000 => hardened (uses priv); else => non-hardened (uses pubkey).
 * ========================================================================= */

__device__ void bip32_ckd_table(uint8_t priv[32], uint8_t chain[32], uint32_t index, const uint32_t* d_g_table) {
    uint8_t msg[37];
    if (index & 0x80000000u) {
        msg[0] = 0x00;
        for (int i = 0; i < 32; i++) msg[1 + i] = priv[i];
    } else {
        uint32_t k[8];
        bytes_to_words(priv, k);
        uint32_t Px[8], Py[8];
        scalar_mul_g_table(k, d_g_table, Px, Py);
        msg[0] = (uint8_t)(0x02 | (Py[7] & 1u));
        uint8_t Xb[32];
        words_to_bytes(Px, Xb);
        for (int i = 0; i < 32; i++) msg[1 + i] = Xb[i];
    }
    msg[33] = (uint8_t)((index >> 24) & 0xffu);
    msg[34] = (uint8_t)((index >> 16) & 0xffu);
    msg[35] = (uint8_t)((index >>  8) & 0xffu);
    msg[36] = (uint8_t)(index & 0xffu);

    uint8_t I[64];
    hmac_sha512(chain, 32, msg, 37, I);

    uint32_t IL_w[8], priv_w[8], new_priv_w[8];
    bytes_to_words(I, IL_w);
    bytes_to_words(priv, priv_w);
    add_mod_n(IL_w, priv_w, new_priv_w);
    words_to_bytes(new_priv_w, priv);
    for (int i = 0; i < 32; i++) chain[i] = I[32 + i];
}

/* =========================================================================
 * GPU-side candidate enumeration.
 *
 * Eliminates CPU candidate generation entirely. Each GPU thread is assigned a
 * 64-bit candidate index; it decodes the missing-word values from the index,
 * computes/validates the BIP39 checksum, builds the mnemonic byte buffer using
 * the embedded wordlist, and runs the full PBKDF2 + BIP32 + secp + hash160
 * + compare pipeline.
 *
 * Wordlist format (passed in d_wordlist):
 *   2048 entries x 12 bytes: [u8 length][u8 chars[8]][3 bytes padding]
 *
 * Candidate index layout for the user-facing missing-word cases:
 *   - last word missing: low bits = (11 - checksum_bits) entropy bits of last word
 *     remaining = sequence of 11-bit free-slot indices (in missing_positions order)
 *   - last word NOT missing: pure sequence of 11-bit free-slot indices; the kernel
 *     checks the BIP39 checksum and returns immediately on mismatch
 * ========================================================================= */

__device__ void sha256_entropy_block(const uint8_t* data, int data_len, uint32_t digest[8]) {
    /* SHA-256 of a short input (<= 55 bytes) into a single 64-byte block. */
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0_256[i];
    uint8_t blk[64];
    for (int i = 0; i < data_len; i++) blk[i] = data[i];
    blk[data_len] = 0x80;
    for (int i = data_len + 1; i < 56; i++) blk[i] = 0;
    uint64_t bit_len = (uint64_t)data_len * 8ULL;
    for (int i = 0; i < 8; i++) blk[56 + i] = (uint8_t)(bit_len >> (56 - i*8));
    sha256_compress(state, blk);
    #pragma unroll
    for (int i = 0; i < 8; i++) digest[i] = state[i];
}

extern "C" __global__ void recovery_enumerate(
    const uint16_t* __restrict__ known_indices,    /* 24 entries, first mnemonic_length used */
    int mnemonic_length,
    int checksum_bits,
    const uint8_t* __restrict__ missing_positions, /* up to 3 0-indexed positions */
    int missing_count,
    int last_is_missing,
    unsigned long long chunk_offset,
    unsigned long long chunk_size,
    const uint8_t* __restrict__ salt,
    int salt_len,
    int iterations,
    const uint32_t* __restrict__ path_indices,
    int path_len,
    const uint8_t* __restrict__ target_hash160,
    const uint8_t* __restrict__ d_wordlist,        /* 2048 * 12 bytes */
    const uint32_t* __restrict__ d_g_table,         /* 64 * 15 * 16 u32 precomputed G multiples */
    long long* __restrict__ d_match_idx            /* output: -1 or absolute cand index */
) {
    /* The packed BIP39 table is only 24 KiB and is read once per candidate before
     * the long PBKDF2 loop. Keeping a 24 KiB copy per block caps SM86 residency, so
     * read it directly from global memory and let L1/L2 cache it. */
    unsigned long long tid_u = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + (unsigned long long)threadIdx.x;
    if (tid_u >= chunk_size) return;
    if (*d_match_idx != -1LL) return;

    unsigned long long cand_idx = chunk_offset + tid_u;

    /* Initialize all 24 slots from known_indices; missing positions will be overwritten. */
    uint16_t indices[24];
    #pragma unroll
    for (int i = 0; i < 24; i++) indices[i] = known_indices[i];

    int last_pos = mnemonic_length - 1;
    int missing_entropy_bits = 11 - checksum_bits;
    int checksum_mask = (1 << checksum_bits) - 1;

    /* Collect the "free" missing positions (excluding the last-word position if it is missing). */
    uint8_t free_pos[3];
    int free_count = 0;
    for (int i = 0; i < missing_count; i++) {
        int p = missing_positions[i];
        if (last_is_missing && p == last_pos) continue;
        free_pos[free_count++] = (uint8_t)p;
    }

    /* Decode candidate index into the missing word values. */
    unsigned long long remaining = cand_idx;
    uint16_t last_entropy = 0;
    if (last_is_missing) {
        last_entropy = (uint16_t)(remaining & (((unsigned long long)1 << missing_entropy_bits) - 1ULL));
        remaining >>= missing_entropy_bits;
    }
    for (int i = 0; i < free_count; i++) {
        uint16_t w = (uint16_t)(remaining & 0x7FFULL);
        indices[free_pos[i]] = w;
        remaining >>= 11;
    }

    /* Either compute the valid last word from the BIP39 checksum, or validate the
     * existing checksum (for the no-last-word-missing case) and bail on mismatch. */
    int total_entropy_bits = mnemonic_length * 11 - checksum_bits;
    int entropy_bytes_len = total_entropy_bits / 8;
    uint8_t entropy[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) entropy[i] = 0;
    int bit_ptr = 0;
    /* Lay down the bits of all known + free words (everything except the last word). */
    for (int i = 0; i < mnemonic_length - 1; i++) {
        uint16_t idx = indices[i];
        for (int b = 10; b >= 0; b--) {
            int bit = (idx >> b) & 1;
            if (bit) entropy[bit_ptr / 8] |= 1 << (7 - (bit_ptr % 8));
            bit_ptr++;
        }
    }
    if (last_is_missing) {
        /* Append the trailing entropy bits of the last word and SHA-256 to derive checksum. */
        for (int b = missing_entropy_bits - 1; b >= 0; b--) {
            int bit = (last_entropy >> b) & 1;
            if (bit) entropy[bit_ptr / 8] |= 1 << (7 - (bit_ptr % 8));
            bit_ptr++;
        }
        uint32_t sha[8];
        sha256_entropy_block(entropy, entropy_bytes_len, sha);
        uint8_t cs = (uint8_t)((sha[0] >> (32 - checksum_bits)) & checksum_mask);
        indices[last_pos] = (uint16_t)((last_entropy << checksum_bits) | cs);
    } else {
        /* Last word is known/decoded; lay down its 11 bits and compare checksum. */
        uint16_t idx = indices[last_pos];
        for (int b = 10; b >= 0; b--) {
            int bit = (idx >> b) & 1;
            if (bit_ptr < total_entropy_bits) {
                if (bit) entropy[bit_ptr / 8] |= 1 << (7 - (bit_ptr % 8));
            }
            bit_ptr++;
        }
        uint32_t sha[8];
        sha256_entropy_block(entropy, entropy_bytes_len, sha);
        uint8_t calc_cs = (uint8_t)((sha[0] >> (32 - checksum_bits)) & checksum_mask);
        uint8_t actual_cs = (uint8_t)(idx & checksum_mask);
        if (calc_cs != actual_cs) return;
    }

    /* Build the mnemonic byte buffer directly from the cached global wordlist. */
    uint8_t mnemonic[256];
    int cursor = 0;
    for (int i = 0; i < mnemonic_length; i++) {
        uint16_t widx = indices[i];
        int wl_off = (int)widx * 12;
        int wlen = d_wordlist[wl_off];
        for (int c = 0; c < wlen; c++) mnemonic[cursor++] = d_wordlist[wl_off + 1 + c];
        if (i < mnemonic_length - 1) mnemonic[cursor++] = (uint8_t)' ';
    }
    int mnemonic_len = cursor;

    /* Run the pipeline. */
    uint8_t seed[64];
    pbkdf2_hmac_sha512_block(mnemonic, mnemonic_len, salt, salt_len, iterations, seed);

    uint8_t I[64];
    {
        const uint8_t key[12] = { 'B','i','t','c','o','i','n',' ','s','e','e','d' };
        hmac_sha512(key, 12, seed, 64, I);
    }
    uint8_t priv[32], chain[32];
    for (int i = 0; i < 32; i++) { priv[i] = I[i]; chain[i] = I[32 + i]; }

    for (int i = 0; i < path_len; i++) {
        bip32_ckd_table(priv, chain, path_indices[i], d_g_table);
    }

    uint32_t k[8], Px[8], Py[8];
    bytes_to_words(priv, k);
    scalar_mul_g_table(k, d_g_table, Px, Py);

    uint8_t pubkey33[33];
    pubkey33[0] = (uint8_t)(0x02 | (Py[7] & 1u));
    {
        uint8_t Xb[32];
        words_to_bytes(Px, Xb);
        for (int i = 0; i < 32; i++) pubkey33[1 + i] = Xb[i];
    }

    uint32_t sha2[8];
    sha256_compressed_pubkey(pubkey33, sha2);
    uint8_t hash160[20];
    ripemd160_of_sha256(sha2, hash160);

    bool match = true;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        if (hash160[i] != target_hash160[i]) { match = false; break; }
    }

    if (match) {
        atomicCAS((unsigned long long*)d_match_idx, (unsigned long long)-1LL, (unsigned long long)cand_idx);
    }
}
