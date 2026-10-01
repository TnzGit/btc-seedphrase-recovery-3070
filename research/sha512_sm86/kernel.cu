typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

__device__ __constant__ uint64_t K512_64[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
};

struct Pair32 { uint32_t lo, hi; };

__device__ __constant__ Pair32 K512_32[80] = {
    {0xd728ae22u,0x428a2f98u},{0x23ef65cdu,0x71374491u},{0xec4d3b2fu,0xb5c0fbcfu},{0x8189dbbcu,0xe9b5dba5u},
    {0xf348b538u,0x3956c25bu},{0xb605d019u,0x59f111f1u},{0xaf194f9bu,0x923f82a4u},{0xda6d8118u,0xab1c5ed5u},
    {0xa3030242u,0xd807aa98u},{0x45706fbeu,0x12835b01u},{0x4ee4b28cu,0x243185beu},{0xd5ffb4e2u,0x550c7dc3u},
    {0xf27b896fu,0x72be5d74u},{0x3b1696b1u,0x80deb1feu},{0x25c71235u,0x9bdc06a7u},{0xcf692694u,0xc19bf174u},
    {0x9ef14ad2u,0xe49b69c1u},{0x384f25e3u,0xefbe4786u},{0x8b8cd5b5u,0x0fc19dc6u},{0x77ac9c65u,0x240ca1ccu},
    {0x592b0275u,0x2de92c6fu},{0x6ea6e483u,0x4a7484aau},{0xbd41fbd4u,0x5cb0a9dcu},{0x831153b5u,0x76f988dau},
    {0xee66dfabu,0x983e5152u},{0x2db43210u,0xa831c66du},{0x98fb213fu,0xb00327c8u},{0xbeef0ee4u,0xbf597fc7u},
    {0x3da88fc2u,0xc6e00bf3u},{0x930aa725u,0xd5a79147u},{0xe003826fu,0x06ca6351u},{0x0a0e6e70u,0x14292967u},
    {0x46d22ffcu,0x27b70a85u},{0x5c26c926u,0x2e1b2138u},{0x5ac42aedu,0x4d2c6dfcu},{0x9d95b3dfu,0x53380d13u},
    {0x8baf63deu,0x650a7354u},{0x3c77b2a8u,0x766a0abbu},{0x47edaee6u,0x81c2c92eu},{0x1482353bu,0x92722c85u},
    {0x4cf10364u,0xa2bfe8a1u},{0xbc423001u,0xa81a664bu},{0xd0f89791u,0xc24b8b70u},{0x0654be30u,0xc76c51a3u},
    {0xd6ef5218u,0xd192e819u},{0x5565a910u,0xd6990624u},{0x5771202au,0xf40e3585u},{0x32bbd1b8u,0x106aa070u},
    {0xb8d2d0c8u,0x19a4c116u},{0x5141ab53u,0x1e376c08u},{0xdf8eeb99u,0x2748774cu},{0xe19b48a8u,0x34b0bcb5u},
    {0xc5c95a63u,0x391c0cb3u},{0xe3418acbu,0x4ed8aa4au},{0x7763e373u,0x5b9cca4fu},{0xd6b2b8a3u,0x682e6ff3u},
    {0x5defb2fcu,0x748f82eeu},{0x43172f60u,0x78a5636fu},{0xa1f0ab72u,0x84c87814u},{0x1a6439ecu,0x8cc70208u},
    {0x23631e28u,0x90befffau},{0xde82bde9u,0xa4506cebu},{0xb2c67915u,0xbef9a3f7u},{0xe372532bu,0xc67178f2u},
    {0xea26619cu,0xca273eceu},{0x21c0c207u,0xd186b8c7u},{0xcde0eb1eu,0xeada7dd6u},{0xee6ed178u,0xf57d4f7fu},
    {0x72176fbau,0x06f067aau},{0xa2c898a6u,0x0a637dc5u},{0xbef90daeu,0x113f9804u},{0x131c471bu,0x1b710b35u},
    {0x23047d84u,0x28db77f5u},{0x40c72493u,0x32caab7bu},{0x15c9bebcu,0x3c9ebe0au},{0x9c100d4cu,0x431d67c4u},
    {0xcb3e42b6u,0x4cc5d4beu},{0xfc657e2au,0x597f299cu},{0x3ad6faecu,0x5fcb6fabu},{0x4a475817u,0x6c44198cu}
};

#define ROTR64(x,n) (((x) >> (n)) | ((x) << (64-(n))))
#define CH64(x,y,z) (((x)&(y)) ^ (~(x)&(z)))
#define MAJ64(x,y,z) (((x)&(y)) ^ ((x)&(z)) ^ ((y)&(z)))
#define BS0(x) (ROTR64((x),28) ^ ROTR64((x),34) ^ ROTR64((x),39))
#define BS1(x) (ROTR64((x),14) ^ ROTR64((x),18) ^ ROTR64((x),41))
#define SS0(x) (ROTR64((x),1) ^ ROTR64((x),8) ^ ((x)>>7))
#define SS1(x) (ROTR64((x),19) ^ ROTR64((x),61) ^ ((x)>>6))

__device__ __forceinline__ Pair32 pxor(Pair32 a, Pair32 b) {
    Pair32 r={a.lo^b.lo,a.hi^b.hi}; return r;
}
__device__ __forceinline__ Pair32 pand(Pair32 a, Pair32 b) {
    Pair32 r={a.lo&b.lo,a.hi&b.hi}; return r;
}
__device__ __forceinline__ Pair32 pnot(Pair32 a) {
    Pair32 r={~a.lo,~a.hi}; return r;
}
__device__ __forceinline__ Pair32 padd(Pair32 a, Pair32 b) {
    Pair32 r;
    asm volatile(
        "add.cc.u32 %0, %2, %4;\n\t"
        "addc.u32 %1, %3, %5;"
        : "=r"(r.lo), "=r"(r.hi)
        : "r"(a.lo), "r"(a.hi), "r"(b.lo), "r"(b.hi)
    );
    return r;
}
__device__ __forceinline__ Pair32 padd4(Pair32 a, Pair32 b, Pair32 c, Pair32 d) {
    return padd(padd(a,b),padd(c,d));
}
__device__ __forceinline__ Pair32 padd5(Pair32 a, Pair32 b, Pair32 c, Pair32 d, Pair32 e) {
    return padd(padd4(a,b,c,d),e);
}
__device__ __forceinline__ Pair32 prot(Pair32 x, int n) {
    Pair32 r;
    if (n < 32) {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.lo) : "r"(x.lo), "r"(x.hi), "r"(n));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.hi) : "r"(x.hi), "r"(x.lo), "r"(n));
    } else {
        int m=n-32;
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.lo) : "r"(x.hi), "r"(x.lo), "r"(m));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.hi) : "r"(x.lo), "r"(x.hi), "r"(m));
    }
    return r;
}
__device__ __forceinline__ Pair32 pshr(Pair32 x, int n) {
    Pair32 r;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r.lo) : "r"(x.lo), "r"(x.hi), "r"(n));
    r.hi=x.hi>>n;
    return r;
}
__device__ __forceinline__ Pair32 pch(Pair32 x, Pair32 y, Pair32 z) {
    return pxor(pand(x,y),pand(pnot(x),z));
}
__device__ __forceinline__ Pair32 pmaj(Pair32 x, Pair32 y, Pair32 z) {
    return pxor(pxor(pand(x,y),pand(x,z)),pand(y,z));
}
__device__ __forceinline__ Pair32 pBS0(Pair32 x) {
    return pxor(pxor(prot(x,28),prot(x,34)),prot(x,39));
}
__device__ __forceinline__ Pair32 pBS1(Pair32 x) {
    return pxor(pxor(prot(x,14),prot(x,18)),prot(x,41));
}
__device__ __forceinline__ Pair32 pSS0(Pair32 x) {
    return pxor(pxor(prot(x,1),prot(x,8)),pshr(x,7));
}
__device__ __forceinline__ Pair32 pSS1(Pair32 x) {
    return pxor(pxor(prot(x,19),prot(x,61)),pshr(x,6));
}
__device__ __forceinline__ Pair32 from64(uint64_t x) {
    Pair32 r={(uint32_t)x,(uint32_t)(x>>32)}; return r;
}
__device__ __forceinline__ uint64_t to64(Pair32 x) {
    return ((uint64_t)x.hi<<32)|(uint64_t)x.lo;
}

__device__ __forceinline__ void sha512_compress_u64(uint64_t s[8], const uint64_t msg[16]) {
    uint64_t w[16];
    #pragma unroll
    for (int i=0;i<16;i++) w[i]=msg[i];
    uint64_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    #pragma unroll 80
    for (int i=0;i<80;i++) {
        int j=i&15;
        uint64_t wi;
        if (i<16) {
            wi=w[j];
        } else {
            wi=SS1(w[(j+14)&15])+w[(j+9)&15]+SS0(w[(j+1)&15])+w[j];
            w[j]=wi;
        }
        uint64_t t1=h+BS1(e)+CH64(e,f,g)+K512_64[i]+wi;
        uint64_t t2=BS0(a)+MAJ64(a,b,c);
        h=g; g=f; f=e; e=d+t1;
        d=c; c=b; b=a; a=t1+t2;
    }
    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d;
    s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

__device__ __forceinline__ void sha512_compress_pair32(Pair32 s[8], const Pair32 msg[16]) {
    Pair32 w[16];
    #pragma unroll
    for (int i=0;i<16;i++) w[i]=msg[i];
    Pair32 a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    #pragma unroll 80
    for (int i=0;i<80;i++) {
        int j=i&15;
        Pair32 wi;
        if (i<16) {
            wi=w[j];
        } else {
            wi=padd4(pSS1(w[(j+14)&15]),w[(j+9)&15],pSS0(w[(j+1)&15]),w[j]);
            w[j]=wi;
        }
        Pair32 t1=padd5(h,pBS1(e),pch(e,f,g),K512_32[i],wi);
        Pair32 t2=padd(pBS0(a),pmaj(a,b,c));
        h=g; g=f; f=e; e=padd(d,t1);
        d=c; c=b; b=a; a=padd(t1,t2);
    }
    s[0]=padd(s[0],a); s[1]=padd(s[1],b); s[2]=padd(s[2],c); s[3]=padd(s[3],d);
    s[4]=padd(s[4],e); s[5]=padd(s[5],f); s[6]=padd(s[6],g); s[7]=padd(s[7],h);
}

__device__ __forceinline__ void init_u64(uint64_t s[8], unsigned int tid) {
    s[0]=0x6a09e667f3bcc908ULL; s[1]=0xbb67ae8584caa73bULL;
    s[2]=0x3c6ef372fe94f82bULL; s[3]=0xa54ff53a5f1d36f1ULL;
    s[4]=0x510e527fade682d1ULL; s[5]=0x9b05688c2b3e6c1fULL;
    s[6]=0x1f83d9abfb41bd6bULL; s[7]=0x5be0cd19137e2179ULL;
    uint64_t mix=((uint64_t)tid<<32)|(uint64_t)tid;
    s[0]^=mix;
}
__device__ __forceinline__ void init_pair(Pair32 s[8], unsigned int tid) {
    uint64_t tmp[8]; init_u64(tmp,tid);
    #pragma unroll
    for (int i=0;i<8;i++) s[i]=from64(tmp[i]);
}

extern "C" __global__ void sha512_u64_kernel(
    const uint64_t* block16, uint64_t* out_state, unsigned int iterations
) {
    unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t msg[16];
    #pragma unroll
    for(int i=0;i<16;i++) msg[i]=block16[i];
    uint64_t mix=((uint64_t)tid<<32)|(uint64_t)tid;
    msg[0]^=mix;
    uint64_t s[8]; init_u64(s,tid);
    for(unsigned int it=0;it<iterations;it++) sha512_compress_u64(s,msg);
    #pragma unroll
    for(int i=0;i<8;i++) out_state[(uint64_t)tid*8+i]=s[i];
}

extern "C" __global__ void sha512_pair32_kernel(
    const uint64_t* block16, uint64_t* out_state, unsigned int iterations
) {
    unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;
    Pair32 msg[16];
    #pragma unroll
    for(int i=0;i<16;i++) msg[i]=from64(block16[i]);
    uint64_t mix=((uint64_t)tid<<32)|(uint64_t)tid;
    msg[0]=pxor(msg[0],from64(mix));
    Pair32 s[8]; init_pair(s,tid);
    for(unsigned int it=0;it<iterations;it++) sha512_compress_pair32(s,msg);
    #pragma unroll
    for(int i=0;i<8;i++) out_state[(uint64_t)tid*8+i]=to64(s[i]);
}
