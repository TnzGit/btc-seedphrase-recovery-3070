# RTX 3070 / SM86 优化验证报告

- 日期：2026-09-23
- 执行者：DSH agent（用户授权）
- 远端：`user@192.168.10.131`（LAN SSH，密钥登录，无密码）
- 远端隔离目录：`/tmp/btc3070bench-j9PBaw`（3.6 GB，**保留中，未删除**）
- 仓库：`https://github.com/TnzGit/btc-seedphrase-recovery-3070`

## 环境

| 项 | 值 |
|---|---|
| SSH endpoint | `user@192.168.10.131` |
| WSL kernel | `6.6.87.2-microsoft-standard-WSL2`, x86_64 |
| Distribution | Ubuntu 26.04 LTS |
| GPU | NVIDIA GeForce RTX 3070, 8192 MiB |
| Driver | 596.36 |
| Power limit | 240.00 W（默认 240，max 250，min 100） |
| Max SM clock | 2115 MHz |
| Max mem clock | 7001 MHz |
| Rust / Cargo | 1.98.1 (48a229cea 2026-09-01) / 1.98.1 (797e8a9bc 2026-08-05) |
| NVRTC | 12.4.127（`apt download` + `dpkg-deb -x` 解包，未安装） |
| nvcc | 不存在（仓库不需要） |

## 精确 commit

| 分支 | commit |
|---|---|
| `main` | `712b199f67791ec11dbb65f732ec3fce18645d2b` |
| `opt/sm86-rtx3070-core` | `60aa092b6fc9ecd984f3f251e639185ef57fd6bf` |
| `opt/sm86-rtx3070-v1` | `bcd76b3a8f3ecad8309ddd384430ba4824034d12` |

三个分支均 `cargo build --locked --release` 成功。main 33.4 s / core 6.3 s / v1 7.3 s。

## 正确性门禁

| 分支 | Device | 自检 | 初始化+自检 wall |
|---|---|---|---|
| main | RTX 3070 | **PASS** (3 BIP84 vectors) | 14.5 s |
| core | RTX 3070 | **PASS** | 26.9 s |
| v1 | RTX 3070 | **PASS** | 27.6 s |

脱敏端到端恢复冒烟测试（公开向量 #0，缺第 12 词，目标 `bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu`）：

| 分支 | exit | RECOVERY SUCCESSFUL | 路径 | 泄漏 |
|---|---|---|---|---|
| main | 0 | ✅ | `m/84'/0'/0'/0/0` | 无 |
| core | 0 | ✅ | `m/84'/0'/0'/0/0` | 无 |
| v1 | 0 | ✅ | `m/84'/0'/0'/0/0` | 无 |

## 结果模板

```text
GPU: NVIDIA GeForce RTX 3070 (GA104, SM86, 8192 MiB)
Driver: 596.36
CUDA toolkit: NVRTC 12.4.127 (runtime JIT; no nvcc on host)
Power limit: 240.00 W (default 240, max 250)
Core clock behavior: 705 MHz idle -> 1807-1861 MHz avg under load, 1980-1995 MHz peak
Memory clock: 7001 MHz (max, stable)
Temperature: 52.7-61.1 C avg, 60-65 C peak

main  (712b199)
  block 64:   283,855 c/s   (5/5 chunks, sum 73.881 s)
  block 128:  290,661 c/s   (5/5, 72.151 s)
  block 256:  299,392 c/s   (5/5, 70.047 s)
  block 512:  *** FAILED TO LAUNCH *** (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)

opt/sm86-rtx3070-core  (60aa092)
  block 64:   297,160 c/s   (5/5, 70.573 s)
  block 128:  295,606 c/s   (5/5, 70.944 s)
  block 256:  307,459 c/s   (5/5, 68.209 s)
  block 512:  *** FAILED TO LAUNCH ***

opt/sm86-rtx3070-v1  (bcd76b3)
  block 64:   293,131 c/s   (5/5, 71.543 s)
  block 128:  295,144 c/s   (5/5, 71.055 s)
  block 256:  301,970 c/s   (5/5, 69.449 s)
  block 512:  *** FAILED TO LAUNCH ***

Best branch/block (as-is): opt/sm86-rtx3070-core @ block 256 = 307,459 c/s
Best candidates/s: 307,459 c/s  (single run; 305,896 c/s ABBA mean; 303,382 c/s repeat mean)
Candidates/s/W: 1,611 c/s/W  (307,459 / 190.8 W avg)

--- 实验结果：__launch_bounds__ 变体（未提交，仅实验） ---
core + __launch_bounds__(256,2) @ block 256 = 318,617 c/s  (+4.5% vs core, +14.4% vs baseline)
core + __launch_bounds__(512,1) @ block 512 = 313,705 c/s  (block 512 首次可运行)
core + __launch_bounds__(256,2) @ block 128 = 318,425 c/s
Candidates/s/W (lb 变体, 按 core b256 功耗估): ~1,670 c/s/W

NCU: 不可用 —— ERR_NVGPUCTRPERM（WSL 无 GPU 性能计数器权限，且远端无免密 sudo）
  以下为 CUDA Driver API (cuFuncGetAttribute) 直接读取的 kernel 属性：
  registers/thread:  main 253 | core 254 | v1 254
  local mem/thread:  main 1088 B | core 1344 B | v1 1344 B
  static shared mem: main 24576 B | core 0 B | v1 0 B
  max threads/block: 256 (all three)
  achieved occupancy: 无法用 ncu 测量；由寄存器推算 = 65536/(256*256) = 1 block/SM = 8 warps/48 = 16.7%
  local load/store: 无 ncu；由 local_size_bytes 间接反映
  L1 hit: 不可用
  L2 hit: 不可用
  dominant stall reasons: 不可用

Correctness self-test: PASS (main / core / v1 / lb 变体 全部 PASS)

Notes:
  - block 512 在 main/core/v1 上物理无法启动（非性能问题），根因见下。
  - __launch_bounds__(256,2) 是本次唯一被证实的可复现提速手段。
  - v1 的 u32 G 表相对 core 无可证实收益。
```

## 关键发现

### 1. block 512 是硬失败，不是慢（handover 的 `let _ =` 缺陷已实际致害）

`src/main.rs:508` 三个分支都是 `let _ = gpu.run_enumeration(...)`，吞掉 launch 错误。
用诊断版二进制（`let _` → `match`）捕获到真实错误：

```
DIAG LAUNCH ERROR: kernel launch: DriverError(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
                  "too many resources requested for launch")
```

静默失败的后果是 `--bench` 打印 `rate = 45293 M c/s` 这类荒谬数字，并在 elapsed=0 时触发
`Runtime error (func=(main), adr=21): Divide by zero`。

**精确根因**：kernel 用 254 regs/thread，`max_threads_per_block = 256`（由 `65536/254` 推出）。
- 256 线程 × 256 regs = 65,536 = 恰好占满整个 SM 寄存器文件 → 每 SM 只能驻留 **1 个 block**
- 512 线程 × 256 regs = 131,072 > 65,536 → `LAUNCH_OUT_OF_RESOURCES`

**推论**：因为每 SM 只有 1 个 block，占用率恒为 16.7%，与 block size 无关 —— 这正好解释了
为什么 64/128/256 的吞吐彼此只差几个百分点（块大小只是改变 block 内的 warp 打包方式，
并不改变每 SM 的驻留 warp 数）。

### 2. handover 的三项改动逐条判定

| 改动 | 判定 | 依据 |
|---|---|---|
| 滚动 SHA-512 调度（去 `W[80]`） | **倾向保留** | 但 local mem 1088→1344 B（+23%），即"去掉数组却引入更多溢出"，正是 handover 第 143 行预警的风险 |
| PBKDF2 fixed-64-byte HMAC 快路径 | **倾向保留** | 与上面合并计为 core vs main 的收益 |
| 移除 24 KiB shared wordlist | **保留** | shared mem 24576 B → **0 B**，已验证生效；core 优于 main |
| `compute_86` NVRTC target | **保留** | 编译成功且自检通过 |
| 默认 block 128 | **建议改为 256** | 三个分支上 b256 一致优于 b128 |
| u32 G 表（仅 v1） | **无法证实，建议回退或再测** | v1 307,493 vs core 305,896（ABBA，+0.5%）；v1 308,163 vs core 303,382（重复，+1.6%）。方向不一致，落在噪声内 |

**core vs main 的净收益**：+2.5%（重复均值 303,382 vs 297,744）到 +2.8%（ABBA 305,896 vs ~297,700）。
三项 kernel 改动合计约 **+2.5~3.5%**，远小于 handover 期望。

### 3. `__launch_bounds__` 是真正的杠杆（本次最有价值的结果）

| 变体 | regs | local | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|
| core 原版 | 254 | 1344 B | 297,160 | 295,606 | 307,459 | ❌ |
| `lb(256,2)` | **128** | 2016 B | 313,789 | 318,425 | **320,606** | ❌ |
| `lb(512,1)` | **128** | 2016 B | — | 316,121 | 316,236 | **313,705 ✅** |
| `lb(256,3)` | 80 | 2608 B | — | — | 303,000 | — |
| `lb(128,4)` | 128 | 2016 B | — | — | ❌ | — |

机制：regs 254→128 使每 SM 驻留 1→2 个 block，占用率 16.7%→33.3%。

**可复现性验证**（交替运行，b256）：

```
round1  lb=321,575  |  core=304,929
round2  lb=316,431  |  core=303,074
round3  lb=317,846  |  core=306,695
均值    lb=318,617  |  core=304,899      -> +4.5%，三轮全胜
```

`lb(256,2)` 变体**已通过自检（3/3 BIP84 PASS）和脱敏恢复冒烟测试**。

**重要**：不是压得越低越好。`lb(256,3)` 把 regs 压到 80，local mem 涨到 2608 B，
反而**变慢**（303,000 < 320,606）。最优点在 regs=128。

`lb(128,4)` 失败的教训：它把 `max_threads` 钉死在 128，而 `SEEDPHRASE_BLOCK=256`
超过该上限 → 再次 OUT_OF_RESOURCES。**`__launch_bounds__` 的第一个参数必须 ≥ 实际使用的 block size。**

### 4. 遥测（矩阵运行期间，1 秒采样）

| 运行 | N | util% | VRAM MiB | temp C | power W | SM MHz |
|---|---:|---:|---:|---:|---:|---:|
| main b64 | 83 | 88.5 | 6930 | 52.7 | 172.8 | 1851 |
| main b128 | 82 | 87.7 | 6925 | 57.8 | 176.3 | 1809 |
| main b256 | 79 | 88.2 | 6962 | 59.4 | 182.1 | 1807 |
| core b64 | 76 | 89.9 | 7010 | 58.6 | 185.9 | 1851 |
| core b128 | 76 | 91.2 | 7120 | 60.2 | 185.5 | 1828 |
| core b256 | 74 | 90.8 | 7093 | 60.9 | 190.8 | 1855 |
| v1 b64 | 78 | 90.0 | 7077 | 58.7 | 180.6 | 1852 |
| v1 b128 | 77 | 90.3 | 7076 | 60.5 | 185.0 | 1852 |
| v1 b256 | 75 | 91.2 | 7066 | 61.1 | 187.3 | 1861 |

峰值：util 100%，power 204.2–218.8 W，SM clock 1980–1995 MHz，temp 60–65 C。

整进程 wall time（含 NVRTC/CUDA 初始化）：core b256 77.8 s，v1 b256 78.8 s，main b256 83.5 s。

### 5. WSL 造成的性能损失（回答你提到的"达不到 100%"）

- GPU util 平均 88–91%，峰值 100% —— 约 10% 时间未被计算占满
- 功耗平均 172–191 W，而 limit 是 240 W —— **功耗上限的 72–80%**
- SM clock 平均 1807–1861 MHz，max 2115 MHz —— **标称的 85–88%**

即：kernel 是访存/整数延迟受限（occupancy 仅 16.7%），而非功耗墙；
WSL 的驱动转发与 `nvidia-smi` 采样开销也占一部分。**在 Windows 原生下应能拿到更高数字**，
但本次全部数据均在 WSL 内测得，同口径可比。

## 局限与未完成项

1. **ncu 完全不可用**（`ERR_NVGPUCTRPERM`）。handover 要求的 L1/L2 hit、stall reasons、
   issue 效率、achieved occupancy 均**无法提供**。已改用 `cuFuncGetAttribute` 给出
   regs/local/shared/max_threads 四个硬指标。要拿全 ncu 需要你在 Windows 侧
   开启性能计数器权限（或给 WSL 用户免密 sudo 以 `--privileged` 方式重试）。
2. **未做功耗扫描**（按你指示跳过）。c/s/W 是单点值。
3. **`main` 的 `--bench` 不打印汇总行**，加权吞吐由我按 `20971520 / Σchunk` 计算。
4. **一次测量被污染**：`abba1-core`（129,942 c/s）与我并发的 ncu 冒烟测试抢 GPU，
   已剔除并在分析中排除。
5. 所有结论仅对上述三个 commit 有效。
6. 每轮配置只跑 1 次矩阵 + 3 次重复；v1 vs core 的差异小于该样本量能分辨的精度。

## 已交付的仓库改动

分支 `opt/sm86-rtx3070-lb`（基于 `opt/sm86-rtx3070-core`，**已 push 到 origin**），两个可独立回退的 commit：

| commit | 内容 |
|---|---|
| `38b6144` | `opt(sm86): cap registers with __launch_bounds__(256,2), default block 256` |
| `25489bc` | `fix(bench): stop discarding run_enumeration errors, guard divide-by-zero` |

**按你的选择，G 表改动未被带入**：`lb` 分支直接基于 `core`，而 `core` 本身就不含 u32 G 表
（handover 第 66 行：core 分支止步于该改动之前）。所以 `lb` 天然满足"回退 G 表、只保留 core 的 kernel 优化"。

### 在真实 3070 上对 push 后的 commit 复验

```
分支 HEAD: 25489bc / 38b6144
cargo build --locked --release : Finished (5.64 s)
GPU self-test (3 BIP84)        : PASS
默认 block（无 env var）        : total 20971520 candidates in 65.529s  weighted = 320032 c/s
SEEDPHRASE_BLOCK=512           : 被 filter 钳制到 256，正常跑完，weighted = 316758 c/s, exit=0
脱敏恢复冒烟                    : RECOVERY SUCCESSFUL / m/84'/0'/0'/0/0 / exit 0 / 无泄漏
```

### bug 修复的对照证据（`results/proof.log`）

用"旧 kernel（254 regs）+ 新错误处理"的构建，`SEEDPHRASE_BLOCK=512`：

```
exit=1
Device: NVIDIA GeForce RTX 3070
chunk #0 FAILED: kernel launch: DriverError(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, ...)
benchmark aborted - refusing to report a rate for a kernel that did not run.
```

同一配置在**原始 core 二进制**（`let _ =`）上：

```
chunk #0  elapsed = 0.002s  rate = 1870.69 M c/s
chunk #1  elapsed = 0.000s  rate = 36473.79 M c/s
```

即：修复前会把"根本没跑"报成 36,000 M c/s；修复后如实失败并给出非零退出码。

## 隐私与网络记录

- 使用了 LAN SSH + GitHub + crates.io(rsproxy 镜像) + Ubuntu apt / NVIDIA 镜像下载。
- 恢复计算本身**未访问任何区块浏览器、余额、RPC 或比特币网络服务**；
  `src/` 全树无 `reqwest/hyper/TcpStream/electrum/blockstream/mempool.space/bitcoind` 引用，
  `Cargo.toml` 依赖中无 HTTP 客户端。
- 仅使用源码内已有的**公开** BIP84 向量，未使用任何真实助记词。
- `Missing Word:` 与 `Complete Seed Phrase:` 两行在捕获输出中已被替换为
  `[REDACTED PUBLIC TEST VECTOR]`，脚本校验 `phrase leaked into output: False`。
- **未安装任何系统包**（NVRTC 与 ncu 均为解包到隔离目录使用）。
- 未修改 SSH 配置。远端仓库工作区保持 clean。
