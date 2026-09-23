# RTX 3070 / SM86 — R2 实验报告

- 起始分支：`opt/sm86-rtx3070-r2`
- 起始 commit：`684943c425e2e56626bd7d62f4c5b9ead46f0f72`
- 结果分支：`opt/sm86-rtx3070-r2`（追加 `2fdb944` 恢复上一轮交付物）
- 未触碰 `main`，未创建 R3（原因见第 6 节）
- 远端 scratch：`/tmp/btc3070bench-j9PBaw`（保留中，未删除）

## 1. 环境

| 项 | 值 |
|---|---|
| SSH endpoint | `user@192.168.10.131` |
| WSL kernel | `6.6.87.2-microsoft-standard-WSL2` |
| Distribution | Ubuntu 26.04 LTS |
| GPU | NVIDIA GeForce RTX 3070, 8192 MiB, GA104 / SM86 |
| Driver | 596.36 |
| CUDA toolkit | NVRTC 12.4.127（运行时 JIT）；静态 ptxas 用 CUDA 12.4.99（CI 容器） |
| Rust / Cargo | 1.98.1 |
| Power limit | 240.00 W（default 240，max 250，min 100）；**全程未修改** |
| 实测 SM clock | 1712–1891 MHz（max 2115） |
| 实测 mem clock | 5929–6672 MHz（max 7001） |
| 实测温度 | 59.5–66.0 C，峰值 70 C |
| 实测功耗 | 198.1–219.6 W，峰值 239.3 W |

## 2. 固定参考配置

所有实验共用：

```bash
export SEEDPHRASE_BLOCK=256
export SEEDPHRASE_LB_MIN_BLOCKS=2
```

A reference（未被修改，仍是 R2 默认）：

```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=0
SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
SEEDPHRASE_NOINLINE_PBKDF2=0
```

吞吐量主指标为 `weighted_steady`（chunk 1–4），`weighted_all` 仅作冷启动参考。

## 3. 正确性门禁（每个配置，相同 env）

`--self-test`，要求 exit 0 且 3/3 BIP84 PASS：

| cfg | flags (sha/hmac/pbkdf2) | exit | self-test | Kernel resources |
|---|---|---|---|---|
| A | 0/0/0 | 0 | **PASS 3/3** | `regs/thread=128 local/thread=2016B shared/block=0B max_threads/block=256` |
| B | 1/0/0 | 0 | **PASS 3/3** | `regs/thread=128 local/thread=1936B shared/block=0B max_threads/block=256` |
| C | 0/1/0 | 0 | **PASS 3/3** | `regs/thread=128 local/thread=2368B shared/block=0B max_threads/block=256` |
| D | 1/1/0 | 0 | **PASS 3/3** | `regs/thread=128 local/thread=1968B shared/block=0B max_threads/block=256` |
| E | 1/0/1 | **1** | **FAIL 向量 0 不匹配** | `regs/thread=128 local/thread=2128B shared/block=0B max_threads/block=256` |
| F | 1/1/1 | 0 | **PASS 3/3** | `regs/thread=128 local/thread=2272B shared/block=0B max_threads/block=256` |

### 3.1 重要缺陷：E 会静默算出错误结果

`SEEDPHRASE_NOINLINE_SHA512_WORDS=1` + `SEEDPHRASE_NOINLINE_FIXED64_HMAC=0` +
`SEEDPHRASE_NOINLINE_PBKDF2=1` 组合下，自检**可复现地失败**（连续 3 次）：

```text
GPU self-test: FAIL: self-test vector 0 FAILED: no match for bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu
exit=1
```

对照组（同一二进制，仅改一个 flag）均 PASS：

| 配置 | self-test |
|---|---|
| sha=0 hmac=0 pbkdf2=1 | PASS |
| sha=1 hmac=0 pbkdf2=1（=E） | **FAIL** |
| sha=1 hmac=1 pbkdf2=1（=F） | PASS |

完整 8 组合 flag 矩阵（`--self-test`，同 block/lb）：

| sha | hmac | pbkdf2 | exit | self-test |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | PASS |
| 0 | 0 | 1 | 0 | PASS |
| 0 | 1 | 0 | 0 | PASS |
| 0 | 1 | 1 | 0 | PASS |
| 1 | 0 | 0 | 0 | PASS |
| **1** | **0** | **1** | **1** | **FAIL** |
| 1 | 1 | 0 | 0 | PASS |
| 1 | 1 | 1 | 0 | PASS |

**只有 `sha=1 hmac=0 pbkdf2=1` 这一种组合失败**，其余 7 种全部 PASS。

#### 根因：已排除一个假设，真正机制尚未定位

我最初怀疑是 `pbkdf2_hmac_sha512_block` 热循环里的 in/out 别名：

```cuda
hmac_sha512_finish_fixed64_words(ipad_state, opad_state, U, U);
```

`msg_words` 与 `out_words` 指向同一个 `U`。**我做了针对性实验来验证**：把该调用改为
独立输出缓冲 + 显式拷贝（`Uout` 后 `U[i] = Uout[i]`），用同一 E 配置重新构建并自检。

结果：**仍然 FAIL**（`exit=1`，向量 0 不匹配），同时 A 配置仍 PASS。
→ **别名假设被实验证伪**，该假设不成立，我已从结论中撤回。

因此：E 的失败是**该 flag 组合在 NVRTC 下的一次真实误编译**，触发条件明确
（仅 sha=1 且 hmac=0 且 pbkdf2=1），但具体机制未定位。诚实结论是**成因未明**，
不是"已找到根因"。

#### 对默认路径的安全性核查

为确认这不是普遍性风险，对 **A（R2 默认）** 做了 12 组 `lb ∈ {1,2,3,4} × block ∈ {64,128,256}` 自检扫描：

**12/12 全部 PASS。** 默认路径（三个 flag 全 0）在所有 launch-bounds 与 block 组合下均正确。

**这是本轮最重要的发现**：E 在 screening 中一度跑出 318,199（高于 A 的 315,262），
如果只看吞吐量就会把一个**结果错误的配置**当成赢家。它证明了 handover 坚持"先过正确性门禁"
的必要性。E 已判定 **REJECT（正确性失败）**，其吞吐数据全部作废。

**建议**：`SEEDPHRASE_NOINLINE_PBKDF2=1` 与 `SEEDPHRASE_NOINLINE_SHA512_WORDS=1`
的组合应在代码层面直接禁止（例如二者同时置位时报错退出），避免今后有人误用该组合
而拿到"更快但算错"的结果。

## 4. Phase 1 screening（顺序执行，仅用于筛选）

每配置 2 次，`weighted_steady`：

| cfg | run 1 | run 2 | mean | Δ vs A | local/thread |
|---|---:|---:|---:|---:|---:|
| **A** | 316,801 | 315,262 | **316,032** | — | 2016 B |
| B | 303,303 | 315,217 | 309,260 | -2.1% | 1936 B |
| C | 305,529 | 312,332 | 308,930 | -2.2% | 2368 B |
| D | 295,310 | 312,943 | 304,126 | -3.8% | 1968 B |
| E | 295,930 | 318,199 | 307,064 | -2.8% | 2128 B |
| F | 290,919 | 297,853 | 294,386 | -6.9% | 2272 B |

screening 方差很大（E 跨度 7.5%），因此**不作为胜负依据**。按 handover 规则，
明显落后且 local/thread 最高的 D、F 在 Phase 2 前淘汰；对 A/B/E/C 进入 Phase 2。

## 5. Phase 2 严格交替测试（`A X X A` × 3 轮）

每对共 6 次 A + 6 次候选，A 与候选交错，使线性热漂移对消。

### A vs B

| run | A | B |
|---|---:|---:|
| r1 | 325,010 / 323,024 | 309,933 / 308,973 |
| r2 | 319,961 / 317,664 | 303,341 / 303,835 |
| r3 | 317,165 / 318,020 | 304,209 / 304,772 |

| | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| A | 6 | **320,141** | 318,990 | 317,165 | 325,010 |
| B | 6 | 305,844 | 304,490 | 303,341 | 309,933 |

**B = -4.5%（mean）/ -4.5%（median）。两组区间完全不重叠（A 最低 317,165 > B 最高 309,933）。**
B 的功耗更高（215.2 W vs 203.1 W）、SM 频率更高（1870 vs 1845 MHz）却更慢 —— 排除了
"频率/功耗/温度解释"。→ **B REJECT**

### A vs E

| run | A | E |
|---|---:|---:|
| r1 | 317,882 / 318,220 | 298,378 / 298,770 |
| r2 | 318,386 / 318,128 | 299,226 / 299,439 |
| r3 | 318,510 / 319,191 | 299,392 / 300,246 |

| | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| A | 6 | **318,386** | 318,303 | 317,882 | 319,191 |
| E | 6 | 299,242 | 299,309 | 298,378 | 300,246 |

**E = -6.0%。且 E 正确性失败，双重 REJECT。**

### A vs C

C 的三轮跨越了一次 GPU 全局状态变化（见第 6 节），必须分段解读：

**变化前（r1）**：

| run | A | C |
|---|---:|---:|
| r1 | 318,220 / 317,873 | 307,260 / 306,415 |

**变化后（r2 + r3）**：

| run | A | C |
|---|---:|---:|
| r2 | 318,037 / 359,791 | 347,027 / 347,088 |
| r3 | 359,707 / 372,650 | 347,138 / 347,391 |

变化后稳态 A 均值（取稳定后的 4 次 A：359,791 / 359,707 / 372,650，及 confirm 段）：

| | n | mean | min | max |
|---|---:|---:|---:|---:|
| A（稳定后） | 4 | **374,254** | 372,960 | 377,875 |
| C（稳定后） | 4 | 347,161 | 347,027 | 347,391 |

**C = -7.2%（稳定后）/ -3.3%（变化前）。→ C REJECT**

### 变化后的独立确认（A vs B，2 轮 ABBA）

| run | A | B |
|---|---:|---:|
| c1 | 377,875 / 373,123 | 357,996 / 356,918 |
| c2 | 373,057 / 372,960 | 355,811 / 355,805 |

| | n | mean | median |
|---|---:|---:|---:|
| A | 4 | **374,254** | 373,090 |
| B | 4 | 356,632 | 356,364 |

**B = -4.7%。排名在两种 GPU 状态下都成立**，说明 A 的胜出不依赖特定机器状态。

## 6. 污染 / 异常 run 的显式记录

`XC-r2-C1` 起，GPU 进入另一种状态并**一直持续**，所有配置同时变快约 14%，
但 `nvidia-smi` 报告的 SM/mem 频率反而**下降**：

| 阶段 | 代表 run | weighted_steady | avg power | avg SM | avg mem |
|---|---|---:|---:|---:|---:|
| 变化前 | `XC-r1-A1` | 318,220 | 202.5 W | 1837 MHz | 6337 MHz |
| 变化后 | `XC-r3-A2` | 372,650 | 207.1 W | 1715 MHz | 5976 MHz |

**吞吐 +17% 而报告频率 -7%**，物理上不自洽 —— 唯一合理解释是 WSL 的
`clocks.sm`/`clocks.mem` 读数在变化后不再反映真实运行频率（WSL 驱动转发失真），
而非真实降频。功耗上限、`SW Power Cap`、`HW Thermal Slowdown` 在变化前后均为 Not Active，
温度区间也一致，所以**不是热节流或功耗墙**。

处理方式（按 handover 要求"标记并剔除，不得静默替换"）：

1. 不删除任何原始 run，全部保留在 `results/r2/logs/abba-summary.txt` 与 `out-*.log`。
2. `A vs C` 按变化前后**分段**统计，不跨状态取均值（跨状态均值会得出 C mean 333,720 这种无意义数字）。
3. 额外跑 `results/r2/logs/confirm-regime.log`（变化后 A vs B，2 轮 ABBA）确认排名不因状态而翻转。
4. 结论只依赖**同一状态内**的交错对比，不依赖任何绝对吞吐数字。

另：`XC-r2-A2`（359,791）与 `XC-r3-A2`（372,650）出现在变化后早期，是过渡期读数，
在稳定后统计中被排除（稳定后 A 取 372,960–377,875 区间）。

## 7. 静态 ptxas 资源（CUDA 12.4, sm_86, lb=(256,2)）

通过项目自带 CI（run `35888411573`）采集，原始输出见 `results/r2/ptxas/ptxas-ci-matrix.txt`。

| cfg | kernel regs | kernel stack | kernel spill | SHA 函数 | HMAC 函数 | PBKDF2 函数 |
|---|---:|---:|---:|---|---|---|
| A | 128 | 1952 B | 0 / 0 | — | 512 / 512 | 856 / 872 |
| B | 128 | 1936 B | **1092 / 1224** | 0 / 0 | — | — |
| C | 128 | 2416 B | 0 / 0 | — | 0 / 0 | **1324 / 1360** |
| D | 128 | 1968 B | **1044 / 1188** | 0 / 0 | 0 / 0 | — |
| E | 128 | 2128 B | 120 / 120 | 0 / 0 | — | 996 / 1128 |
| F | 128 | 2272 B | 84 / 84 | 0 / 0 | 0 / 0 | 992 / 1116 |

（spill 为 stores / loads 字节数）

**关键结论：noinline 没有消除 spill，只是把 spill 从函数内部搬到 kernel 边界。**
A 的 kernel 自身 spill 为 0/0，而 B/D 让 kernel 背上 1092/1224、1044/1188 的 spill，
总 spill 反而增加。这与硬件结果（B/D 更慢）完全一致。

本地也尝试过用解包的 CUDA 12.4 工具链直接跑 ptxas，但宿主 gcc 15 与 CUDA 12.4 头文件
不兼容（`__is_array` / `__bfloat16_t` 报错），故改用 CI 容器取得数据。

## 8. 每个 candidate 的最终判定

| cfg | 配置 | self-test | 相对 A | 判定 | 理由 |
|---|---|---|---|---|---|
| **A** | 全 inline | PASS | — | **KEEP（参考）** | R2 默认，最快且正确 |
| B | SHA noinline | PASS | **-4.5%** | **REJECT** | 6v6 区间不重叠；功耗/频率更高却更慢；spill 转移非消除 |
| C | HMAC noinline | PASS | **-7.2%** | **REJECT** | 两种 GPU 状态下均落后；PBKDF2 spill 升至 1324/1360 |
| D | SHA+HMAC | PASS | -3.8%（screening） | **REJECT** | screening 落后且 kernel spill 1044/1188；未进入 Phase 2 |
| E | SHA+PBKDF2 | **FAIL** | -6.0%（且数据作废） | **REJECT（正确性）** | 8 组合中唯一失败；成因未定位；建议代码层禁用该组合 |
| F | 三边界 | PASS | -6.9%（screening） | **REJECT** | screening 最差，local/thread 最高（2272 B） |

**没有任何 noinline/boundary 配置获胜。** 按 handover 第 111–121 行，R2 默认行为保持不变，
下一步应转向 partial-unroll / live-range reduction。

## 9. Recovery smoke test（R2 默认路径）

对 A（R2 默认）执行公开向量脱敏端到端恢复：

```text
=== exit status: 0
=== CHECKS ===
RECOVERY SUCCESSFUL present: True
target address present    : True
default BIP84 path present: true   (m/84'/0'/0'/0/0)
exit status zero          : True
phrase leaked into output : False
```

## 10. 未完成 / 阻塞项

1. **未做功耗曲线**：按 handover 第 9 节，应在代码 winner 确定后再做。由于本轮无 winner
   （A 即原参考），且 GPU 出现了第 6 节的状态漂移，功耗曲线会与代码选择混淆，故**本轮不做**，
   留待有真实代码 winner 后执行。
2. **未创建 R3**：无候选胜出，按 handover 第 8 节保持 R2 默认路径不变。
3. **ncu 仍不可用**：`ERR_NVGPUCTRPERM`，权限条件未改变，按指示未重复尝试。
   achieved occupancy / L1/L2 hit / stall reasons 依然缺失。
4. **GPU 状态漂移未根治**：第 6 节现象在 WSL 下无法从内部修正；已用同状态内交错对比规避。
   若后续需要绝对吞吐可比性，建议在 Windows 侧固定时钟或重启 WSL 后重测。

## 11. 下一阶段建议

按 handover 第 8 节，保持 R2 默认路径，进行**单一**受控 partial-unroll / live-range 实验：

- 保留 16-word rolling schedule，禁止重新引入动态 `W[80]` / `W[16]`
- 目标仍是 128 regs / 2 resident blocks
- 重点降低 PBKDF2 hot-path 的 856/872 spill
- 每次先看 ptxas，再 self-test，再上硬件

**建议的第一条**：`pbkdf2_hmac_sha512_block` 的 U2..U2048 循环中，
`hmac_sha512_finish_fixed64_words(..., U, U)` 的 in/out 别名迫使编译器同时保留
8 个输入与 8 个输出寄存器。若拆成独立 in/out 缓冲并让编译器看清复制，
可能降低该循环的活跃区间。

**注意**：该 alias 改动我已实测，**对 E 的误编译无效**（见 3.1），但它作为
live-range 实验本身仍未做过硬件吞吐测试，可作为候选之一。

**另需优先处理**：把 `sha=1 + hmac=0 + pbkdf2=1` 这个错误组合在代码层禁用，
避免今后误用。

## 12. 交付物

- 本报告：`SM86_RESULTS_R2.md`
- 精简日志：`results/r2/`
  - `logs/` — screening / phase2 / confirm / ABBA summary / 每个 run 的原始输出 / 每个配置的 self-test
  - `telemetry/` — 每次 run 的 1 Hz GPU 遥测 CSV（util/power/sm/mem/temp）
  - `ptxas/ptxas-ci-matrix.txt` — 静态 ptxas 资源矩阵
- 上一轮找回物：`results/r1-recovered/`（commit `2fdb944`）

未提交任何多 GB profiler/scratch 数据；未提交任何真实 seed、助记词或私钥。
