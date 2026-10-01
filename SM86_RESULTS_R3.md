# RTX 3070 / SM86 — R3 硬件几何实验报告

执行角色：本地执行 agent（按 `NEXT_AGENT_PROMPT_R3.md` 执行；未改动任何 CUDA/Rust 热路径代码）。

- 分支：`opt/sm86-rtx3070-r3-hw`
- 起始 commit：`70d9c5195cdee45e15309c726f9dbb54f4f1752c`（历史包含 `dae5aaa295d4a5b123274c664d984f9bc78fb454`，已用 `git merge-base --is-ancestor` 验证）
- 未触碰 `main`，未重写 `opt/sm86-rtx3070-r2`
- 远端 scratch：`/home/user/r3hw/btc-seedphrase-recovery-3070`（保留中）

## 1. 环境

| 项 | 值 |
|---|---|
| SSH endpoint | `user@192.168.10.131` |
| WSL kernel | `6.6.87.2-microsoft-standard-WSL2` |
| Distribution | Ubuntu 26.04 LTS |
| GPU | NVIDIA GeForce RTX 3070, 8192 MiB, GA104 / SM86 |
| Driver | 596.36 |
| NVRTC | 12.9.86（`/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib` + `/usr/lib/wsl/lib`，运行时 JIT `compute_86`） |
| Rust / Cargo | 1.93.1（发行版版本化工具链，`RUSTC=/usr/bin/rustc-1.93 cargo-1.93 build --locked --release`，构建耗时 3m16s，exit 0） |
| Power limit | 240.00 W（默认值，全程未修改；本轮按 prompt 未做功率扫描） |
| 实测 SM clock | 均值 1954–1965 MHz（bench 全程 100% util 段） |
| 实测 mem clock | 6801 MHz（bench 段） |
| 实测功耗 | 均值 216.8–226.4 W，峰值 232.1 W |
| 实测温度 | 峰值 66–70 °C |

静态 ptxas 参考（CI run `36861467496` / `36861467478`，均在 `dae5aaa`）见
`results/r3/ptxas/ptxas-static-r3-reference.txt`：G128/G192 几何均为 168 regs、kernel spill 0/0、
HMAC 340/340 B、PBKDF2 456/456 B；A(256,2) 维持 128 regs、HMAC 512/512、PBKDF2 856/872。

## 2. 候选配置与精确环境变量

所有实验 flag 一律 OFF（每 run 全新 shell）：

```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=0
SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
SEEDPHRASE_NOINLINE_PBKDF2=0
SEEDPHRASE_PBKDF2_SCALAR_UT=0
SEEDPHRASE_PBKDF2_T_SHARED=0
SEEDPHRASE_PBKDF2_STATE_SHARED=0
SEEDPHRASE_SHA512_HALF_SHARED_SCHEDULE=0
```

| cfg | SEEDPHRASE_LB_MAX_THREADS | SEEDPHRASE_LB_MIN_BLOCKS | SEEDPHRASE_BLOCK | 额外 flag |
|---|---|---|---|---|
| A（R2 参考） | 256 | 2 | 256 | 无 |
| G128 | 128 | 3 | 128 | 无 |
| G192 | 192 | 2 | 192 | 无 |
| G128-T | 128 | 3 | 128 | `SEEDPHRASE_PBKDF2_T_SHARED=1` |
| G128-HS | 128 | 3 | 128 | `SEEDPHRASE_SHA512_HALF_SHARED_SCHEDULE=1` |

G192 被运行时接受（`LB_MAX_THREADS=192` 无拒绝），确认 checkout 含 `dae5aaa`。

## 3. 正确性门禁（每配置，相同 env，`--self-test`）

全部 exit 0、3/3 BIP84 GPU 向量 PASS（日志 `results/r3/logs/results_selftest_*.log`）：

| cfg | exit | self-test | Kernel resources |
|---|---|---|---|
| A | 0 | PASS 3/3 | `regs/thread=128 local/thread=2016B shared/block=0B max_threads/block=256` |
| G128 | 0 | PASS 3/3 | `regs/thread=168 local/thread=1744B shared/block=0B max_threads/block=128` |
| G192 | 0 | PASS 3/3 | `regs/thread=168 local/thread=1744B shared/block=0B max_threads/block=192` |
| G128-T | 0 | PASS 3/3 | `regs/thread=168 local/thread=1696B shared/block=8192B max_threads/block=128` |
| G128-HS | 0 | PASS 3/3 | `regs/thread=168 local/thread=1680B shared/block=8192B max_threads/block=128` |

已知错误的 R2 noinline 组合由 Rust/CUDA 硬门禁拦截，本轮未绕过。

## 4. Phase 1 — 硬件筛查（顺序 A G128 G192 → G192 G128 A，各 2 run）

主指标 `weighted_steady`（chunk 1–4），`weighted_all` 仅冷启动参考。原始输出
`results/r3/logs/out-scr-*.log`，遥测 `results/r3/telemetry/tel-scr-*.csv`（1 Hz）。

| run | cfg | weighted_steady | weighted_all | exit | wall_s |
|---|---|---|---|---|---|
| scr-r1-A | A | 308593 | 309083 | 0 | 68 |
| scr-r1-G128 | G128 | 308944 | 308709 | 0 | 69 |
| scr-r1-G192 | G192 | 292789 | 292920 | 0 | 72 |
| scr-r2-G192 | G192 | 291612 | 291698 | 0 | 73 |
| scr-r2-G128 | G128 | 307748 | 308065 | 0 | 69 |
| scr-r2-A | A | 307813 | 307983 | 0 | 69 |

- **G192：早期拒绝** — 同 regime 两次分别比 A 慢 5.1% / 5.2%（>5% 阈值）。
- **G128：存活** — 与 A 差 +0.11% / −0.02%，在 R2 观测的筛查噪声（数个百分点）之内，进入严格测试。
- regime 检查：六 run 遥测 avg SM clock 1954–1965 MHz、avg power 216.8–226.4 W、峰值温度 66–70 °C，
  无 regime 边界，可同 regime 对比。

### 4.1 Phase 1b — shared 探针（G128 几何，各 2 run；先 self-test）

| run | cfg | weighted_steady | vs A 筛查均值(308203) |
|---|---|---|---|
| scr-r3-G128T-1 | G128-T | 307726 | −0.15% |
| scr-r3-G128T-2 | G128-T | 305748 | −0.80% |
| scr-r3-G128HS-1 | G128-HS | 307249 | −0.31% |
| scr-r3-G128HS-2 | G128-HS | 307302 | −0.29% |

两个 shared 探针均不快于 A（G128-T 均值 −0.48%，G128-HS 均值 −0.30%），无竞争力证据，未进入 Phase 2。
按 prompt：未测 shared HMAC-state，未组合 shared 探针。

## 5. Phase 2 — 严格对决：G128 vs A（3 轮 A X X A）

六 run A、六 run G128。原始输出 `results/r3/logs/out-p2-*.log`，遥测 `results/r3/telemetry/tel-p2-*.csv`。

| round | A1 | G128-1 | G128-2 | A2 | 轮内配对 Δ（G 均值 vs A 均值） |
|---|---|---|---|---|---|
| 1 | 307934 | 308450 | 311060 | 309966 | **+0.261%** |
| 2 | 309992 | 310320 | 310318 | 308191 | **+0.397%** |
| 3 | 308488 | 309586 | 310299 | 309427 | **+0.319%** |

配对单 run Δ（vs 同轮 A 均值）：−0.162%, +0.683% / +0.397%, +0.397% / +0.203%, +0.434%。

汇总（各 n=6）：

| cfg | mean | median | min | max |
|---|---|---|---|---|
| A | 308999.7 | 308957.5 | 307934 | 309992 |
| G128 | 310005.5 | 310308.5 | 308450 | 311060 |

**配对增益：mean +0.326%，median +0.437%。三轮全部为正向 → G128 增益方向一致、可复现，但幅度远低于 ≥2% 晋升门槛。**

电源/时钟排查：G128 run 的 avg power（222–226 W）不低于 A（217–225 W），SM clock 均值两者一致
（1954–1965 MHz），温度峰值同级（68–70 °C）——该 +0.3% 差异无电源/温度/时钟混杂解释，是真效应，但低于门槛。

## 6. Recovery smoke（公开/redacted BIP84 向量 #0，`scripts/recovery_smoke_r3.py`）

A 与 G128 两个配置均：`RECOVERY SUCCESSFUL`，地址 `bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu`，
路径 `m/84'/0'/0'/0/0`，exit 0，phrase 未泄漏（日志 `results/r3/logs/smoke-A.log`、`smoke-G128.log`）。

## 7. 污染/排除 run

无。全部 20 个 bench run + 2 个 smoke run exit 0，遥测均显示持续 100% util 的 GPU 工作段；
无被排除或标注的 run。

## 8. 每候选最终裁定

| 候选 | 裁定 | 依据 |
|---|---|---|
| A | **keep**（生产参考，R2 语义不变） | 基线本身 |
| G128 | **reject（不晋升）**；可作为 lab 几何保留观察 | 正确性+smoke 全过、三轮一致 +0.33%/+0.44%，但 mean/median 均 <2% 门槛 |
| G192 | **reject** | 同 regime 一致慢 5.2%（>5% 早拒阈值），且与 G128 静态分配相同却更慢 |
| G128-T | **reject** | 不比 A 快（均值 −0.48%），shared 流量代价可见 |
| G128-HS | **reject** | 不比 A 快（均值 −0.30%） |

## 9. G128 / G192 是否确立真实硬件胜利

- **G128：确立了一个真实但微小的硬件增益**（+0.33% mean / +0.437% median，三轮方向一致，
  无电源/时钟混杂）。**未达到 ≥2% 晋升门槛**，因此不晋升。
- **G192：未确立胜利**——同静态资源分配下比 A 慢 5.2%，比 G128 慢 5.7%；192×2 的 384 驻留线程
  几何在该 kernel 上是净劣势。

## 10. shared 探针是否值得继续

不值得。两个探针在 168-reg 几何下把 kernel spill 从 0/0 变为 432/432 B 并引入 8 KiB smem 流量，
硬件实测比 A 慢 0.3–0.5%（方向与静态资源"改善"相反）。本轮无证据支持进一步的 shared 工作。

## 11. 剩余阻塞 / 交给远端 agent 的事项

1. **无硬件阻塞**；环境、构建、正确性全部正常。
2. G128 的 +0.3% 增益真实存在但低于门槛。若要把它变成可晋升候选，需要**代码级**（远端 agent 负责）
   把该几何的收益做大——例如在 (128,3) 几何上重新审视 spill 布局/noinline 组合——而不是继续纯几何扫描；
   本轮已证明单纯几何切换在 RTX 3070 上不产生 ≥2% 效果。
3. R2 已知错误组合（sha=1+pbkdf2=1+hmac=0）仍被硬门禁拦截，行为符合预期。
4. NVRTC 版本本轮为 12.9.86（R2 报告为 12.4.127）；如需严格跨轮对比，远端 agent 应确认版本差异
   不影响 JIT 结果（本机可用工具链即为此版本）。
