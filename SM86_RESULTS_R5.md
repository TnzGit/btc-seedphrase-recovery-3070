# RTX 3070 / SM86 — R5 scalar-U/T 硬件小筛报告

执行角色：本地执行 agent（按 `NEXT_AGENT_PROMPT_R5.md`；本轮未改动 CUDA/Rust 热路径代码）。

## 1. 分支 / 基线验证

- 分支：`opt/sm86-rtx3070-r5-scalar-hw`（远端 tip `c587365211ea555c1160f36b9ebd6ad9495e2a50`）
- 双 ancestor 验证（本地与主机均通过）：
  - `git merge-base --is-ancestor a8d6482a21aef5751a04788535b8be96d876fc23 HEAD`（R4 结果件）✓
  - `git merge-base --is-ancestor 6c9fb8e55f02e5841f15cd7410d9865c99edc085 HEAD`（遥测修正/harness 修复）✓
- 未修改 `main`，未重写 R2/R3/R4 结果分支
- 主机侧经 git-bundle 传输更新（主机直连 GitHub 仍不稳定），构建一次成功
  （`RUSTC=/usr/bin/rustc-1.93 cargo-1.93 build --locked --release`，Rust 1.93.1）

## 2. 运行时环境（仅 N129）

- NVRTC 12.9.86：`/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib`
- 每个 run 日志头部内嵌同一目录同一 LD_LIBRARY_PATH 的 `nvrtcVersion=12.9` 探测
- N124 未混入本轮 ✓
- 无关 flag 全部显式 OFF：
  `NOINLINE_SHA512_WORDS / NOINLINE_FIXED64_HMAC / NOINLINE_PBKDF2 / PBKDF2_T_SHARED /
   PBKDF2_STATE_SHARED / SHA512_HALF_SHARED_SCHEDULE = 0`
- 远端 review 已关闭的方向本轮全部未再耗硬件时间：G192、T_SHARED/half-shared、shared HMAC state、
  noinline 变体、N124 pinning、fixed64 in-place HMAC（SASS_IDENTICAL=1）。

## 3. 正确性门禁（4 配置，`--self-test`）

全部 exit 0、`GPU self-test: PASS (3/3 BIP84 reference vectors)`：

| 配置 | Kernel resources（gate 日志 `results/r5/logs/gate-correctness.log`） |
|---|---|
| A | `regs/thread=128 local/thread=2016B shared/block=0B max_threads/block=256` |
| A-S | 同 A（标量 U/T 不改变资源分配） |
| G128 | `regs/thread=168 local/thread=1744B shared/block=0B max_threads/block=128` |
| G128-S | 同 G128 |

与静态 ptxas 预测一致：标量 U/T 不是 spill 分配赌注（A: 128 regs / PBKDF2 spill 856/872 B；
G128: 168 regs / 456/456 B —— 运行时资源与对应 reference 完全相同）。

## 4. Phase-1 原始数据（2 条平衡通道，8 run，全部 exit 0、kernel_rows>0）

通道 1：A → A-S → G128 → G128-S；通道 2：G128-S → G128 → A-S → A。

| run | 配置 | weighted_steady | kernel-phase 遥测（util≥99%） |
|---|---|---|---|
| p1-A | A | 317405 | avg 224.0 W / 1973 MHz / max 65 °C / 62 rows |
| p1-AS | A-S | 313940 | avg 222.6 W / 1964 MHz / max 67 °C / 63 rows |
| p1-G128 | G128 | 314204 | avg 222.5 W / 1965 MHz / max 68 °C / 62 rows |
| p1-G128S | G128-S | 313476 | avg 226.2 W / 1955 MHz / max 69 °C / 63 rows |
| p2-G128S | G128-S | 315807 | avg 227.0 W / 1952 MHz / max 70 °C / 62 rows |
| p2-G128 | G128 | 315634 | avg 227.5 W / 1951 MHz / max 70 °C / 62 rows |
| p2-AS | A-S | 317191 | avg 228.2 W / 1948 MHz / max 70 °C / 62 rows |
| p2-A | A | 320055 | avg 229.0 W / 1946 MHz / max 71 °C / 61 rows |

分组汇总（各 n=2）：A mean 318730.0（317405–320055）；A-S mean 315565.5（313940–317191）；
G128 mean 314919.0（314204–315634）；G128-S mean 314641.5（313476–315807）。

## 5. 同几何增益

- **A-S vs A：-0.993%**（两轮 A-S 均慢于配对的 A run：313940<317405、317191<320055）
- **G128-S vs G128：-0.088%**（一轮慢一轮快，纯噪声区间，无信号）

## 6. G128-S vs 生产 A

**-1.283%**，且两条 G128-S run 均慢于两条 A run → 按早停规则 #2 直接排除。

## 7. Phase-2 ABBA

**未执行**：没有任何候选存活（A-S 两轮皆慢 → 早停；G128-S 在 G128 下无信号且 vs A 更差）。

## 8. 排除 / 污染 run

无。8/8 exit 0；无 warm-up 泄漏；`Bench config:` 行逐 run 核实了 `pbkdf2_scalar_ut` 值正确传入
（本轮 harness 修复：commit 版 `bench_sm86_r4.sh` 只把额外 env 写进日志而未传给进程，
本轮 wrapper `scripts/bench_sm86_r5.sh` 用 `env $envline` 真实下发，已验证生效）。

## 9. 最终裁定

| 候选 | 裁定 | 依据 |
|---|---|---|
| A-S（scalar-U/T @ A 几何） | **reject** | 同几何 -0.99%，两轮全慢 |
| G128-S（scalar-U/T @ G128） | **reject** | vs G128 -0.088% 无信号；vs 生产 A -1.283% 全慢 |
| **A（生产几何，SCALAR_UT=0）** | **keep（维持生产配置）** | 本轮最快正确配置 |

无晋升 → recovery smoke 无需新增（A 的 smoke 已在 R3/R4 留档 PASS）。

## 10. 明确声明

**scalar-U/T 就此关闭为一个优化方向**：它不改变寄存器/spill 分配，硬件上也无指令调度层面的
可观收益（A 几何下甚至回退 ~1%）。**低成本 live-range 家族至此全部耗尽**
（G192、T_SHARED、half-shared、shared HMAC state、noinline/boundary、fixed64 in-place HMAC
[SASS 字节相同]、NVRTC 12.4 pinning、scalar-U/T —— 全部已关闭）。
是否进入更大的 SHA-512 指令级重设计，由远端 agent 决定。

## 11. 给远端 agent 的备注

- 本轮唯一的工具性异常：committed harness（`scripts/bench_sm86_r4.sh` @ 6c9fb8e）接受额外
  env 赋值但只写日志、不下发进程——若未来还要按 flag 跑变体，需沿用 `env $envline` 下发行
  （`scripts/bench_sm86_r5.sh` 已示范）。
- A-S 的功率画像与 A 几乎相同（222–228 W vs 224–229 W），调度差异不来自功耗路径，
  与"分配相同、仅调度不同"的静态结论一致。
