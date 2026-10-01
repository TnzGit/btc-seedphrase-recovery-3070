# RTX 3070 / SM86 — R4 NVRTC 环境隔离实验报告

执行角色：本地执行 agent（按 `NEXT_AGENT_PROMPT_R4.md` 执行；未改动 CUDA/Rust 热路径，kernel.cu 未打补丁）。

- 分支：`opt/sm86-rtx3070-r4-env`（基线结果 commit `4ef1905a485a7f00eac2b1eda4c4164757c1b09a`，
  已用 `git merge-base --is-ancestor` 验证；远端 checkout 经 git-bundle 传输，因主机直连 GitHub 网络抖动）
- 未修改 `main`，未重写 R2/R3 结果分支
- 远端工作目录：`/home/user/r3hw/btc-seedphrase-recovery-3070`；N124 私有 scratch：`/home/user/r4env/`
- **全程使用同一个构建产物**（`cargo build --locked --release`，源不变 → 0.25s no-op，二进制即 R3 构建件）

## 1. 两个 NVRTC 环境（证据见 `results/r4/env-facts.txt`）

| | N129（R3 已知环境） | N124（重建 R2 风格环境） |
|---|---|---|
| 目录 | `/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib` | `/home/user/r4env/nvrtc124/usr/lib/x86_64-linux-gnu` |
| ctypes `nvrtcVersion` | **12.9**（文件 12.9.86） | **12.4**（文件 12.4.127，与 R2 记录一致） |
| builtins 同目录 | `libnvrtc-builtins.so.12.9.86` ✓ | `libnvrtc-builtins.so.12.4.127` ✓ |
| 来源 | R3 遗留目录 | `apt-get download libnvrtc12 libnvrtc-builtins12.4`（12.4.127~12.4.1-8）+ `dpkg-deb -x`，**未装系统包、未动驱动/机器级 CUDA** |
| LD_LIBRARY_PATH | `<dir>:/usr/lib/wsl/lib` | `<dir>:/usr/lib/wsl/lib` |

- R2 scratch `/tmp/btc3070bench-j9PBaw` 已不存在（主机重启后 tmpfs 清空），故按下述方式重建 N124。
- 防混杂验证：`/usr/lib/wsl/lib` 只有 `libcuda*`（无 libnvrtc），不可能遮蔽任一工具包；
  `LD_DEBUG=files` 证明每个环境的 libnvrtc 与 builtins 都从**同一私有目录**加载
  （`results/r4/logs/gate-correctness.log` 保存了每个环境的 "calling init" 行）。
- 每个 bench run 日志头部都内嵌了同一目录、同一 LD_LIBRARY_PATH 下的 `nvrtcVersion` 探测
  （`scripts/nvrtc_ver.py`），即"进程所用的 NVRTC 版本"的直接证据。

## 2. GPU / 驱动 / Rust 环境

RTX 3070 8192 MiB (GA104/SM86)；驱动 596.36；功率上限 240 W（全程未改）；
WSL kernel `6.6.87.2-microsoft-standard-WSL2`；Ubuntu 26.04；Rust/Cargo 1.93.1。

## 3. 正确性门禁（4 个组合，`--self-test`）

全部 exit 0、3/3 BIP84 PASS（gate 日志含 LD_DEBUG 加载证据）：

| 组合 | self-test | Kernel resources |
|---|---|---|
| N124 + A | PASS 3/3 | `regs/thread=128 local/thread=2016B shared/block=0B max_threads/block=256` |
| N129 + A | PASS 3/3 | 同上 |
| N124 + G128 | PASS 3/3 | `regs/thread=168 local/thread=1744B shared/block=0B max_threads/block=128` |
| N129 + G128 | PASS 3/3 | 同上 |

**运行时资源属性在两个编译器下完全一致** → 按本轮判据，这不是 major resource finding。
（首轮 gate 曾因我漏传 LB 环境变量而全部跑成 A 几何，已重做并留档，不影响最终数据。）

## 4. Phase 1 sanity（8 run，交错 N124A N129A N124G128 N129G128 N129G128 N124G128 N129A N124A）

| run | 组合 | weighted_steady |
|---|---|---|
| p1-r1-N124A | N124+A | 312287 |
| p1-r1-N129A | N129+A | 309941 |
| p1-r1-N124G128 | N124+G128 | 310002 |
| p1-r1-N129G128 | N129+G128 | 309035 |
| p1-r2-N129G128 | N129+G128 | 311481 |
| p1-r2-N124G128 | N124+G128 | 320256 |
| p1-r2-N129A | N129+A | 316713 |
| p1-r2-N124A | N124+A | 318403 |

无灾难性回归；G128 在两个编译器下都在 A 的 ±2% 内 → Phase 2B 成立。

## 5. Phase 2A（主要结论）：N124-A vs N129-A，3 轮 X X A A

| round | N124-A1 | N129-A1 | N129-A2 | N124-A2 | 轮内配对 Δ(N124 vs N129) |
|---|---|---|---|---|---|
| 1 | 317897 | 316495 | 314146 | 316371 | **+0.575%** |
| 2 | 308514 | 304180 | 303683 | 303706 | **+0.717%** |
| 3 | 303382 | 302870 | 303771 | 305277 | **+0.333%** |

汇总（各 n=6）：N124-A mean 309191.2 / median 306895.5 / min 303382 / max 318403；
N129-A mean 307524.2 / median 303975.5 / min 302870 / max 316713。

**配对增益 N124 vs N129：mean +0.542%，median +0.961%。三轮方向全部为 N124 优，但均 <2%。**

### 5.1 遥测画像（真实内核差异信号）

系统性差异：**N124 run 平均 SM clock 1822–1898 MHz、平均功率 193–205 W；
N129 run 平均 SM clock ~1950 MHz、平均功率 220–233 W**（温度峰值同为 70 °C，功率上限从未触发）。
即同一份源码在两个 NVRTC 下 JIT 出的 SASS 调度特性不同（N124 内核每瓦特产出更高、时钟更低却更快），
这与 throughput +0.5% 的方向自洽——但幅度低于门槛，不构成可晋升的工具链结论。
本轮**未改时钟、未做 ncu/nsys 深度 profiling**（按 prompt）。

### 5.2 regime / 污染

整个会话存在连续热漂移（round1 ~317k → round3 ~303k，温度顶到 70 °C 目标位），
属同一 regime 类内的连续漂移而非离散边界；所有比较都在轮内配对完成。
无排除/标注 run；全部 26 个 bench run + 2 个 smoke run exit 0。

## 6. Phase 2B：G128 vs A 在两个编译器下（各 2 轮 A G128 G128 A）

| 轮 | N124: A 均值 | N124: G128 均值 | G-vs-A | N129: A 均值 | N129: G128 均值 | G-vs-A |
|---|---|---|---|---|---|---|
| 1 | 307470.5 | 309268.0 | **+0.585%** | 309646.5 | 310562.0 | **+0.296%** |
| 2 | 306893.0 | 309509.5 | **+0.853%** | 308541.0 | 309713.0 | **+0.380%** |

**编译器版本不改变 G128-vs-A 的方向**（两个编译器下 G128 都小幅胜出，N124 下幅度略大：
N124 两轮均值 +0.72%，N129 两轮均值 +0.34%），与 R3 的 +0.326% 同向。
所有数值仍 <2% → 无一达到晋升门槛。

## 7. Recovery smoke（公开 BIP84 向量 #0，两环境均跑）

N129+A 与 N124+A 均：`RECOVERY SUCCESSFUL`、目标地址、默认路径、exit 0、无 phrase 泄漏
（`results/r4/logs/smoke-N129-A.log`、`smoke-N124-A.log`）。

## 8. 最终裁定

| 环境 | 裁定 | 依据 |
|---|---|---|
| **N129（保留为默认）** | **keep** | 与 R3 一致、已就位；N124 的 +0.5~1% 增益低于 2% 门槛 → 按判据视为 operating noise；换环境的运维收益不成立 |
| **N124** | **reject（不晋升/不 pin）** | 三轮一致更快（mean +0.542% / median +0.961%），方向真实、有功率/时钟画像佐证，但幅度 <2%；运行时资源无差异 |

## 9. 交给远端 agent 的 setup/README 建议

1. README/setup.sh 应写明 **NVRTC 与 builtins 必须来自同一 toolkit 目录**、`/usr/lib/wsl/lib` 仅提供
   libcuda 的配对规则，以及 `nvrtcVersion` 探测脚本的用法（本轮已提供 `scripts/nvrtc_ver.py`）。
2. setup.sh 目前不 pin NVRTC 版本；如未来要固定工具链，建议按本轮方式
   （`apt-get download` + `dpkg-deb -x` 私有目录）而非系统安装，保留双版本对照能力。
3. 跨轮绝对吞吐对比必须注明 NVRTC 版本与遥测画像（本轮证明 12.4/12.9 的功率-时钟画像系统性不同，
   混用会污染跨轮比较）。
4. 主机直连 GitHub 不稳定（R4 期间 `git fetch` 超时 135 s），传 bundle 可行；README 可提示。
