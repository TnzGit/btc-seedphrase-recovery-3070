#!/usr/bin/env python3
"""Experiment: force lower register pressure with __launch_bounds__.

Correct placement for a C-linkage kernel is:

    extern "C" __global__ void __launch_bounds__(MAX_T, MIN_B) name(args)

i.e. the attribute goes after `__global__ void` and before the function name.
The earlier attempt put it before `extern "C"`, which NVRTC rejects.
"""
import pathlib, re, sys

max_t = sys.argv[1] if len(sys.argv) > 1 else "256"
min_b = sys.argv[2] if len(sys.argv) > 2 else "2"

cu = pathlib.Path("src/gpu/cuda/kernel.cu")
s = cu.read_text()

# Remove any previously injected attribute in either position.
s = re.sub(r'__launch_bounds__\([^)]*\)\s*\n?\s*extern "C" __global__ void recovery_enumerate\(',
           'extern "C" __global__ void recovery_enumerate(', s)
s = re.sub(r'extern "C" __global__ void __launch_bounds__\([^)]*\) recovery_enumerate\(',
           'extern "C" __global__ void recovery_enumerate(', s)

sig = 'extern "C" __global__ void recovery_enumerate('
assert sig in s, "kernel signature not found"

new_sig = f'extern "C" __global__ void __launch_bounds__({max_t}, {min_b}) recovery_enumerate('
s = s.replace(sig, new_sig, 1)
cu.write_text(s)
print(f"patched: __launch_bounds__({max_t}, {min_b})")
