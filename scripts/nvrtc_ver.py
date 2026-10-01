#!/usr/bin/env python3
"""Report the active NVRTC version for a given library dir (ctypes nvrtcVersion)."""
import ctypes, os, sys, glob
d = sys.argv[1]
maj = ctypes.c_int(); minr = ctypes.c_int()
lib = ctypes.CDLL(os.path.join(d, "libnvrtc.so.12"))
lib.nvrtcVersion(ctypes.byref(maj), ctypes.byref(minr))
print(f"dir={d} nvrtcVersion={maj.value}.{minr.value} loaded={lib._name}")
bl = sorted(glob.glob(os.path.join(d, "libnvrtc-builtins.so.*")))
print("builtins_in_same_dir=", [os.path.basename(b) for b in bl])
