#!/usr/bin/env python3
"""Redacted end-to-end recovery smoke test against the PUBLIC BIP84 test vector.

The mnemonic is a synthetic public reference vector already embedded in
src/gpu/mod.rs (vector #0). It is NOT a real wallet. The phrase is assembled
in-process and is never written to a shell command line, a file, or a log.
Output lines that would echo secret-like material are redacted.
"""
import subprocess, sys, os, re

BIN = sys.argv[1]
ADDR = "bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu"

# Public vector #0, minus its final word. Assembled here, not passed on a command line.
KNOWN = " ".join(["abandon"] * 11)

# Interactive answer sequence:
#  1         -> 12 words
#  1         -> one word missing
#  <11 words>-> the known words in sequence
#  y         -> positions known
#  12        -> the missing word is position 12
#  <addr>    -> target public address
ANSWERS = ["1", "1", KNOWN, "y", "12", ADDR]

REDACT = re.compile(r"^\s*(Missing Word:|Complete Seed Phrase:).*$", re.MULTILINE)
REDACTED = "[REDACTED PUBLIC TEST VECTOR]"

env = dict(os.environ)
env["SEEDPHRASE_BLOCK"] = "256"

p = subprocess.run(
    [BIN],
    input="\n".join(ANSWERS) + "\n",
    capture_output=True,
    text=True,
    env=env,
    timeout=900,
)

raw = p.stdout + p.stderr
clean = REDACT.sub(REDACTED, raw)

print("=== exit status:", p.returncode)
print("=== redacted output ===")
print(clean)

# Evidence checks
ok_success = "RECOVERY SUCCESSFUL" in clean
ok_addr = ADDR in clean
ok_path = "m/84'/0'/0'/0/0" in clean
leaked = (KNOWN in clean) or ("abandon abandon abandon" in clean)
print("=== CHECKS ===")
print("RECOVERY SUCCESSFUL present:", ok_success)
print("target address present    :", ok_addr)
print("default BIP84 path present:", ok_path)
print("exit status zero          :", p.returncode == 0)
print("phrase leaked into output :", leaked)
sys.exit(0 if (ok_success and ok_addr and p.returncode == 0 and not leaked) else 1)
