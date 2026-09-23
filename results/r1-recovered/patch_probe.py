#!/usr/bin/env python3
"""Patch src/main.rs and src/gpu/mod.rs to print CUDA launch attributes."""
import sys, pathlib

mod = pathlib.Path("src/gpu/mod.rs")
s = mod.read_text()
anchor = "    pub fn device_name(&self)"
probe = """    pub fn probe_launch_attrs(&self) -> Vec<String> {
        let mut out = Vec::new();
        match self.enum_kernel.num_regs() {
            Ok(n) => out.push(format!("num_regs={}", n)),
            Err(e) => out.push(format!("num_regs ERR {e}")),
        }
        out
    }

"""
if "probe_launch_attrs" not in s:
    assert anchor in s, "device_name anchor missing"
    s = s.replace(anchor, probe + anchor, 1)
    mod.write_text(s)

main = pathlib.Path("src/main.rs")
m = main.read_text()
old = '    println!("Device: {}", gpu.device_name());'
new = '''    println!("Device: {}", gpu.device_name());
    for line in gpu.probe_launch_attrs() {
        println!("PROBE {}", line);
    }'''
if "PROBE {}" not in m:
    assert old in m, "device print anchor missing"
    m = m.replace(old, new, 1)
    main.write_text(m)

print("patched ok")
