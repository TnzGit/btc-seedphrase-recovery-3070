#!/usr/bin/env python3
"""Patch to dump full CUDA function attributes for the recovery kernel."""
import pathlib

mod = pathlib.Path("src/gpu/mod.rs")
s = mod.read_text()
anchor = "    pub fn device_name(&self)"
probe = """    pub fn probe_launch_attrs(&self) -> Vec<String> {
        let mut out = Vec::new();
        let k = &self.enum_kernel;
        let mut push = |name: &str, r: Result<i32, _>| {
            match r {
                Ok(v) => out.push(format!("{}={}", name, v)),
                Err(e) => out.push(format!("{} ERR {}", name, e)),
            }
        };
        push("num_regs", k.num_regs());
        push("local_size_bytes", k.local_size_bytes());
        push("shared_size_bytes", k.shared_size_bytes());
        push("const_size_bytes", k.const_size_bytes());
        push("max_threads_per_block", k.max_threads_per_block());
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
