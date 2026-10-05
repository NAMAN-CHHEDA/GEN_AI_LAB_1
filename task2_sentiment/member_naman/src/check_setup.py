"""Tiny sanity check: configs load, seeding works, a device is found, the logger writes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import utils` works from anywhere

import torch

from utils import Logger, get_device, load_config, log_config, set_seed


def report(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return ok


def main():
    results = []

    cfgs = {}
    for name in ("local", "gpu"):
        try:
            cfgs[name] = load_config(name)
            results.append(report(f"load_config('{name}')", True))
        except Exception as e:
            print(f"  error: {e}")
            results.append(report(f"load_config('{name}')", False))

    if len(cfgs) == 2:
        same = cfgs["local"]["data"].keys() == cfgs["gpu"]["data"].keys() and \
            cfgs["local"]["train"].keys() == cfgs["gpu"]["train"].keys() and \
            cfgs["local"]["models"].keys() == cfgs["gpu"]["models"].keys()
        results.append(report("local and gpu configs have the same keys", same))

    try:
        set_seed(42)
        a = torch.rand(1).item()
        set_seed(42)
        results.append(report("set_seed gives repeatable numbers", a == torch.rand(1).item()))
    except Exception as e:
        print(f"  error: {e}")
        results.append(report("set_seed", False))

    try:
        device, device_name = get_device()
        print(f"  device: {device} ({device_name})")
        results.append(report("get_device", True))
    except Exception as e:
        print(f"  error: {e}")
        results.append(report("get_device", False))

    try:
        from jupyter_client.kernelspec import KernelSpecManager
        kernels = KernelSpecManager().find_kernel_specs()
        print(f"  kernels found: {sorted(kernels)}")
        results.append(report("ipykernel: a 'python3' kernel is installed", "python3" in kernels))
    except Exception as e:
        print(f"  error: {e}")
        results.append(report("ipykernel kernel check", False))

    if "local" in cfgs:
        try:
            log_path = cfgs["local"]["paths"]["output_dir"] / "setup_check.log"
            logger = Logger(log_path)
            log_config(logger, cfgs["local"])
            results.append(report(f"Logger wrote {log_path.name}", log_path.exists()))
        except Exception as e:
            print(f"  error: {e}")
            results.append(report("Logger", False))

    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
