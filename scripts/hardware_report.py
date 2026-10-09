#!/usr/bin/env python3
"""Collect a read-only hardware and Python environment report.

The script intentionally uses only the Python standard library. It tolerates a
missing or unhealthy NVIDIA runtime and records that limitation instead of
guessing VRAM or CUDA visibility.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "hardware_report.json"


def run_command(command: Sequence[str], timeout: float = 5.0) -> dict[str, Any]:
    """Run a read-only command and retain enough output for auditability."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {
            "command": list(command),
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "returncode": None,
            "ok": False,
            "stdout": _text_or_empty(exc.stdout),
            "stderr": f"timed out after {timeout:g}s",
        }
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def parse_human_bytes(value: str) -> int | None:
    """Parse values such as ``8192 MiB`` or ``8 GiB`` into bytes."""

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)", value, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "b": 1,
        "kb": 1_000,
        "kib": 1024,
        "mb": 1_000_000,
        "mib": 1024**2,
        "gb": 1_000_000_000,
        "gib": 1024**3,
        "tb": 1_000_000_000_000,
        "tib": 1024**4,
    }
    multiplier = multipliers.get(unit)
    return int(amount * multiplier) if multiplier is not None else None


def parse_nvidia_query(stdout: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi CSV query output without requiring nvidia-smi to exist."""

    devices: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or not fields[0]:
            continue
        memory_text = fields[1]
        devices.append(
            {
                "model": fields[0],
                "vram": {
                    "total_bytes": parse_human_bytes(memory_text),
                    "reported": memory_text,
                    "measured": True,
                },
                "driver_version": fields[2],
            }
        )
    return devices


def parse_lspci_graphics(stdout: str) -> list[dict[str, Any]]:
    """Extract conservative GPU identity and driver facts from lspci output."""

    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        header = re.match(
            r"^(?P<address>[0-9a-f:.]+)\s+.*?(VGA compatible controller|3D controller|Display controller).*?:\s*(?P<description>.+)$",
            line,
            re.IGNORECASE,
        )
        if header:
            match = re.search(r"\[(GeForce [^\]]+)\]", header.group("description"), re.IGNORECASE)
            model = match.group(1) if match else header.group("description")
            current = {
                "pci_address": header.group("address"),
                "description": header.group("description"),
                "model": model,
                "driver_in_use": None,
            }
            devices.append(current)
            continue
        if current is not None:
            driver_match = re.search(r"Kernel driver in use:\s*(.+)$", line.strip(), re.IGNORECASE)
            if driver_match:
                current["driver_in_use"] = driver_match.group(1).strip()
    return devices


def parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    """Read the standard Linux OS release file when available."""

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _lscpu_values(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def collect_cpu() -> dict[str, Any]:
    result = run_command(["lscpu"])
    values = _lscpu_values(result["stdout"])
    return {
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "model": values.get("Model name") or platform.processor() or None,
        "physical_cores": _as_int(values.get("Core(s) per socket")),
        "sockets": _as_int(values.get("Socket(s)")),
        "threads_per_core": _as_int(values.get("Thread(s) per core")),
        "raw_command": result,
    }


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def collect_memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, remainder = line.partition(":")
            if not remainder:
                continue
            match = re.search(r"(\d+)\s*kB", remainder)
            if match:
                values[key] = int(match.group(1)) * 1024
    except OSError:
        pass
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
    }


def collect_disk() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO_ROOT)
    return {
        "path": str(REPO_ROOT),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def collect_python() -> dict[str, Any]:
    interpreters: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in (sys.executable, shutil.which("python3"), shutil.which("python")):
        if not candidate:
            continue
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        version = run_command([candidate, "--version"])
        interpreters.append(
            {
                "path": candidate,
                "version": (version["stdout"] or version["stderr"]).strip(),
                "raw_command": version,
            }
        )
    return {
        "current_executable": sys.executable,
        "current_version": platform.python_version(),
        "interpreters": interpreters,
    }


def collect_pytorch() -> dict[str, Any]:
    result: dict[str, Any] = {
        "installed": importlib.util.find_spec("torch") is not None,
        "import_error": None,
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "devices": [],
    }
    if not result["installed"]:
        return result
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
            result["version"] = torch.__version__
            result["cuda_build"] = torch.version.cuda
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["cuda_device_count"] = int(torch.cuda.device_count())
            if result["cuda_available"]:
                for index in range(result["cuda_device_count"]):
                    properties = torch.cuda.get_device_properties(index)
                    result["devices"].append(
                        {
                            "index": index,
                            "name": properties.name,
                            "vram_total_bytes": int(properties.total_memory),
                            "compute_capability": f"{properties.major}.{properties.minor}",
                        }
                    )
    except Exception as exc:  # diagnostic script must report broken installs
        result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect_gpu() -> dict[str, Any]:
    smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    smi_devices = parse_nvidia_query(smi["stdout"]) if smi["ok"] else []
    lspci = run_command(["lspci", "-nnk"])
    pci_devices = parse_lspci_graphics(lspci["stdout"]) if lspci["ok"] else []
    proc_version = run_command(["cat", "/proc/driver/nvidia/version"])
    driver_match = re.search(r"NVRM version:\s*([^\n]+)", proc_version["stdout"])
    driver_raw = driver_match.group(1).strip() if driver_match else None
    numeric_driver_match = re.search(r"\b(\d+\.\d+\.\d+)\b", driver_raw or "")
    driver_version = numeric_driver_match.group(1) if numeric_driver_match else driver_raw

    devices: list[dict[str, Any]] = []
    if smi_devices:
        devices = smi_devices
        for index, device in enumerate(devices):
            if index < len(pci_devices):
                device.update(pci_devices[index])
    else:
        devices = pci_devices
        for device in devices:
            device["vram"] = {
                "total_bytes": None,
                "reported": None,
                "measured": False,
                "status": "not measured: nvidia-smi query failed",
            }
    for device in devices:
        device.setdefault("driver_version", driver_version)
        device.setdefault("driver_in_use", None)

    return {
        "detected": bool(devices),
        "devices": devices,
        "driver_version": driver_version,
        "driver_version_raw": driver_raw,
        "nvidia_smi_ok": smi["ok"],
        "nvidia_smi_error": smi["stderr"].strip() or None,
        "raw_commands": {"nvidia_smi": smi, "lspci": lspci, "proc_driver_version": proc_version},
    }


def collect_cuda_toolkit() -> dict[str, Any]:
    nvcc = run_command(["nvcc", "--version"])
    release_match = re.search(r"release\s+([0-9.]+)", nvcc["stdout"], re.IGNORECASE)
    return {
        "nvcc_available": nvcc["ok"],
        "nvcc_version": release_match.group(1) if release_match else None,
        "raw_command": nvcc,
    }


def collect_report() -> dict[str, Any]:
    """Collect all diagnostics without changing system or repository state."""

    return {
        "report_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO_ROOT),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "distribution": parse_os_release(),
        },
        "cpu": collect_cpu(),
        "system_memory": collect_memory(),
        "gpu": collect_gpu(),
        "nvidia_driver": {
            "proc_version_available": Path("/proc/driver/nvidia/version").exists(),
        },
        "cuda_toolkit": collect_cuda_toolkit(),
        "python": collect_python(),
        "pytorch": collect_pytorch(),
        "disk": collect_disk(),
    }


def _gib(value: int | None) -> str:
    return f"{value / 1024**3:.2f} GiB" if value is not None else "unknown"


def print_human_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable summary while JSON remains authoritative."""

    operating_system = report["operating_system"]
    distribution = operating_system["distribution"]
    os_name = distribution.get("PRETTY_NAME") or operating_system["system"]
    cpu = report["cpu"]
    memory = report["system_memory"]
    disk = report["disk"]
    pytorch = report["pytorch"]
    toolkit = report["cuda_toolkit"]
    print(f"OS: {os_name} ({operating_system['release']})")
    print(f"CPU: {cpu['model']} | logical CPUs: {cpu['logical_cpus']}")
    print(f"System RAM: {_gib(memory['total_bytes'])} | available: {_gib(memory['available_bytes'])}")
    print(f"Disk ({disk['path']}): {_gib(disk['free_bytes'])} free of {_gib(disk['total_bytes'])}")
    gpu = report["gpu"]
    if gpu["devices"]:
        for device in gpu["devices"]:
            vram = device.get("vram", {})
            vram_text = _gib(vram.get("total_bytes"))
            if not vram.get("measured", False):
                vram_text += " (not measured)"
            print(f"GPU: {device.get('model')} | VRAM: {vram_text}")
    else:
        print("GPU: no graphics device detected by the diagnostic")
    print(f"NVIDIA driver: {gpu.get('driver_version') or 'unavailable'} | nvidia-smi: {gpu['nvidia_smi_ok']}")
    print(f"CUDA toolkit (nvcc): {toolkit['nvcc_version'] or 'unavailable'}")
    print(
        f"PyTorch: {pytorch['version'] or 'not installed'} | build CUDA: {pytorch['cuda_build'] or 'none'} | "
        f"CUDA visible: {pytorch['cuda_available']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report destination (default: results/hardware_report.json)",
    )
    args = parser.parse_args(argv)
    report = collect_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_human_report(report)
    print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
