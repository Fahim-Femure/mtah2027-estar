"""Compare the running interpreter and installed packages with requirements.lock.

    python scripts/check_environment.py

requirements.lock (Data Engineering, verified Python 3.11 x64 rebuild) is the
team's single dependency lock. Prints versions; exit 0 when everything matches
the lock, 1 otherwise. No network access, no installation.
"""
from importlib import metadata
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock"
PYTHON = (3, 11, 9)
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s=]+)$")


def parse_constraints(text: str) -> dict:
    pins = {}
    for raw in text.lstrip("﻿").splitlines():   # tolerate a UTF-8 BOM (PowerShell 5.1)
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN.match(line)
        if not match:
            raise ValueError(f"lock entries must be exact name==version pins: {line!r}")
        pins[match.group(1).lower()] = match.group(2)
    return pins


def compare(pins: dict, installed: dict) -> list:
    problems = []
    for name, version in pins.items():
        found = installed.get(name)
        if found is None:
            problems.append(f"{name}: locked {version}, not installed")
        elif found != version:
            problems.append(f"{name}: locked {version}, installed {found}")
    return problems


def python_problems(version_info) -> list:
    running = tuple(version_info[:3])
    if running == PYTHON:
        return []
    return [f"python: locked {'.'.join(map(str, PYTHON))}, running {'.'.join(map(str, running))}"]


def installed_versions(names) -> dict:
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def read_lock() -> dict:
    return parse_constraints(LOCK.read_text(encoding="utf-8-sig"))


def main() -> int:
    pins = read_lock()
    installed = installed_versions(pins)
    print(f"python {platform.python_version()} {platform.architecture()[0]} ({sys.executable})")
    for name in pins:
        print(f"{name} {installed.get(name, 'not installed')}")
    problems = python_problems(sys.version_info) + compare(pins, installed)
    if problems:
        print("MISMATCH")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("environment matches requirements.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
