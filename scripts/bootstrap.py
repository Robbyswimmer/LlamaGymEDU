#!/usr/bin/env python3
"""Cross-platform environment bootstrapper for LlamaGym EDU.

This script creates (or reuses) a virtual environment, installs the
project's pinned dependencies, and optionally installs the package itself.
It understands macOS, Windows, and Linux defaults so students can get up
and running with a single command:

    python scripts/bootstrap.py

Use ``python scripts/bootstrap.py --help`` to see all available options.
"""

from __future__ import annotations

import argparse
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"
DEFAULT_VENV = PROJECT_ROOT / ".venv"
AVAILABLE_REQUIREMENT_EXTRAS = {
    "linux": REQUIREMENTS_DIR / "linux.txt",
    "textworld": REQUIREMENTS_DIR / "textworld.txt",
}


def _quote(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _python_from_venv(venv_path: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _ensure_venv(venv_path: Path, *, recreate: bool, interpreter: str, dry_run: bool) -> Path:
    if recreate and venv_path.exists():
        print(f"🧹 Removing existing virtual environment at {venv_path}")
        if not dry_run:
            shutil.rmtree(venv_path)

    python_in_venv = _python_from_venv(venv_path)
    if python_in_venv.exists():
        return python_in_venv

    print(f"🐍 Creating virtual environment at {venv_path}")
    cmd = [interpreter, "-m", "venv", str(venv_path)]
    print(f"  $ {_quote(cmd)}")
    if dry_run:
        return python_in_venv

    subprocess.run(cmd, check=True)

    python_in_venv = _python_from_venv(venv_path)
    if not python_in_venv.exists():
        raise RuntimeError(
            f"Virtual environment missing expected python executable at {python_in_venv}"
        )
    return python_in_venv


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    print(f"  $ {_quote(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def _pip(python_executable: Path, args: Iterable[str], *, dry_run: bool) -> None:
    _run([str(python_executable), "-m", "pip", *args], dry_run=dry_run)


def _resolve_extras(user_selection: Iterable[str] | None, skip_platform: bool) -> Set[str]:
    resolved: Set[str] = set()
    if user_selection:
        resolved.update(user_selection)
    elif not skip_platform and platform.system() == "Linux":
        # Linux machines can take advantage of bitsandbytes for quantization.
        resolved.add("linux")

    # Filter out extras that do not have requirement files.
    return {extra for extra in resolved if extra in AVAILABLE_REQUIREMENT_EXTRAS}


def _install_requirements(python_executable: Path, *, extras: Set[str], dry_run: bool) -> None:
    base_req = REQUIREMENTS_DIR / "base.txt"
    if not base_req.exists():
        raise FileNotFoundError(f"Missing requirements file: {base_req}")

    print("📦 Installing core dependencies")
    _pip(python_executable, ["install", "-r", str(base_req)], dry_run=dry_run)

    for extra in sorted(extras):
        req_path = AVAILABLE_REQUIREMENT_EXTRAS[extra]
        print(f"📦 Installing optional '{extra}' dependencies")
        _pip(python_executable, ["install", "-r", str(req_path)], dry_run=dry_run)


def _install_project(python_executable: Path, *, extras: Set[str], mode: str, dry_run: bool) -> None:
    if mode == "skip":
        print("⏭️  Skipping LlamaGym package installation as requested")
        return

    extra_suffix = ""
    if extras:
        extra_suffix = "[" + ",".join(sorted(extras)) + "]"

    target = f".{extra_suffix}"
    if mode == "editable":
        print("🛠️  Installing LlamaGym in editable mode")
        _pip(python_executable, ["install", "-e", target], dry_run=dry_run)
    elif mode == "wheel":
        print("🛠️  Installing LlamaGym as a wheel")
        _pip(python_executable, ["install", target], dry_run=dry_run)
    else:
        raise ValueError(f"Unknown install mode: {mode}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a cross-platform environment for LlamaGym.",
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
        help="Location of the virtual environment to create/use (default: .venv)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter used to create the virtual environment",
    )
    parser.add_argument(
        "--system-python",
        action="store_true",
        help="Use the current interpreter instead of creating a virtual environment",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the virtual environment before installing",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        choices=sorted(AVAILABLE_REQUIREMENT_EXTRAS.keys()),
        help="Optional dependency groups to install (defaults to OS-aware selection)",
    )
    parser.add_argument(
        "--no-platform-extras",
        action="store_true",
        help="Disable automatic extras based on the detected operating system",
    )
    parser.add_argument(
        "--install-mode",
        choices=["editable", "wheel", "skip"],
        default="editable",
        help="How to install the LlamaGym package itself",
    )
    parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        default=True,
        help="Upgrade pip, setuptools, and wheel before installing requirements",
    )
    parser.add_argument(
        "--no-upgrade-pip",
        action="store_false",
        dest="upgrade_pip",
        help="Skip upgrading pip/setuptools/wheel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.system_python:
        python_executable = Path(sys.executable)
        print("⚙️  Using current Python interpreter")
    else:
        venv_path = Path(args.venv).expanduser()
        python_executable = _ensure_venv(
            venv_path,
            recreate=args.recreate,
            interpreter=args.python,
            dry_run=args.dry_run,
        )
        print(f"⚙️  Using virtual environment python at {python_executable}")

    if args.upgrade_pip:
        print("⬆️  Upgrading pip/setuptools/wheel")
        _pip(
            python_executable,
            ["install", "--upgrade", "pip", "setuptools", "wheel"],
            dry_run=args.dry_run,
        )

    extras = _resolve_extras(args.extras, args.no_platform_extras)
    if extras:
        print(f"✨ Installing extras: {', '.join(sorted(extras))}")

    _install_requirements(python_executable, extras=extras, dry_run=args.dry_run)
    _install_project(python_executable, extras=extras, mode=args.install_mode, dry_run=args.dry_run)

    print("✅ Environment bootstrap complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
