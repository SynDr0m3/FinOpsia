"""Runtime bootstrap for Prophet's CmdStan dependency on Windows."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cmdstanpy
import importlib_resources


CMDSTAN_VERSION = "2.33.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CMDSTAN_DIR = PROJECT_ROOT / "data" / "cmdstan" / f"cmdstan-{CMDSTAN_VERSION}"
_CONFIGURED_PATH: Path | None = None


def _is_cmdstan_runtime(path: Path) -> bool:
    if not path.is_dir():
        return False
    bin_dir = path / "bin"
    return (bin_dir / "stanc.exe").exists() or (bin_dir / "stanc").exists()


def _ensure_makefile(path: Path) -> Path:
    makefile = path / "makefile"
    if not makefile.exists():
        makefile.write_text("# FinOpsia CmdStan runtime shim\n", encoding="ascii")
    return path


def _project_cmdstan_copy(source: Path) -> Path:
    PROJECT_CMDSTAN_DIR.parent.mkdir(parents=True, exist_ok=True)
    if PROJECT_CMDSTAN_DIR.exists():
        return _ensure_makefile(PROJECT_CMDSTAN_DIR)
    shutil.copytree(source, PROJECT_CMDSTAN_DIR)
    return _ensure_makefile(PROJECT_CMDSTAN_DIR)


def _bundled_prophet_runtime() -> Path | None:
    bundled = importlib_resources.files("prophet") / "stan_model" / f"cmdstan-{CMDSTAN_VERSION}"
    candidate = Path(str(bundled))
    return candidate if _is_cmdstan_runtime(candidate) else None


def _fallback_runtime_candidates() -> list[Path]:
    candidates: list[Path] = []

    for env_name in ("FINOPSIA_CMDSTAN_HOME", "CMDSTAN"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw))

    candidates.append(PROJECT_CMDSTAN_DIR)

    bundled = _bundled_prophet_runtime()
    if bundled is not None:
        candidates.append(bundled)

    system_drive = Path(sys.executable).anchor or "C:\\"
    candidates.extend(
        Path(p) for p in Path(system_drive).glob(r"Python*\Lib\site-packages\prophet\stan_model\cmdstan-*")
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def ensure_cmdstan_runtime() -> Path:
    """Configure cmdstanpy to use a local CmdStan runtime if the env lacks one."""
    global _CONFIGURED_PATH

    if _CONFIGURED_PATH is not None:
        return _CONFIGURED_PATH

    try:
        _CONFIGURED_PATH = Path(cmdstanpy.cmdstan_path())
        return _CONFIGURED_PATH
    except ValueError:
        pass

    for candidate in _fallback_runtime_candidates():
        if not _is_cmdstan_runtime(candidate):
            continue

        runtime = candidate
        if not (runtime / "makefile").exists():
            runtime = _ensure_makefile(runtime) if runtime == PROJECT_CMDSTAN_DIR else _project_cmdstan_copy(runtime)

        cmdstanpy.set_cmdstan_path(str(runtime))
        os.environ["CMDSTAN"] = str(runtime)
        _CONFIGURED_PATH = runtime
        return runtime

    raise RuntimeError(
        "Prophet requires a CmdStan runtime, but none was found. "
        "Set CMDSTAN or FINOPSIA_CMDSTAN_HOME to a valid CmdStan directory."
    )
