"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

import os
from pathlib import Path

# Point acados at the repo's bundled `acados/` build before any acados_template import, so
# it stops printing "Did not find environment variable ACADOS_SOURCE_DIR ..." on every
# solver build. setdefault keeps any explicit override; we only set it when the dir exists,
# so on machines where acados lives elsewhere acados_template falls back to its own guess.
_acados_dir = Path(__file__).resolve().parent.parent / "acados"
if _acados_dir.is_dir():
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(_acados_dir))

from crazyflow.utils import enable_cache

import lsy_drone_racing.envs  # noqa: F401, register environments with gymnasium

enable_cache()  # Enable persistent caching of jax functions
