"""
Application package initializer.

Ensures the repository root is placed on ``sys.path`` so that local modules
like ``lion_linker`` are imported from the working tree even when the API
is launched from a nested directory (e.g., ``app/``). This prevents Python
from falling back to an outdated site-packages installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
