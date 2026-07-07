"""Make ``src/`` importable when the package is not pip-installed.

Lets ``pytest`` run straight from a checkout (``pytest tests/``) without an
editable install. When the package *is* installed, the installed copy still
wins because this only appends to ``sys.path``.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
