import os
import sys

# Add parent directory to path to import version
_parent_dir = os.path.dirname(os.path.abspath(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from version import FLAGSCALE_VERSION

__version__ = FLAGSCALE_VERSION
