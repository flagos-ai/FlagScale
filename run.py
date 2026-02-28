#!/usr/bin/env python
"""Backward compatibility wrapper for flagscale.run.

This file is kept for backward compatibility with existing scripts and documentation.
New code should use: flagscale run ... or python -m flagscale.run
"""

from flagscale.run import main

if __name__ == "__main__":
    main()
