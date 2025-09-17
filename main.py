#!/usr/bin/env python3
"""Legacy entry point delegating to the packaged CLI."""

from __future__ import annotations

import sys

from mindthread_app.cli import main as cli_main


if __name__ == "__main__":
    sys.exit(cli_main())
