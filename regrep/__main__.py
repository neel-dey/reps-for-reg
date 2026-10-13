"""``python -m regrep`` runs the command line."""

import sys

from regrep.cli import main

# The guard keeps the batch workers, which start by importing this module,
# from running the command line again.
if __name__ == "__main__":
    sys.exit(main())
