"""Compatibility entrypoint for walk log preprocessing.

This wrapper keeps existing command usage intact while delegating to the
maintained implementation in preprocess_walk_log_v2.py.
"""

from preprocess_walk_log_v2 import main


if __name__ == "__main__":
    main()
