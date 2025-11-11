import os
import sys

# Ensure the project parent directory is on sys.path so imports like
# `from toysim import simulation` resolve when tests are executed from the
# project directory. This mirrors running pytest from the repository parent.
ROOT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_PARENT not in sys.path:
    sys.path.insert(0, ROOT_PARENT)
