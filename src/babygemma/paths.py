"""Repository-root-anchored paths, so modules and scripts resolve data the same way
regardless of where they live in the tree."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def results(*parts):
    return os.path.join(ROOT, *parts)
