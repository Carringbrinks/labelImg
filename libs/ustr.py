import sys


def ustr(x):
    """py3 unicode passthrough helper (kept for API compatibility).

    Qt5 has no QString wrapper: strings arrive as native str. None passes
    through untouched so that callers can keep their None checks.
    """
    if x is None or isinstance(x, str):
        return x
    return str(x)
