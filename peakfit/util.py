import errno
import os
import os.path


def make_dirs(path=None):
    """Make the directory if needed"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
