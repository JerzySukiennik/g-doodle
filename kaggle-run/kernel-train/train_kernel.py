"""Bootstrap for G-Doodle training.

The corpus arrives through kernel_sources rather than as a Dataset, which skips
an 800 MB round trip through a home connection. 02-train.py globs for it
recursively because that mount sits deeper than a Dataset's.
"""

import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-doodle.git"
BOOT = "/tmp/gd-boot"

subprocess.run(["git", "clone", "--depth", "1", REPO, BOOT], check=True)
subprocess.run([sys.executable, f"{BOOT}/kaggle/02-train.py"], check=True)
