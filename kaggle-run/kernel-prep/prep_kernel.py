"""Bootstrap for the G-Doodle corpus build. CPU only, so it costs no GPU quota."""

import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-doodle.git"
BOOT = "/tmp/gd-boot"

subprocess.run(["git", "clone", "--depth", "1", REPO, BOOT], check=True)
subprocess.run([sys.executable, f"{BOOT}/kaggle/01-prep.py"], check=True)
