import subprocess
from datetime import datetime
import time
import sys
import os

# --- Get venv Python executable ---
python_exe = sys.executable  # ensures it uses the same Python interpreter as this script

# --- Scheduled start time ---
START_TIME = datetime.fromisoformat("2025-12-12 17:13:40.497741")
print("Start time scheduled:", START_TIME)

# Wait until start time
while datetime.now() < START_TIME:
    time.sleep(0.001)  # 1 ms resolution

# --- Paths to scripts ---
lockin_script = os.path.join(os.path.dirname(__file__), "lockin_with_timestamp.py")
dc_script = os.path.join(os.path.dirname(__file__), "dc_monitor_with_timestamp.py")

# --- Launch both scripts using the venv Python ---
proc_lockin = subprocess.Popen([python_exe, lockin_script])
proc_dc = subprocess.Popen([python_exe, dc_script])

# --- Wait for both to finish ---
proc_lockin.wait()
proc_dc.wait()

print("✓ Both acquisitions finished")
