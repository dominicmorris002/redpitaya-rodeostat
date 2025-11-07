"""
Auto-create RedPitaya WebScope YAML and arm the scope
Shows IN1 and OUT1 in colorful overlay
"""

import os
import yaml
from pyrpl import Pyrpl

# ---------- Config ----------
HOSTNAME = "rp-f073ce.local"
YAML_FILE = "scope_config.yml"

# ---------- YAML content ----------
config = {
    'redpitaya_hostname': HOSTNAME,
    'modules': ['scope', 'asg0'],
    'scope': {
        'input1': 'in1',
        'input2': 'out1',
        'decimation': 256,         # smaller = more samples = zoomed in
        'average': False,
        'trigger_source': 'immediately',
        'trigger_edge': 'rising',
        'trigger_level': 0.0,
        'color_ch1': 'red',
        'color_ch2': 'cyan'
    },
    'asg0': {
        'waveform': 'sin',
        'frequency': 1000,
        'amplitude': 0.5,
        'offset': 0.0,
        'output_direct': 'out1',
        'trigger_source': 'immediately'
    }
}

# ---------- Create YAML ----------
if not os.path.exists(YAML_FILE):
    with open(YAML_FILE, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Created YAML config: {YAML_FILE}")
else:
    print(f"ℹ️ YAML already exists: {YAML_FILE}")

# ---------- Connect to RedPitaya ----------
rp = Pyrpl(config=YAML_FILE)
scope = rp.rp.scope
asg = rp.rp.asg0

# ---------- Arm scope ----------
scope.single()  # triggers the scope once
print("📊 Scope armed. Open WebScope in your browser:")
print(f"http://{HOSTNAME}:8000")
