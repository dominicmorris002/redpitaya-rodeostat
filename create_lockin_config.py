"""
create_lockin_config.py - Generate correct YAML config for lock-in amplifier
Run this first to create the config file, then run the lock-in code.
"""

import yaml
import os

# ------------------------- Configuration Parameters -------------------------
HOSTNAME = 'rp-f073ce.local'
YAML_FILE = 'lockin_scope_config.yml'

REF_FREQUENCY = 100        # Hz
REF_AMPLITUDE = 0.4        # V
OUTPUT_CHANNEL = 'out1'    # 'out1' or 'out2'
FILTER_BANDWIDTH = 10      # Hz
PHASE_OFFSET = 0           # degrees
DECIMATION = 128           # MUST be one of: 1, 8, 64, 1024, 8192, 65536

# ----------------------------------------------------------------------------

def create_config():
    """Create YAML config file for lock-in amplifier"""
    
    # Validate decimation
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]
    if DECIMATION not in allowed_decimations:
        print(f"ERROR: Decimation {DECIMATION} is invalid!")
        print(f"Allowed values: {allowed_decimations}")
        return False
    
    config = {
        'redpitaya_hostname': HOSTNAME,
        'modules': ['scope', 'iq2'],
        'scope': {
            'ch1_active': True,
            'ch2_active': True,
            'input1': 'out1',
            'input2': 'in1',
            'threshold': 0.0,
            'hysteresis': 0.0,
            'duration': 0.01,
            'trigger_delay': 0.0,
            'trigger_source': 'immediately',
            'running_state': 'running_continuous',
            'average': False,
            'decimation': DECIMATION
        },
        'iq2': {
            'input': 'in1',
            'frequency': REF_FREQUENCY,
            'bandwidth': FILTER_BANDWIDTH,
            'gain': 0.0,
            'phase': PHASE_OFFSET,
            'acbandwidth': 0,
            'amplitude': REF_AMPLITUDE,
            'output_direct': OUTPUT_CHANNEL,
            'output_signal': 'quadrature',
            'quadrature_factor': 1
        }
    }
    
    # Write YAML file
    with open(YAML_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("=" * 60)
    print(f"✓ Created config file: {YAML_FILE}")
    print("=" * 60)
    print("Configuration:")
    print(f"  Hostname: {HOSTNAME}")
    print(f"  Scope decimation: {DECIMATION} (sample rate: {125e6/DECIMATION:.2f} Hz)")
    print(f"  IQ2 frequency: {REF_FREQUENCY} Hz")
    print(f"  IQ2 amplitude: {REF_AMPLITUDE} V → {OUTPUT_CHANNEL}")
    print(f"  IQ2 input: in1")
    print(f"  Filter bandwidth: {FILTER_BANDWIDTH} Hz")
    print("=" * 60)
    print(f"Sample rate: {125e6/DECIMATION:.2f} Hz")
    print(f"Duration per capture: 0.01 s")
    print(f"Samples per capture: {int(0.01 * 125e6/DECIMATION)}")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = create_config()
    if success:
        print("\n✓ Config file created successfully!")
        print(f"  Now you can run the lock-in amplifier code.")
        print(f"  Make sure OUT1 is connected to IN1.")
    else:
        print("\n✗ Failed to create config file!")
