import sys
sys.path.insert(0, 'scripts')
from utils.hdf5_reader import load_xrf, get_channel, list_element_channels
import numpy as np

data = load_xrf('data/Data_May2026/SMW_UA1_P1_250um_10ms_12000_0_001.hdf5')
print(f"{'Channel':<10} {'mean':>8} {'max':>10} {'n_unique':>10}")
print('-' * 42)
for name in list_element_channels(data):
    ch = get_channel(data, name)
    uv = len(np.unique(ch))
    print(f"{name:<10} {ch.mean():>8.1f} {ch.max():>10.1f} {uv:>10}")
