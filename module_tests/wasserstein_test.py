# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
root = str(script_dir.parent)
print(root)
sys.path.insert(0, root + '/modules')

# import required modules
import wasserstein as ws
import numpy as np 

ws.compute_cost_evolution(ens_file='data/sde_evolve_test_2d_n.h5', save_path='data/sde_evolve_test_2d_n_cost_2.h5')