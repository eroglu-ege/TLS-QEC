"""
io.py
=====
Save and load simulation results.

Format: HDF5 (.h5) via h5py.
Why HDF5:
  - Stores arrays + metadata (params) in one file
  - Fast, compressed, readable by Python/MATLAB/Julia
  - Standard in physics/QEC research
  - Human-inspectable with HDFView or h5py

File structure:
  results.h5
  ├── t              (float64 array)
  ├── P_e_q          (float64 array)
  ├── P_e_t          (float64 array)
  ├── coherence      (float64 array, Lindblad only)
  ├── P_e_analytic   (float64 array, closed system only)
  └── params/
        wq, wt, g, gamma_q, gamma_t, ...  (scalar attributes)

Usage:
    from utils.io import save, load
    save(result, 'data/single_trajectory/lindblad_resonant.h5')
    result = load('data/single_trajectory/lindblad_resonant.h5')
"""

import h5py
import numpy as np
from pathlib import Path


def save(result: dict, path: str):
    """
    Save a simulation result dictionary to HDF5.

    Parameters
    ----------
    result : dict  (output of lindblad.evolve, solomon.evolve, etc.)
    path   : str   file path, should end in .h5
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        # Save all numpy arrays
        array_keys = ['t', 'P_e_q', 'P_e_t', 'coherence',
                      'P_e_analytic', 'ISE_integrand']
        for key in array_keys:
            if key in result and result[key] is not None:
                f.create_dataset(key, data=np.array(result[key]),
                                 compression='gzip')

        # Save scalar results
        scalar_keys = ['Gamma', 'max_diff', 'ISE',
                       'coherence_max', 'coherence_integral']
        for key in scalar_keys:
            if key in result:
                f.attrs[key] = result[key]

        # Save params dict as attributes in a group
        if 'params' in result:
            grp = f.create_group('params')
            for k, v in result['params'].items():
                grp.attrs[k] = v

    print(f"Saved → {path}")


def load(path: str) -> dict:
    """
    Load a simulation result from HDF5.

    Returns
    -------
    dict with same structure as the original result dict
    """
    path = Path(path)
    result = {}

    with h5py.File(path, 'r') as f:
        # Load arrays
        for key in f.keys():
            if key != 'params':
                result[key] = f[key][:]

        # Load scalar attributes
        for key in f.attrs:
            result[key] = f.attrs[key]

        # Load params
        if 'params' in f:
            result['params'] = dict(f['params'].attrs)

    return result


def save_sweep(sweep_results: dict, path: str):
    """
    Save a parameter sweep result.

    sweep_results should contain:
        'sweep_values' : array of swept parameter values
        'ISE'          : array of ISE values
        'max_diff'     : array of max differences
        'coherence_max': array of peak coherences
        'sweep_param'  : string label for the swept parameter
        'fixed_params' : dict of fixed parameters
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        for key in ['sweep_values', 'ISE', 'max_diff', 'coherence_max',
                    'gamma1_L', 'gamma2_L', 'gamma1_S', 'gamma2_S']:
            if key in sweep_results:
                f.create_dataset(key, data=np.array(sweep_results[key]),
                                 compression='gzip')

        if 'sweep_param' in sweep_results:
            f.attrs['sweep_param'] = sweep_results['sweep_param']

        if 'fixed_params' in sweep_results:
            grp = f.create_group('fixed_params')
            for k, v in sweep_results['fixed_params'].items():
                grp.attrs[k] = v

    print(f"Saved sweep → {path}")


def load_sweep(path: str) -> dict:
    """Load a sweep result from HDF5."""
    path = Path(path)
    result = {}

    with h5py.File(path, 'r') as f:
        for key in f.keys():
            if key != 'fixed_params':
                result[key] = f[key][:]
        for key in f.attrs:
            result[key] = f.attrs[key]
        if 'fixed_params' in f:
            result['fixed_params'] = dict(f['fixed_params'].attrs)

    return result


def save_phase_diagram(grid_result: dict, path: str):
    """
    Save a 2D phase diagram result.

    grid_result should contain:
        'x_values' : 1D array (e.g. g/Delta values)
        'y_values' : 1D array (e.g. g/gamma_t values)
        'ISE_grid' : 2D array of shape (len(x), len(y))
        'x_label'  : string
        'y_label'  : string
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        for key in ['x_values', 'y_values', 'ISE_grid',
                    'coherence_grid', 'max_diff_grid']:
            if key in grid_result:
                f.create_dataset(key, data=np.array(grid_result[key]),
                                 compression='gzip')
        for key in ['x_label', 'y_label']:
            if key in grid_result:
                f.attrs[key] = grid_result[key]

    print(f"Saved phase diagram → {path}")


def load_phase_diagram(path: str) -> dict:
    """Load a phase diagram from HDF5."""
    path = Path(path)
    result = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            result[key] = f[key][:]
        for key in f.attrs:
            result[key] = f.attrs[key]
    return result
