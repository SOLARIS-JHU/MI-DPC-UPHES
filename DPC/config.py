"""Configuration: physical constants, training hyperparameters, preprocess.pkl loader."""

import dill as pickle
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
DT = 3600.0        # Time step (seconds)
RHO = 1000.0       # Water density (kg/m^3)
G = 9.81            # Gravitational acceleration (m/s^2)
ETA = 0.8           # System efficiency (dimensionless)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LR = 3e-4  # Keep baseline learning rate
BATCH_SIZE = 32
GRAD_CLIP = 1.0

# Optimizer & Scheduler options
WEIGHT_DECAY = 1e-2      # for AdamW; decoupled L2 regularisation
WARMUP_EPOCHS = 10       # linear warmup before cosine decay (10% of 100-epoch default)

# Temperature scheduling
TAU_START = 10.0
TAU_END = 0.1  # TUNED: reduced from 0.3 for sharper mode decisions
# tau decays over 75% of total epochs (set dynamically in train.py)

# Curriculum
EPOCHS_PER_STAGE = 200
TOTAL_STAGES = 6
STEPS_PER_STAGE = 4   # nsteps grows as stage * STEPS_PER_STAGE

# Loss weights
HEAD_PENALTY = 50.0
C_OP = 0.4

# Mode bias target
TARGET_PROBS = [0.40, 0.15, 0.45]  # pump, idle, turbine

# Data
PRICE_NOISE_STD = 5.0    # TUNED: increased from 2.0 for broader price coverage
MIN_PRICE = -100.0       # FIX: allow negative prices (was 0.1, which blinded policy to neg-price days)
NUM_TRAIN_SAMPLES = 10000  # TUNED: increased from 5000 for better training diversity
EVAL_TAU = 0.5

# ---------------------------------------------------------------------------
# preprocess.pkl loader
# ---------------------------------------------------------------------------

def _move_defaults_to_device(fn, device):
    """Move any torch.Tensor default arguments of *fn* to *device* in-place."""
    if not callable(fn) or not hasattr(fn, '__defaults__'):
        return
    defaults = fn.__defaults__
    if defaults is None:
        return
    new_defaults = []
    changed = False
    for d in defaults:
        if isinstance(d, torch.Tensor) and d.device != device:
            new_defaults.append(d.to(device))
            changed = True
        else:
            new_defaults.append(d)
    if changed:
        fn.__defaults__ = tuple(new_defaults)


def _retarget_pickled_function(fn, device):
    """Retarget a pickled callable to the requested device in-place."""
    if hasattr(fn, '__globals__'):
        fn.__globals__['torch'] = torch
        fn.__globals__['device'] = device
    _move_defaults_to_device(fn, device)
    return fn


def _validate_inverse_upc_function(fn, name, device):
    """Eagerly validate an inverse UPC callable with a small sample call."""
    sample_q = torch.tensor([3.0, -3.0], device=device)
    sample_h = torch.tensor([77.0, 77.0], device=device)
    try:
        sample_output = fn(sample_q, sample_h)
    except Exception as exc:
        raise ValueError(f"{name} failed validation call: {exc}") from exc
    if not isinstance(sample_output, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor, got {type(sample_output)!r}")
    if sample_output.shape != sample_q.shape:
        raise ValueError(
            f"{name} returned shape {tuple(sample_output.shape)} for sample input "
            f"{tuple(sample_q.shape)}"
        )
    return fn


def _load_inverse_upc_functions(inverse_pkl_path, device):
    """Load inverse UPC callables from the separate inverse artifact."""
    _torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault('map_location', device)
        kwargs.setdefault('weights_only', False)
        return _torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
    try:
        with open(inverse_pkl_path, 'rb') as f:
            payload = pickle.load(f)
    finally:
        torch.load = _torch_load

    if not isinstance(payload, (tuple, list)):
        raise TypeError(f"Expected inverse UPC pickle payload to be tuple/list, got {type(payload)!r}")
    if len(payload) != 3:
        raise ValueError(f"Expected inverse UPC pickle payload with 3 items, got {len(payload)}")

    upc_inv_tur, upc_inv_pump, inverse_meta = payload
    if not callable(upc_inv_tur):
        raise TypeError(f"UPC_inv_tur must be callable, got {type(upc_inv_tur)!r}")
    if not callable(upc_inv_pump):
        raise TypeError(f"UPC_inv_pump must be callable, got {type(upc_inv_pump)!r}")

    upc_inv_tur = _retarget_pickled_function(upc_inv_tur, device)
    upc_inv_pump = _retarget_pickled_function(upc_inv_pump, device)
    _validate_inverse_upc_function(upc_inv_tur, 'UPC_inv_tur', device)
    _validate_inverse_upc_function(upc_inv_pump, 'UPC_inv_pump', device)
    return upc_inv_tur, upc_inv_pump, inverse_meta


def load_system_params(pkl_path, device=None, physics_mode='nonlinear', inverse_pkl_path=None):
    """Load system parameters from preprocess.pkl.

    Args:
        pkl_path: Path to preprocess.pkl
        device: Target torch device for coefficient tensors in the pickled
                functions. Defaults to 'cuda' if available, else 'cpu'.
        physics_mode: 'nonlinear' (default) uses polynomial UPC and nonlinear
                      v-h curves; 'linear' replaces all physics with linear
                      approximations at the operating point.
        inverse_pkl_path: Optional path to the separate inverse UPC artifact.

    Returns dict with all functions and scalar bounds needed by dynamics,
    objectives, and dataset modules.
    """
    if physics_mode not in ('nonlinear', 'linear'):
        raise ValueError(f"physics_mode must be 'nonlinear' or 'linear', got '{physics_mode}'")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # The pickle may contain CUDA-backed tensor storages from the environment
    # where it was generated. Force those storages onto the requested device so
    # CPU-only evaluation/training remains possible.
    _torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault('map_location', device)
        kwargs.setdefault('weights_only', False)
        return _torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
    try:
        with open(pkl_path, 'rb') as f:
            (v_up_to_h, h_to_v_up, v_low_to_h, h_to_v_low,
             coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin,
             UPC_linear_tur, UPC_linear_pump,
             UPC_poly_tur, UPC_poly_pump,
             neg_min, neg_max, pos_min, pos_max,
             h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit,
             DA_price_hour, DA_price_quarter,
             prepare_and_fit_model, get_UPC_bound, LR_UPC_bound,
             max_vol_up, max_vol_low, max_vol, head_min, head_max,
             nl_h_to_v_low, nl_v_low_to_h, nl_h_v_coeffs, nl_v_low_h_coeffs) = pickle.load(f)
    finally:
        torch.load = _torch_load

    # Fix pickled closures: inject globals and move default-arg tensors to
    # the target device.  The functions were pickled from an IPython/CUDA
    # session and carry coefficient tensors on cuda:0 as default arguments.
    _all_fns = [v_up_to_h, h_to_v_up, v_low_to_h, h_to_v_low,
                UPC_poly_tur, UPC_poly_pump, UPC_linear_tur, UPC_linear_pump,
                neg_min, neg_max, pos_min, pos_max,
                nl_h_to_v_low, nl_v_low_to_h]
    for fn in _all_fns:
        if hasattr(fn, '__globals__'):
            fn.__globals__['torch'] = torch
            fn.__globals__['device'] = device
            fn.__globals__['max_vol_up'] = max_vol_up
            fn.__globals__['max_vol_low'] = max_vol_low
            fn.__globals__['v_up_to_h'] = v_up_to_h
            fn.__globals__['h_to_v_up'] = h_to_v_up
            fn.__globals__['v_low_to_h'] = v_low_to_h
            fn.__globals__['h_to_v_low'] = h_to_v_low
            fn.__globals__['nl_h_v_coeffs'] = nl_h_v_coeffs
            fn.__globals__['nl_v_low_h_coeffs'] = nl_v_low_h_coeffs
        _move_defaults_to_device(fn, device)

    # Move nonlinear coefficient tensors to target device
    nl_h_v_coeffs = nl_h_v_coeffs.to(device)
    nl_v_low_h_coeffs = nl_v_low_h_coeffs.to(device)

    upc_inv_tur = None
    upc_inv_pump = None
    inverse_meta = None
    if inverse_pkl_path is not None:
        inverse_pkl_path = Path(inverse_pkl_path)
        upc_inv_tur, upc_inv_pump, inverse_meta = _load_inverse_upc_functions(inverse_pkl_path, device)

    # Derived initial / target values — use mode-appropriate inverse v-h function
    h_init = 77.0
    _hv_init_fn = nl_h_to_v_low if physics_mode == 'nonlinear' else h_to_v_low
    v_init = _hv_init_fn(torch.tensor(h_init, device=device))
    if isinstance(v_init, torch.Tensor):
        v_init = v_init.item()
    target_head = h_init
    target_vol_low = v_init

    # --- Select physics functions based on mode ---
    if physics_mode == 'linear':
        # UPC: use linear approximations (already loaded from pickle)
        upc_tur = UPC_linear_tur
        upc_pump = UPC_linear_pump

        # Volume-to-head: linearize at operating point
        eps = 1.0  # m^3
        v_init_t = torch.tensor(v_init, device=device)
        h_plus = v_low_to_h(v_init_t + eps)
        h_minus = v_low_to_h(v_init_t - eps)
        if isinstance(h_plus, torch.Tensor):
            h_plus = h_plus.item()
        if isinstance(h_minus, torch.Tensor):
            h_minus = h_minus.item()
        dh_dv = (h_plus - h_minus) / (2.0 * eps)

        def v_low_to_h_linear(v):
            """Linear tangent approximation of v_low_to_h at operating point."""
            return h_init + dh_dv * (v - v_init)

        def h_to_v_low_linear(h):
            """Inverse of linear v-h approximation."""
            return v_init + (h - h_init) / dh_dv

        vh_fn = v_low_to_h_linear
        hv_fn = h_to_v_low_linear

        # Power bounds: constants evaluated at h_init
        h_init_t = torch.tensor(h_init, device=device)
        _pos_min_val = float(pos_min(h_init_t))
        _pos_max_val = float(pos_max(h_init_t))
        _neg_min_val = float(neg_min(h_init_t))
        _neg_max_val = float(neg_max(h_init_t))

        def const_pos_min(h): return torch.full_like(h, _pos_min_val)
        def const_pos_max(h): return torch.full_like(h, _pos_max_val)
        def const_neg_min(h): return torch.full_like(h, _neg_min_val)
        def const_neg_max(h): return torch.full_like(h, _neg_max_val)

        pmin_fn = const_pos_min
        pmax_fn = const_pos_max
        nmin_fn = const_neg_min
        nmax_fn = const_neg_max

        print(f"[config] Physics mode: LINEAR")
        print(f"  v-h slope: dh/dv = {dh_dv:.6f} m/m^3")
        print(f"  Power bounds at h={h_init}m: "
              f"pos=[{_pos_min_val:.2f}, {_pos_max_val:.2f}] MW, "
              f"neg=[{_neg_min_val:.2f}, {_neg_max_val:.2f}] MW")
    else:
        upc_tur = UPC_poly_tur
        upc_pump = UPC_poly_pump
        vh_fn = nl_v_low_to_h
        hv_fn = nl_h_to_v_low
        pmin_fn = pos_min
        pmax_fn = pos_max
        nmin_fn = neg_min
        nmax_fn = neg_max
        print(f"[config] Physics mode: NONLINEAR (polynomial v-h)")

    return {
        # Conversion functions
        'v_up_to_h': v_up_to_h,
        'h_to_v_up': h_to_v_up,
        'v_low_to_h': vh_fn,
        'h_to_v_low': hv_fn,
        # UPC polynomials (linear or polynomial depending on mode)
        'UPC_poly_tur': upc_tur,
        'UPC_poly_pump': upc_pump,
        'UPC_inv_tur': upc_inv_tur,
        'UPC_inv_pump': upc_inv_pump,
        # Power bound functions (constant or head-dependent)
        'pos_min': pmin_fn,
        'pos_max': pmax_fn,
        'neg_min': nmin_fn,
        'neg_max': nmax_fn,
        # Scalar bounds
        'head_min': float(head_min),
        'head_max': float(head_max),
        'max_vol_low': float(max_vol_low),
        'max_vol_up': float(max_vol_up),
        # Derived
        'h_init': h_init,
        'v_init': v_init,
        'target_head': target_head,
        'target_vol_low': target_vol_low,
        # Metadata
        'physics_mode': physics_mode,
        'inverse_pkl_path': str(inverse_pkl_path) if inverse_pkl_path is not None else None,
        'inverse_meta': inverse_meta,
    }


def get_energy_conversion(target_head):
    """Convert volume deficit (m^3) to energy (MWh) at given head."""
    return (RHO * G * ETA * target_head) / 3.6e9
