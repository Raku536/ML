#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_sst.py

SST-only ConvLSTM training for TSUBAME with optional full-domain mode:
- Land/missing handled via "mask channel" (SST NaN -> 0, mask=0)
- Optional topo static features from ROMS grid file (h, gx, gy, slope)
- Resume / checkpoint style: per-tile done_flag + per-tile model file
- Time-budget test mode + result saving + auto judgement
- Option to exclude boundary (e.g., 302x302 -> inner 300x300)

Key points:
- We fill NaN SST with 0 but add an explicit mask channel, so the model can learn land boundaries.
- Target is sampled only where original SST is finite (mask==1).
- Per-tile models are trained to control memory and allow huge domains.
- Resume: if reports/done_tile_{j0}_{i0}.txt exists, skip that tile.
"""

import os
import re
import gc
import json
from time import time
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import psutil

from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, ConvLSTM2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import argparse


# ============================================================
# CLI
# ============================================================
# NOTE:
#   This block defines all runtime knobs. For TSUBAME full-domain runs,
#   the main change is to enable --full_domain so the entire grid is used
#   as one tile. Other parameters (batch_size, max_samples_per_tile, window)
#   remain useful for memory control or speeding up experiments.
#   See "full_domain" in parse_args() and its use in main().
def parse_args():
    p = argparse.ArgumentParser(
        description="SST-only ConvLSTM training on TSUBAME (tile or full-domain, with topo/resume/time budget/test)."
    )

    # --- paths ---
    p.add_argument("--data_root", type=str, required=True,
                   help="e.g., /gs/bs/tga-NakamuLab/raku/ML/data")
    p.add_argument("--out_root", type=str, required=True,
                   help="e.g., /gs/bs/tga-NakamuLab/raku/ML/outputs/sst/train_run001")
    p.add_argument("--grid_file", type=str, default="",
                   help="optional: ROMS grid file path (Yaeyama2_grd_v11.2.nc) for topo features")

    # --- data scanning ---
    p.add_argument("--ocean_dir", type=str, default="COAWST_OUTPUT",
                   help="subdir under data_root that contains ocean netcdf files")
    p.add_argument("--ocean_pattern", type=str, default="temp2layer_*.nc",
                   help="file pattern for ocean data")
    p.add_argument("--year_min", type=int, default=1994)
    p.add_argument("--year_max", type=int, default=2001)  # inclusive
    p.add_argument("--train_years", type=str, default="1994-2001",
                   help="e.g., 1994-2001 or 1994,1995,1996")

    # --- variables ---
    p.add_argument("--sst_var", type=str, default="SST",
                   help="preferred SST variable name. If not found, code tries fallbacks: temp, sst, ...")
    p.add_argument("--time_var", type=str, default="ocean_time",
                   help="time dimension name (default ocean_time, fallback to time)")

    # --- model / window ---
    p.add_argument("--window", type=int, default=56,
                   help="time window length, e.g. 56 for 56 hours when dt=1h")
    p.add_argument("--tile_size", type=int, default=60)
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tile_epochs", type=int, default=4)
    p.add_argument("--max_samples_per_tile", type=int, default=6000)

    # --- features ---
    p.add_argument("--use_month_sincos", action="store_true")
    p.add_argument("--use_doy_sincos", action="store_true")
    p.add_argument("--use_sst_lag1", action="store_true",
                   help="include SST(t-1) as an extra channel")
    p.add_argument("--use_topo", action="store_true",
                   help="include depth/slope static features (requires --grid_file)")

    # --- domain cropping / boundary exclusion ---
    p.add_argument("--exclude_boundary", action="store_true",
                   help="exclude 1-cell boundary on all sides: use [1:ny-1, 1:nx-1] tiles (e.g., 302->300)")
    p.add_argument("--boundary_width", type=int, default=1,
                   help="boundary width to exclude if --exclude_boundary is enabled (default 1)")

    # --- full-domain override ---
    # When enabled, the code will override tile_size/stride to cover the full
    # spatial extent in one go. This keeps the original tiling logic intact
    # (so the same script can still run in tiled mode on low-resource machines)
    # while letting TSUBAME run with full-domain continuity.
    p.add_argument("--full_domain", action="store_true",
                   help="override tile_size/stride to cover the entire domain as a single tile")

    # --- optimization ---
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--loss", type=str, default="huber")

    # --- scalers ---
    p.add_argument("--standardize_target", action="store_true",
                   help="standardize target SST (recommended for stable training)")
    p.add_argument("--warmup_per_year", type=int, default=1500,
                   help="number of warmup samples per year for fitting scalers")

    # --- resume ---
    p.add_argument("--resume", action="store_true",
                   help="resume from existing OUT_ROOT: skip tiles that have done_flag")
    p.add_argument("--resume_by_model", action="store_true",
                   help="if resume: also skip when model file exists (in case done_flag missing)")

    # --- test / time budget ---
    p.add_argument("--time_budget_sec", type=int, default=0,
                   help="hard time budget for whole run. 0 disables budget")
    p.add_argument("--test_mode", action="store_true",
                   help="run a small subset of tiles/years, save test report and exit")
    p.add_argument("--test_tiles", type=int, default=3,
                   help="number of tiles to run in test_mode")
    p.add_argument("--test_years", type=int, default=1,
                   help="number of years to run in test_mode")

    # --- optional numeric stability ---
    p.add_argument("--disable_onednn", action="store_true",
                   help="set TF_ENABLE_ONEDNN_OPTS=0 (more deterministic, slightly slower)")

    return p.parse_args()


# ============================================================
# Small helpers
# ============================================================
# Guidance:
# - now_str(): use for consistent timestamp logging in reports.
# - parse_years(): accepts flexible CLI year formats.
# - time_left_ok(): used to implement a global time-budget guard.
def now_str():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_years(s: str):
    """
    Accept:
      - "1994-2001"
      - "1994,1995,1996"
      - "1994"
    """
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(s)]


def generate_calendar_from_ocean_time(ds: xr.Dataset, time_dim: str, year_fallback: int):
    """
    Prefer deriving month/doy from the dataset's time coordinate if possible.
    We open with decode_times=False in main data reads (for speed and reliability),
    so here we do a light attempt:
    - If time coordinate exists and has units, use xr.decode_cf on a small slice.
    - Otherwise fallback to a simple hourly range from Jan 1.
    """
    # Fallback: hourly range starting Jan 1
    nt = int(ds.sizes.get(time_dim, 0))
    dates = pd.date_range(start=f"{year_fallback}-01-01", periods=nt, freq="H")
    month = dates.month.values.astype(np.int16)
    doy = dates.dayofyear.values.astype(np.int16)
    return dates, month, doy


def time_left_ok(t0, budget_sec):
    if budget_sec <= 0:
        return True
    return (time() - t0) < budget_sec


# ============================================================
# SST variable picking
# ============================================================
# Guidance:
# - If your dataset uses a custom SST variable name, pass --sst_var.
# - The fallback list keeps the script robust across ROMS/COAWST outputs.
def pick_sst_var(ds: xr.Dataset, prefer: str) -> str:
    if prefer in ds.data_vars:
        return prefer

    for v in ["SST", "sst", "temp", "TEMP", "ST", "surface_temp", "TS"]:
        if v in ds.data_vars:
            return v

    # last resort: any 3D var including ocean_time or time
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim == 3 and (("ocean_time" in da.dims) or ("time" in da.dims)):
            return v

    raise KeyError(f"Cannot find SST var. prefer={prefer}. available_vars={list(ds.data_vars)[:30]}...")


# ============================================================
# Feature counting and model
# ============================================================
# Guidance:
# - estimate_feature_count(): ensures model input shape matches feature flags.
# - build_model(): change filters, layers, or dropout here if you scale up
#   for a larger GPU/multi-GPU environment.
def estimate_feature_count(use_month, use_doy, use_lag1, use_topo):
    base = 2  # SST_filled + mask
    add = 0
    if use_month:
        add += 2
    if use_doy:
        add += 2
    if use_lag1:
        add += 1
    static = 4 if use_topo else 0  # h, gx, gy, slope
    return base + add + static


def build_model(input_shape, lr):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), activation="relu",
                         padding="same", return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), activation="relu",
                         padding="same", return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1.0), metrics=["mae"])
    return model


# ============================================================
# Optional topo loading
# ============================================================
# Guidance:
# - Set --use_topo and provide --grid_file to add static depth/slope features.
# - Topo is broadcast over time for each tile (or full-domain) and can help
#   the model understand land/sea and bathymetry.
def load_topo(grid_file):
    ds = xr.open_dataset(grid_file)
    if "h" not in ds:
        raise KeyError(f"grid_file has no 'h' variable: {grid_file}")

    h = ds["h"].values.astype(np.float32)
    gy, gx = np.gradient(h)
    slope = np.sqrt(gx**2 + gy**2).astype(np.float32)

    ds.close()
    return h, gx.astype(np.float32), gy.astype(np.float32), slope


# ============================================================
# Build arrays for a year+tile
# ============================================================
# Guidance:
# - This function reads a single year and spatial tile. For full-domain mode,
#   the tile is the whole grid, so memory usage can spike. Use test_mode or
#   reduce max_samples_per_tile to check feasibility before a long run.
# - NaN SST is filled with 0 and accompanied by a mask channel so the model
#   can still learn land boundaries.
def build_tile_arrays_sst_only(
    ocean_path, year, j0, i0, j1, i1,
    window,
    sst_var_prefer, time_dim,
    use_month, use_doy, use_lag1,
    topo_pack=None
):
    """
    Read a whole year/tile into numpy arrays:
      X_full: (T,H,W,C)
      Y_full: (T,H,W) filled SST
      Y_mask: (T,H,W) validity mask (1 finite, 0 NaN)
    """
    with xr.open_dataset(ocean_path, decode_times=False) as ods:
        # time dim fallback
        if time_dim not in ods.dims:
            if "time" in ods.dims:
                time_dim = "time"
            else:
                raise KeyError(f"time dim not found in {ocean_path}. dims={ods.dims}")

        nt = int(ods.sizes.get(time_dim, 0))
        if nt <= window:
            return None

        sst_name = pick_sst_var(ods, sst_var_prefer)

        # Read tile slice (may include s_rho)
        temp = ods[sst_name].isel(
            {time_dim: slice(0, nt),
             "eta_rho": slice(j0, j1),
             "xi_rho": slice(i0, i1)}
        ).values.astype(np.float32)

        # Case A: (T,H,W)
        if temp.ndim == 3:
            SST = temp

        # Case B: (T,s_rho,H,W) -> take surface s_rho=0
        elif temp.ndim == 4:
            SST = temp[:, 0, :, :]
            if SST.ndim != 3:
                raise ValueError(f"SST extraction failed after s_rho selection: {SST.shape}")

        else:
            raise ValueError(f"Unsupported SST ndim={temp.ndim}, shape={temp.shape} in {ocean_path} var={sst_name}")

        # Calendar features (simple fallback)
        # NOTE: You verified ocean_time diff=3600s, so hourly is consistent.
        # We use year-based range for month/doy features (optional).
        T, H, W = SST.shape
        dates = pd.date_range(start=f"{year}-01-01", periods=T, freq="H")
        month = dates.month.values.astype(np.int16)
        doy = dates.dayofyear.values.astype(np.int16)

    # mask + fill
    mask = np.isfinite(SST).astype(np.float32)
    SST_filled = np.where(np.isfinite(SST), SST, 0.0).astype(np.float32)

    parts = []
    parts.append(SST_filled[..., None])  # (T,H,W,1)
    parts.append(mask[..., None])        # (T,H,W,1)

    # month sin/cos
    if use_month:
        ms = np.sin(2*np.pi*month/12.0).astype(np.float32)
        mc = np.cos(2*np.pi*month/12.0).astype(np.float32)
        Fm = np.stack([ms, mc], axis=-1)[:, None, None, :]  # (T,1,1,2)
        parts.append(np.broadcast_to(Fm, (T, H, W, 2)))

    # doy sin/cos
    if use_doy:
        dsin = np.sin(2*np.pi*doy/366.0).astype(np.float32)
        dcos = np.cos(2*np.pi*doy/366.0).astype(np.float32)
        Fd = np.stack([dsin, dcos], axis=-1)[:, None, None, :]
        parts.append(np.broadcast_to(Fd, (T, H, W, 2)))

    # lag1
    if use_lag1:
        sst_lag = np.concatenate([SST_filled[:1], SST_filled[:-1]], axis=0)
        parts.append(sst_lag[..., None])  # (T,H,W,1)

    # topo static features
    if topo_pack is not None:
        h, gx, gy, slope = topo_pack
        tile_h = h[j0:j1, i0:i1]
        tile_gx = gx[j0:j1, i0:i1]
        tile_gy = gy[j0:j1, i0:i1]
        tile_s  = slope[j0:j1, i0:i1]
        STATIC = np.stack([tile_h, tile_gx, tile_gy, tile_s], axis=-1).astype(np.float32)  # (H,W,4)
        parts.append(np.broadcast_to(STATIC[None, ...], (T, H, W, 4)))

    X_full = np.concatenate(parts, axis=-1).astype(np.float32)  # (T,H,W,C)
    Y_full = SST_filled.astype(np.float32)
    Y_mask = mask.astype(np.float32)
    return X_full, Y_full, Y_mask


# ============================================================
# Sample generator
# ============================================================
# Guidance:
# - Yields per-time, per-pixel 3x3 spatial blocks with a temporal window.
# - Full-domain mode does not change this logic; it just makes the spatial
#   sweep cover the entire grid rather than a single tile.
# - If you want fewer samples (for speed), reduce max_samples_per_tile.
def iter_samples_from_arrays(X_full, Y_full, Y_mask, window, max_samples=None):
    """
    Yield:
      block: (window,3,3,C)
      target: float (center)
    Only sample where target is valid (Y_mask==1).
    """
    T, H, W, C = X_full.shape

    X_pad = np.pad(X_full, ((0,0),(1,1),(1,1),(0,0)), constant_values=0.0)
    Ym_pad = np.pad(Y_mask, ((0,0),(1,1),(1,1)), constant_values=0.0)
    Yv_pad = np.pad(Y_full, ((0,0),(1,1),(1,1)), constant_values=0.0)

    count = 0
    for t in range(window, T):
        slice_t = X_pad[t-window:t, ...]     # (window,H+2,W+2,C)
        tgt_m = Ym_pad[t, 1:-1, 1:-1]        # (H,W)
        if not np.any(tgt_m > 0.5):
            continue

        jj, ii = np.where(tgt_m > 0.5)
        for j, i in zip(jj, ii):
            block = slice_t[:, j:j+3, i:i+3, :]
            target = float(Yv_pad[t, j+1, i+1])
            yield block, target
            count += 1
            if max_samples is not None and count >= max_samples:
                return


# ============================================================
# Main
# ============================================================
# Guidance:
# - Full-domain mode is applied after inferring grid size (ny, nx) so it can
#   compute the correct full spatial extent and override tile_size/stride.
# - Time budget is enforced with a margin to allow logs to be saved safely.
def main():
    args = parse_args()
    t0 = time()

    # TF determinism option
    if args.disable_onednn:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    tf.random.set_seed(42)
    np.random.seed(42)

    # ---------- training years ----------
    # Guidance: adjust --train_years to control the temporal span of training.
    # For quick checks on TSUBAME, consider --test_mode with --test_years.
    years = parse_years(args.train_years)
    years = [y for y in years if args.year_min <= y <= args.year_max]
    if not years:
        raise ValueError("No training years selected. Check --train_years / --year_min / --year_max.")

    # ---------- scan ocean files ----------
    # Guidance: if some years are missing, the warning lists them and those
    # years will be skipped. Ensure filenames contain a 4-digit year.
    ocean_root = os.path.join(args.data_root, args.ocean_dir)
    ocean_paths = sorted(glob(os.path.join(ocean_root, args.ocean_pattern)))
    if not ocean_paths:
        raise FileNotFoundError(f"No ocean files found: {ocean_root}/{args.ocean_pattern}")

    year_to_file = {}
    year_pat = re.compile(r"(\d{4})")
    for p in ocean_paths:
        m = year_pat.search(os.path.basename(p))
        if m:
            y = int(m.group(1))
            year_to_file[y] = p

    missing = [y for y in years if y not in year_to_file]
    if missing:
        print(f"[WARN] missing ocean file for years: {missing}")

    # ---------- output directories ----------
    ensure_dir(args.out_root)
    report_dir = os.path.join(args.out_root, "reports")
    model_dir = os.path.join(args.out_root, "models")
    ensure_dir(report_dir)
    ensure_dir(model_dir)

    # ---------- topo ----------
    # Guidance: topo features are loaded once and sliced per tile. If the grid
    # is very large, ensure the topo array fits in memory.
    topo_pack = None
    if args.use_topo:
        if not args.grid_file:
            raise ValueError("--use_topo requires --grid_file")
        topo_pack = load_topo(args.grid_file)
        ny, nx = topo_pack[0].shape
    else:
        first_file = next(iter(year_to_file.values()))
        with xr.open_dataset(first_file, decode_times=False) as ds:
            if "eta_rho" in ds.dims and "xi_rho" in ds.dims:
                ny, nx = int(ds.sizes["eta_rho"]), int(ds.sizes["xi_rho"])
            else:
                dims2 = [d for d in ds.dims if d not in {args.time_var, "time", "ocean_time"}]
                if len(dims2) >= 2:
                    ny, nx = int(ds.sizes[dims2[-2]]), int(ds.sizes[dims2[-1]])
                else:
                    raise ValueError(f"Cannot infer grid dims from {ds.dims}")

    # ---------- feature count ----------
    # Guidance: The model input shape depends on feature flags, so changing
    # use_month/use_doy/use_lag1/use_topo must be consistent here.
    n_features = estimate_feature_count(
        args.use_month_sincos,
        args.use_doy_sincos,
        args.use_sst_lag1,
        args.use_topo
    )
    input_shape = (args.window, 3, 3, n_features)

    # ---------- scalers ----------
    # Guidance: StandardScaler is fit on a warmup sample. For full-domain
    # training, you may want to increase --warmup_per_year so scalers see
    # enough variability across the large grid.
    X_scaler = StandardScaler()
    y_scaler = StandardScaler() if args.standardize_target else None

    # ---------- test mode limitations ----------
    test_years = years[:max(1, args.test_years)]

    # ---------- tile positions (optionally exclude boundary) ----------
    # Guidance:
    # - exclude_boundary removes edge cells that may be poorly resolved.
    # - full_domain overrides tile_size/stride to produce a single tile that
    #   spans the entire remaining grid.
    tile_positions = []

    if args.exclude_boundary:
        bw = int(args.boundary_width)
        J_START, I_START = bw, bw
        J_END, I_END = ny - bw, nx - bw   # exclusive
    else:
        J_START, I_START = 0, 0
        J_END, I_END = ny, nx

    if args.full_domain:
        # Full-domain override: this turns the tiling loop into a single tile
        # covering the entire spatial extent. Keep tile logic intact so
        # resume/test_mode still works without a separate code path.
        args.tile_size = max(J_END - J_START, I_END - I_START)
        args.stride = args.tile_size
        print(f"[{now_str()}] Full-domain mode enabled: tile_size={args.tile_size}, stride={args.stride}")

    for j0 in range(J_START, J_END, args.stride):
        for i0 in range(I_START, I_END, args.stride):
            j1 = min(j0 + args.tile_size, J_END)
            i1 = min(i0 + args.tile_size, I_END)
            if (j1 - j0) <= 0 or (i1 - i0) <= 0:
                continue
            tile_positions.append((j0, i0, j1, i1))

    if args.test_mode:
        tile_positions = tile_positions[:max(1, args.test_tiles)]

    # ============================================================
    # Warmup scalers
    # ============================================================
    # Guidance:
    # - Warmup uses only one tile (or full-domain) per year to fit scalers.
    # - If warmup is too small, scaling can be unstable; consider increasing
    #   --warmup_per_year on TSUBAME.
    print(f"[{now_str()}] Warmup scalers...")
    warmup_stats = {"years": {}, "total_blocks": 0}
    warmup_year_list = test_years if args.test_mode else years

    for y in warmup_year_list:
        if not time_left_ok(t0, args.time_budget_sec):
            print("[TIME] budget exceeded during warmup, stopping warmup early.")
            break

        ocean_file = year_to_file.get(y)
        if not ocean_file:
            warmup_stats["years"][str(y)] = 0
            continue

        # warmup tile: use INNER region if boundary excluded (avoid using boundary in scaler)
        j0 = J_START
        i0 = I_START
        j1 = min(j0 + args.tile_size, J_END)
        i1 = min(i0 + args.tile_size, I_END)

        out = build_tile_arrays_sst_only(
            ocean_file, y, j0, i0, j1, i1,
            window=args.window,
            sst_var_prefer=args.sst_var,
            time_dim=args.time_var,
            use_month=args.use_month_sincos,
            use_doy=args.use_doy_sincos,
            use_lag1=args.use_sst_lag1,
            topo_pack=topo_pack
        )
        if out is None:
            warmup_stats["years"][str(y)] = 0
            continue

        X_full, Y_full, Y_mask = out

        cnt = 0
        for xb, yb in iter_samples_from_arrays(X_full, Y_full, Y_mask, args.window, max_samples=args.warmup_per_year):
            X_scaler.partial_fit(xb.reshape(-1, xb.shape[-1]))
            if y_scaler is not None:
                y_scaler.partial_fit(np.asarray(yb, dtype=np.float32).reshape(-1, 1))
            cnt += 1

        warmup_stats["years"][str(y)] = cnt
        warmup_stats["total_blocks"] += cnt

        del X_full, Y_full, Y_mask
        gc.collect()

    joblib.dump(X_scaler, os.path.join(args.out_root, "X_scaler.pkl"))
    if y_scaler is not None:
        joblib.dump(y_scaler, os.path.join(args.out_root, "y_scaler.pkl"))

    print(f"[{now_str()}] Warmup done. total_blocks={warmup_stats['total_blocks']}")

    # ============================================================
    # Training loop over tiles (with resume)
    # ============================================================
    # Guidance:
    # - In full-domain mode, this loop will typically run once.
    # - Resume logic is still useful if a long run is interrupted.
    callbacks = [
        EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=2, min_lr=1e-5)
    ]
    perf_log = []
    n_tiles_done = 0
    n_tiles_skipped = 0

    # time budget guard: keep margin for saving final report
    budget_save_margin = 120  # sec

    print(f"[{now_str()}] Start training tiles: tile_size={args.tile_size}, stride={args.stride}, "
          f"window={args.window}, features={n_features}, exclude_boundary={args.exclude_boundary}")

    y_list = test_years if args.test_mode else years
    y_list = [y for y in y_list if y in year_to_file]

    for (j0, i0, j1, i1) in tile_positions:
        # stop if near time budget
        if args.time_budget_sec > 0 and (time() - t0) > (args.time_budget_sec - budget_save_margin):
            print("[TIME] budget nearly exceeded, stopping tile loop.")
            break

        # ------------------------------------------------------------
        # RESUME LOGIC: skip if already done
        # ------------------------------------------------------------
        tile_name = f"tile_{j0}_{i0}.keras"
        tile_model_path = os.path.join(model_dir, tile_name)
        done_flag = os.path.join(report_dir, f"done_tile_{j0}_{i0}.txt")

        if args.resume:
            if os.path.exists(done_flag):
                n_tiles_skipped += 1
                print(f"[{now_str()}] SKIP tile ({j0}:{j1},{i0}:{i1}) done_flag exists: {done_flag}")
                continue
            if args.resume_by_model and os.path.exists(tile_model_path):
                n_tiles_skipped += 1
                print(f"[{now_str()}] SKIP tile ({j0}:{j1},{i0}:{i1}) model exists: {tile_model_path}")
                continue

        mem0 = psutil.Process().memory_info().rss / 1024**2
        tile_t0 = time()

        X_tile, y_tile = [], []

        # collect samples for this tile across years (until max_samples_per_tile)
        # Guidance:
        # - For full-domain training, max_samples_per_tile effectively caps
        #   the total number of samples, which can control memory/runtime.
        # - For larger GPUs, increase max_samples_per_tile and batch_size.
        for y in y_list:
            if args.time_budget_sec > 0 and (time() - t0) > (args.time_budget_sec - budget_save_margin):
                break

            ocean_file = year_to_file[y]
            out = build_tile_arrays_sst_only(
                ocean_file, y, j0, i0, j1, i1,
                window=args.window,
                sst_var_prefer=args.sst_var,
                time_dim=args.time_var,
                use_month=args.use_month_sincos,
                use_doy=args.use_doy_sincos,
                use_lag1=args.use_sst_lag1,
                topo_pack=topo_pack
            )
            if out is None:
                continue

            X_full, Y_full, Y_mask = out

            for xb, yb in iter_samples_from_arrays(X_full, Y_full, Y_mask, args.window, max_samples=None):
                X_tile.append(xb)
                y_tile.append(yb)
                if len(X_tile) >= args.max_samples_per_tile:
                    break

            del X_full, Y_full, Y_mask
            gc.collect()

            if len(X_tile) >= args.max_samples_per_tile:
                break

        n_samp = len(X_tile)
        if n_samp == 0:
            n_tiles_skipped += 1
            print(f"[{now_str()}] SKIP tile ({j0}:{j1},{i0}:{i1}) because n_samples=0")
            continue

        # pack to numpy
        X_np = np.asarray(X_tile, dtype=np.float32)  # (N,T,3,3,C)
        y_np = np.asarray(y_tile, dtype=np.float32)  # (N,)
        del X_tile, y_tile
        gc.collect()

        # scale X and y
        X_scaled = X_scaler.transform(X_np.reshape(-1, X_np.shape[-1])).reshape(X_np.shape)
        if y_scaler is not None:
            y_scaled = y_scaler.transform(y_np.reshape(-1, 1)).ravel()
        else:
            y_scaled = y_np

        # sanity
        if not np.isfinite(X_scaled).all() or not np.isfinite(y_scaled).all():
            del X_np, y_np, X_scaled, y_scaled
            gc.collect()
            n_tiles_skipped += 1
            print(f"[{now_str()}] SKIP tile ({j0}:{j1},{i0}:{i1}) due to non-finite values")
            continue

        # train model
        # Guidance:
        # - On TSUBAME, consider increasing tile_epochs and batch_size to
        #   leverage more compute/memory.
        model = build_model(input_shape=input_shape, lr=args.lr)
        hist = model.fit(
            X_scaled, y_scaled,
            batch_size=min(args.batch_size, len(X_scaled)),
            epochs=args.tile_epochs,
            validation_split=0.1 if not args.test_mode else 0.2,
            shuffle=True,
            callbacks=callbacks,
            verbose=1
        )

        best_val_mae = float(np.nanmin(hist.history.get("val_mae", [np.nan])))
        last_loss = float(hist.history["loss"][-1])
        last_mae = float(hist.history["mae"][-1])

        # save model
        model.save(tile_model_path)

        # write done flag ONLY after successful save
        with open(done_flag, "w", encoding="utf-8") as f:
            f.write(f"OK {now_str()} best_val_mae={best_val_mae:.6f} samples={n_samp} tile=({j0}:{j1},{i0}:{i1})\n")

        mem1 = psutil.Process().memory_info().rss / 1024**2
        dur = time() - tile_t0

        perf_log.append({
            "tile_j0": int(j0), "tile_i0": int(i0),
            "tile_j1": int(j1), "tile_i1": int(i1),
            "samples": int(n_samp),
            "best_val_mae": best_val_mae,
            "last_loss": last_loss,
            "last_mae": last_mae,
            "duration_sec": float(dur),
            "mem_mb_start": float(mem0),
            "mem_mb_end": float(mem1),
            "model_path": tile_model_path,
            "done_flag": done_flag,
        })

        del X_np, y_np, X_scaled, y_scaled, model, hist
        gc.collect()

        n_tiles_done += 1
        print(f"[{now_str()}] DONE tile ({j0}:{j1},{i0}:{i1}) samples={n_samp} "
              f"best_val_mae={best_val_mae:.4f} dur={dur/60:.1f}min mem={mem0:.0f}->{mem1:.0f}MB")

        if args.test_mode and n_tiles_done >= max(1, args.test_tiles):
            break

    # ============================================================
    # Save logs + summary judgement
    # ============================================================
    # Guidance:
    # - Summary JSON and CSV are saved for later analysis.
    # - Judgement is a quick heuristic; adjust thresholds as needed.
    log_path = os.path.join(report_dir, "tile_training_log.csv")
    pd.DataFrame(perf_log).to_csv(log_path, index=False)

    judgement = {"status": "UNKNOWN", "reasons": [], "suggestions": []}
    if len(perf_log) == 0:
        judgement["status"] = "FAIL"
        judgement["reasons"].append("No tile finished training. Likely no valid samples or time budget too small.")
        judgement["suggestions"].append("Increase --time_budget_sec, reduce --tile_size/--stride, check SST var with --sst_var.")
    else:
        v = [r["best_val_mae"] for r in perf_log if np.isfinite(r["best_val_mae"])]
        if not v:
            judgement["status"] = "WARN"
            judgement["reasons"].append("val_mae is NaN for finished tiles.")
            judgement["suggestions"].append("Check scaling, SST validity, and mask/fill logic.")
        else:
            med = float(np.median(v))
            judgement["status"] = "PASS"
            judgement["reasons"].append(f"Finished tiles={len(perf_log)}, median(best_val_mae)={med:.4f}")
            if med > 2.0:
                judgement["status"] = "WARN"
                judgement["reasons"].append("Median val_mae is large; may underfit or units mismatch.")
                judgement["suggestions"].append("Confirm SST units, increase epochs, increase samples, enable lag1/topo.")

    summary = {
        "time_start": pd.Timestamp.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
        "time_end": now_str(),
        "elapsed_sec": float(time() - t0),
        "args": vars(args),
        "warmup_stats": warmup_stats,
        "tiles_done": int(n_tiles_done),
        "tiles_skipped": int(n_tiles_skipped),
        "log_csv": log_path,
        "judgement": judgement
    }

    summary_path = os.path.join(report_dir, "run_summary.json")
    save_json(summary_path, summary)

    print(f"[{now_str()}] Saved log: {log_path}")
    print(f"[{now_str()}] Saved summary: {summary_path}")
    print(f"[{now_str()}] Judgement: {judgement['status']} | " + " ; ".join(judgement["reasons"]))


if __name__ == "__main__":
    main()
