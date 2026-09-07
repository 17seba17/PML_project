import os

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

# CONSTANTS & FOURIER BASIS


N_MODES_BASE = 4
N_MODES_TREND = 1
omega = 2 * np.pi / 365.25

def fourier_design_matrix(doy, n_modes):
    
    t = np.asarray(doy, dtype=float)
    cols = []
    for k in range(1, n_modes + 1):
        cols.append(np.cos(k * omega * t))
        cols.append(np.sin(k * omega * t))
    return np.column_stack(cols)

def main():
    
    # LOAD DATA (vae_dataset.nc)
    
    ds = xr.open_dataset("dataset/vae_dataset.nc")
    
    START_YEAR = 1980
    SPLIT_YEAR = 2000
    times = pd.to_datetime(ds["time"].values)
    mask_train = (times.year >= START_YEAR) & (times.year <= SPLIT_YEAR)
    
    ds_train = ds.isel(time=mask_train)
    times_train = pd.to_datetime(ds_train["time"].values)

    
    # PRECOMPUTE MATRICES & ARRAYS
    
    doy_train = times_train.dayofyear.values
    
    tg_train_stack = ds_train["tg"].values.astype(np.float32) 
    T_train_arr = ds_train["fgmt"].values.astype(np.float32) 
    n_time_train, n_points = tg_train_stack.shape
    cells = np.arange(n_points)

    harmonics_base = fourier_design_matrix(doy_train, N_MODES_BASE)
    harmonics_trend = fourier_design_matrix(doy_train, N_MODES_TREND)

    a0_int_hat = np.zeros(n_points, dtype=np.float32)
    base_harmonic_coefs = np.zeros((2 * N_MODES_BASE, n_points), dtype=np.float32)
    gmt_intercept_hat = np.zeros(n_points, dtype=np.float32)
    gmt_harmonic_coefs = np.zeros((2 * N_MODES_TREND, n_points), dtype=np.float32)
    sigma_hat = np.zeros(n_points, dtype=np.float32)

    
    # BAYESIAN MODEL PER GRID POINT (OPTIMIZED)
    
    with pm.Model() as model:
        y_obs_data = pm.Data('y_obs_data', np.zeros(n_time_train, dtype=np.float32))

        a0_intercept = pm.Normal("a0_intercept", mu=15.0, sigma=10.0)
        base_sd = np.repeat([1.0 / (2 * k - 1) for k in range(1, N_MODES_BASE + 1)], 2) * 5
        a_b_intercepts = pm.Normal("a_b_intercepts", mu=0, sigma=base_sd, shape=2 * N_MODES_BASE)

        beta_0_intercept = pm.Normal("beta_0_intercept", mu=1.4, sigma=0.4)
        a_b_slopes = pm.Normal("a_b_slopes", mu=0.0, sigma=0.15, shape=2 * N_MODES_TREND)

        sigma = pm.HalfNormal("sigma", sigma=3.0)

        seasonal_base = pm.math.dot(harmonics_base, a_b_intercepts)
        climate_sensitivity = beta_0_intercept + pm.math.dot(harmonics_trend, a_b_slopes)
        
        mu_train = a0_intercept + seasonal_base + climate_sensitivity * T_train_arr

        pm.Normal("y_obs", mu=mu_train, sigma=sigma, observed=y_obs_data)

    print(f"Inizio training su {n_points} punti (MAP estimation)...")
    for i in range(n_points):
        y_train_orig = tg_train_stack[:, i]
        
        with model:
            pm.set_data({'y_obs_data': y_train_orig})
            map_estimate = pm.find_MAP(progressbar=False)

        a0_int_hat[i] = map_estimate["a0_intercept"]
        base_harmonic_coefs[:, i] = map_estimate["a_b_intercepts"]
        gmt_intercept_hat[i] = map_estimate["beta_0_intercept"]
        gmt_harmonic_coefs[:, i] = map_estimate["a_b_slopes"]
        sigma_hat[i] = map_estimate["sigma"]

        if (i+1) % 50 == 0 or (i+1) == n_points:
            print(f"Processing point {i+1}/{n_points}")

    
    # SAVE PARAMETERS TO NETCDF
    
    ds_params = xr.Dataset(
        data_vars={
            "a0_intercept": (["point"], a0_int_hat),
            "base_harmonic_coefs": (["base_mode", "point"], base_harmonic_coefs),
            "gmt_intercept": (["point"], gmt_intercept_hat),
            "gmt_harmonic_coefs": (["trend_mode", "point"], gmt_harmonic_coefs),
            "sigma": (["point"], sigma_hat),
        },
        coords={
            "point": cells,
            "base_mode": np.arange(2 * N_MODES_BASE),
            "trend_mode": np.arange(2 * N_MODES_TREND),
        }
    )

    output_path = "dataset/blr_posterior.nc"
    ds_params.to_netcdf(output_path)

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    main()
