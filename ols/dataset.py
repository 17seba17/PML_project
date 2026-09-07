import gc

import numpy as np
import torch
import xarray as xr


# BUILDING TENSORS PER MODELLI (CVAE / BLR)
def buildingTensors(file_path="vae_dataset.nc", start_year=1980, split_year=2000):
    ds = xr.open_dataset(file_path)

    if start_year is not None:
        ds = ds.isel(time=(ds.time.dt.year >= start_year).values)

    min_year = int(ds.time.dt.year.min().values)
    max_year = int(ds.time.dt.year.max().values)
    print(f"Anni presenti nel dataset: {min_year} -> {max_year} | split_year impostato: {split_year}")

    train_mask = (ds.time.dt.year <= split_year).values
    test_mask = (ds.time.dt.year > split_year).values

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"Osservazioni Train: {len(train_idx)} | Test: {len(test_idx)}")

    if len(test_idx) == 0:
        raise ValueError(
            f"Errore: Nessun dato per il Test Set! Il dataset finisce nel {max_year}, "
            f"ma hai scelto split_year={split_year}. Imposta uno split_year inferiore a {max_year}."
        )

    ds_train = ds.isel(time=train_idx)
    ds_test = ds.isel(time=test_idx)

    del train_mask, test_mask, train_idx, test_idx
    gc.collect()

    N_train = ds_train.sizes["time"]
    N_test = ds_test.sizes["time"]

    # TEMPERATURE - TARGET Y
    tg_clim = ds_train["tg"].groupby("time.dayofyear").mean(dim="time")

    tg_anom_train = (
        ds_train["tg"].groupby("time.dayofyear") - tg_clim
    ).values.astype(np.float32)
    tg_anom_test = (
        ds_test["tg"].groupby("time.dayofyear") - tg_clim
    ).values.astype(np.float32)

    tg_dim = tg_anom_train.shape[1]

    tg_std = tg_anom_train.std(axis=0, keepdims=True) + 1e-6
    tg_train_norm = tg_anom_train / tg_std
    tg_test_norm = tg_anom_test / tg_std

    Y_train = torch.from_numpy(tg_train_norm)
    Y_test = torch.from_numpy(tg_test_norm)

    del tg_anom_train, tg_anom_test, tg_train_norm, tg_test_norm
    gc.collect()

    # PRESSURE (PP) - PREDICTORS
    pp_clim = ds_train["pp"].groupby("time.dayofyear").mean(dim="time")

    pp_anom_train = (
        ds_train["pp"].groupby("time.dayofyear") - pp_clim
    ).values.astype(np.float32)
    pp_anom_test = (
        ds_test["pp"].groupby("time.dayofyear") - pp_clim
    ).values.astype(np.float32)

    pp_std = pp_anom_train.std(axis=0, keepdims=True) + 1e-6

    pp_train_norm = pp_anom_train / pp_std
    pp_test_norm = pp_anom_test / pp_std

    del pp_anom_train, pp_anom_test
    gc.collect()

    # DOY - FOURIER HARMONICS
    omega = 2.0 * np.pi / 365.25
    doy_train = ds_train.time.dt.dayofyear.values.astype(np.float32)
    doy_test = ds_test.time.dt.dayofyear.values.astype(np.float32)

    doy_feat_train = np.stack([
        np.sin(omega * doy_train),
        np.cos(omega * doy_train)
    ], axis=1)

    doy_feat_test = np.stack([
        np.sin(omega * doy_test),
        np.cos(omega * doy_test)
    ], axis=1)

    # concatenation of the predictor
    pred_train = np.concatenate([pp_train_norm, doy_feat_train], axis=1)
    pred_test = np.concatenate([pp_test_norm, doy_feat_test], axis=1)

    pp_dim = pred_train.shape[1]
    cond_dim = pp_dim + 1

    del pp_train_norm, pp_test_norm, doy_feat_train, doy_feat_test
    gc.collect()

    # fGMT
    fgmt_train_raw = ds_train["fgmt"].values.reshape(N_train, 1).astype(np.float32)
    fgmt_test_raw = ds_test["fgmt"].values.reshape(N_test, 1).astype(np.float32)

    fgmt_mean = fgmt_train_raw.mean(axis=0, keepdims=True)
    fgmt_std = fgmt_train_raw.std(axis=0, keepdims=True) + 1e-6

    fgmt_train_norm = (fgmt_train_raw - fgmt_mean) / fgmt_std
    fgmt_test_norm = (fgmt_test_raw - fgmt_mean) / fgmt_std

    del fgmt_train_raw, fgmt_test_raw
    gc.collect()

    # building tensors ...
    X_train_mat = np.concatenate([pred_train, fgmt_train_norm], axis=1)
    X_test_mat = np.concatenate([pred_test, fgmt_test_norm], axis=1)

    X_train = torch.from_numpy(X_train_mat)
    X_test = torch.from_numpy(X_test_mat)

    del pred_train, pred_test, fgmt_train_norm, fgmt_test_norm, X_train_mat, X_test_mat
    gc.collect()

    norm_stats = {
        "tg_std": tg_std.squeeze(),
        "pp_std": pp_std.squeeze(),
        "fgmt_mean": fgmt_mean.squeeze(),
        "fgmt_std": fgmt_std.squeeze(),
    }

    return (
        X_train,
        Y_train,
        X_test,
        Y_test,
        tg_dim,
        pp_dim,
        cond_dim,
        norm_stats,
        ds,
    )
