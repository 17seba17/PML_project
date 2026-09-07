import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from blr.evaluate import predict_scenarios
from cvae.evaluate import generate_counterfactuals, predict_pure
from cvae.models import CVAE
from ols.counterfactual import (
    calcola_impatto,
    costruisci_gmta_counterfactual,
    indice_impatto_periodo,
)

from ols.dataset import buildingTensors
from ols.grafici import (
    grafico_beta,
    grafico_beta_affidabile,
    grafico_gmta_counterfactual,
    grafico_impatto,
    grafico_r2,
    grafico_smoothing,
)
from ols.ols_train import fit_ols

# GENERAL CONFIGURATION & UNIFIED BASELINE
DATASET_PATH = "dataset/vae_dataset.nc" if os.path.exists("dataset/vae_dataset.nc") else "vae_dataset.nc"
BLR_TRACE_PATH = "dataset/blr_posterior.nc" if os.path.exists("dataset/blr_posterior.nc") else "blr_posterior.nc"
CVAE_WEIGHTS_PATH = "cvae/cvae_model.pt" if os.path.exists("cvae/cvae_model.pt") else "cvae_model.pt"

START_YEAR = 1980
SPLIT_YEAR = 2000

GMT_COUNTERFACTUAL_VAL = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("graphs", exist_ok=True)


def calculate_metrics(y_true, y_pred):
    errors = y_true - y_pred
    mae = float(np.nanmean(np.nanmean(np.abs(errors), axis=0)))
    rmse = float(np.nanmean(np.sqrt(np.nanmean(errors ** 2, axis=0))))
    
    sst = np.nansum((y_true - np.nanmean(y_true, axis=0)) ** 2, axis=0)
    sse = np.nansum(errors ** 2, axis=0)
    
    global_sse = np.nansum(sse)
    global_sst = np.nansum(sst)
    r2 = 1.0 - (global_sse / global_sst) if global_sst != 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_timescales(y_true, y_pred, times_index):
    da_true = xr.DataArray(y_true, coords={"time": times_index, "point": np.arange(y_true.shape[1])}, dims=["time", "point"])
    da_pred = xr.DataArray(y_pred, coords={"time": times_index, "point": np.arange(y_pred.shape[1])}, dims=["time", "point"])

    m_daily = calculate_metrics(da_true.values, da_pred.values)

    da_true_m = da_true.resample(time="1MS").mean()
    da_pred_m = da_pred.resample(time="1MS").mean()
    m_monthly = calculate_metrics(da_true_m.values, da_pred_m.values)

    da_true_s = da_true.resample(time="QS-DEC").mean()
    da_pred_s = da_pred.resample(time="QS-DEC").mean()
    m_seasonal = calculate_metrics(da_true_s.values, da_pred_s.values)

    return {
        "Daily": m_daily,
        "Monthly": m_monthly,
        "Seasonal": m_seasonal,
    }


def points_to_grid(values_1d, ds):
    n_lat = ds.attrs["orig_n_lat"]
    n_lon = ds.attrs["orig_n_lon"]
    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])

    matrix = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)
    matrix[lat_indices, lon_indices] = values_1d

    if lat_grid[0] > lat_grid[-1]:
        lat_grid = lat_grid[::-1]
        matrix = matrix[::-1, :]

    return matrix, lon_grid, lat_grid


def main():
    print("=" * 75)
    print(f"ATTRIBUTION MODELS COMPARISON: OLS vs BLR vs CVAE (Train: {START_YEAR}-{SPLIT_YEAR} | Test: {SPLIT_YEAR+1}+)")
    print("=" * 75)

    (
        _X_train, _Y_train, X_test, Y_test,
        tg_dim, pp_dim, _cond_dim, norm_stats, ds
    ) = buildingTensors(DATASET_PATH, start_year=START_YEAR, split_year=SPLIT_YEAR)

    times = pd.to_datetime(ds["time"].values)
    train_mask = (times.year >= START_YEAR) & (times.year <= SPLIT_YEAR)
    test_mask = (times.year > SPLIT_YEAR)

    times_train = times[train_mask]
    times_test = times[test_mask]
    
    doy_test = times_test.dayofyear.values

    tg_raw = ds["tg"].values
    fgmt_raw = ds["fgmt"].values

    tg_train_raw = tg_raw[train_mask]
    fgmt_train_raw = fgmt_raw[train_mask]
    
    tg_test_raw = tg_raw[test_mask]
    fgmt_test_raw = fgmt_raw[test_mask]

    ds_train = ds.isel(time=train_mask)
    clim_doy = ds_train["tg"].groupby("time.dayofyear").mean(dim="time")
    clim_train = clim_doy.sel(dayofyear=times_train.dayofyear.values).values
    clim_test = clim_doy.sel(dayofyear=doy_test).values

    tg_anom_train = tg_train_raw - clim_train

    print("\n[SHARED COUNTERFACTUAL SCENARIO]")
    print(f"Reference Counterfactual GMTA: {GMT_COUNTERFACTUAL_VAL:.4f} °C\n")

    # OLS INFERENCE
    print("OLS Training and Inference (Monthly Model)...")
    ols_fact_test = np.full_like(tg_test_raw, np.nan)
    ols_cf_test = np.full_like(tg_test_raw, np.nan)
    ols_impact = np.full_like(tg_test_raw, np.nan)
    
    for m in range(1, 13):
        mask_train_m = (times_train.month == m)
        mask_test_m = (times_test.month == m)
        if not np.any(mask_train_m) or not np.any(mask_test_m): 
            continue
        
        tg_anom_train_m = tg_anom_train[mask_train_m]
        fgmt_train_m = fgmt_train_raw[mask_train_m]
        alpha_m, beta_m = fit_ols(tg_anom_train_m, fgmt_train_m)
        
        fgmt_test_m = fgmt_test_raw[mask_test_m]
        clim_test_m = clim_test[mask_test_m]
        
        anom_fact_m = alpha_m + fgmt_test_m[:, np.newaxis] * beta_m
        fact_m = clim_test_m + anom_fact_m
        
        impact_m, cf_m = calcola_impatto(
            beta_map=beta_m,
            gmt_factual=fgmt_test_m,
            gmt_counterfactual=GMT_COUNTERFACTUAL_VAL,
            tg_factual=fact_m
        )
        
        ols_fact_test[mask_test_m] = fact_m
        ols_cf_test[mask_test_m] = cf_m
        ols_impact[mask_test_m] = impact_m

    # BLR INFERENCE
    print("BLR Inference...")
    if not os.path.exists(BLR_TRACE_PATH):
        raise FileNotFoundError(f"BLR parameters file not found at '{BLR_TRACE_PATH}'.")

    trace = az.from_netcdf(BLR_TRACE_PATH)
    
    blr_res = predict_scenarios(
        doy=doy_test,
        temp=tg_test_raw,
        gmt=fgmt_test_raw,
        trace=trace,
        baseline_gmt=GMT_COUNTERFACTUAL_VAL
    )
    blr_fact_test = blr_res["mu_factual"]
    blr_cf_test = blr_res["mu_counterfactual"]
    blr_impact = blr_res["warming_effect"]

    # CVAE INFERENCE
    print("CVAE Inference (Dual Evaluation: Pure vs Inverted Reconstruction)...")
    cvae_model = CVAE(tg_dim=tg_dim, pp_dim=pp_dim, hidden_dim=256, latent_dim=32).to(DEVICE)
    if not os.path.exists(CVAE_WEIGHTS_PATH):
        raise FileNotFoundError(f"CVAE weights not found at '{CVAE_WEIGHTS_PATH}'.")
    cvae_model.load_state_dict(torch.load(CVAE_WEIGHTS_PATH, map_location=DEVICE))

    cvae_fact_recon_norm, cvae_cf_recon_norm, _, cvae_impact_recon = generate_counterfactuals(
        cvae_model, X_test, Y_test, norm_stats, fgmt_cf_val=GMT_COUNTERFACTUAL_VAL, device=DEVICE
    )
    cvae_fact_recon = cvae_fact_recon_norm + clim_test
    cvae_cf_recon = cvae_cf_recon_norm + clim_test

    cvae_fact_pure_norm, cvae_cf_pure_norm, cvae_impact_pure = predict_pure(
        cvae_model, X_test, norm_stats, fgmt_cf_val=GMT_COUNTERFACTUAL_VAL, device=DEVICE
    )
    cvae_fact_pure = cvae_fact_pure_norm + clim_test
    cvae_cf_pure = cvae_cf_pure_norm + clim_test

    # metrics
    scale_ols = evaluate_timescales(tg_test_raw, ols_fact_test, times_test)
    scale_blr = evaluate_timescales(tg_test_raw, blr_fact_test, times_test)
    scale_cvae_pure = evaluate_timescales(tg_test_raw, cvae_fact_pure, times_test)
    scale_cvae_recon = evaluate_timescales(tg_test_raw, cvae_fact_recon, times_test)

    scale_names = ["Daily", "Monthly", "Seasonal"]
    
    print("\n" + "=" * 88)
    print(f"MULTI-SCALE METRICS COMPARISON ON TEST SET ({SPLIT_YEAR + 1} - Present)")
    print("=" * 88)

    records = []
    for scale in scale_names:
        for metric in ["MAE", "RMSE", "R2"]:
            records.append({
                "Scale": scale,
                "Metric": metric,
                "OLS": scale_ols[scale][metric],
                "BLR": scale_blr[scale][metric],
                "CVAE (Pure: Z=0)": scale_cvae_pure[scale][metric],
                "CVAE (Recon: with Y)": scale_cvae_recon[scale][metric],
            })

    df_multi = pd.DataFrame(records).set_index(["Scale", "Metric"])
    print(df_multi.round(4))
    print("=" * 88)

    month_names_long = [
        "1. January", "2. February", "3. March", "4. April",
        "5. May", "6. June", "7. July", "8. August",
        "9. September", "10. October", "11. November", "12. December"
    ]

    # EVOLUTION MONTHLY MAP (OLS, BLR, CVAE) 
    def plot_monthly_grid(impact_arr, title, filename, cmap="Reds"):
        mat_months = []
        for m in range(1, 13):
            mask_m = (times_test.month == m)
            imp_m = np.nanmean(impact_arr[mask_m], axis=0)
            mat, lon_g, lat_g = points_to_grid(imp_m, ds)
            mat_months.append(mat)
        vmin = float(np.nanmin(mat_months))
        vmax = float(np.nanmax(mat_months))

        fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharex=True, sharey=True)
        for idx, ax in enumerate(axes.flat):
            im = ax.imshow(
                mat_months[idx], origin="lower",
                extent=[lon_g.min(), lon_g.max(), lat_g.min(), lat_g.max()],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax.set_title(month_names_long[idx], fontsize=11, fontweight="bold")
            if idx % 3 == 0: ax.set_ylabel("Latitude", fontsize=10)
            if idx >= 9: ax.set_xlabel("Longitude", fontsize=10)

        fig.subplots_adjust(right=0.86, hspace=0.18, wspace=0.08)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Climate Impact ($T_{fact} - T_{cf}$) [°C]", fontsize=11)
        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.93)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

    print("\nGenerating 4x3 monthly grids...")
    plot_monthly_grid(ols_impact, "Month-by-Month Evolution (Monthly OLS)", "graphs/monthly_comparison_12_months_ols.png")
    plot_monthly_grid(blr_impact, "Month-by-Month Evolution (BLR)", "graphs/monthly_comparison_12_months_blr.png")
    plot_monthly_grid(cvae_impact_recon, "Month-by-Month Evolution (CVAE)", "graphs/monthly_comparison_12_months_cvae.png")

    # ANNUAL SYNUSOID
    print("Generating annual continuity plot (Jan -> Dec)...")
    months_x = np.arange(1, 13)
    month_names_short = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    mean_ols_cycle = [np.nanmean(ols_impact[times_test.month == m]) for m in range(1, 13)]
    mean_blr_cycle = [np.nanmean(blr_impact[times_test.month == m]) for m in range(1, 13)]
    mean_cvae_recon_cycle = [np.nanmean(cvae_impact_recon[times_test.month == m]) for m in range(1, 13)]
    mean_cvae_pure_cycle = [np.nanmean(cvae_impact_pure[times_test.month == m]) for m in range(1, 13)]

    plt.figure(figsize=(12, 6))
    plt.plot(months_x, mean_blr_cycle, label="BLR (Harmonics)", color="tab:green", marker="o", linewidth=2.5, zorder=4)
    plt.plot(months_x, mean_cvae_recon_cycle, label="CVAE Recon (Inverted with Y)", color="tab:red", marker="^", linewidth=2.0, zorder=3)
    plt.plot(months_x, mean_cvae_pure_cycle, label="CVAE Pure (Z=0)", color="tab:orange", linestyle="--", marker="x", linewidth=1.5, zorder=3)
    plt.plot(months_x, mean_ols_cycle, label="OLS (Monthly)", color="tab:blue", linestyle=":", marker="s", linewidth=1.8, alpha=0.8, zorder=2)

    plt.axhline(0, color="black", linestyle=":", alpha=0.5, zorder=1)
    plt.title("Mean Climate Impact Profile across 12 Months (Continuity Check)", fontsize=13, fontweight="bold")
    plt.xlabel("Month", fontsize=11)
    plt.ylabel("Mean Impact ($T_{fact} - T_{cf}$) [°C]", fontsize=11)
    plt.xticks(months_x, month_names_short)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(0.0, 3.0)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig("graphs/annual_cycle_continuity_12_months.png", dpi=150)
    plt.close()

    # EXTREME EVENT IN AUGUST 2024
    TARGET_LAT, TARGET_LON = 46.07, 13.23  # Udine
    lats = ds["lat"].values
    lons = ds["lon"].values
    idx_target = int(np.argmin((lats - TARGET_LAT) ** 2 + (lons - TARGET_LON) ** 2))

    print(f"Generating full August 2024 plot for [Lat: {lats[idx_target]:.2f}, Lon: {lons[idx_target]:.2f}]...")
    mask_aug2024 = (times_test >= "2024-08-01") & (times_test <= "2024-08-31")
    days_aug = times_test[mask_aug2024].day

    t_obs_aug = tg_test_raw[mask_aug2024, idx_target]

    t_ols_fact = ols_fact_test[mask_aug2024, idx_target]
    t_ols_cf = ols_cf_test[mask_aug2024, idx_target]

    t_blr_fact = blr_fact_test[mask_aug2024, idx_target]
    t_blr_cf = blr_cf_test[mask_aug2024, idx_target]
    
    t_cvae_pure_fact = cvae_fact_pure[mask_aug2024, idx_target]
    t_cvae_pure_cf = cvae_cf_pure[mask_aug2024, idx_target]
    
    t_cvae_recon_fact = cvae_fact_recon[mask_aug2024, idx_target]
    t_cvae_recon_cf = cvae_cf_recon[mask_aug2024, idx_target]

    plt.figure(figsize=(16, 8.5))

    plt.plot(days_aug, t_ols_fact, color="tab:blue", linestyle="-", linewidth=1.5, alpha=0.35, label="OLS Factual", zorder=2)
    plt.plot(days_aug, t_ols_cf, color="tab:blue", linestyle="--", linewidth=1.5, alpha=0.35, label="OLS CF ($fGMT=0$)", zorder=2)

    plt.plot(days_aug, t_blr_fact, color="tab:green", linestyle="-", linewidth=2.0, alpha=0.9, label="BLR Factual", zorder=5)
    plt.plot(days_aug, t_blr_cf, color="tab:green", linestyle="--", linewidth=2.0, alpha=0.9, label="BLR CF ($fGMT=0$)", zorder=5)

    plt.plot(days_aug, t_cvae_pure_fact, color="tab:orange", linestyle="-", linewidth=2.0, label="CVAE Pure Factual (Z=0)", zorder=6)
    plt.plot(days_aug, t_cvae_pure_cf, color="tab:orange", linestyle="--", linewidth=2.0, label="CVAE Pure CF ($fGMT=0$)", zorder=6)

    plt.plot(days_aug, t_cvae_recon_fact, color="tab:red", linestyle="-", linewidth=2.5, label="CVAE Recon Factual (with Y)", zorder=7)
    plt.plot(days_aug, t_cvae_recon_cf, color="purple", linestyle="--", linewidth=2.5, label="CVAE Recon CF ($fGMT=0$)", zorder=7)

    plt.plot(days_aug, t_obs_aug, color="black", linewidth=3.5, marker="o", markersize=6, label="Observed Real (E-OBS)", zorder=10)

    plt.title(
        f"Heatwave Event - August 2024 | Udine (Lat {lats[idx_target]:.2f}, Lon {lons[idx_target]:.2f})\n"
        f"Full Model Comparison: Factual (Solid Line) vs Counterfactual (Dashed Line)",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Day of August 2024", fontsize=12)
    plt.ylabel("Surface Temperature (°C)", fontsize=12)
    plt.xticks(np.arange(1, 32, 2))
    plt.grid(True, linestyle=":", alpha=0.7)
    y_max = np.nanmax(t_obs_aug)
    plt.ylim(top=y_max + 2.0)
    plt.legend(
        ncol=5, 
        loc="upper right",
        framealpha=0.95,
        fontsize=10,
        title="Legend: Solid = Factual | Dashed = Counterfactual (Pre-industrial world)"
    )
    plt.tight_layout()
    plt.savefig("graphs/comparison_august_2024_single_coordinate.png", dpi=150)
    plt.close()

    # OLS graphs (Further analysis on OLS beyond the previous aspects)
    print("Generating OLS diagnostic plots...")
    da_fgmt = xr.DataArray(fgmt_raw, coords={"time": times}, dims=["time"])
    gmt_m = da_fgmt.resample(time="1MS").mean()
    gmt_6m = gmt_m.rolling(time=6, min_periods=1).mean()
    gmt_12m = gmt_m.rolling(time=12, min_periods=1).mean()
    gmt_24m = gmt_m.rolling(time=24, min_periods=1).mean()

    grafico_smoothing(gmt_m, gmt_6m, gmt_12m, gmt_24m, save_path="graphs/ols_gmta_smoothing.png")

    _, da_cf_gmt = costruisci_gmta_counterfactual(
        gmt_12m, 
        inizio_riferimento="1951-01-01", 
        fine_riferimento="1960-12-31"
    )
    grafico_gmta_counterfactual(gmt_12m, da_cf_gmt, save_path="graphs/ols_gmta_factual_vs_cf.png")


    var_tg = np.nanvar(tg_anom_train, axis=0)
    var_tg[var_tg == 0] = np.nan
    tg_diff = tg_anom_train - np.nanmean(tg_anom_train, axis=0)
    fgmt_diff = (fgmt_train_raw - np.nanmean(fgmt_train_raw))[:, None]
    
    cov_tg_fgmt = np.nanmean(tg_diff * fgmt_diff, axis=0)
    beta_1d = cov_tg_fgmt / np.nanvar(fgmt_train_raw)
    r2_1d = (cov_tg_fgmt ** 2) / (np.nanvar(fgmt_train_raw) * var_tg)
    beta_grid, lon_g, lat_g = points_to_grid(beta_1d, ds)
    r2_grid, _, _ = points_to_grid(r2_1d, ds)
    da_beta = xr.DataArray(beta_grid, coords={"latitude": lat_g, "longitude": lon_g}, dims=["latitude", "longitude"])
    da_r2 = xr.DataArray(r2_grid, coords={"latitude": lat_g, "longitude": lon_g}, dims=["latitude", "longitude"])

    grafico_beta(da_beta, save_path="graphs/ols_map_beta.png")
    grafico_r2(da_r2, save_path="graphs/ols_map_r2.png")
    grafico_beta_affidabile(da_beta, da_r2, soglia=0.20, save_path="graphs/ols_map_beta_reliable.png")


    da_ols_impact_points = xr.DataArray(
        ols_impact,
        coords={"time": times_test, "point": np.arange(ols_impact.shape[1])},
        dims=["time", "point"]
    )
    mean_imp_period = indice_impatto_periodo(
        da_ols_impact_points,
        data_inizio="2015-01-01",
        data_fine="2024-12-31"
    ).values
    imp_grid, _, _ = points_to_grid(mean_imp_period, ds)
    da_imp = xr.DataArray(imp_grid, coords={"latitude": lat_g, "longitude": lon_g}, dims=["latitude", "longitude"])
    grafico_impatto(da_imp, save_path="graphs/ols_map_impact_2015_2024.png")

    print("\n[COMPLETED] All graphs and tables have been generated successfully.")

if __name__ == "__main__":
    main()
