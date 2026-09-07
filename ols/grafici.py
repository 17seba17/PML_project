import matplotlib.pyplot as plt


# GMTA AND SMOOTHING
def grafico_smoothing(gmt_monthly, gmt_6m, gmt_12m, gmt_24m, save_path="grafico_smoothing"):
    plt.figure(figsize=(12, 6))
    plt.plot(gmt_monthly.time, gmt_monthly, linewidth=0.5, label="Monthly")
    plt.plot(gmt_6m.time, gmt_6m, linewidth=1.2, label="6 months")
    plt.plot(gmt_12m.time, gmt_12m, linewidth=1.8, label="12 months")
    plt.plot(gmt_24m.time, gmt_24m, linewidth=2, label="24 months")
    plt.xlabel("Year")
    plt.ylabel("Global Mean Temperature Anomaly (K)")
    plt.title("Global Mean Temperature Anomaly")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

# BETA MAP
def grafico_beta(beta_map, save_path="grafico_beta"):
    plt.figure(figsize=(9, 10))
    beta_map.plot(
        x="longitude", y="latitude", robust=True,
        cbar_kwargs={"label": "β (local °C / global K)"}
    )
    plt.title("OLS - Local Sensitivity to GMTA (12 months)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

# R² MAP
def grafico_r2(r2_map, save_path="grafico_r2"):
    plt.figure(figsize=(9, 10))
    r2_map.plot(x="longitude", y="latitude", cbar_kwargs={"label": "R²"})
    plt.title("OLS - R² of 12-Month GMTA Model")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

# BETA MAP WITH R^2 THRESHOLD
def grafico_beta_affidabile(beta_map, r2_map, soglia=0.20, save_path="grafico_beta_affidabile"):
    beta_affidabile = beta_map.where(r2_map >= soglia)
    plt.figure(figsize=(9, 10))
    beta_affidabile.plot(
        x="longitude", y="latitude", robust=True,
        cbar_kwargs={"label": "β (local °C / global K)"}
    )
    plt.title(f"OLS - Local Sensitivity to GMTA\ngrid cells with R² >= {soglia}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

# GMTA FACTUAL VS COUNTERFACTUAL
def grafico_gmta_counterfactual(gmt_factual, gmt_counterfactual, save_path="grafico_gmta_counterfactual"):
    plt.figure(figsize=(12, 5))
    plt.plot(gmt_factual.time, gmt_factual, label="Factual GMTA - 12 months")
    plt.plot(gmt_counterfactual.time, gmt_counterfactual, linewidth=2, label="Counterfactual GMTA")
    plt.xlabel("Year")
    plt.ylabel("Global Mean Temperature Anomaly (K)")
    plt.title("Factual GMTA and Counterfactual Scenario")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

# IMPACT MAP
def grafico_impatto(impatto, save_path="grafico_impatto"):
    plt.figure(figsize=(9, 10))
    impatto.plot(
        x="longitude", y="latitude", robust=True,
        cbar_kwargs={"label": "Factual - Counterfactual (°C)"}
    )
    plt.title("OLS - Mean Factual - Counterfactual Difference\n 2015–2024")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
