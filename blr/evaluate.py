import numpy as np
from scipy.stats import norm

N_MODES_BASE = 4
N_MODES_TREND = 1
OMEGA = 2 * np.pi / 365.25

def fourier_design_matrix(doy, n_modes):
    t = np.asarray(doy, dtype=float)
    cols = []
    for k in range(1, n_modes + 1):
        cols.append(np.cos(k * OMEGA * t))
        cols.append(np.sin(k * OMEGA * t))
    return np.column_stack(cols)

def predict_scenarios(
    doy, 
    temp, 
    gmt, 
    trace, 
    baseline_gmt=0.0
):
    
    ds = trace.posterior if hasattr(trace, "posterior") else trace

    def _get_mean(possible_names):
        for name in possible_names:
            if name in ds:
                arr = ds[name].values
                if len(arr.shape) > 1 and hasattr(ds[name], "dims"):
                    if "chain" in ds[name].dims and "draw" in ds[name].dims:
                        return arr.mean(axis=(0, 1))
                return arr

    
    a0 = _get_mean(["a0_intercept", "a0"])                            
    harmonic_coefs = _get_mean(["base_harmonic_coefs", "harmonic_coefs"])    
    beta_0 = _get_mean(["gmt_intercept", "beta_0"])                    
    slope_harmonics = _get_mean(["gmt_harmonic_coefs", "slope_harmonics"])  
    sigma_hat = _get_mean(["sigma"])                  

    
    harmonics_base = fourier_design_matrix(doy, N_MODES_BASE)   
    harmonics_trend = fourier_design_matrix(doy, N_MODES_TREND) 

    
    seasonal_base = a0 + harmonics_base @ harmonic_coefs
    climate_sensitivity = beta_0 + harmonics_trend @ slope_harmonics

    
    mu_f = seasonal_base + climate_sensitivity * gmt[:, np.newaxis]
    mu_cf = seasonal_base + climate_sensitivity * baseline_gmt

    
    
    z_f = (temp - mu_f) / sigma_hat
    
    
    q = norm.cdf(np.clip(z_f, -4.0, 4.0))
    
    
    z_cf = norm.ppf(np.clip(q, 1e-6, 1.0 - 1e-6))
    
    
    temp_cf = mu_cf + z_cf * sigma_hat

    
    warming_effect = temp - temp_cf

    return {
        "mu_factual": mu_f,               
        "mu_counterfactual": mu_cf,       
        "counterfactual_weather": temp_cf,     
        "warming_effect": warming_effect       
    }
