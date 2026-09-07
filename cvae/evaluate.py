
import numpy as np
import torch


def compute_physical_sensitivity_stats(model, X_test, norm_stats, device="cpu"):
    model.eval()
    X_test = X_test.to(device)
    pp_test = X_test[:, :model.pp_dim]
    with torch.no_grad():
        sens_norm = model.decoder.get_sensitivity(pp_test).cpu().numpy()
        sens_norm_1d = np.mean(sens_norm, axis=0)
    tg_std = norm_stats["tg_std"]
    if isinstance(tg_std, torch.Tensor):
        tg_std = tg_std.cpu().numpy()
    fgmt_std = float(norm_stats["fgmt_std"])
    sens_phys = sens_norm_1d * (tg_std / fgmt_std)
    mean_s = float(np.mean(sens_phys))
    std_s = float(np.std(sens_phys))
    min_s = float(np.min(sens_phys))
    max_s = float(np.max(sens_phys))
    return mean_s, std_s, min_s, max_s, sens_phys

def generate_counterfactuals(model, X_test, Y_test, norm_stats, fgmt_cf_val=0.0, device="cpu"):
    model = model.to(device)
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    pp_test = X_test[:, : model.pp_dim]
    fgmt_test = X_test[:, model.pp_dim :]

    with torch.no_grad():
        sensitivity = model.decoder.get_sensitivity(pp_test)
        y_dyn_test = Y_test - sensitivity * fgmt_test

        mu_z, _ = model.encoder(y_dyn_test, pp_test)
        z = mu_z

        mu_fact, _, _, _ = model.decoder(z, pp_test, fgmt_test)

        fgmt_cf_norm = float((fgmt_cf_val - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"])
        fgmt_cf_tensor = torch.full_like(fgmt_test, fgmt_cf_norm)
        
        mu_cf, _, _, _ = model.decoder(z, pp_test, fgmt_cf_tensor)

    y_factual = mu_fact.cpu().numpy() * norm_stats["tg_std"]
    y_counterfactual = mu_cf.cpu().numpy() * norm_stats["tg_std"]
    y_true = Y_test.cpu().numpy() * norm_stats["tg_std"]

    climate_impact = y_factual - y_counterfactual
    return y_factual, y_counterfactual, y_true, climate_impact
def predict_pure(model, X_test, norm_stats, fgmt_cf_val=0.0, device="cpu"):
    model = model.to(device)
    model.eval()
    X_test = X_test.to(device)

    pp_test = X_test[:, : model.pp_dim]
    fgmt_test = X_test[:, model.pp_dim :]
    N = X_test.shape[0]
    latent_dim = model.encoder.fc_mu.out_features

    with torch.no_grad():
        z_pure = torch.zeros((N, latent_dim), device=device)

        mu_fact, _, _, _ = model.decoder(z_pure, pp_test, fgmt_test)

        fgmt_cf_norm = float((fgmt_cf_val - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"])
        fgmt_cf_tensor = torch.full_like(fgmt_test, fgmt_cf_norm)
        mu_cf, _, _, _ = model.decoder(z_pure, pp_test, fgmt_cf_tensor)

    tg_std = norm_stats["tg_std"]
    if isinstance(tg_std, torch.Tensor):
        tg_std = tg_std.cpu().numpy()

    y_factual_pure = mu_fact.cpu().numpy() * tg_std
    y_counterfactual_pure = mu_cf.cpu().numpy() * tg_std
    climate_impact_pure = y_factual_pure - y_counterfactual_pure

    return y_factual_pure, y_counterfactual_pure, climate_impact_pure
