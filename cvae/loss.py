import torch


def vae_loss_function(
    mu_y, logvar_y, y_true, mu_z, logvar_z, sensitivity, mu_dyn,
    beta, 
    target_mu_norm, 
    target_std_norm, 
    lambda_moments,
    lambda_dyn

):
    var_y = torch.exp(logvar_y)
    recon_nll = 0.5 * torch.mean(logvar_y + ((y_true - mu_y) ** 2) / var_y)
    kl_div = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    sens_mean = torch.mean(sensitivity)
    sens_std = torch.std(sensitivity)
    loss_mean = (sens_mean - target_mu_norm) ** 2
    loss_std =  (sens_std - target_std_norm) ** 2
    moment_loss = loss_mean + loss_std

    dyn_loss = torch.mean(mu_dyn ** 2)

    total_loss = recon_nll + beta * kl_div + lambda_moments * moment_loss + lambda_dyn * dyn_loss

    return total_loss, recon_nll, kl_div
