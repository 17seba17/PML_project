import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluate import (
    compute_physical_sensitivity_stats,
    generate_counterfactuals,
    predict_pure,
)

from loss import vae_loss_function
from models import CVAE
from scales import scale

from ols.dataset import buildingTensors

if __name__ == "__main__":
    epochs = 50
    batch_size = 64
    lr = 1e-3
    beta = 0.1
    split_year = 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Creating tensors from dataset on device: {device}...")
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        tg_dim,
        pp_dim,
        cond_dim,
        norm_stats,
        ds,
    ) = buildingTensors("dataset/vae_dataset.nc", start_year=1980, split_year=2000)

    target_mu_norm, target_std_norm, base_sens_init = scale(
        norm_stats,
        lat_min=ds.attrs["lat_min"],
        lat_max=ds.attrs["lat_max"],
        lon_min=ds.attrs["lon_min"],
        lon_max=ds.attrs["lon_max"],
        split_year=split_year
    )


    print(f"Instantiating CVAE | Total spatial points: {tg_dim}...")
    model = CVAE(
        tg_dim=tg_dim,
        pp_dim=pp_dim,
        hidden_dim=256,
        latent_dim=32,
        base_sensitivity=base_sens_init
    ).to(device)

    base_param = [model.decoder.base_sensitivity]
    base_id = set(map(id, base_param))
    general_params = [p for p in model.parameters() if id(p) not in base_id]

    optimizer = torch.optim.Adam([
        {'params': general_params, 'lr': lr},
        {'params': base_param, 'lr': 0.2*lr}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=max(1, epochs), 
        eta_min=1e-5
    )

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\n--- Starting {epochs} Epochs Evolution Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        total_grad_norm = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            mu_y, logvar_y, mu_z, logvar_z, sensitivity, mu_dyn = model(batch_y, batch_x)

            loss, _, _ = vae_loss_function(
                mu_y, logvar_y, batch_y, mu_z, logvar_z, sensitivity, mu_dyn,
                beta=beta,
                target_mu_norm=target_mu_norm,
                target_std_norm=target_std_norm,
                lambda_moments=1.0,
                lambda_dyn=0.0
            )

            loss.backward()

            batch_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_grad_norm += float(batch_grad_norm)

        epoch_grad_norm = total_grad_norm / len(loader)
        epoch_loss = total_loss / len(X_train)
        scheduler.step()

        mean_s, std_s, min_s, max_s, _ = compute_physical_sensitivity_stats(
            model, X_test, norm_stats, device=device
        )
        _, _, y_true, _ = generate_counterfactuals(
            model, X_test, Y_test, norm_stats, device=device
        )
        y_fact_pure, _, _ = predict_pure(
            model, X_test, norm_stats, device=device
        )
        pure_rmse = np.sqrt(np.mean((y_true - y_fact_pure) ** 2))
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Test RMSE: {pure_rmse:.2f} °C | "
            f"Sens: {mean_s:.4f} ± {std_s:.4f} °C/°C (Min: {min_s:.2f}, Max: {max_s:.2f}) | "
            f"Pendenza: {epoch_grad_norm:.4f}"
        )

    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), "cvae/cvae_model.pt")
    print("\nModello salvato con successo in 'cvae/cvae_model.pt'.")
