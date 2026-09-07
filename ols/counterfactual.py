import numpy as np
import xarray as xr

# COSTRUZIONE DELLA GMTA CONTROFATTUALE

def costruisci_gmta_counterfactual(
    gmt_factual,
    inizio_riferimento="1951-01-01",
    fine_riferimento="1960-12-31"
):
    if isinstance(gmt_factual, xr.DataArray):
        periodo_riferimento = gmt_factual.sel(
            time=slice(
                inizio_riferimento,
                fine_riferimento
            )
        )
        gmt_riferimento = float(
            periodo_riferimento
            .mean()
            .values
        )
        gmt_counterfactual = xr.full_like(
            gmt_factual,
            gmt_riferimento
        )
        gmt_counterfactual.name = (
            "gmt_counterfactual"
        )
    else:
        gmt_riferimento = float(np.nanmean(gmt_factual))
        gmt_counterfactual = np.full_like(gmt_factual, gmt_riferimento)

    return (
        gmt_riferimento,
        gmt_counterfactual
    )

# CALCOLO DELL'IMPATTO

def calcola_impatto(
    beta_map,
    gmt_factual,
    gmt_counterfactual,
    tg_factual
):
    delta_gmt = (
        gmt_factual
        -
        gmt_counterfactual
    )

    if isinstance(tg_factual, xr.DataArray):
        impatto_tempo = (
            beta_map
            *
            delta_gmt
        )
        if {"latitude", "longitude"}.issubset(impatto_tempo.dims):
            impatto_tempo = impatto_tempo.transpose(
                "time",
                "latitude",
                "longitude"
            )
        impatto_tempo.name = "climate_impact"

        tg_counterfactual = (
            tg_factual
            -
            impatto_tempo
        )
        tg_counterfactual.name = "counterfactual_temperature_anomaly"
    else:
        if delta_gmt.ndim == 1:
            delta_gmt = delta_gmt[:, np.newaxis]
        impatto_tempo = delta_gmt * beta_map 
        tg_counterfactual = tg_factual - impatto_tempo

    return (
        impatto_tempo,
        tg_counterfactual
    )

# INDICE DI IMPATTO SU UN PERIODO

def indice_impatto_periodo(
    impatto,
    data_inizio,
    data_fine
):
    periodo = impatto.sel(
        time=slice(
            data_inizio,
            data_fine
        )
    )

    if periodo.sizes["time"] == 0:
        raise ValueError(
            "Il periodo selezionato non contiene dati."
        )

    return periodo.mean(
        dim="time",
        skipna=True
    )

