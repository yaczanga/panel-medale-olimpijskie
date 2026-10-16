# -*- coding: utf-8 -*-
"""Przebieg wszystkich modeli na gotowym panelu; wynik do reports/model_results.txt.

Skrypt nie buduje panelu od danych surowych - czyta zapisany
`data/processed/panel_analityczny.csv`. To jest jego sens: szybki przebieg
samych modeli, bez wczytywania 95 MB wyników olimpijskich. Panel buduje
notebooks/01_panel_medale.ipynb.

Uruchomienie z korzenia projektu:

    python scripts/run_models.py
"""

import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data_load import PROJECT_ROOT
from src.inference import compare_gdp_effect, dispersion_parts, irr_table, mundlak_test
from src.model import TERMS, fit_all
from src.pre_processing import describe_panel, load_model_frame

OUTPUT = PROJECT_ROOT / "reports" / "model_results.txt"


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    df = load_model_frame()
    fits = fit_all(df)

    d = describe_panel(df)
    print(f"N = {d['N']} | krajow = {d['krajow']} | edycji = {d['edycji']}")
    print(f"lata: {d['rok_min']} - {d['rok_max']}")
    print(f"udzial zer: {round(d['udzial_zer'], 4)}")
    print("medals describe:")
    print(df["medals"].describe())
    print(f"mean medals: {d['srednia']:.3f} var medals: {d['wariancja']:.3f}")

    pois = fits["pooled"]
    chi2_p, df_p, disp = dispersion_parts(pois)
    print("\n=== 4a Pooled Poisson ===")
    print(pois.summary())
    print(f"Pearson chi2/df = {chi2_p:,.1f} / {df_p} = {disp:.3f}")
    print("AIC pois:", round(pois.aic, 1))

    print("\n=== 4b Pooled Poisson + cluster ===")
    pois_cl = fits["pooled_cluster"]
    print(irr_table(pois_cl.params, pois_cl.bse).round(4))

    negbin = fits["negbin"]
    print("\n=== 4c NegBin ===")
    print(negbin.summary())
    print("alpha =", float(negbin.params.get("alpha", np.nan)))
    print("AIC negbin:", round(negbin.aic, 1))

    fepois = fits["fe"]
    print("\n=== 4d FE-Poisson ===")
    print(fepois.summary())
    print("N used FE:", fepois._N)

    ga = fits["pa"]
    print("\n=== 4e PA Poisson (GEE) ===")
    print(ga.summary())
    print("IRR:", dict(np.exp(ga.params[TERMS]).round(4)))

    cre = fits["cre"]
    print("\n=== 4f CRE / Mundlak ===")
    print(cre.summary())
    chi2, pval = mundlak_test(cre, fits["mundlak_bars"])
    print(f"Mundlak test chi2= {chi2} p= {pval} df= {len(fits['mundlak_bars'])}")

    zip_res = fits["zip"]
    if zip_res is not None:
        print("\n=== 4g ZIP ===")
        print(zip_res.summary())
        print(
            "AIC: Poisson", round(pois.aic, 1),
            "NegBin", round(negbin.aic, 1),
            "ZIP", round(zip_res.aic, 1),
        )

    print("\n=== 5 Porownanie efektu PKB ===")
    print(compare_gdp_effect(fits).round(4).to_string())

    print("\n=== IRR pelna tabela pooled-cluster ===")
    print(irr_table(pois_cl.params, pois_cl.bse).round(4).to_string())

    print("\n=== IRR FE-Poisson ===")
    for t in TERMS:
        c = fepois.coef().get(t, np.nan)
        print(
            t, "coef", round(c, 4),
            "se", round(fepois.se().get(t, np.nan), 4),
            "p", round(fepois.pvalue().get(t, np.nan), 4),
            "IRR", round(np.exp(c), 4),
        )

    print("\nDONE")


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    sys.stdout = io.open(OUTPUT, "w", encoding="utf-8")
    try:
        main()
    finally:
        sys.stdout.close()
        sys.stdout = original
    print("Zapisano:", OUTPUT)
