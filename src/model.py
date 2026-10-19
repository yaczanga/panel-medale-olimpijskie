"""Specyfikacje i estymacja panelowych modeli liczebnosci.

Modul tworzy oszacowania; to, co sie z nich czyta, jest w `inference`.
Odpowiedniki polecen Stata sa w komorce tekstowej notebooka.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.pre_processing import add_mundlak_means

TERMS = ["log_gdp_pc", "log_pop", "host", "medals_lag"]
FORMULA = "medals ~ " + " + ".join(TERMS)

# Efekty stale kraju pochlaniaja trwala heterogenicznosc (tradycja, ustroj),
# efekty igrzysk - wspolne szoki edycji (liczba konkurencji, bojkoty).
FE_FORMULA = FORMULA + " | country_noc + edition_id"

CLUSTER = "country_noc"

# Zmienne w rownaniu inflacyjnym ZIP: co decyduje o "zerze strukturalnym".
INFLATION_VARS = ["log_gdp_pc", "log_pop"]


def pooled_poisson(df: pd.DataFrame):
    """Przekrojowy Poisson - punkt odniesienia i podstawa oceny dyspersji."""
    return smf.poisson(FORMULA, data=df).fit(disp=0)


def pooled_poisson_cluster(df: pd.DataFrame):
    """Poisson z macierza klastrowana po kraju - rekomendacja Wooldridge'a.

    Zgodnosc beta wymaga tylko poprawnej specyfikacji rownania sredniej;
    klastrowa macierz koryguje wnioskowanie przy nadmiernej dyspersji.
    """
    return smf.poisson(FORMULA, data=df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df[CLUSTER]}
    )


def negative_binomial(df: pd.DataFrame):
    """NegBin2 - jawne modelowanie nadmiernej dyspersji parametrem alpha."""
    return smf.negativebinomial(FORMULA, data=df).fit(disp=0)


def fe_poisson(df: pd.DataFrame):
    """FE-Poisson z efektami kraju i igrzysk, SE klastrowane po kraju (CRV1)."""
    import pyfixest as pf

    return pf.fepois(FE_FORMULA, data=df, vcov={"CRV1": CLUSTER})


def pa_poisson(df: pd.DataFrame):
    """Population-averaged Poisson (GEE) z wymienialna struktura korelacji."""
    return sm.GEE.from_formula(
        FORMULA,
        groups=CLUSTER,
        data=df,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()


def cre_poisson(df: pd.DataFrame):
    """CRE / Mundlak: regresory plus ich srednie wewnatrzkrajowe.

    Zwraca (wynik, ramke ze srednimi, liste nazw kolumn `*_bar`).
    """
    cre_df = add_mundlak_means(df, TERMS)
    bars = [c + "_bar" for c in TERMS]
    formula = "medals ~ " + " + ".join(TERMS + bars)
    res = smf.poisson(formula, data=cre_df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": cre_df[CLUSTER]}
    )
    return res, cre_df, bars


def zero_inflated_poisson(df: pd.DataFrame):
    """ZIP: czesc liczebnosciowa plus logit prawdopodobienstwa zera strukturalnego.

    Zwraca `None`, jesli optymalizacja nie zbiegla - bez przerywania przebiegu,
    bo pozostale modele sa od niego niezalezne.
    """
    from patsy import dmatrices
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    y, x = dmatrices(FORMULA, df, return_type="dataframe")
    z = sm.add_constant(df[INFLATION_VARS])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ZeroInflatedPoisson(y, x, exog_infl=z, inflation="logit").fit(
                method="bfgs", maxiter=500, disp=0
            )
    except Exception as exc:
        print("ZIP nie zbiegł:", repr(exc))
        return None


def fit_all(df: pd.DataFrame) -> dict:
    """Wszystkie specyfikacje w jednym przebiegu.

    Zwraca slownik z oszacowaniami oraz ramka CRE i nazwami srednich Mundlaka,
    ktorych potrzebuja `inference` i `visualisation`.
    """
    cre, cre_df, bars = cre_poisson(df)
    return {
        "pooled": pooled_poisson(df),
        "pooled_cluster": pooled_poisson_cluster(df),
        "negbin": negative_binomial(df),
        "fe": fe_poisson(df),
        "pa": pa_poisson(df),
        "cre": cre,
        "cre_df": cre_df,
        "mundlak_bars": bars,
        "zip": zero_inflated_poisson(df),
    }
