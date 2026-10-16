"""Specyfikacje i estymacja panelowych modeli liczebności.

Siedem specyfikacji z zakresu wykładu, w odpowiednikach Stata:

    pooled          poisson
    pooled_cluster  poisson, vce(cluster)
    negbin          nbreg
    fe              xtpoisson, fe
    pa              xtpoisson, pa
    cre             CRE Poisson (Mundlak)
    zip             zip

Moduł **tworzy** oszacowania. To, co się z nich czyta (IRR, porównania,
diagnostyka), jest w `inference`.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.pre_processing import add_mundlak_means

TERMS = ["log_gdp_pc", "log_pop", "host", "medals_lag"]
FORMULA = "medals ~ " + " + ".join(TERMS)

# Efekty stałe kraju pochłaniają trwałą heterogeniczność (tradycja, ustrój),
# efekty igrzysk - wspólne szoki edycji (liczba konkurencji, bojkoty).
FE_FORMULA = FORMULA + " | country_noc + edition_id"

CLUSTER = "country_noc"

# Zmienne w równaniu inflacyjnym ZIP: co decyduje o "zerze strukturalnym".
INFLATION_VARS = ["log_gdp_pc", "log_pop"]


def pooled_poisson(df: pd.DataFrame):
    """Przekrojowy Poisson - punkt odniesienia i podstawa oceny dyspersji."""
    return smf.poisson(FORMULA, data=df).fit(disp=0)


def pooled_poisson_cluster(df: pd.DataFrame):
    """Poisson z macierzą klastrowaną po kraju - rekomendacja Wooldridge'a.

    Zgodność beta wymaga tylko poprawnej specyfikacji równania średniej;
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
    """Population-averaged Poisson (GEE) z wymienialną strukturą korelacji."""
    return sm.GEE.from_formula(
        FORMULA,
        groups=CLUSTER,
        data=df,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()


def cre_poisson(df: pd.DataFrame):
    """CRE / Mundlak: regresory plus ich średnie wewnątrzkrajowe.

    Zwraca (wynik, ramkę ze średnimi, listę nazw kolumn `*_bar`).
    """
    cre_df = add_mundlak_means(df, TERMS)
    bars = [c + "_bar" for c in TERMS]
    formula = "medals ~ " + " + ".join(TERMS + bars)
    res = smf.poisson(formula, data=cre_df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": cre_df[CLUSTER]}
    )
    return res, cre_df, bars


def zero_inflated_poisson(df: pd.DataFrame):
    """ZIP: część liczebnościowa plus logit prawdopodobieństwa zera strukturalnego.

    Zwraca `None`, jeśli optymalizacja nie zbiegła - bez przerywania przebiegu,
    bo pozostałe modele są od niego niezależne.
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

    Zwraca słownik z oszacowaniami oraz ramką CRE i nazwami średnich Mundlaka,
    których potrzebują `inference` i `visualisation`. Na panelu rzędu dwóch
    tysięcy obserwacji całość liczy się poniżej dwóch sekund.
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
