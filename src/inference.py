"""Co sie czyta z oszacowanego modelu: IRR, dyspersja, testy, porownania.

Oddzielone od `model`, zeby dopasowanie wykonac raz, a wnioskowanie powtarzac.
"""

import numpy as np
import pandas as pd

from src.model import TERMS

# Etykiety polskie - wspolne dla tabel HTML i wydrukow.
LABELS = {
    "log_gdp_pc": "log PKB per capita",
    "log_pop": "log populacji",
    "host": "Gospodarz (host)",
    "medals_lag": "medals_lag (kontrola)",
    "Intercept": "Stała",
}

# Progi oceny dyspersji: Pearson chi2/df istotnie powyzej 1 oznacza Var > E.
DISPERSION_THRESHOLD = 1.5


def dispersion_parts(pooled) -> tuple[float, float, float]:
    """Statystyka Pearsona, liczba stopni swobody i ich iloraz.

    Zwracane sa tez same skladniki, bo iloraz 6,06 przy statystyce rzedu
    dwunastu tysiecy mowi cos innego niz ten sam iloraz na malej probie.
    """
    chi2 = float((pooled.resid_pearson**2).sum())
    df = float(pooled.df_resid)
    return chi2, df, chi2 / df


def mundlak_test(cre, bars: list[str]) -> tuple[float, float]:
    """Laczny test istotnosci srednich Mundlaka - odpowiednik testu Hausmana."""
    w = cre.wald_test(", ".join(f"{b} = 0" for b in bars), use_f=False, scalar=True)
    return float(w.statistic), float(w.pvalue)


def irr_table(params, bse, names: list[str] = None) -> pd.DataFrame:
    """Wspolczynnik, IRR = exp(beta), blad standardowy, statystyka z i p.

    Wartosc p z rozkladu normalnego: przy macierzy klastrowej z jest
    asymptotycznie normalna.
    """
    from scipy.stats import norm

    names = names or TERMS
    p = params.reindex(names)
    s = bse.reindex(names)
    z = p / s
    return pd.DataFrame(
        {"coef": p, "IRR": np.exp(p), "se": s, "z": z, "p": 2 * (1 - norm.cdf(np.abs(z)))}
    )


def _coef_se(res, key: str) -> tuple[float, float]:
    """Wspolczynnik i blad standardowy, niezaleznie od interfejsu estymatora.

    `pyfixest` udostepnia je metodami `coef()`/`se()`, statsmodels atrybutami.
    """
    if hasattr(res, "coef") and callable(res.coef):
        return float(res.coef().get(key, np.nan)), float(res.se().get(key, np.nan))
    return float(res.params.get(key, np.nan)), float(res.bse.get(key, np.nan))


# Kolejnosc: od modelu najbardziej do najmniej naiwnego wobec heterogenicznosci.
COMPARISON_ORDER = [
    ("Pooled Poisson (klaster)", "pooled_cluster"),
    ("PA Poisson (GEE)", "pa"),
    ("Negative Binomial", "negbin"),
    ("FE-Poisson", "fe"),
    ("CRE / Mundlak", "cre"),
]


def compare_gdp_effect(fits: dict, key: str = "log_gdp_pc") -> pd.DataFrame:
    """Efekt PKB per capita w kolejnych estymatorach."""
    rows = [
        (label, *_coef_se(fits[name], key)) for label, name in COMPARISON_ORDER
    ]
    comp = pd.DataFrame(rows, columns=["model", "coef_log_gdp_pc", "se"])
    comp["IRR"] = np.exp(comp["coef_log_gdp_pc"])
    comp["IRR_+10%PKB"] = comp["coef_log_gdp_pc"].apply(lambda b: np.exp(b * np.log(1.10)))
    return comp
