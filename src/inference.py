"""Co się czyta z oszacowanego modelu: IRR, dyspersja, testy, porównania.

Oddzielone od `model`, żeby kosztowne dopasowanie wykonać raz, a wnioskowanie
powtarzać dowolnie.
"""

import numpy as np
import pandas as pd

from src.model import TERMS

# Etykiety polskie - wspólne dla tabel HTML i wydruków.
LABELS = {
    "log_gdp_pc": "log PKB per capita",
    "log_pop": "log populacji",
    "host": "Gospodarz (host)",
    "medals_lag": "medals_lag (kontrola)",
    "Intercept": "Stała",
}

# Progi oceny dyspersji: Pearson chi2/df istotnie powyżej 1 oznacza Var > E.
DISPERSION_THRESHOLD = 1.5


def dispersion_parts(pooled) -> tuple[float, float, float]:
    """Statystyka Pearsona, liczba stopni swobody i ich iloraz.

    Iloraz istotnie powyżej 1 oznacza nadmierną dyspersję: oszacowania beta
    pozostają zgodne, ale błędy standardowe największej wiarygodności są
    błędne. Zwracane są też same składniki, bo iloraz 6,06 przy statystyce
    rzędu dwunastu tysięcy mówi coś innego niż ten sam iloraz na małej próbie.
    """
    chi2 = float((pooled.resid_pearson**2).sum())
    df = float(pooled.df_resid)
    return chi2, df, chi2 / df


def mundlak_test(cre, bars: list[str]) -> tuple[float, float]:
    """Łączny test istotności średnich Mundlaka - odpowiednik testu Hausmana.

    Odrzucenie H0 oznacza, że efekt stały jest skorelowany z regresorami,
    czyli FE jest właściwsze niż RE.
    """
    w = cre.wald_test(", ".join(f"{b} = 0" for b in bars), use_f=False, scalar=True)
    return float(w.statistic), float(w.pvalue)


def irr_table(params, bse, names: list[str] = None) -> pd.DataFrame:
    """Współczynnik, IRR = exp(beta), błąd standardowy, statystyka z i p.

    Wartość p z rozkładu normalnego, bo przy klastrowej macierzy wariancji
    statystyka z jest asymptotycznie normalna.
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
    """Współczynnik i błąd standardowy, niezależnie od interfejsu estymatora.

    `pyfixest` udostępnia je metodami `coef()`/`se()`, statsmodels atrybutami.
    """
    if hasattr(res, "coef") and callable(res.coef):
        return float(res.coef().get(key, np.nan)), float(res.se().get(key, np.nan))
    return float(res.params.get(key, np.nan)), float(res.bse.get(key, np.nan))


# Kolejność w tabeli i na wykresie: od modelu najbardziej do najmniej naiwnego
# wobec heterogeniczności kraju.
COMPARISON_ORDER = [
    ("Pooled Poisson (klaster)", "pooled_cluster"),
    ("PA Poisson (GEE)", "pa"),
    ("Negative Binomial", "negbin"),
    ("FE-Poisson", "fe"),
    ("CRE / Mundlak", "cre"),
]


def compare_gdp_effect(fits: dict, key: str = "log_gdp_pc") -> pd.DataFrame:
    """Efekt PKB per capita w kolejnych estymatorach.

    Spadek współczynnika od modelu pooled do FE pokazuje, w jakiej części
    efekt zamożności jest efektem **między** krajami, a nie wewnątrz kraju
    w czasie.
    """
    rows = [
        (label, *_coef_se(fits[name], key)) for label, name in COMPARISON_ORDER
    ]
    comp = pd.DataFrame(rows, columns=["model", "coef_log_gdp_pc", "se"])
    comp["IRR"] = np.exp(comp["coef_log_gdp_pc"])
    comp["IRR_+10%PKB"] = comp["coef_log_gdp_pc"].apply(lambda b: np.exp(b * np.log(1.10)))
    return comp
