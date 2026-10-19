"""Generator syntetycznego panelu o kontrolowanych wlasnosciach.

Panel sluzy testom wnioskowania: proces generujacy jest znany, wiec da sie
sprawdzic, czy diagnostyka wykrywa to, co ma wykrywac.

Trzy pokretla, kazde odpowiadajace jednej wlasnosci:

- `country_sd` - rozrzut efektu kraju. Zero oznacza, ze przekrojowy Poisson
  jest poprawnie wyspecyfikowany, wiec dyspersja musi wyjsc okolo jedynki.
  Wartosc dodatnia to nieuwzgledniona heterogenicznosc, czyli nadmierna
  dyspersja - dlatego panel z efektem kraju NIE jest panelem „czystym".
- `gamma_mixing` - domieszka gamma, czyli jawny rozklad ujemny dwumianowy.
- `gdp_effect_loading` - sila korelacji zamoznosci z efektem kraju. Dodatnia
  wartosc to sytuacja, w ktorej RE jest obciazony i test Mundlaka musi zapalic.
"""

import numpy as np
import pandas as pd
import pytest

SEED = 20261002

# Panel na tyle duzy, zeby estymatory zbiegaly, i na tyle maly, zeby testy
# wykonywaly sie w sekundach.
N_COUNTRIES = 60
N_EDITIONS = 10

# Wspolczynniki procesu generujacego - znane, wiec testy moga sie do nich odnosic.
TRUE_GDP = 0.5
TRUE_POP = 0.4
TRUE_HOST = 0.3
INTERCEPT = -6.0

# Udzial obserwacji z rola gospodarza. Losowy, nie wyznaczony z indeksu edycji:
# gospodarz powiazany deterministycznie z edycja bylby wspolliniowy z jej
# efektem stalym i FE-Poisson przestawalby sie szacowac.
HOST_SHARE = 0.08


def make_panel(
    country_sd: float = 0.0,
    gamma_mixing: bool = False,
    gdp_effect_loading: float = 0.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Syntetyczny panel kraj x igrzyska z `medals` o znanej sredniej warunkowej."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(N_COUNTRIES):
        effect = rng.normal(0.0, country_sd) if country_sd > 0 else 0.0
        gdp_level = 9.0 + gdp_effect_loading * effect + rng.normal(0.0, 0.6)
        pop_level = 16.0 + rng.normal(0.0, 1.2)
        for e in range(N_EDITIONS):
            gdp = gdp_level + rng.normal(0.0, 0.08)
            pop = pop_level + rng.normal(0.0, 0.05)
            host = int(rng.random() < HOST_SHARE)
            lin = INTERCEPT + TRUE_GDP * gdp + TRUE_POP * pop + TRUE_HOST * host + effect
            mu = float(np.exp(np.clip(lin, -6.0, 4.0)))
            if gamma_mixing:
                mu *= rng.gamma(1.0, 1.0)
            rows.append(
                {
                    "country_noc": f"C{c:02d}",
                    "edition_id": e,
                    "year": 1960 + 4 * e,
                    "medals": int(rng.poisson(mu)),
                    "log_gdp_pc": gdp,
                    "log_pop": pop,
                    "host": host,
                }
            )

    df = pd.DataFrame(rows).sort_values(["country_noc", "year"]).reset_index(drop=True)
    df["medals_lag"] = df.groupby("country_noc")["medals"].shift(1)
    df = df.dropna(subset=["medals_lag"]).reset_index(drop=True)
    df["country"] = df["country_noc"].astype("category")
    return df


@pytest.fixture(scope="session")
def panel_czysty():
    """Bez efektu kraju: przekrojowy Poisson poprawnie wyspecyfikowany."""
    return make_panel()


@pytest.fixture(scope="session")
def panel_nadmierna_dyspersja():
    """Domieszka gamma: jawna nadmierna dyspersja."""
    return make_panel(gamma_mixing=True)


@pytest.fixture(scope="session")
def panel_skorelowany_efekt():
    """Zamoznosc skorelowana z efektem kraju: RE obciazony, CRE konieczne."""
    return make_panel(country_sd=1.0, gdp_effect_loading=1.5)
