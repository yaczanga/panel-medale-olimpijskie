"""Generator syntetycznego panelu o kontrolowanych własnościach.

Panel służy testom wnioskowania: proces generujący jest znany, więc da się
sprawdzić, czy diagnostyka wykrywa to, co ma wykrywać.

Trzy pokrętła, każde odpowiadające jednej własności:

- `country_sd` - rozrzut efektu kraju. Zero oznacza, że przekrojowy Poisson
  jest poprawnie wyspecyfikowany, więc dyspersja musi wyjść około jedynki.
  Wartość dodatnia to nieuwzględniona heterogeniczność, czyli nadmierna
  dyspersja - dlatego panel z efektem kraju NIE jest panelem „czystym".
- `gamma_mixing` - domieszka gamma, czyli jawny rozkład ujemny dwumianowy.
- `gdp_effect_loading` - siła korelacji zamożności z efektem kraju. Dodatnia
  wartość to sytuacja, w której RE jest obciążony i test Mundlaka musi zapalić.
"""

import numpy as np
import pandas as pd
import pytest

SEED = 20261002

# Panel na tyle duży, żeby estymatory zbiegały, i na tyle mały, żeby testy
# wykonywały się w sekundach.
N_COUNTRIES = 60
N_EDITIONS = 10

# Współczynniki procesu generującego - znane, więc testy mogą się do nich odnosić.
TRUE_GDP = 0.5
TRUE_POP = 0.4
TRUE_HOST = 0.3
INTERCEPT = -6.0

# Udział obserwacji z rolą gospodarza. Losowy, nie wyznaczony z indeksu edycji:
# gospodarz powiązany deterministycznie z edycją byłby współliniowy z jej
# efektem stałym i FE-Poisson przestawałby się szacować.
HOST_SHARE = 0.08


def make_panel(
    country_sd: float = 0.0,
    gamma_mixing: bool = False,
    gdp_effect_loading: float = 0.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Syntetyczny panel kraj x igrzyska z `medals` o znanej średniej warunkowej."""
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
    """Zamożność skorelowana z efektem kraju: RE obciążony, CRE konieczne."""
    return make_panel(country_sd=1.0, gdp_effect_loading=1.5)
