# Testy

Testy sprawdzają **gwarancje metodologiczne**, bo na nich stoją wnioski projektu.
Nie weryfikują konkretnych oszacowań - to byłoby testowanie `statsmodels`.

Wszystkie działają na małej próbie syntetycznej, bez danych źródłowych, więc
wykonują się w sekundy i przechodzą w CI, gdzie 95 MB wyników olimpijskich nie
istnieje.

## Uruchomienie

```bash
pip install -r requirements-dev.txt
pytest
```

## Co jest sprawdzane

`test_pre_processing.py` - budowa panelu:

- **deduplikacja medali drużynowych** - drużyna jedenastu piłkarzy daje jedno
  złoto, nie jedenaście;
- **zera dla uczestników bez medalu** - panel powstaje z listy uczestnictw, nie
  z listy medali, bo model liczebności wymaga zer;
- **brak przecieku w opóźnieniu** - `medals_lag` liczone wewnątrz kraju, po
  uporządkowaniu po roku, nigdy z przyszłości ani z innego kraju;
- **jednoznaczność mapowania NOC na ISO3** - dwa kody nie mogą wskazywać tego
  samego kraju, bo scaliłyby dwie reprezentacje w jedną;
- **średnie Mundlaka w obrębie kraju**, nie na całej ramce.

`test_inference.py` - wnioskowanie na panelu syntetycznym (generator w
`conftest.py`):

- diagnostyka dyspersji reaguje na dyspersję i nie reaguje na jej brak;
- test Mundlaka wykrywa efekt kraju skorelowany z regresorami;
- tabele wynikowe nie zawierają liczb innych niż w samych oszacowaniach;
- po dodaniu efektów stałych kraju współczynnik przy PKB maleje - mechanizm,
  na którym stoi główny wniosek projektu.

## Weryfikacja mutacyjna

Test, który zawsze przechodzi, nie jest testem. Suita została sprawdzona przez
celowe uszkodzenie ośmiu gwarancji na kopii kodu: usunięcie deduplikacji,
budowanie panelu z medali zamiast z uczestnictw, opóźnienie bez podziału na
kraje, opóźnienie bez sortowania, opóźnienie z przyszłości, średnie Mundlaka
liczone globalnie, IRR bez eksponenty, łączenie z WDI po NOC zamiast po ISO3.
Każde uszkodzenie zapaliło test tej właśnie gwarancji.

Weryfikację warto powtórzyć po każdej zmianie w `src/pre_processing.py` lub
`src/inference.py`.

## Kompletność zależności testowych

`requirements-dev.txt` jest węższy od `requirements.txt`, więc trzeba sprawdzać,
czy nie brakuje w nim czegoś, co testy mimo wszystko wołają:

```bash
python tests/sprawdz_zaleznosci.py
```

Skrypt blokuje import bibliotek, których `pip install -r requirements-dev.txt`
nie zainstaluje, i uruchamia testy. Tak wyszło na jaw, że `pyfixest` jest
w testach potrzebny, bo `fit_all` szacuje FE-Poissona.

Blokada celowo **nie** obejmuje zależności przechodnich. `matplotlib` przychodzi
przez `pyfixest` i `seaborn`, a `lxml` przez `maketables` i `python-docx` -
blokowanie ich dawałoby fałszywe trafienia. Uwaga przy zmianie wersji
`pyfixest`: ładuje on `matplotlib.pyplot` leniwie, przy pierwszym wywołaniu
`fepois`, więc brak tej biblioteki ujawnia się dopiero na estymacji, a nie na
imporcie.
