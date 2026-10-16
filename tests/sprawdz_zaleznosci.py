"""Blokada importow pakietow NIEOBECNYCH w domknieciu requirements-dev.txt.

Blokowane sa tylko te biblioteki, ktorych `pip install -r requirements-dev.txt`
NIE zainstaluje - ani wprost, ani przechodnio. Blokowanie zaleznosci
przechodnich (matplotlib przez pyfixest -> seaborn, lxml przez maketables ->
python-docx) dawaloby falszywe trafienia.
"""
import importlib.abc, sys

ZABLOKOWANE = {'wbgapi', 'seleniumbase', 'nbclient', 'nbformat', 'ipykernel'}


class Blokada(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ZABLOKOWANE:
            raise ImportError(f"'{fullname}' nie jest w requirements-dev.txt")
        return None


sys.meta_path.insert(0, Blokada())
import pytest
sys.exit(pytest.main(['-q']))
