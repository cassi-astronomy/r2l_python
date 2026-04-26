# Jak správně získat soubor `analyza_jasu.py` (bez HTML)

Pokud Python hlásí chybu jako:

```
SyntaxError: invalid character '·' (U+00B7)
```

a v souboru je řádek s `<title>...`, znamená to, že byl omylem uložen **HTML obsah stránky z GitHubu**, ne skutečný `.py` soubor.

## Správný postup stažení

1. Otevři soubor na GitHubu.
2. Klikni na tlačítko **Raw**.
3. Teprve stránku `Raw` ulož jako `analyza_jasu.py`.

## Rychlá kontrola souboru

Na začátku správného souboru by měly být Python importy, například:

```python
import os
import json
import rawpy
```

Pokud první řádky začínají `<html`, `<!doctype`, nebo obsahují `<title>`, máš uložené HTML místo Pythonu.

## Doporučené soubory

Použij soubory z této složky:

- `fixed/analyza_jasu.py`
- `fixed/config.json`

Tyto soubory jsou připravené jako čisté textové zdrojáky bez diff značek.
