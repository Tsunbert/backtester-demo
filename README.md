# backtester-demo

**Prognozele mai bune ale volatilității duc la o strategie mai bună?** Acest repo prognozează volatilitatea SPY cu persistență naivă, HAR-RV, GARCH, GJR-GARCH și două variante de LSTM, pe cinci estimatori ai volatilității realizate. Apoi testează prognozele în [vectorbt](https://vectorbt.dev) ca strategie de vol-targeting (țintirea volatilității), alături de o strategie simplă long/short pe medii mobile 50/200.

Proiectul pornește de la [BogdanT54/Neural-networks-volatility-prediction](https://github.com/BogdanT54/Neural-networks-volatility-prediction), care compara prognoze LSTM / Seq2Seq / GARCH pe acțiuni NYSE, și adaugă pasul de la prognoză la profit și pierdere (P&L).

## Pornire rapidă

```bash
uv sync
uv run pytest -q -W ignore          # testele anti-look-ahead
uv run backtester-demo              # construiește grila de prognoze (~6 min prima dată) și afișează rezultatele
uv run jupyter lab notebooks/demo.ipynb
```

## Ce conține notebook-ul demo

1. **Vizualizări OHLC**: lumânări japoneze cu medii mobile și volum, perioadele de antrenare/validare/test colorate, distribuția randamentelor, gruparea (clustering-ul) volatilității.
2. **Spectrul volatilității**: close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang. Arată nivelul fiecărui estimator (cei doar intraday ratează saltul de peste noapte) și cât de zgomotos este.
3. **Grila de prognoze**: 6 modele × 5 estimatori × orizonturi de 1/5/20 zile.
4. **Evaluare pe setul de test ținut deoparte** (2023→azi): skill față de naiv, QLIKE, Mincer–Zarnowitz, Diebold–Mariano față de HAR și diagnostice pentru „copia lag-1” (corelația schimbărilor, profilul de lag).
5. **Backtest vectorbt**: vol-targeting la 12% anualizat, executat la deschiderea următoare, cu comisioane și slippage. Comparat cu buy & hold, cu long/short-ul pe medii mobile (simplu și scalat cu volatilitatea) și cu un ORACLE cu viziune perfectă, ca plafon.

## Cum evităm privirea în viitor (look-ahead)

| Componentă | Protecție |
|---|---|
| Împărțirea datelor | antrenare ≤ 2018, validare 2019–2022, test 2023→. Toate cifrele raportate sunt doar din test. |
| Ținte | ținta(t, h) = volatilitatea realizată pe t+1…t+h. Rândurile a căror țintă trece peste o graniță între perioade sunt eliminate (embargo). |
| HAR | reestimat la fiecare 21 de zile pe fereastră extinsă, doar cu ținte deja observate la data reestimării. |
| GARCH | reestimat la fiecare 21 de zile pe datele de până atunci, apoi parametri înghețați și recursie a varianței pas cu pas. Transpus pe estimatorii non-close-to-close printr-o calibrare estimată doar pe validare. |
| LSTM | scalarea, normalizarea țintei și ponderile sunt estimate pe antrenare; validarea servește doar la oprirea timpurie. |
| Backtest | ponderile decise la închiderea zilei *t* se execută la deschiderea zilei *t+1*. |

`tests/test_no_lookahead.py` verifică asta empiric. Pentru fiecare model, amestecă toate prețurile de după o dată-limită și verifică faptul că nicio prognoză făcută înainte de acea dată nu se schimbă.

### Cum evităm capcana „LSTM-ul învață lag-1”

Volatilitatea zilnică realizată este foarte persistentă și foarte zgomotoasă. Un LSTM antrenat cu MSE pe volatilitatea de mâine, care primește ca intrare volatilitatea de azi, poate ajunge să dea o copie netezită a zilei de azi. Copia arată convingător pe grafic, dar nu aduce nimic. Repo-ul tratează problema în trei feluri:

- `lstm_residual` prezice reziduul față de o ancoră HAR estimată pe antrenare. Dacă nu învață nimic, revine la HAR, nu la lag-1.
- `lstm_direct` (formularea naivă) este păstrat ca grup de control.
- Fiecare model este evaluat prin **skill față de naiv**, **corelația schimbărilor** (prezice *mișcarea* volatilității?) și un test **Diebold–Mariano** față de HAR.

## Rezultate (perioada de test 2023-01-03 → 2026-11-02, rulare din 2026-11-03)

**Prognoză:** skill mediu față de naiv la h=5, mediat peste estimatori (1 − MSE/MSE_naiv):

| lstm_direct | lstm_residual | gjr_garch | har | garch |
|---|---|---|---|---|
| 0.669 | 0.665 | 0.650 | 0.602 | 0.586 |

Ambele LSTM-uri bat HAR semnificativ la h=1 și h=5 (Diebold–Mariano). Ele prezic și *schimbarea* volatilității mai bine decât HAR, cu o corelație a schimbărilor de ~0,62 la h=1 pe Garman–Klass. Deci niciunul nu este o copie lag-1. Estimatorii bazați pe interval (range) sunt mult mai puțin zgomotoși decât close-to-close: autocorelația de lag 1 a log-volatilității zilnice este ~0,66 pentru ei, față de 0,14 pentru close-to-close.

**Backtest (țintă de volatilitate 12%, doar long, execuție la deschiderea următoare, costuri 5 bps):**

| strategie | Sharpe | vol. anualizată | drawdown maxim | vol-of-vol |
|---|---|---|---|---|
| buy & hold | 1.34 | 14.9% | −18.8% | 5.5% |
| vol-target, HAR / Garman–Klass | 1.36 | 11.9% | −13.2% | 1.7% |
| vol-target, LSTM-residual / Garman–Klass | 1.32 | 12.0% | −14.3% | 1.8% |
| vol-target, naiv / close-to-close | 1.01 | 12.6% | −19.4% | 3.1% |
| vol-target, ORACLE (viziune perfectă) | 2.11 | 11.2% | −9.5% | 1.2% |
| long/short pe medii mobile 50/200 | 0.58 | 14.9% | −27.0% | 5.4% |

Ce arată: vol-targeting păstrează Sharpe-ul strategiei buy & hold, dar reduce drawdown-ul cu ~30% și face volatilitatea realizată de ~3× mai stabilă. Alegerea unui **estimator bun** (bazat pe interval, nu close-to-close) contează mai mult pentru P&L decât alegerea unui **model** mai sofisticat. Avantajul LSTM-ului față de HAR la prognoză este real statistic, dar nu se transformă în Sharpe pe o singură traiectorie de 4 ani. Rândul ORACLE arată cât ar mai fi de câștigat cu un prognozator mult mai bun.
