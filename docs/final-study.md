# Final study: does volatility timing with "safe" assets protect stock investors?

This page states the project's final research goal and what the evidence
shows. It ties the original ETF study to the paper we adapted:
Kaczmarek, Będowska-Sójka, Grobelny and Perez (2022), “False Safe Haven
Assets: Evidence From the Target Volatility Strategy Based on Recurrent
Neural Network” ([paper](https://doi.org/10.1016/j.ribaf.2021.101610)).

## The research goal

**Can a U.S. stock investor protect against market falls better than by
simply holding less stock?** We test the paper's answer: forecast stock
volatility, hold less stock when forecast risk is high, and in the most
stressed periods move some cash into a "safe haven" asset.

Three questions, each answered with rules fixed before its results were seen:

1. **Method.** Can we rebuild the paper's forecasting models and strategy on
   public data, and get similar numbers?
2. **Timing.** Once trading is realistic, does volatility timing beat a fixed
   SPY/cash mix holding the same average amount of stock?
3. **Safe havens.** Do Treasuries or gold add protection beyond cash, and does
   that protection last, including 2022–2025?

This follows directly from the original project. Phase 3 showed that an
eight-ETF mix did not clearly beat SPY plus cash, and the protection study
found no ETF reliably beat cash in falling months. The paper offers a smarter
alternative to fixed holdings; this study checks whether it changes that answer.

## What we did

| Step | Data | Rules | Where |
|---|---|---|---|
| Rebuild the paper (2010–2020) | Oxford-Man S&P 500 bipower variation; public price proxies for all 13 defensive assets | Paper's own idealized rules: zero fees, zero cash interest, full-test stress cutoffs | `outputs/paper-reconstruction-20261021/evaluation` |
| Fix the richer GRU | Same | Learns log volatility so forecasts are always positive; added after results were seen, labelled post hoc | `.../gru_univar_log`, `.../gru_multivar_log`, `.../evaluation_fixed` |
| Realistic test (2011–2020) | Same saved forecasts; WRDS/CRSP SPY, TLT, GLD, DBC total returns; French cash rate | Trade one day after each forecast, 0.1% fees, cash earns interest, cutoffs from past forecasts only, six pre-chosen tests | `outputs/paper-realistic-20261021` |
| 2022–2025 check | WRDS/CRSP ETF returns, 2007–2025 | Fixed holdings versus cash in months when SPY fell | `outputs/protection-long-final` |

The five forecasting models are the paper's: ARFIMA, GARCH, GJR-GARCH, a
one-input GRU and a richer GRU with gold, oil and yield inputs. Every model's
settings were chosen with pre-2010 data only. The strategy holds
`min(10% / predicted volatility, 100%)` in stocks and the rest in cash. When
predicted volatility is above its 90th percentile, up to 20% moves from cash
into the defensive asset.

## What we found

### 1. The rebuild works

| Model | Paper forecast error (MSE) | Ours | Paper Sharpe | Ours |
|---|---:|---:|---:|---:|
| ARFIMA | 0.0034 | 0.0035 | 1.00 | 1.01 |
| GARCH | 0.0127 | 0.0102 | 0.95 | 1.01 |
| GJR-GARCH | 0.0146 | 0.0105 | 0.88 | 0.98 |
| One-input GRU | 0.0043 | 0.0034 | 0.95 | 1.03 |
| Richer GRU | 0.0031 | 0.0040 | 0.97 | 0.98 |

ARFIMA, the paper's best portfolio model, matches almost exactly. The
one-input GRU ties ARFIMA on forecast error (not statistically different).
Our raw richer GRU did worse than the paper's because its forecasts went
negative on 5.5% of days, almost all in the very calm years 2017–2018.

**The fix.** Learning log volatility removes every negative forecast. Both
fixed GRUs keep the same settings menu, seeds and pre-2010 selection, and both
chose the same profile (two layers of 32 units).

| Model | Forecast error (MSE) | Percentage error (MAPE) | Negative forecasts | Sharpe |
|---|---:|---:|---:|---:|
| Richer GRU, original | 0.0040 | 36.0% | 5.5% of days | 0.98 |
| Richer GRU, fixed | 0.0035 | 28.6% | none | 1.01 |
| One-input GRU, fixed | 0.0033 | 29.2% | none | 1.01 |
| Paper's richer GRU | 0.0031 | 26.2% | not reported | 0.97 |

The fixed richer GRU ties ARFIMA on forecast error and has the lowest
percentage error of all seven models, so it is much closer to the paper.
It still does not beat the simpler models on forecast error as the paper's
did, so we cannot confirm that extra inputs help. Because the fix came after
we saw the original results, it is reported beside the original, not instead of it.

As in the paper, **a better forecast did not mean a better portfolio**: GARCH's
forecast error was three times ARFIMA's, yet their Sharpe ratios were almost equal.

### 2. Timing mainly changes *how* you lose, not *how much* you earn per unit of risk

Realistic test, 2011–2020, 0.1% fees, cash earning interest:

| Portfolio | Annual growth | Sharpe | Worst fall | COVID fall | 2015–16 fall |
|---|---:|---:|---:|---:|---:|
| SPY | 13.8% | 0.80 | −33.7% | −33.7% | −13.0% |
| ARFIMA timing | 7.4% | 0.76 | −14.5% | −9.7% | −11.6% |
| Fixed mix, same average stock (56%) | 8.0% | 0.80 | −19.6% | −19.6% | −7.3% |
| One-input GRU timing | 8.1% | 0.80 | −14.5% | −10.2% | −11.8% |
| Fixed mix, same average stock (59%) | 8.4% | 0.80 | −20.5% | −20.5% | −7.7% |

- **Sharpe: no gain.** Timing minus the fixed mix is −0.04 (ARFIMA) and −0.005
  (GRU); neither is close to significant.
- **Fees decide it.** Timing trades about 116 times its own value over ten
  years. Before fees it beat the fixed mix (ARFIMA Sharpe 0.89 against 0.80);
  at 0.1% they tie, and at 0.25% timing loses (0.57 against 0.80).
- **Better forecasts did not rescue it.** The fixed GRUs forecast best but
  traded even more (about 137 times their value); after 0.1% fees their Sharpe
  was 0.75, below their fixed mixes' 0.80.
- **Fast crashes versus slow falls.** Timing halved the COVID loss, because
  volatility jumped first. It lost more in the slow 2015–16 decline, because
  volatility rose too late to help.
- **It depended on the period.** Timing beat the fixed mix in 2016–2020
  (Sharpe 1.06 against 0.78) and lost in 2011–2015 (0.47 against 0.83).

### 3. Safe havens: Treasuries helped until 2021, then failed

**Paper-style rebuild (2010–2020, 13 assets):**
- Most "safe" assets made things worse: oil, silver, platinum, copper, cocoa,
  corn, wheat and the Swiss and German bonds lowered Sharpe for every model.
- **Gold** had the largest average gain (+0.015 Sharpe), helping with four of
  five models (+0.01 to +0.03; GARCH −0.01). It slightly worsened the COVID fall.
- **U.S. Treasuries** added a little. The 10-year proxy (IEF) was the only
  asset that helped with all five models (+0.01 to +0.02); the 30-year proxy
  (TLT) helped with four. Long Treasuries were the best crash protector:
  +1.4 points against cash in COVID and +1.5 in 2011.

**Realistic test (2011–2020):** adding TLT or GLD in stress raised Sharpe by
about 0.02–0.04, and TLT cut the COVID loss from 9.7% to 7.6%. None of the
four safe-haven tests survived the multiple-testing adjustment: the effects
are too small to separate from luck over ten years.

**2022–2025: average return minus cash in months when SPY fell (fixed holdings):**

| Period | Falling months | TLT | GLD |
|---|---:|---:|---:|
| 2010–2014 | 19 | +3.64 pp | −0.18 pp |
| 2015–2019 | 15 | +2.10 pp | +2.26 pp |
| 2020–2021 | 8 | +1.71 pp | −1.10 pp |
| 2022–2025 | 17 | **−4.31 pp** | +0.67 pp |

The paper's one true safe haven, long Treasuries, came from a period when
bonds reliably rose as stocks fell. In 2022–2025 stocks and bonds fell
together, and Treasuries did worse than cash in falling months.

## The answer

**Volatility timing is a way to avoid big, fast crashes, not a way to earn
more per unit of risk.** After realistic fees it matched a simple fixed
SPY/cash mix on Sharpe, with much smaller worst falls but lower growth. And
**no single asset is a reliable safe haven**: Treasuries protected through
2021 and then failed, while gold helped over time but not in every crash.
This supports the paper's warning about false safe havens, and extends it:
even the paper's surviving safe haven depended on the period.

## How sure are we

- **Small effects.** The safe-haven gains are a few hundredths of Sharpe, and
  no primary test is significant after the Holm adjustment.
- **History already seen.** 2010–2020 was examined in earlier runs, so this is
  a careful historical check, not a test on untouched data.
- **Substitute data.** Public prices replace the paper's Datastream series;
  commodity futures prices are not investable returns. The rebuild uses
  price-only bond ETFs, close to the paper's clean prices; the realistic test
  uses total returns, which include coupons.
- **Idealized versus realistic.** The paper-style rebuild uses its zero-fee,
  same-close, full-test-cutoff rules. Only the realistic test describes
  something an investor could have done.
- **The richer GRU fix was added after we saw the results.** It is reported
  separately and never replaces the original run.
- **Not covered.** The paper's international markets, and forecasts after
  2020 (the volatility archive ends in February 2022).
