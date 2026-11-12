# Roblox Up-and-Coming top-10 forecast

Entry for make.fun `roblox-up-and-coming-forecast-01`: a program that reads
the Up-and-Coming chart history and forecasts its top 10, place by place,
for every hour of the 72 hours after entries close. Scored by log score
against the hourly charts; a prize needs a score strictly above the house
baselines.

## Run

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` (`task.json` + `history.ndjson`, defaults to
`/input`) and writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (defaults to
`/output`). Python 3.13 standard library only, no network, deterministic:
two runs produce byte-identical output.

## Model

Each game's rank is modelled as a drift-free random walk. The per-hour rank
volatility is measured from the history itself (median absolute hourly rank
change, MAD-scaled, shrunk by 0.7), and each game's rank at hour `t` is
`Normal(current rank, sigma * sqrt(t))`. Per-place probabilities are the
marginals of that model averaged uniformly over the scored window — exactly
how the scorer uses the forecast.

Two findings from the practice history drove the design:

- **No drift.** Recent rank velocity barely persists (autocorrelation ~0.2)
  and player-count growth is uncorrelated with rank movement (r ~ 0.02), so
  a drift term only overfits noise.
- **A large unnamed reserve.** About 15% of the games that take a top-10
  place come from rank 41+ or from outside the chart, so each place keeps
  0.25 of its mass unnamed (shared by every game not named there) and named
  mass is normalised to 0.75.

## Backtest (practice window)

Scored against the 10 archive reads in the practice window
(2026-11-07T23:00Z to 2026-11-10T23:00Z), using the scorer as specified in
the competition rules (mean over reads of the sum over places of
`ln max(P, 0.0001)`):

| forecast | log score |
| --- | --- |
| persistence baseline (house starter) | -43.512 |
| this program | -33.301 |

Mean top-10 hits 6.4 of 10, same as the baseline — the gain comes from
honest, calibrated probabilities rather than from guessing a different order.
