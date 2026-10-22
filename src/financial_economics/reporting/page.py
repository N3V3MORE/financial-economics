"""Write one HTML page that shows the saved results of all four phases."""

from html import escape
import json
from pathlib import Path

import pandas as pd

from financial_economics.datasets.public import price_source_description

NAMES = {"SPY": "SPY", "Equal weight": "Equal-weight mix", "SPY + cash": "SPY + cash"}
CLASSES = {"SPY": "spy", "Equal weight": "mix", "SPY + cash": "cash"}
WIDTH = 720


def _read(folder: Path, name: str, **options) -> pd.DataFrame:
    return pd.read_csv(folder / name, **options)


def _load(outputs: Path) -> dict:
    """Read the saved phase outputs and check that they came from the same snapshot."""
    folders = [outputs / f"phase{phase}" for phase in [1, 2, 3, 4]]
    for phase, folder in enumerate(folders, start=1):
        if not (folder / "run.json").is_file():
            raise FileNotFoundError(f"Run Phase {phase} first; {folder / 'run.json'} is missing.")
    runs = [json.loads((folder / "run.json").read_text(encoding="utf-8")) for folder in folders]
    if any(run["snapshot_files"] != runs[0]["snapshot_files"] or run["study"] != runs[0]["study"]
           for run in runs):
        raise ValueError("The four phase outputs come from different snapshots or settings; rerun them.")
    phase1, phase2, phase3, phase4 = folders
    # Phase 4 must explain the same comparison Phase 3 made: same test period and same split.
    calibration = json.loads((phase3 / "calibration.json").read_text(encoding="utf-8"))
    if runs[3]["calibration"] != calibration:
        raise ValueError("Phases 3 and 4 used different test periods or SPY + cash splits; rerun them.")
    return {
        "study": runs[0]["study"], "files": runs[0]["snapshot_files"],
        "sources": runs[0].get("sources", {}),
        "seed": runs[1]["seed_date"], "threshold": runs[1]["drawdown_threshold"],
        "exposure": _read(phase3, "stock_exposure.csv"),
        "returns1": _read(phase1, "daily_returns.csv", index_col=0, parse_dates=True),
        "performance1": _read(phase1, "performance.csv", index_col=0),
        "episodes2": _read(phase2, "episodes.csv"),
        "comparison2": _read(phase2, "episode_comparison.csv"),
        "calibration": calibration,
        "returns3": _read(phase3, "evaluation_returns.csv", index_col=0, parse_dates=True),
        "performance3": _read(phase3, "performance.csv", index_col=0),
        "comparison3": _read(phase3, "episode_comparison.csv"),
        "uncertainty": _read(phase4, "uncertainty.csv"),
        "contributions": _read(phase4, "contributions.csv"),
        "correlation": _read(phase4, "bond_correlation.csv", index_col=0).correlation,
        "settings": runs[3]["settings"],
    }


def _keys(names: list[str]) -> str:
    keys = "".join(f'<span><span class="key {CLASSES[name]}"></span>{NAMES[name]}</span>' for name in names)
    return f'<p class="key-row">{keys}</p>'


def _span(start: str, end: str) -> str:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.year == end.year:
        return f"{start:%b}–{end:%b %Y}"
    return f"{start:%b %Y}–{end:%b %Y}"


def _money(value: float, thousands: bool = False) -> str:
    return f"${value / 1000:g}k" if thousands and value else f"${value:,.0f}"


def _growth_chart(returns: pd.DataFrame, seed: str, label: str) -> str:
    """Growth of $1,000, sampled weekly, with direct end labels and a hover readout."""
    wealth = pd.concat([pd.DataFrame(1000.0, index=[pd.Timestamp(seed)], columns=returns.columns),
                        1000 * (1 + returns).cumprod()])
    weekly = ~wealth.index.to_period("W").duplicated(keep="last")
    weekly[0] = True
    wealth = wealth.loc[weekly]
    left, right, top, bottom, height = 52, WIDTH - 104, 12, 262, 290
    days = (wealth.index - wealth.index[0]).days.to_numpy()
    top_value = wealth.to_numpy().max()
    step = 1000 if top_value < 4000 else 2500
    ymax = step * (int(top_value // step) + 1)

    def x(day):
        return left + (right - left) * day / days[-1]

    def y(value):
        return bottom - (bottom - top) * value / ymax

    parts = []
    for value in range(0, ymax + 1, step):
        parts.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{y(value):.1f}" y2="{y(value):.1f}"/>'
                     f'<text class="tick" x="{left - 8}" y="{y(value) + 4:.1f}" text-anchor="end">'
                     f'{_money(value, ymax >= 10_000)}</text>')
    for year in range(wealth.index[0].year + 1, wealth.index[-1].year + 1):
        if year % (5 if days[-1] > 3000 else 2) == 0:
            day = (pd.Timestamp(f"{year}-01-01") - wealth.index[0]).days
            parts.append(f'<text class="tick" x="{x(day):.1f}" y="{bottom + 20}" text-anchor="middle">{year}</text>')
    ends = []
    for name in wealth:
        points = " ".join(f"{x(day):.1f},{y(value):.1f}" for day, value in zip(days, wealth[name], strict=True))
        parts.append(f'<polyline class="line {CLASSES[name]}" points="{points}"/>')
        ends.append([y(wealth[name].iloc[-1]), name])
    # Spread the end labels so they never overlap.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 30)
    for label_y, name in ends:
        parts.append(f'<text class="end {CLASSES[name]}-text" x="{right + 8}" y="{label_y - 2:.1f}">{NAMES[name]}</text>'
                     f'<text class="end-value" x="{right + 8}" y="{label_y + 13:.1f}">'
                     f'{_money(wealth[name].iloc[-1])}</text>')
    data = {"left": left, "right": right, "top": top, "bottom": bottom,
            "x": [round(x(day), 1) for day in days],
            "dates": [date.strftime("%d %b %Y").lstrip("0") for date in wealth.index],
            "series": [{"name": NAMES[name], "class": CLASSES[name],
                        "y": [round(y(value), 1) for value in wealth[name]],
                        "values": [round(value) for value in wealth[name]]} for name in wealth]}
    return (f'<figure class="chart growth" data-chart=\'{json.dumps(data)}\'>'
            f'<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">'
            f'{"".join(parts)}<line class="cursor" y1="{top}" y2="{bottom}" x1="0" x2="0"/></svg>'
            '<div class="readout" hidden></div></figure>')


def _axis(bottom: int, ticks: list, scale) -> str:
    parts = []
    for value, label in ticks:
        parts.append(f'<line class="grid" x1="{scale(value):.1f}" x2="{scale(value):.1f}" y1="0" y2="{bottom}"/>'
                     f'<text class="tick" x="{scale(value):.1f}" y="{bottom + 18}" text-anchor="middle">{label}</text>')
    return "".join(parts)


def _loss_chart(episodes: pd.DataFrame, comparison: pd.DataFrame, names: list[str], label: str) -> str:
    """One row per SPY fall: each portfolio's loss over the same peak-to-trough dates."""
    left, right, row = 104, WIDTH - 20, 30
    bottom = row * len(episodes) + 6
    xmax = 40

    def x(loss):
        return left + (right - left) * loss / xmax

    parts = [_axis(bottom, [(v, f"{v}%") for v in range(0, xmax + 1, 10)], x)]
    losses = comparison.pivot(index="episode", columns="asset", values="peak_to_trough_return") * -100
    for i, episode in enumerate(episodes.itertuples()):
        cy = row * i + row / 2
        peak = pd.Timestamp(episode.peak_date)
        values = [losses.loc[episode.episode, name] for name in names]
        parts.append(f'<text class="row-label" x="0" y="{cy + 4:.1f}">{peak:%b %Y}</text>'
                     f'<line class="link" x1="{x(min(values)):.1f}" x2="{x(max(values)):.1f}" y1="{cy}" y2="{cy}"/>')
        for name, value in zip(names, values, strict=True):
            parts.append(f'<circle class="dot {CLASSES[name]}" cx="{x(value):.1f}" cy="{cy}" r="5">'
                         f'<title>{NAMES[name]}: lost {value:.1f}% from the {peak:%b %Y} SPY peak</title></circle>')
    return (f'<figure class="chart"><svg viewBox="0 -4 {WIDTH} {bottom + 26}" role="img" '
            f'aria-label="{escape(label)}">{"".join(parts)}</svg></figure>')


def _range_chart(rows: pd.DataFrame, title: str, limit: float, scale: float,
                 unit: str) -> str:
    """Observed difference (dot) and its 95% range (bar), with zero marked."""
    left, right, row, width = 150, 340, 40, 360
    bottom = row * len(rows) + 4

    def x(value):
        return left + (right - left) * (scale * value + limit) / (2 * limit)

    ticks = [(-limit / scale, f"{-limit:g}"), (0, "0"), (limit / scale, f"+{limit:g}")]
    parts = [_axis(bottom, ticks, x),
             f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="0" y2="{bottom}"/>']
    for i, item in enumerate(rows.itertuples()):
        cy = row * i + row / 2
        text = (f"Mix minus {item.reference}: {scale * item.estimate:+.3g}{unit}, "
                f"95% range {scale * item.ci_low:+.3g} to {scale * item.ci_high:+.3g}{unit}")
        parts.append(f'<text class="row-label" x="0" y="{cy + 4:.1f}">vs {item.reference}</text>'
                     f'<line class="range" x1="{x(item.ci_low):.1f}" x2="{x(item.ci_high):.1f}" y1="{cy}" y2="{cy}">'
                     f'<title>{text}</title></line>'
                     f'<circle class="dot mix" cx="{x(item.estimate):.1f}" cy="{cy}" r="5"><title>{text}</title></circle>')
    return (f'<figure class="chart small"><figcaption>{title}</figcaption>'
            f'<svg viewBox="0 -4 {width} {bottom + 26}" role="img" aria-label="{escape(title)}">'
            f'{"".join(parts)}</svg></figure>')


def _bar_chart(values: pd.Series, title: str, limit: float, scale: float, spec: str,
               highlight: set = frozenset()) -> str:
    """Horizontal bars from zero: left for negative values, right for positive ones."""
    left, right, row, width = 52, 340, 22, 360
    bottom = row * len(values) + 4
    zero = (left + right) / 2

    def x(value):
        return zero + (right - left) / 2 * scale * value / limit

    parts = [f'<line class="zero" x1="{zero}" x2="{zero}" y1="0" y2="{bottom}"/>']
    for i, (name, value) in enumerate(values.items()):
        top = row * i + 3
        start, end = sorted([zero, x(value)])
        shown = format(scale * value, spec)
        kind = "bar strong" if name in highlight else "bar"
        anchor, label_x = ("start", end + 5) if value >= 0 else ("end", start - 5)
        parts.append(f'<text class="row-label" x="0" y="{top + 12}">{name}</text>'
                     f'<rect class="{kind}" x="{start:.1f}" y="{top}" width="{max(end - start, 1):.1f}" '
                     f'height="{row - 7}" rx="2"><title>{name}: {shown}</title></rect>'
                     f'<text class="bar-value" x="{label_x:.1f}" y="{top + 12}" text-anchor="{anchor}">{shown}</text>')
    return (f'<figure class="chart small"><figcaption>{title}</figcaption>'
            f'<svg viewBox="0 -4 {width} {bottom + 8}" role="img" aria-label="{escape(title)}">'
            f'{"".join(parts)}</svg></figure>')


def _table(header: list[str], rows: list[list[str]], text_columns: int = 1) -> str:
    """A table whose first text_columns hold words; the rest hold numbers."""
    def cells(row, tag):
        return "".join(f'<{tag} class="text">{cell}</{tag}>' if i < text_columns else f"<{tag}>{cell}</{tag}>"
                       for i, cell in enumerate(row))
    body = "".join(f"<tr>{cells(row, 'td')}</tr>" for row in rows)
    return f"<table><thead><tr>{cells(header, 'th')}</tr></thead><tbody>{body}</tbody></table>"


def _metrics_table(performance: pd.DataFrame, names: list[str]) -> str:
    rows = []
    for label, column, spec in [("$1,000 grows to", "cumulative_return", "money"),
                                ("Yearly growth (CAGR)", "cagr", ".2%"),
                                ("Yearly fluctuation", "annualized_volatility", ".2%"),
                                ("Sharpe ratio", "sharpe", ".3f"),
                                ("Worst decline", "max_drawdown", ".2%")]:
        cells = []
        for name in names:
            value = performance.loc[name, column]
            cells.append(f"${1000 * (1 + value):,.0f}" if spec == "money" else format(value, spec))
        rows.append([label, *cells])
    return _table(["", *[f'<span class="key {CLASSES[n]}"></span>{NAMES[n]}' for n in names]], rows)


def _write_html(data: dict) -> str:
    study, calibration = data["study"], data["calibration"]
    perf1, perf3 = data["performance1"], data["performance3"]
    three = ["SPY", "Equal weight", "SPY + cash"]
    losses2 = data["comparison2"].pivot(index="episode", columns="asset", values="peak_to_trough_return")
    fewer = int((losses2["Equal weight"] > losses2["SPY"]).sum())
    uncertainty = data["uncertainty"]
    sharpe = uncertainty.loc[uncertainty.measure.eq("sharpe") & uncertainty.reference.eq("SPY + cash")].iloc[0]
    contributions = data["contributions"]
    windows = [window for window in dict.fromkeys(contributions.window) if window.endswith("fall")]
    falls = {window: contributions.loc[contributions.window.eq(window)].set_index("asset").contribution
             for window in windows}
    deepest = sorted(sorted(windows, key=lambda window: falls[window].sum())[:2], key=windows.index)
    bond = data["settings"]["bond_fund"]
    highlight = {bond, *[falls[window].idxmin() for window in deepest],
                 *[falls[window].idxmax() for window in deepest]}
    bond_text = " and ".join(f"{100 * falls[window][bond]:+.1f} points in the {window}" for window in deepest)

    # Headlines follow the data, so a rerun on other data cannot keep a stale claim.
    mix1, spy1 = perf1.loc["Equal weight"], perf1.loc["SPY"]
    smoother = mix1.annualized_volatility < spy1.annualized_volatility and mix1.cagr < spy1.cagr
    mix3, cash3 = perf3.loc["Equal weight"], perf3.loc["SPY + cash"]
    tradeoff = (mix3.cagr > cash3.cagr and mix3.annualized_volatility > cash3.annualized_volatility
                and mix3.max_drawdown < cash3.max_drawdown and mix3.sharpe < cash3.sharpe)
    chance = sharpe.ci_low < 0 < sharpe.ci_high
    first, last = calibration["evaluation_start"][:4], calibration["evaluation_end"][:4]
    test_years = first if first == last else f"{first}–{last}"
    fund_count = len(study["assets"])
    block = data["settings"]["mean_block_length"]
    beta = data["exposure"].set_index(["period", "asset"]).beta
    mix_beta, cash_beta = beta[("evaluation", "Equal weight")], beta[("evaluation", "SPY + cash")]
    first_hash = next(iter(data["files"].values()))["sha256"][:12]

    episode_rows = []
    for episode in data["episodes2"].itertuples():
        rows = data["comparison2"].loc[data["comparison2"].episode.eq(episode.episode)].set_index("asset")
        days = [rows.loc[name, "recovery_trading_days"] for name in ["SPY", "Equal weight"]]
        episode_rows.append([_span(episode.peak_date, episode.trough_date),
                             f"{rows.loc['SPY', 'peak_to_trough_return']:.1%}",
                             f"{rows.loc['Equal weight', 'peak_to_trough_return']:.1%}",
                             *["not yet" if pd.isna(day) else f"{int(day)}" for day in days]])
    range_rows = [[f"Mix minus {row.reference}", label,
                   f"{scale * row.estimate:+.{digits}f}", f"{scale * row.ci_low:+.{digits}f} to {scale * row.ci_high:+.{digits}f}",
                   f"{row.share_favoring_portfolio:.0%}"]
                  for measure, label, scale, digits in [("sharpe", "Sharpe ratio", 1, 3),
                                                        ("cagr", "Yearly growth (points)", 100, 2),
                                                        ("annualized_volatility", "Yearly fluctuation (points)", 100, 2)]
                  for row in uncertainty.loc[uncertainty.measure.eq(measure)].itertuples()]
    fund_rows = [[fund, *[f"{100 * falls[window][fund]:+.1f}" for window in windows]]
                 for fund in falls[windows[0]].index] if windows else []
    fund_rows.append(["Mix", *[f"{100 * falls[window].sum():+.1f}" for window in windows]])
    whole_name = next(window for window in dict.fromkeys(contributions.window) if window not in windows)
    whole = contributions.loc[contributions.window.eq(whole_name)].set_index("asset").contribution
    correlation = data["correlation"]
    correlation.index = correlation.index.astype(str)

    sections = [
        f"""<section>
<h2>{'A smoother ride than SPY, with slower growth' if smoother else 'How the mix compared with SPY'}</h2>
<p>Over {study['start'][:4]}–{study['end'][:4]} the mix grew {perf1.loc['Equal weight', 'cagr']:.2%} a year against
{perf1.loc['SPY', 'cagr']:.2%} for SPY. Its returns varied {'less' if mix1.annualized_volatility < spy1.annualized_volatility else 'more'}: a yearly fluctuation of
{perf1.loc['Equal weight', 'annualized_volatility']:.2%} against {perf1.loc['SPY', 'annualized_volatility']:.2%}.</p>
{_growth_chart(data['returns1'][['SPY', 'Equal weight']], data['seed'], f"Growth of $1,000, {study['start'][:4]} to {study['end'][:4]}")}
{_metrics_table(perf1, ['SPY', 'Equal weight'])}
</section>""",
        f"""<section>
<h2>{'The mix lost less in every market fall' if fewer == len(losses2) else f'The mix lost less in {fewer} of {len(losses2)} market falls'}</h2>
<p>Each row is a time SPY fell at least {data['threshold']:.0%} from a peak. Both portfolios are measured from the same SPY peak
to the same SPY trough. The mix lost less in {fewer} of {len(losses2)} falls.</p>
{_keys(['SPY', 'Equal weight'])}
{_loss_chart(data['episodes2'], data['comparison2'], ['SPY', 'Equal weight'], 'Loss in each SPY fall')}
{_table(['SPY peak to trough', 'SPY loss', 'Mix loss', 'SPY days to recover', 'Mix days to recover'], episode_rows)}
<p class="note">Days to recover are trading days from the trough until each portfolio regained its value at the SPY peak.</p>
</section>""",
        f"""<section>
<h2>{'Holding less SPY did about as well' if tradeoff else 'Compared with holding less SPY'}</h2>
<p>A smaller loss than SPY could just mean the mix held less stock. So we built a simpler portfolio with the
same stock sensitivity: using {calibration['training_start'][:4]}–{calibration['training_end'][:4]} only, the mix moved about {calibration['raw_beta']:.2f}% for each 1%
move in SPY, which sets {calibration['stock_weight']:.2%} SPY and {calibration['cash_weight']:.2%} cash.
That split was then frozen and tested on {test_years}.</p>
{_growth_chart(data['returns3'], calibration['evaluation_seed_date'], f"Growth of $1,000, {calibration['evaluation_start'][:4]} to {calibration['evaluation_end'][:4]}")}
{_metrics_table(perf3, three)}
{'<p>The mix grew a little faster than SPY + cash, but it also fluctuated more, fell further at its worst, and had a slightly lower Sharpe ratio.</p>' if tradeoff else ''}
<p>The match was only approximate. Over {test_years} the mix moved {mix_beta:.2f}% for each 1% move in SPY,
against {cash_beta:.2f}% for SPY + cash. Part of any difference
comes from that gap in stock exposure, not only from diversification.</p>
</section>""",
        f"""<section>
<h2>{'Chance alone could explain the difference' if chance else 'The Sharpe gap is larger than these rebuilds explain'}</h2>
<p>The {test_years} test period is one history. We rebuilt it {data['settings']['resamples']:,} times from runs of real trading
days, {block:g} days long on average, keeping every portfolio on the same days, and recalculated the comparison each time. The bars show
the middle 95% of the results. The Sharpe gap against SPY + cash was {sharpe.estimate:+.3f}, and the mix came
out ahead in {sharpe.share_favoring_portfolio:.0%} of the rebuilt histories.</p>
<div class="pair">
{_range_chart(uncertainty.loc[uncertainty.measure.eq('sharpe')], 'Sharpe ratio, mix minus each', .3, 1, '')}
{_range_chart(uncertainty.loc[uncertainty.measure.eq('cagr')], 'Yearly growth, mix minus each (points)', 8, 100, '')}
</div>
{_table(['Comparison', 'Measure', 'Observed', '95% range', 'Mix better in'], range_rows, 2)}
<p class="note">For fluctuation, lower counts as better. These ranges reuse the same days, so they show how much
the result could move by chance; they are not a formal significance test. A range that includes zero means these
years cannot tell the portfolios apart; it does not show they are equal. The rebuilds reshuffle the finished portfolio
returns with the {calibration['stock_weight']:.2%} SPY split held fixed. They do not redo the choice of funds or the
fitting of that split.</p>
</section>""",
        f"""<section>
<h2>Which funds drove the deepest falls</h2>
<p>Each bar is one fund's gain or loss, per $1 in the mix at the SPY peak, through the trough. The bars add up
to the mix's loss. The bond fund ({bond}) contributed {bond_text}. Dark bars mark the bond fund and the biggest
gain and loss in each fall.</p>
<div class="pair">
{''.join(_bar_chart(falls[window], f'{window} · mix {100 * falls[window].sum():+.1f} points', 8, 100, '+.1f', highlight) for window in deepest)}
</div>
{_table(['Fund', *windows], fund_rows)}
<p>A negative correlation means the bond fund tended to rise on days stocks fell. The dark bars mark years when
it was positive: {', '.join(year for year, value in correlation.items() if value > 0) or 'none'}.</p>
<div class="pair">
{_bar_chart(correlation, f'SPY/{bond} daily correlation by year', .8, 1, '+.2f', {year for year, value in correlation.items() if value > 0})}
{_bar_chart(whole, f"{whole_name}, whole period · mix {100 * whole.sum():+.0f} points", 40, 100, '+.1f')}
</div>
</section>""",
        f"""<section>
<h2>What this can't show</h2>
<p>These are {fund_count} funds picked with hindsight, and the full history had already been looked at before the
{test_years} test, so it is a historical check rather than a test on unseen data. Market falls are dated after the
fact. Trading costs and taxes are zero. Contributions show where gains and losses came from, not what a different
mix would have earned.</p>
</section>""",
    ]
    funds = ", ".join(study["assets"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETF diversification study</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<header>
<p class="kicker">ETF diversification study · {study['start'][:4]}–{study['end'][:4]}</p>
<h1>{'The mix reduced risk relative to SPY, but showed no clear risk-adjusted edge over SPY plus cash' if smoother and chance else f'Did {fund_count} ETFs spread risk better than holding less SPY?'}</h1>
<p class="lede">The mix holds {len(study['assets'])} funds ({funds}) in equal amounts, reset each month. This page
follows one question: do different ETFs help spread risk, especially when markets fall?</p>
{_keys(three)}
</header>
{''.join(sections)}
<footer>
<p>ETF source: {escape(price_source_description(data))}. Risk-free rate and cash: Kenneth French's daily factors.
Built from saved snapshot <code>{first_hash}</code> with <code>uv run fecon page</code>. Full tables and
definitions are in <code>outputs/phase1</code> to <code>outputs/phase4</code>.</p>
</footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""


def write_page(outputs: Path, destination: Path) -> None:
    """Read the four phase outputs and write a self-contained HTML page."""
    page = _write_html(_load(Path(outputs)))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")


STYLE = """
:root { color-scheme: light dark;
  --paper: #F5F1E9; --ink: #252925; --muted: #62665F; --rule: #D8D3C9; --bar: #C9C3B7;
  --spy: #B4502F; --mix: #1F63B0; --cash: #9A7B0A; }
@media (prefers-color-scheme: dark) { :root {
  --paper: #1B1D1A; --ink: #E9E5DC; --muted: #A8A69E; --rule: #3A3D38; --bar: #55584F;
  --spy: #D0623F; --mix: #4A8AD6; --cash: #B08E26; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 18px/1.6 "Iowan Old Style", Charter, Georgia, serif; }
main { max-width: 760px; margin: 0 auto; padding: 56px 20px 72px; }
h1 { font-size: 40px; line-height: 1.15; font-weight: 600; margin: 8px 0 20px; letter-spacing: -0.01em; }
h2 { font-size: 26px; line-height: 1.25; font-weight: 600; margin: 0 0 12px; }
section { border-top: 1px solid var(--rule); padding-top: 32px; margin-top: 40px; }
p { margin: 0 0 16px; max-width: 64ch; }
.kicker, .note, figcaption, table, footer, .key-row, svg text, .readout {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
.kicker { color: var(--muted); font-size: 15px; margin: 0; }
.lede { font-size: 20px; color: var(--ink); }
.note, footer { color: var(--muted); font-size: 14px; }
.key-row { display: flex; gap: 20px; font-size: 15px; color: var(--muted); }
.key { display: inline-block; width: 18px; height: 3px; border-radius: 2px; margin-right: 7px; vertical-align: middle; }
.key.spy { background: var(--spy); } .key.mix { background: var(--mix); } .key.cash { background: var(--cash); }
figure { margin: 20px 0; position: relative; }
figure svg { display: block; width: 100%; height: auto; overflow: visible; }
figcaption { font-size: 14px; color: var(--muted); margin-bottom: 6px; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 8px 32px; }
.grid { stroke: var(--rule); stroke-width: 1; }
.zero { stroke: var(--muted); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 12px; }
.row-label { fill: var(--ink); font-size: 13px; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.line.spy { stroke: var(--spy); } .line.mix { stroke: var(--mix); } .line.cash { stroke: var(--cash); }
.end { font-size: 12px; } .end-value { fill: var(--ink); font-size: 14px; font-weight: 600; }
.spy-text { fill: var(--spy); } .mix-text { fill: var(--mix); } .cash-text { fill: var(--cash); }
.dot { stroke: var(--paper); stroke-width: 2; }
.dot.spy { fill: var(--spy); } .dot.mix { fill: var(--mix); } .dot.cash { fill: var(--cash); }
.link { stroke: var(--rule); stroke-width: 2; }
.range { stroke: var(--mix); stroke-opacity: .35; stroke-width: 10; stroke-linecap: round; }
.bar { fill: var(--bar); } .bar.strong { fill: var(--ink); }
.bar-value { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.cursor { stroke: var(--muted); stroke-width: 1; opacity: 0; }
.readout { position: absolute; top: 0; pointer-events: none; background: var(--paper); border: 1px solid var(--rule);
  border-radius: 4px; padding: 6px 10px; font-size: 13px; line-height: 1.5; white-space: nowrap; }
table { width: 100%; border-collapse: collapse; font-size: 15px; margin: 8px 0 20px;
  font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--rule); text-align: right; }
th.text, td.text { text-align: left; }
th { font-weight: 600; color: var(--muted); font-size: 13px; }
code { font-size: 14px; }
footer { border-top: 1px solid var(--rule); margin-top: 48px; padding-top: 20px; }
@media (max-width: 560px) { h1 { font-size: 30px; } body { font-size: 17px; }
  .chart:not(.small) { overflow-x: auto; } .chart:not(.small) svg { min-width: 560px; }
  table { font-size: 13px; } th, td { padding-right: 6px; } }
"""

SCRIPT = """
for (const figure of document.querySelectorAll('figure.growth')) {
  const data = JSON.parse(figure.dataset.chart);
  const svg = figure.querySelector('svg'), cursor = svg.querySelector('.cursor');
  const readout = figure.querySelector('.readout');
  svg.addEventListener('pointermove', (event) => {
    const box = svg.getBoundingClientRect(), scale = box.width / svg.viewBox.baseVal.width;
    const x = (event.clientX - box.left) / scale;
    if (x < data.left || x > data.right) { readout.hidden = true; cursor.style.opacity = 0; return; }
    let i = data.x.findIndex((value) => value >= x);
    if (i < 0) i = data.x.length - 1;
    cursor.setAttribute('x1', data.x[i]); cursor.setAttribute('x2', data.x[i]); cursor.style.opacity = 1;
    readout.innerHTML = '<b>' + data.dates[i] + '</b><br>' + data.series.map((s) =>
      '<span class="key ' + s['class'] + '"></span>' + s.name + ' $' + s.values[i].toLocaleString()).join('<br>');
    readout.hidden = false;
    const left = data.x[i] * scale;
    readout.style.left = (left > box.width / 2 ? left - readout.offsetWidth - 12 : left + 12) + 'px';
  });
  svg.addEventListener('pointerleave', () => { readout.hidden = true; cursor.style.opacity = 0; });
}
"""
