"""One captioned film following the diversification question through three phases."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from manim import (
    Arrow, Circle, DashedLine, DashedVMobject, Dot, FadeIn, FadeOut, FadeTransform, Group,
    Line, Rectangle, Scene, Text, ValueTracker, VGroup, VMobject, linear,
)

from financial_economics.presentation import time_prefix
from financial_economics.unified_presentation import load_story


ROOT = Path(__file__).resolve().parents[1]
PAPER, INK, MUTED, GRID = "#F5F1E9", "#252925", "#62665F", "#D8D3C9"
ASSETS = ["SPY", "Equal weight", "SPY + cash"]
NAMES = ["SPY", "8-fund mix", "SPY + cash"]
COLORS = ["#B45232", "#285E89", "#30766B"]
DASH = .16


def words(value, size=28, color=INK, headline=False):
    result = Text(str(value), font="Georgia" if headline else "Segoe UI",
                  font_size=size, color=color, disable_ligatures=True)
    if result.width > 12.2:
        result.scale_to_fit_width(12.2)
    return result


def left_at(mobject, x, y):
    return mobject.move_to([x + mobject.width / 2, y, 0])


class SharedTimeline(VGroup):
    """A shared calendar cursor reveals every portfolio through the same date."""

    left, right, bottom, top = -5.45, 4.45, -1.85, 1.45

    def __init__(self, dates, values, ylim, ticks, yticks):
        super().__init__()
        self.dates = pd.DatetimeIndex(dates)
        self.times = (self.dates - self.dates[0]).days.to_numpy(dtype=float)
        self.values, self.ylim = np.asarray(values), ylim
        axes = VGroup()
        for value, label in yticks:
            axes.add(Line(self.point(0, value), self.point(self.times[-1], value),
                          stroke_color=MUTED if value == 100 else GRID, stroke_width=.8))
            axes.add(words(label, 23, MUTED).move_to([-6.05, self.point(0, value)[1], 0]))
        for date, label in ticks:
            day = (pd.Timestamp(date) - self.dates[0]).days
            axes.add(words(label, 23, MUTED).move_to([self.point(day, 0)[0], -2.22, 0]))
        self.curves = VGroup(*(VMobject(stroke_color=color, stroke_width=3.3)
                               for color in COLORS[:self.values.shape[1]]))
        self.dots = VGroup(*(Dot(radius=.05, color=color)
                            for color in COLORS[:self.values.shape[1]]))
        self.cursor = Line([self.left, self.bottom, 0], [self.left, self.top, 0],
                           stroke_color=MUTED, stroke_width=.8, stroke_opacity=.5)
        self.add(axes, self.cursor, self.curves, self.dots)
        self.seek(0)

    def point(self, day, value):
        x = self.left + (self.right - self.left) * day / self.times[-1]
        y = self.bottom + (self.top - self.bottom) * (value - self.ylim[0]) / np.ptp(self.ylim)
        return np.array([x, y, 0.0])

    def seek(self, day):
        dates, values = time_prefix(self.times, self.values, day)
        for column, (curve, dot) in enumerate(zip(self.curves, self.dots)):
            points = np.array([self.point(x, y) for x, y in zip(dates, values[:, column])])
            path = VMobject(stroke_color=curve.get_stroke_color(), stroke_width=3.3)
            path.set_points_as_corners(np.repeat(points, 2, axis=0) if len(points) == 1 else points)
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            if column == 2 and length > DASH:
                # Dash the whole revealed path so every dash has the same length.
                path = DashedVMobject(path, num_dashes=int(length / DASH), dashed_ratio=.55,
                                      color=COLORS[2]).set_stroke(width=3.3)
            curve.become(path)
            dot.move_to(points[-1])
        x = self.point(dates[-1], 0)[0]
        self.cursor.put_start_and_end_on([x, self.bottom, 0], [x, self.top, 0])
        return self


class ProjectResults(Scene):
    def construct(self):
        self.camera.background_color = PAPER
        self.data = load_story(ROOT)
        expected = {"SPY", "QQQ", "VNQ", "GLD", "XLE", "XLY", "VIG", "TLT"}
        if (set(self.data.phase1.metadata["study"]["assets"]) != expected
                or self.data.phase1.benchmark != "SPY"
                or self.data.calibration["requested_evaluation_start"] != "2018-01-01"):
            raise ValueError("Review the film's story before changing the universe or evaluation split.")
        self.chapters, self.cues = [], []
        self.body = VGroup()
        self.title = left_at(words("What did the mix change?", 48, headline=True), -6.15, 3.05)
        self.subtitle = left_at(words("A diversification study in three steps", 27, MUTED), -6.15, 2.35)
        self.caption = words("Eight funds. Equal weights. Reset monthly.", 28).move_to([0, -3.05, 0])
        self.footer = words("Selected funds · Historical returns · Zero trading costs and taxes", 17, MUTED)
        self.footer.move_to([0, -3.72, 0])
        self.play(FadeIn(self.title), FadeIn(self.subtitle), FadeIn(self.caption),
                  FadeIn(self.footer), run_time=1)
        universe = words("SPY   QQQ   VNQ   GLD   XLE   XLY   VIG   TLT", 30, COLORS[1])
        self.body = VGroup(universe)
        self.play(FadeIn(self.body), run_time=.7)
        self.cue("opening")
        self.wait(3.5)

        self.baseline()
        self.historical_downturns()
        self.calibration()
        self.evaluation_growth()
        self.common_2020()
        self.evaluation_downturns()
        self.tradeoff()
        self.exposure()

        self.section("Extra growth came with extra risk.",
                     "The lesson from all three phases", "Lower losses versus SPY were only the first comparison.")
        conclusion = VGroup(
            left_at(words("The mix added growth versus SPY + cash.", 33), -5.55, .9),
            left_at(words("It also added risk.", 38, COLORS[1], headline=True), -5.55, -.05),
            left_at(words("No clear advantage over the simpler control.", 31), -5.55, -1.05),
        )
        self.body = conclusion
        self.play(FadeIn(conclusion), run_time=.8)
        self.caption_to("The full history was already examined. No significance test.")
        self.cue("conclusion")
        self.wait(7)
        self.safe(Group(*self.mobjects))
        output = ROOT / "outputs/unified"
        output.mkdir(parents=True, exist_ok=True)
        evidence = {**self.data.evidence, "chapters": self.chapters, "review_cues": self.cues,
                    "duration_seconds": round(self.time, 3),
                    "path_display": "Weekly growth endpoints plus the final day; daily 2020 detail and all numerical estimates.",
                    "common_episode": "2020-02-19 close through 2020-08-31 close; no extra rebalance.",
                    "limitation": "Already-inspected historical sample; zero costs and taxes; descriptive comparison, not a superiority test."}
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    def section(self, title, subtitle, caption):
        if len(self.body):
            self.play(FadeOut(self.body), run_time=.45)
            self.remove(*self.body.get_family())
            self.body = VGroup()
        self.chapters.append({"title": title, "start_seconds": round(self.time, 3)})
        replacements = [left_at(words(title, 47, headline=True), -6.15, 3.05),
                        left_at(words(subtitle, 27, MUTED), -6.15, 2.35),
                        words(caption, 28).move_to([0, -3.05, 0])]
        self.safe(Group(*replacements))
        self.play(*(FadeTransform(old, new) for old, new in zip(
            [self.title, self.subtitle, self.caption], replacements)), run_time=.7)
        # FadeTransform restores the old text afterwards; remove it so it cannot linger.
        self.remove(self.title, self.subtitle, self.caption)
        self.title, self.subtitle, self.caption = replacements

    def caption_to(self, value):
        replacement = words(value, 28).move_to([0, -3.05, 0])
        self.safe(replacement)
        self.play(FadeTransform(self.caption, replacement), run_time=.45)
        self.remove(self.caption)
        self.caption = replacement

    def cue(self, name):
        self.safe(Group(*self.mobjects))
        self.cues.append({"name": name, "seconds": round(self.time, 3)})

    def legend(self, count):
        group = VGroup()
        for i in range(count):
            x = -5.35 + 3.8 * i
            swatch = DashedLine if i == 2 else Line
            group.add(swatch([x, 1.93, 0], [x + .35, 1.93, 0],
                            stroke_color=COLORS[i], stroke_width=3.3),
                      left_at(words(NAMES[i], 25, COLORS[i]), x + .5, 1.93))
        return group

    def reveal(self, plot, end=None, seconds=6):
        start = getattr(plot, "last_day", 0)
        end = plot.times[-1] if end is None else end
        tracker = ValueTracker(start)
        plot.add_updater(lambda obj: obj.seek(tracker.get_value()))
        self.play(tracker.animate.set_value(end), run_time=seconds, rate_func=linear)
        plot.clear_updaters()
        plot.seek(end)
        plot.last_day = end
        self.remove(tracker)

    def labels(self, plot, day=None, losses=False):
        day = plot.times[-1] if day is None else day
        values = np.array([np.interp(day, plot.times, column) for column in plot.values.T])
        anchors = [plot.point(day, value) for value in values]
        order = np.argsort(values)
        ys = np.array([anchors[i][1] for i in order])
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + .82)
        ys -= max(0, ys[-1] - 1.25)
        ys += max(0, -1.5 - ys[0])
        result = VGroup()
        label_x = anchors[0][0] + .6 if losses else 4.8
        for rank, i in enumerate(order):
            value = f"{values[i] - 100:.1f}%" if losses else f"${values[i]:,.0f}"
            tag = VGroup(words(NAMES[i], 22, COLORS[i]), words(value, 29))
            tag.arrange(np.array([0, -1, 0]), buff=.08, aligned_edge=np.array([-1, 0, 0]))
            left_at(tag, label_x, ys[rank])
            result.add(Line(anchors[i], [label_x - .15, ys[rank], 0], stroke_color=COLORS[i],
                            stroke_width=1), tag)
        self.safe(result)
        return result

    def growth_plot(self, returns, seed, count):
        wealth = pd.concat([pd.DataFrame(1000., index=[pd.Timestamp(seed)], columns=returns.columns),
                            1000 * (1 + returns).cumprod()])
        selected = ~wealth.index.to_period("W").duplicated(keep="last")
        selected[0] = selected[-1] = True
        sampled = wealth.loc[selected]
        years = sorted(set([wealth.index[1].year, 2015, 2020, wealth.index[-1].year]))
        ticks = [(f"{year}-01-01", str(year)) for year in years
                 if wealth.index[0].year <= year <= wealth.index[-1].year]
        scale = 2000 if wealth.to_numpy().max() > 5000 else 1000
        ymax = np.ceil(wealth.to_numpy().max() / scale) * scale
        step = ymax / 4 if ymax > 5000 else 1000
        plot = SharedTimeline(sampled.index, sampled.to_numpy(), (0, ymax), ticks,
                              [(value, "$0" if value == 0 else f"${value / 1000:g}k")
                               for value in np.arange(0, ymax + .1, step)])
        self.body = VGroup(plot, self.legend(count))
        self.play(FadeIn(self.body), run_time=.7)
        return plot

    def baseline(self):
        self.section("A smoother ride, with slower growth.", "Phase 1 · 2010–2025 · $1,000 at the start",
                     "Same dates. Dividends included. No optimized weights.")
        seed = self.data.evidence["full_seed_date"]
        plot = self.growth_plot(self.data.phase1_returns[ASSETS[:2]], seed, 2)
        self.reveal(plot, seconds=7)
        labels = self.labels(plot)
        self.body.add(labels)
        self.play(FadeIn(labels), run_time=.5)
        vol = self.data.phase1.performance.loc[ASSETS[:2], "annualized_volatility"]
        self.caption_to(f"Yearly fluctuation: SPY {vol.iloc[0]:.1%} · mix {vol.iloc[1]:.1%}.")
        self.cue("baseline")
        self.wait(4)

    def downturn_marks(self, episodes, comparisons, count):
        group = self.legend(count)
        left, right = -3.7, 5.35
        for percentage in [0, 10, 20, 30, 40]:
            x = left + (right - left) * percentage / 40
            group.add(Line([x, -1.9, 0], [x, 1.4, 0], stroke_color=GRID, stroke_width=.7),
                      words(f"{percentage}%", 22, MUTED).move_to([x, -2.22, 0]))
        group.add(words("Loss from the same SPY peak to its trough", 23, MUTED).move_to([.2, -2.62, 0]))
        self.play(FadeIn(group), run_time=.6)
        for y, episode in zip(np.linspace(1.2, -1.65, len(episodes)), episodes.itertuples()):
            peak, trough = pd.Timestamp(episode.peak_date), pd.Timestamp(episode.trough_date)
            label = peak.strftime("%b %Y") if peak.year == trough.year else f"{peak.year}–{str(trough.year)[2:]}"
            rows = comparisons.loc[comparisons.episode.eq(episode.episode)].set_index("asset")
            points = [left + (right - left) * (-100 * rows.loc[asset, "peak_to_trough_return"]) / 40
                      for asset in ASSETS[:count]]
            row = VGroup(left_at(words(label, 22, MUTED), -6.1, y),
                         Line([min(points), y, 0], [max(points), y, 0], stroke_color=GRID, stroke_width=2))
            for i, x in enumerate(points):
                mark = (Circle(radius=.065, color=COLORS[i], stroke_width=2.3).move_to([x, y, 0])
                        if i == 2 else Dot([x, y, 0], radius=.065, color=COLORS[i]))
                row.add(mark)
            group.add(row)
            self.play(FadeIn(row), run_time=.35)
        self.body = group

    def historical_downturns(self):
        self.section("The mix lost less in market falls.",
                     "Phase 2 · 2010–2025 · every SPY decline reaching 10%",
                     "Each pair uses the same SPY peak and trough dates.")
        self.downturn_marks(self.data.phase2_episodes, self.data.phase2_comparison, 2)
        pivot = self.data.phase2_comparison.pivot(index="episode", columns="asset", values="peak_to_trough_return")
        count = int((pivot["Equal weight"] > pivot["SPY"]).sum())
        self.caption_to(f"The mix lost less than SPY in {count} of {len(pivot)} downturns.")
        self.cue("all_historical_downturns")
        self.wait(5)

    def calibration(self):
        self.section("Could holding less SPY do the same?", "Phase 3 · choose the allocation using earlier returns",
                     "Match the mix's earlier sensitivity to SPY, approximately.")
        group = VGroup()
        for x, label, action in [(-2.8, "2010–2017", "Choose the weights"),
                                  (2.8, "2018–2025", "Evaluate the result")]:
            group.add(words(label, 30).move_to([x, 1.45, 0]),
                      words(action, 25, MUTED).move_to([x, .85, 0]))
        group.add(Line([0, .55, 0], [0, 1.8, 0], stroke_color=GRID, stroke_width=1.5))
        self.body = group
        self.play(FadeIn(group), run_time=.8)
        self.wait(3)
        weight = self.data.calibration["stock_weight"]
        bar = VGroup()
        for start, width, color, label in [(-5.3, 10.6 * weight, COLORS[0], f"{weight:.1%} SPY"),
                                          (-5.3 + 10.6 * weight, 10.6 * (1 - weight), GRID,
                                           f"{1 - weight:.1%} cash")]:
            bar.add(Rectangle(width=width, height=.42, stroke_width=0,
                              fill_color=color, fill_opacity=1).move_to([start + width / 2, -.35, 0]),
                    words(label, 29, INK).move_to([start + width / 2, -.98, 0]))
        self.body.add(bar)
        self.play(FadeIn(bar), run_time=.8)
        self.caption_to("Cash earns French's daily risk-free return.")
        self.cue("allocation")
        self.wait(4)
        self.caption_to("Freeze the targets. Reset monthly. Let holdings drift between resets.")
        self.wait(4)

    def evaluation_growth(self):
        self.section("Now compare all three portfolios.", "Phase 3 · 2018–2025 · $1,000 at the same starting close",
                     "The SPY/cash allocation uses no later returns.")
        plot = self.growth_plot(self.data.phase3_returns, self.data.calibration["evaluation_seed_date"], 3)
        self.reveal(plot, seconds=7)
        labels = self.labels(plot)
        self.body.add(labels)
        self.play(FadeIn(labels), run_time=.5)
        self.caption_to("The mix finished a little above SPY + cash.")
        self.cue("evaluation_growth")
        self.wait(4)

    def common_2020(self):
        self.section("Look again at the 2020 fall.", "19 February–31 August 2020 · all start at $100",
                     "Existing holdings continue. Rebasing adds no rebalance.")
        returns = self.data.phase3_returns
        wealth = (1 + returns).cumprod()
        window = wealth.loc["2020-02-19":"2020-08-31"]
        window = 100 * window / window.iloc[0]
        plot = SharedTimeline(window.index, window.to_numpy(), (60, 110),
                              [("2020-02-19", "19 Feb"), ("2020-03-23", "23 Mar"),
                               ("2020-06-01", "1 Jun"), ("2020-08-31", "31 Aug")],
                              [(value, f"${value}") for value in [60, 80, 100]])
        self.body = VGroup(plot, self.legend(3))
        self.play(FadeIn(self.body), run_time=.7)
        trough = (pd.Timestamp("2020-03-23") - window.index[0]).days
        self.reveal(plot, trough, seconds=3)
        labels = self.labels(plot, trough, losses=True)
        self.body.add(labels)
        self.play(FadeIn(labels), run_time=.5)
        self.caption_to("SPY + cash lost less than the mix.")
        self.cue("2020_trough")
        self.wait(5)
        self.play(FadeOut(labels), run_time=.4)
        self.body.remove(labels)
        self.reveal(plot, seconds=3)
        self.caption_to("The mix recovered first; SPY + cash followed, then SPY.")
        self.cue("2020_recovery")
        self.wait(3)

    def evaluation_downturns(self):
        self.section("The simpler control also softened the falls.",
                     "2018–2025 · all five SPY declines reaching 10%",
                     "Compare every portfolio over identical peak-to-trough dates.")
        self.downturn_marks(self.data.phase3_episodes, self.data.phase3_comparison, 3)
        pivot = self.data.phase3_comparison.pivot(index="episode", columns="asset", values="peak_to_trough_return")
        count = int((pivot["SPY + cash"] > pivot["Equal weight"]).sum())
        self.caption_to(f"SPY + cash lost less than the mix in {count} of {len(pivot)} falls.")
        self.cue("evaluation_downturns")
        self.wait(4)
        recovery = self.data.phase3_comparison.pivot(index="episode", columns="asset", values="recovery_trading_days").dropna()
        faster = int((recovery["Equal weight"] < recovery["SPY + cash"]).sum())
        self.caption_to(f"The mix recovered sooner in {faster} of {len(recovery)}. The 2022 recovery was later.")
        self.wait(5)

    def tradeoff(self):
        self.section("More growth came with more risk.", "2018–2025 · the whole evaluation period",
                     "Yearly growth means compounded growth per year.")
        group = VGroup(words("8-fund mix", 29, COLORS[1]).move_to([-.1, 1.65, 0]),
                       words("SPY + cash", 29, COLORS[2]).move_to([3.8, 1.65, 0]))
        rows = [("Yearly growth", "cagr", ".2%"), ("Yearly fluctuation", "annualized_volatility", ".2%"),
                ("Worst decline", "max_drawdown", ".2%"), ("Sharpe ratio", "sharpe", ".3f")]
        self.body = group
        self.play(FadeIn(group), run_time=.6)
        for y, (label, metric, spec) in zip([.85, .05, -.75, -1.55], rows):
            row = VGroup(left_at(words(label, 27, MUTED), -5.7, y))
            for asset, x in [("Equal weight", -.1), ("SPY + cash", 3.8)]:
                value = self.data.phase3_performance.loc[asset, metric]
                row.add(words(format(value, spec), 34).move_to([x, y, 0]))
            row.add(Line([-5.7, y - .37, 0], [5.5, y - .37, 0], stroke_color=GRID, stroke_width=.7))
            group.add(row)
            self.play(FadeIn(row), run_time=.5)
        self.caption_to("Worst decline uses each portfolio's own earlier peak.")
        self.cue("evaluation_metrics")
        self.wait(4)
        self.caption_to("Sharpe compares excess return with its day-to-day variability.")
        self.cue("sharpe_explanation")
        self.wait(5)

    def exposure(self):
        self.section("The stock sensitivity did not stay matched.",
                     "Beta measures sensitivity to SPY's daily excess returns",
                     "The match was chosen before evaluation; later exposure could change.")
        group = VGroup(words("2010–2017", 27, MUTED).move_to([-1.4, 1.5, 0]),
                       words("2018–2025", 27, MUTED).move_to([3.6, 1.5, 0]))
        exposure = self.data.exposure.set_index(["period", "asset"])
        for asset, label, color, y in [("Equal weight", "8-fund mix", COLORS[1], .55),
                                        ("SPY + cash", "SPY + cash", COLORS[2], -.7)]:
            group.add(left_at(words(label, 28, color), -5.8, y),
                      words(f"{exposure.loc[('training', asset), 'beta']:.3f}", 40, color).move_to([-1.4, y, 0]),
                      Arrow([-.3, y, 0], [2.5, y, 0], color=GRID, stroke_width=2),
                      words(f"{exposure.loc[('evaluation', asset), 'beta']:.3f}", 40, color).move_to([3.6, y, 0]))
        self.body = group
        self.play(FadeIn(group), run_time=.8)
        self.cue("realized_exposure")
        self.wait(7)

    def safe(self, group):
        if (group.get_left()[0] < -6.9 or group.get_right()[0] > 6.9
                or group.get_top()[1] > 3.9 or group.get_bottom()[1] < -3.9):
            bad = [(type(obj).__name__, str(obj)[:70], obj.get_left()[0], obj.get_right()[0],
                    obj.get_bottom()[1], obj.get_top()[1]) for obj in self.mobjects
                   if obj.get_left()[0] < -6.9 or obj.get_right()[0] > 6.9
                   or obj.get_top()[1] > 3.9 or obj.get_bottom()[1] < -3.9]
            raise ValueError(f"A presentation element is outside the safe frame: {bad}")
