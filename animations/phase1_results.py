"""A continuous, captioned comparison drawn from the saved daily results."""

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    Dot, FadeIn, FadeOut, FadeTransform, Group, LEFT, Line, Scene, Text, Transform,
    UP, ValueTracker, VGroup, VMobject, linear,
)

from financial_economics.presentation import load_presentation, rebased_window, time_prefix


ROOT = Path(__file__).resolve().parents[1]
PAPER = "#F5F1E9"
INK = "#252925"
MUTED = "#62665F"
GRID = "#D8D3C9"
COLORS = ["#B45232", "#285E89"]
NAMES = ["SPY", "8-fund mix"]
ASSETS = ["SPY", "Equal weight"]


def text(words, size=30, color=INK, headline=False):
    return Text(words, font="Georgia" if headline else "Segoe UI",
                font_size=size, color=color, disable_ligatures=True)


def left_at(mobject, x, y):
    return mobject.move_to([x + mobject.width / 2, y, 0])


class TimelinePlot(VGroup):
    """Two paths and one shared time cursor; all movement follows the x-axis."""

    left, right, bottom, top = -5.45, 4.55, -1.65, 1.55

    def __init__(self, times, values, ylim, xticks, yticks):
        super().__init__()
        self.times, self.values, self.ylim = times, values, ylim
        self.axes = self.make_axes(xticks, yticks)
        self.curves = VGroup(*(VMobject(stroke_color=c, stroke_width=3.5) for c in COLORS))
        self.dots = VGroup(*(Dot(radius=0.055, color=c) for c in COLORS))
        self.cursor = Line([0, self.bottom, 0], [0, self.top, 0],
                           stroke_color=MUTED, stroke_width=1, stroke_opacity=0.5)
        self.add(self.axes, self.cursor, self.curves, self.dots)
        self.seek(times[0])

    def point(self, when, value):
        x = self.left + (self.right - self.left) * (when - self.times[0]) / np.ptp(self.times)
        y = self.bottom + (self.top - self.bottom) * (value - self.ylim[0]) / np.ptp(self.ylim)
        return np.array([x, y, 0.0])

    def make_axes(self, xticks, yticks):
        axes = VGroup()
        for value, word in yticks:
            baseline = value == 100 and self.ylim[1] < 200
            axes.add(Line(self.point(self.times[0], value), self.point(self.times[-1], value),
                          stroke_color=MUTED if baseline else GRID,
                          stroke_width=1.4 if baseline else 0.8))
            tick = text(word, 23, MUTED)
            tick.move_to([self.left - 0.55, self.point(self.times[0], value)[1], 0])
            axes.add(tick)
        for when, word in xticks:
            axes.add(text(word, 23, MUTED).move_to([self.point(when, 0)[0], self.bottom - 0.37, 0]))
        return axes

    def seek(self, when):
        dates, values = time_prefix(self.times, self.values, when)
        for i, (curve, dot) in enumerate(zip(self.curves, self.dots)):
            points = np.array([self.point(day, value) for day, value in zip(dates, values[:, i])])
            if len(points) == 1:
                points = np.repeat(points, 2, axis=0)
            curve.set_points_as_corners(points)
            dot.move_to(points[-1])
        x = self.point(dates[-1], values[-1, 0])[0]
        self.cursor.put_start_and_end_on([x, self.bottom, 0], [x, self.top, 0])
        return self


class PhaseOneResults(Scene):
    """Full-period growth and a common-date 2020 example."""

    def construct(self):
        self.camera.background_color = PAPER
        self.data = load_presentation(Path(os.environ.get("FECON_RESULTS_DIR", ROOT / "outputs/phase1")))
        if (set(self.data.metadata["study"]["assets"])
                != {"SPY", "QQQ", "VNQ", "GLD", "XLE", "XLY", "VIG", "TLT"}
                or self.data.benchmark != "SPY"):
            raise ValueError("This film supports the eight-fund SPY universe in study.toml.")
        self.chapter_times = []
        self.title = left_at(text("What did the mix change?", 49, headline=True), -6.15, 3.05)
        self.subtitle = left_at(text("SPY versus eight equally weighted funds", 29, MUTED), -6.15, 2.35)
        self.caption = text("Reset to equal weights every month.", 29, MUTED).move_to([0, -2.95, 0])
        self.legend = VGroup()
        for i, (name, color) in enumerate(zip(NAMES, COLORS)):
            x = -5.3 + 3.6 * i
            swatch = Line([x, 1.95, 0], [x + 0.36, 1.95, 0], stroke_color=color, stroke_width=4)
            self.legend.add(swatch, left_at(text(name, 27, color), x + 0.55, 1.95))
        self.play(FadeIn(self.title, shift=UP * 0.12), FadeIn(self.subtitle), run_time=1)
        self.play(FadeIn(self.legend), FadeIn(self.caption), run_time=0.7)
        self.wait(2)

        self.chapter("Full-period growth")
        times, values, xticks = self.full_period()
        ymax = np.ceil(values.max() / 2000) * 2000
        self.full_spec = (times, values, (0, ymax), xticks,
                          [(v, "$0" if v == 0 else f"${v / 1000:g}k")
                           for v in np.linspace(0, ymax, 5)])
        self.plot = TimelinePlot(*self.full_spec)
        self.change_words("Start with $1,000.", "2010–2025 · dividends included",
                          "Follow both portfolios through the same dates.")
        self.play(FadeIn(self.plot), run_time=0.8)
        self.advance(times[0], times[-1], 8)
        self.ends = self.endpoint_labels()
        self.play(FadeIn(self.ends), run_time=0.6)
        lower_growth = values[-1, 1] < values[-1, 0]
        self.change_caption("The mix finished lower." if lower_growth else "The mix finished higher.")
        self.wait(2.8)

        self.chapter("A common-date 2020 example")
        window = rebased_window(self.data, "2020-02-19", "2020-08-31")[ASSETS]
        days = (window.index - window.index[0]).days.to_numpy(dtype=float)
        ticks = [(0, "19 Feb"), (33, "23 Mar"), (103, "1 Jun"), (194, "31 Aug")]
        self.change_plot((days, window.to_numpy(), (60, 120), ticks,
                          [(v, f"${v}") for v in [60, 80, 100, 120]]),
                         "Watch the 2020 fall.", "19 Feb–31 Aug · both start at $100",
                         "Same starting date. Same monthly portfolio rule.")
        trough_date = window["SPY"].idxmin()
        trough = float((trough_date - window.index[0]).days)
        self.advance(0, trough, 4)
        self.ends = self.endpoint_labels(position=trough, loss=True)
        date_label = text(trough_date.strftime("%d %B %Y").lstrip("0"), 27, MUTED).move_to([0, -2.48, 0])
        self.play(FadeIn(self.ends), FadeIn(date_label), run_time=0.6)
        self.change_caption("The mix lost less from the shared starting date.")
        self.wait(4.5)
        self.play(FadeOut(self.ends), FadeOut(date_label), run_time=0.4)
        self.ends = VGroup()
        self.advance(trough, days[-1], 5)
        self.change_caption("By 31 August, both were back above $100.")
        self.wait(3)

        self.chapter("What the full sample shows")
        vol = self.data.performance.loc[ASSETS, "annualized_volatility"]
        growth_word = "Less" if lower_growth else "More"
        risk_word = "Less" if vol.iloc[1] < vol.iloc[0] else "More"
        self.change_plot(self.full_spec, f"{growth_word} growth. {risk_word} fluctuation.",
                         "2010–2025 · the full sample", "Same dates. Same starting amount.", complete=True)
        self.ends = self.endpoint_labels()
        self.play(FadeIn(self.ends), FadeOut(self.caption), run_time=0.5)
        comparison = text(f"Annual volatility:  SPY {vol.iloc[0]:.1%}  ·  mix {vol.iloc[1]:.1%}", 29)
        comparison.move_to([0, -2.7, 0])
        qualification = text("Selected funds · Historical returns · Trading costs and taxes excluded", 19, MUTED)
        qualification.move_to([0, -3.75, 0])
        self.play(FadeIn(comparison), run_time=0.6)
        self.wait(2.5)
        self.play(FadeIn(qualification), run_time=0.6)
        self.wait(3.5)
        self.check_bounds(Group(*self.mobjects))
        out = ROOT / "outputs/animation"
        out.mkdir(parents=True, exist_ok=True)
        (out / "chapters.json").write_text(json.dumps(self.chapter_times, indent=2), encoding="utf-8")
        evidence = {
            "episode": "2020-02-19 close through 2020-08-31 close",
            "initial_value": 100,
            "trough_date": str(trough_date.date()),
            "trough_values": window.loc[trough_date].to_dict(),
            "end_values": window.iloc[-1].to_dict(),
            "qualification": "Illustrative common-date losses, not full-sample maximum drawdowns.",
            "sources": ["daily_returns.csv", "performance.csv", "run.json"],
        }
        (out / "episode.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    def chapter(self, title):
        self.chapter_times.append({"chapter": title, "start_seconds": round(self.time, 2)})

    def full_period(self):
        dates = self.data.dates
        weekly = np.flatnonzero(~dates.to_period("W").duplicated(keep="last")) + 1
        positions = np.r_[0, weekly]
        values = self.data.wealth[ASSETS].iloc[positions].to_numpy()
        tick_years = [dates[0].year, 2015, 2020, dates[-1].year]
        xticks = [(int(np.flatnonzero(dates.year == year)[0]) + 1, str(year))
                  for year in sorted(set(tick_years)) if year in dates.year]
        return positions.astype(float), values, xticks

    def advance(self, start, end, seconds):
        tracker = ValueTracker(start)
        self.plot.add_updater(lambda plot: plot.seek(tracker.get_value()))
        self.play(tracker.animate.set_value(end), run_time=seconds, rate_func=linear)
        self.plot.clear_updaters()
        self.plot.seek(end)
        self.remove(tracker)

    def change_words(self, title, subtitle, caption):
        replacements = [left_at(text(title, 49, headline=True), -6.15, 3.05),
                        left_at(text(subtitle, 29, MUTED), -6.15, 2.35),
                        text(caption, 29, MUTED).move_to([0, -3.05, 0])]
        self.check_bounds(Group(*replacements))
        self.play(*(FadeTransform(old, new) for old, new in zip(
            [self.title, self.subtitle, self.caption], replacements)), run_time=0.8)
        self.title, self.subtitle, self.caption = replacements

    def change_caption(self, words):
        replacement = text(words, 29, MUTED).move_to([0, -3.05, 0])
        self.check_bounds(Group(replacement))
        self.play(FadeTransform(self.caption, replacement), run_time=0.5)
        self.caption = replacement

    def change_plot(self, spec, title, subtitle, caption, complete=False):
        target = TimelinePlot(*spec)
        position = spec[0][-1] if complete else spec[0][0]
        target.seek(position)
        self.play(FadeOut(self.ends), run_time=0.35)
        self.change_words(title, subtitle, caption)
        self.play(Transform(self.plot.axes, target.axes), Transform(self.plot.curves, target.curves),
                  Transform(self.plot.dots, target.dots), Transform(self.plot.cursor, target.cursor),
                  run_time=1.1)
        self.plot.times, self.plot.values, self.plot.ylim = spec[:3]
        self.plot.seek(position)
        self.ends = VGroup()

    def endpoint_labels(self, position=None, loss=False):
        position = self.plot.times[-1] if position is None else position
        values = [np.interp(position, self.plot.times, col) for col in self.plot.values.T]
        anchors = [self.plot.point(position, value) for value in values]
        labels = VGroup()
        for i, (name, color, value, anchor) in enumerate(zip(NAMES, COLORS, values, anchors)):
            if loss:
                y = -0.35 if i == 1 else -1.25
                words = f"{name}: ${value:.0f}  ({value / 100 - 1:.1%})"
                tag = left_at(text(words, 27, color), anchor[0] + 0.5, y)
                connector = Line(anchor, tag.get_left() + LEFT * 0.12,
                                 stroke_color=color, stroke_width=1.2)
                labels.add(connector, tag)
            else:
                tag = VGroup(text(name, 26, color), text(f"${value:,.0f}", 31))
                tag.arrange(np.array([0, -1, 0]), buff=0.11, aligned_edge=LEFT)
                left_at(tag, 4.8, anchor[1])
                labels.add(tag)
        self.check_bounds(labels)
        return labels

    def check_bounds(self, group):
        for obj in group:
            if (obj.get_left()[0] < -6.9 or obj.get_right()[0] > 6.9
                    or obj.get_top()[1] > 3.9 or obj.get_bottom()[1] < -3.9):
                raise ValueError(f"Presentation object is outside the safe frame: {obj}")
