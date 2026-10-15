"""A silent, captioned film of the saved ETF baseline. No network calls or LaTeX."""

import os
from pathlib import Path

import numpy as np
from manim import (
    AnimationGroup, Create, Dot, FadeIn, FadeOut, Group, GrowFromEdge, LEFT, Line,
    Rectangle, RoundedRectangle, Scene, Text, UP, VGroup, VMobject, linear,
)

from financial_economics.presentation import load_presentation


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#0C1424"
TEXT = "#F1F5FA"
MUTED = "#A6B4C8"
GRID = "#27354B"
SPY = "#91B8FF"
PORTFOLIO = "#64E0C4"
ACCENT = "#EACB88"
FONT = "Segoe UI"


def label(text, size=26, color=TEXT):
    return Text(text, font=FONT, font_size=size, color=color, disable_ligatures=True)


def at_left(mobject, x, y):
    return mobject.move_to([x + mobject.width / 2, y, 0])


class PhaseOneResults(Scene):
    """Four chapters, with every displayed estimate read from the saved outputs."""

    def construct(self):
        self.camera.background_color = BACKGROUND
        folder = Path(os.environ.get("FECON_RESULTS_DIR", ROOT / "outputs" / "phase1"))
        self.results = load_presentation(folder)
        if (set(self.results.metadata["study"]["assets"])
                != {"SPY", "QQQ", "VNQ", "GLD", "XLE", "XLY", "VIG", "TLT"}
                or self.results.benchmark != "SPY"):
            raise ValueError("This film supports the eight-fund SPY baseline in study.toml.")
        self.chapter_times = []
        self.introduction()
        self.growth()
        self.risk()
        self.interpretation()
        # This also catches accidental cropping during the low-resolution draft.
        folder = ROOT / "outputs" / "animation"
        folder.mkdir(parents=True, exist_ok=True)
        import json
        (folder / "chapters.json").write_text(
            json.dumps(self.chapter_times, indent=2), encoding="utf-8"
        )

    def check_bounds(self, group):
        for obj in group:
            if (obj.get_left()[0] < -6.85 or obj.get_right()[0] > 6.85
                    or obj.get_top()[1] > 3.90 or obj.get_bottom()[1] < -3.90):
                raise ValueError(f"Presentation object is outside the safe frame: {obj}")

    def header(self, chapter, title, footer):
        self.chapter_times.append({"chapter": title, "start_seconds": round(self.time, 2)})
        brand = at_left(label("FINANCIAL ECONOMICS  /  PHASE 1", 17, MUTED), -6.3, 3.45)
        page = label(f"0{chapter} / 04", 17, MUTED).move_to([5.95, 3.45, 0])
        rule = Line([-6.3, 3.1, 0], [6.3, 3.1, 0], stroke_color=GRID, stroke_width=1)
        foot = label(footer, 15, MUTED).move_to([0, -3.65, 0])
        group = VGroup(brand, page, rule, foot)
        self.check_bounds(group)
        self.play(FadeIn(group), run_time=0.7)
        return group

    def clear_chapter(self):
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.7)

    def introduction(self):
        data = self.results
        n = len(data.metadata["study"]["assets"])
        years = f"{data.dates[0].year}–{data.dates[-1].year}"
        self.header(1, "The question", "Retrospectively selected funds • Historical comparison")
        title = label("What did diversification change?", 46)
        at_left(title, -6.3, 2.45)
        subtitle = at_left(label(f"{n} funds · {years} · {len(data.dates):,} daily observations", 25, MUTED), -6.3, 1.7)
        self.play(FadeIn(title, shift=UP * 0.15), FadeIn(subtitle), run_time=1.0)

        tiles = VGroup()
        for i, (ticker, name) in enumerate(data.metadata["study"]["assets"].items()):
            x = -4.8 + (i % 4) * 3.2
            y = 0.65 - (i // 4) * 1.25
            tile = RoundedRectangle(width=2.9, height=1.05, corner_radius=0.13,
                                    stroke_width=1, stroke_color=GRID)
            tile.move_to([x, y, 0])
            ticker_text = label(ticker, 29, SPY if ticker == data.benchmark else TEXT)
            ticker_text.move_to([x, y + 0.18, 0])
            # Use short, readable categories; full instrument names stay in study.toml.
            category = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "VNQ": "Real estate",
                        "GLD": "Gold", "XLE": "Energy", "XLY": "Consumer stocks",
                        "VIG": "Dividend growth", "TLT": "Long Treasuries"}.get(ticker, name)
            name_text = label(category, 18, MUTED)
            if name_text.width > 2.6:
                name_text.scale_to_fit_width(2.6)
            name_text.move_to([x, y - 0.25, 0])
            tiles.add(VGroup(tile, ticker_text, name_text))
        self.check_bounds(tiles)
        self.play(AnimationGroup(*(FadeIn(tile, shift=UP * 0.12) for tile in tiles),
                                 lag_ratio=0.13), run_time=2.2)
        allocation = label(f"{100 / n:g}% in each fund. Reset every month.", 32, PORTFOLIO)
        allocation.move_to([0, -1.8, 0])
        benchmark = label(f"Compare the mix with {data.benchmark}, the S&P 500 fund.", 25, MUTED)
        benchmark.move_to([0, -2.55, 0])
        self.check_bounds(VGroup(title, subtitle, allocation, benchmark))
        self.play(FadeIn(allocation), run_time=0.8)
        self.wait(2.8)
        self.play(FadeIn(benchmark), run_time=0.8)
        self.wait(3.8)
        self.clear_chapter()

    def growth(self):
        data = self.results
        self.header(2, "Growth", "Yahoo Finance adjusted prices • Weekly chart points • Daily statistics")
        title = at_left(label("How $1,000 grew", 43), -6.3, 2.45)
        subtitle = at_left(label("Same starting amount. Same dates. Dividends included.", 24, MUTED), -6.3, 1.82)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.9)
        left, right, bottom, top = -5.65, 4.5, -1.8, 1.3
        selected = [data.benchmark, "Equal weight"]
        ceiling = np.ceil(data.wealth[selected].to_numpy().max() / 2000) * 2000

        def point(day, value):
            return np.array([left + (right - left) * day / len(data.dates),
                             bottom + (top - bottom) * value / ceiling, 0])

        axes = VGroup()
        for value in np.linspace(0, ceiling, 5):
            line = Line(point(0, value), point(len(data.dates), value),
                        stroke_color=GRID, stroke_width=1)
            tick = label(f"${value / 1000:g}k" if value else "$0", 18, MUTED)
            tick.move_to([left - 0.47, point(0, value)[1], 0])
            axes.add(line, tick)
        tick_years = list(range(data.dates[0].year, data.dates[-1].year, 4))
        tick_years.append(data.dates[-1].year)
        for year in tick_years:
            index = int(np.flatnonzero(data.dates.year == year)[0]) + 1
            axes.add(label(str(year), 19, MUTED).move_to([point(index, 0)[0], bottom - 0.3, 0]))
        self.play(FadeIn(axes), run_time=0.8)

        weekly_positions = (np.flatnonzero(~data.dates.to_period("W").duplicated(keep="last")) + 1)
        positions = np.r_[0, weekly_positions]
        curves = VGroup()
        for asset, color in zip(selected, [SPY, PORTFOLIO]):
            curve = VMobject(stroke_color=color, stroke_width=3)
            curve.set_points_as_corners([point(i, data.wealth.iloc[i][asset]) for i in positions])
            curves.add(curve)
        self.play(*(Create(curve) for curve in curves), run_time=8, rate_func=linear)

        endpoints = VGroup()
        for asset, name, color in zip(selected, [data.benchmark, "Eight funds"], [SPY, PORTFOLIO]):
            final = data.wealth.iloc[-1][asset]
            end = point(len(data.dates), final)
            tag = VGroup(label(name, 20, color), label(f"${final:,.0f}", 27, TEXT))
            tag.arrange(direction=np.array([0, -1, 0]), buff=0.13, aligned_edge=LEFT)
            at_left(tag, 4.73, end[1])
            endpoints.add(Dot(end, radius=0.045, color=color), tag)
        annual = label(
            f"Annual growth   {data.benchmark}: {data.performance.loc[data.benchmark, 'cagr']:.2%}"
            f"    /    Eight funds: {data.performance.loc['Equal weight', 'cagr']:.2%}", 25)
        annual.move_to([0, -2.6, 0])
        winner = (data.benchmark if data.wealth.iloc[-1][data.benchmark]
                  > data.wealth.iloc[-1]["Equal weight"] else "The fund mix")
        caption = label(f"{winner} grew more over this period.", 26, MUTED)
        caption.move_to([0, -3.15, 0])
        self.check_bounds(VGroup(title, subtitle, axes, endpoints, annual, caption))
        self.play(FadeIn(endpoints), FadeIn(annual), run_time=1.0)
        self.wait(3)
        self.play(FadeIn(caption), run_time=0.7)
        self.wait(4.2)
        self.clear_chapter()

    def risk(self):
        data = self.results
        self.header(3, "Risk", "Worst falls are each portfolio's own extreme; they need not occur together")
        smoother = (data.performance.loc["Equal weight", "annualized_volatility"]
                    < data.performance.loc[data.benchmark, "annualized_volatility"])
        smaller_fall = (data.performance.loc["Equal weight", "max_drawdown"]
                        > data.performance.loc[data.benchmark, "max_drawdown"])
        heading = "The mix had a smoother ride" if smoother else "How the risks compared"
        title = at_left(label(heading, 43), -6.3, 2.45)
        legend = VGroup(label(data.benchmark, 23, SPY), label("Eight funds", 23, PORTFOLIO))
        legend.arrange(buff=0.65).move_to([3.95, 1.68, 0])
        self.play(FadeIn(title), FadeIn(legend), run_time=0.9)
        panels = [(-6.3, "Return fluctuation", "Annualized volatility", "annualized_volatility"),
                  (0.5, "Worst fall from a peak", "Maximum drawdown", "max_drawdown")]
        all_bars, descriptions = VGroup(), VGroup()
        for x, heading, detail, field in panels:
            heading_text = at_left(label(heading, 29), x, 0.96)
            detail_text = at_left(label(detail, 20, MUTED), x, 0.45)
            descriptions.add(heading_text, detail_text)
            # Identical zero-based scales within each paired comparison.
            values = [abs(data.performance.loc[asset, field]) for asset in [data.benchmark, "Equal weight"]]
            maximum = np.ceil(max(values) * 100 / 5) * 5 / 100
            for row, (value, color) in enumerate(zip(values, [SPY, PORTFOLIO])):
                y = -0.27 - row * 0.95
                track = Rectangle(width=3.6, height=0.28, stroke_width=0,
                                  fill_color=GRID, fill_opacity=1)
                at_left(track, x, y)
                bar = Rectangle(width=3.6 * value / maximum, height=0.28,
                                stroke_width=0, fill_color=color, fill_opacity=1)
                at_left(bar, x, y)
                number = at_left(label(f"{value:.2%}", 30, color), x + 3.8, y)
                self.add(track)
                all_bars.add(VGroup(bar, number))
        self.check_bounds(VGroup(title, legend, descriptions, all_bars))
        self.play(FadeIn(descriptions), run_time=0.8)
        for bars in [all_bars[:2], all_bars[2:]]:
            self.play(*(GrowFromEdge(pair[0], LEFT) for pair in bars),
                      *(FadeIn(pair[1]) for pair in bars), run_time=1.6)
            self.wait(2.3)
        conclusion = ("Less fluctuation, with a smaller worst decline."
                      if smoother and smaller_fall else "Risk depends on the allocation and the period.")
        caption = label(conclusion, 29, PORTFOLIO)
        caption.move_to([0, -2.35, 0])
        explanation = label("Substantial losses still occurred in this historical sample.", 23, MUTED)
        explanation.move_to([0, -3.0, 0])
        self.check_bounds(VGroup(caption, explanation))
        self.play(FadeIn(caption), run_time=0.8)
        self.wait(3)
        self.play(FadeIn(explanation), run_time=0.7)
        self.wait(4.0)
        self.clear_chapter()

    def interpretation(self):
        data = self.results
        self.header(4, "Interpretation", "French U.S. market factor • HAC inference • Zero trading costs and taxes")
        title = at_left(label("What the market model says", 43), -6.3, 2.45)
        beta = float(data.capm.loc["Equal weight", "beta"])
        pvalue = float(data.capm.loc["Equal weight", "alpha_pvalue"])
        self.play(FadeIn(title), run_time=0.8)
        beta_number = at_left(label(f"{beta:.2f}", 73, PORTFOLIO), -6.3, 0.85)
        beta_label = at_left(label("Portfolio beta", 26, MUTED), -6.3, -0.05)
        definition = VGroup(label("A 1 percentage point move in", 27),
                            label("market excess return was associated", 27),
                            label(f"with about {beta:.2f} percentage points", 27),
                            label("in portfolio excess return, on average.", 27))
        definition.arrange(direction=np.array([0, -1, 0]), aligned_edge=LEFT, buff=0.2)
        at_left(definition, -2.7, 0.65)
        self.check_bounds(VGroup(title, beta_number, beta_label, definition))
        self.play(FadeIn(beta_number), FadeIn(beta_label), FadeIn(definition), run_time=1.0)
        self.wait(5.2)

        alpha_text = "No clear evidence of CAPM alpha." if pvalue >= 0.05 else "Evidence of CAPM alpha at the 5% level."
        alpha = at_left(label(alpha_text, 31, ACCENT), -6.3, -1.3)
        alpha_detail = at_left(label(f"Alpha p-value: {pvalue:.3f}  ·  5% significance threshold", 23, MUTED), -6.3, -1.9)
        self.check_bounds(VGroup(alpha, alpha_detail))
        self.play(FadeIn(alpha), FadeIn(alpha_detail), run_time=0.9)
        self.wait(4.5)
        self.play(FadeOut(VGroup(beta_number, beta_label, definition, alpha, alpha_detail)), run_time=0.6)

        lower_vol = (data.performance.loc["Equal weight", "annualized_volatility"]
                     < data.performance.loc[data.benchmark, "annualized_volatility"])
        lower_growth = (data.performance.loc["Equal weight", "cagr"]
                        < data.performance.loc[data.benchmark, "cagr"])
        takeaway = VGroup(label("Less fluctuation." if lower_vol else "More fluctuation.", 44, PORTFOLIO),
                          label("Less growth." if lower_growth else "More growth.", 44, SPY),
                          label("A trade-off worth studying.", 37))
        takeaway.arrange(direction=np.array([0, -1, 0]), aligned_edge=LEFT, buff=0.28)
        at_left(takeaway, -6.3, 0.3)
        limit = at_left(label("Historical results. Future performance remains an open question.", 23, MUTED), -6.3, -2.0)
        self.check_bounds(VGroup(takeaway, limit))
        self.play(FadeIn(takeaway, shift=UP * 0.15), run_time=0.9)
        self.play(FadeIn(limit), run_time=0.6)
        self.wait(5.0)
