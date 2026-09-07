#!/usr/bin/env python3
"""Plot the request-composition pie chart (main-agent vs sub-agent requests)
for the semianalysisai/cc-traces-weka-062126 replay study.

Reads the authoritative counts from data/conversion_stats.json (produced by
convert_weka_to_agentic_mooncake.py) so the chart always matches the report's
numbers exactly, and saves a PNG under figures/.
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = ROOT / "data" / "conversion_stats.json"
FIGURES_DIR = ROOT / "figures"
OUTPUT_PATH = FIGURES_DIR / "request_composition_pie.png"

# Droid Sans Fallback ships on this host and covers CJK glyphs; fall back to
# the matplotlib default if it is not installed elsewhere.
_CJK_FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


def _configure_cjk_font() -> None:
    """Register a CJK-capable font as a fallback alongside DejaVu Sans.

    Droid Sans Fallback only covers CJK glyphs (no Latin letters/digits), so
    it must be combined with a Latin-covering font via matplotlib's
    per-glyph font-family fallback list (matplotlib >= 3.6), rather than set
    as the sole font.family.
    """
    if Path(_CJK_FONT_PATH).exists():
        fm.fontManager.addfont(_CJK_FONT_PATH)
        cjk_name = fm.FontProperties(fname=_CJK_FONT_PATH).get_name()
        plt.rcParams["font.family"] = ["DejaVu Sans", cjk_name]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    stats = json.loads(STATS_PATH.read_text())
    main_rows = stats["main_rows"]
    child_rows = stats["child_rows"]
    total_rows = stats["total_rows"]
    child_groups = stats["child_groups"]
    assert main_rows + child_rows == total_rows

    _configure_cjk_font()

    values = [main_rows, child_rows]
    colors = ["#4C72B0", "#DD8452"]
    short_labels = ["主 Agent 请求", "子 Agent 请求"]

    fig, ax = plt.subplots(figsize=(9, 10.5))

    def _autopct(pct: float) -> str:
        count = int(round(pct / 100.0 * total_rows))
        return f"{pct:.1f}%\n{count:,} 条"

    wedges, _texts, _autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=1.15,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        autopct=_autopct,
        pctdistance=0.62,
        textprops={"fontsize": 15, "color": "white", "ha": "center", "va": "center"},
    )

    ax.set_title(
        f"数据集请求构成占比（总计 {total_rows:,} 条请求）\n"
        f"semianalysisai/cc-traces-weka-062126",
        fontsize=14,
        pad=16,
    )

    # Wrap each legend entry so long descriptions do not force the canvas
    # to stretch horizontally; entries stack as multi-line paragraphs below
    # the pie instead.
    wrap_width = 52
    legend_labels = [
        textwrap.fill(
            f"{short_labels[0]}（{main_rows:,} 条，{100*main_rows/total_rows:.1f}%）— "
            f"会话主线上的顶层请求：原始数据中不带 type=\"subagent\" 标记的普通模型调用，"
            f"即每个会话自身的主时间线请求",
            width=wrap_width,
        ),
        textwrap.fill(
            f"{short_labels[1]}（{child_rows:,} 条，{100*child_rows/total_rows:.1f}%）— "
            f"归入 {child_groups:,} 个子 Agent 组（child_groups）内部的执行请求：由主 Agent "
            f"派生（spawn）出来，各子 Agent 组平均含 {child_rows/child_groups:.1f} 条内部请求",
            width=wrap_width,
        ),
    ]
    legend = ax.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=10.5,
        frameon=False,
        labelspacing=1.6,
        handlelength=1.5,
        handletextpad=0.8,
    )

    fig.subplots_adjust(top=0.88, bottom=0.30)
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"  主 Agent 请求: {main_rows:,} ({100*main_rows/total_rows:.2f}%)")
    print(f"  子 Agent 请求: {child_rows:,} ({100*child_rows/total_rows:.2f}%)")
    print(f"  总请求数:     {total_rows:,}")


if __name__ == "__main__":
    main()
