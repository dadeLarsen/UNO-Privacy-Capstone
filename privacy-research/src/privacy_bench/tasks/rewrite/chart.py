"""keypoints.png and .svg: what the judge found in each run's rewrites, drawn for readers new to the metrics."""

from pathlib import Path
import textwrap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .judging import VERDICTS

COLORS = {"kept": "#168174", "partial": "#e3a83a", "missing": "#b9c3cd", "contradicted": "#bd4863"}
NAMES = {"kept": "Kept", "partial": "Partly kept", "missing": "Left out", "contradicted": "Contradicted"}
TEXT, MUTED = "#172b42", "#526379"


def run_label(row) -> str:
    return row["legend_label"] or f"{row['model']} · {row['precision'].upper()}"


def stacked(ax, rows, scope, heading):
    for y, row in enumerate(rows):
        if not row["judged_rewrites"]:
            ax.text(1, y, "No valid rewrites to judge", va="center", color=MUTED, fontsize=10)
            continue
        left = 0
        for verdict in VERDICTS:
            value = 100 * row[f"{scope}_{verdict}_share"]
            ax.barh(y, value, left=left, color=COLORS[verdict], height=0.64)
            if value >= 6:
                ax.text(left + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=9.5,
                        color="white" if verdict in ("kept", "contradicted") else TEXT)
            left += value
    ax.set_xlim(0, 100)
    ax.set_title(heading, loc="left", weight="bold", fontsize=12.5, pad=10)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])


def draw_keypoints(rows, output_dir, *, title, context, footnotes) -> list:
    """Write keypoints.png and .svg; returns their paths."""
    labels = [run_label(row) for row in rows]
    notes = [line for note in footnotes for line in textwrap.wrap(note, 200)]
    height = 2.3 + 0.52 * len(rows) + 0.2 * len(notes)
    fig = Figure(figsize=(17, height))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3, sharey=True, gridspec_kw={"width_ratios": [3.2, 3.2, 1.7], "wspace": 0.12})
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.75 / height, bottom=(0.45 + 0.2 * len(notes)) / height)
    stacked(axes[0], rows, "keypoints", "All key points")
    stacked(axes[1], rows, "critical", "Critical key points")
    misleading = axes[2]
    for y, row in enumerate(rows):
        if row["misleading_rate"] is None:
            continue
        value = 100 * row["misleading_rate"]
        misleading.barh(y, value, color=COLORS["contradicted"], height=0.64)
        misleading.text(value + 2, y, f"{value:.0f}%", va="center", fontsize=9.5, color=TEXT)
    misleading.set_xlim(0, 100)
    misleading.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    misleading.set_title("Rewrites that could mislead", loc="left", weight="bold", fontsize=12.5, pad=10)
    # The panels share one y axis, so its labels are set once and hidden on the panels to the right.
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for ax in axes:
        ax.tick_params(length=0, labelsize=10.5, colors=TEXT, labelleft=ax is axes[0])
        ax.grid(axis="x", color="#e2e7ed", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.012, 1 - 0.45 / height, title, fontsize=19, weight="bold", color=TEXT)
    fig.text(0.012, 1 - 0.8 / height, context, fontsize=11, color=TEXT)
    fig.legend(handles=[Patch(color=COLORS[verdict], label=NAMES[verdict]) for verdict in VERDICTS],
               loc="upper left", bbox_to_anchor=(0.2, 1 - 0.95 / height), ncol=len(VERDICTS), frameon=False,
               fontsize=11)
    for i, note in enumerate(notes):
        fig.text(0.012, (0.18 + 0.2 * (len(notes) - 1 - i)) / height, note, fontsize=9, color=MUTED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"keypoints.{extension}" for extension in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    return paths
