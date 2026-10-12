"""Bar matrix of quality and cost panels: one bar per run, grouped by condition."""

import math
from pathlib import Path
import textwrap

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

COLORS = ("#3568b0", "#168174", "#bc6724", "#8857a5", "#bd4863", "#626c78", "#9a7b1f", "#2f8fb3")
PRECISION_COLORS = {"fp32": "#bd4863", "bf16": "#8857a5", "fp16": "#bc6724", "int8": "#3568b0", "nf4": "#168174"}
DIRECTIONS = {"higher": "higher is better", "lower": "lower is better", "neutral": "token cost"}


def common_footnotes(rows) -> list:
    """Notes that hold for every task: what a bar is, and what the cost panels measure."""
    notes = ["Bars show individual runs; no confidence intervals or averaging of repetitions."]
    outputs = [row["mean_output_tokens"] for row in rows if row["mean_output_tokens"] is not None]
    if outputs:
        notes.append(f"Mean output: {min(outputs):.2f}–{max(outputs):.2f} tokens where recorded.")
    notes.append("GPU allocation: PyTorch peak allocated bytes on CUDA device 0, including resident weights and "
                 "inference intermediates; excludes unused cached memory, other allocations, and CPU RAM.")
    return notes


def series_key(row):
    return row["model"], row["revision"], row["precision"]


def ordered(values):
    return list(dict.fromkeys(values))


def series_labels(series, rows):
    several_models = len({row["model"] for row in rows}) > 1
    labels = []
    for key in series:
        members = [row for row in rows if series_key(row) == key]
        overrides = {row["legend_label"] for row in members if row["legend_label"]}
        if len(overrides) > 1:
            raise ValueError(f"Runs in one series need the same legend label: {sorted(overrides)}")
        if overrides:
            labels.append(next(iter(overrides)))
            continue
        model, revision, precision = key
        parts = ([model] if several_models else []) + [precision.upper()]
        if len({row["revision"] for row in rows if (row["model"], row["precision"]) == (model, precision)}) > 1:
            parts.append(revision[:10])
        labels.append(" · ".join(parts))
    return labels


def series_colors(series):
    colors = {}
    for key in series:
        color = PRECISION_COLORS.get(key[2])
        if color is None or color in colors.values():
            color = next((c for c in COLORS if c not in colors.values()), COLORS[len(colors) % len(COLORS)])
        colors[key] = color
    return colors


def bar_label(value, style):
    if style == "percent":
        return f"{value:.0f}%" if float(value).is_integer() else f"{value:.1f}%"
    if style in ("score", "seconds"):
        return f"{value:.3f}"
    if style == "memory":
        return f"{value:.2f}"
    return f"{value:.1f}"


def draw_panel(ax, panel, rows, groups, series, colors):
    cells = {(series_key(row), row["condition_label"]): row for row in rows}
    measured = [row for row in rows if row[panel.field] is not None]
    panel_groups = [group for group in groups if any(row["condition_label"] == group for row in measured)]
    panel_series = [key for key in series if any(series_key(row) == key for row in measured)]
    values = [row[panel.field] * panel.scale for row in measured]
    ceiling = 100 if panel.style == "percent" else 1 if panel.style == "score" else max(values or [1]) or 1
    ax.set_ylim(0, ceiling * 1.17)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e7ed", linewidth=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cbd3dd")
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_title(f"{panel.title} · {DIRECTIONS[panel.better]}", loc="left", pad=14, weight="bold", fontsize=12)
    ax.set_ylabel(panel.axis, fontsize=10)
    ax.set_xticks(range(len(panel_groups)), panel_groups)
    ax.set_xlim(-0.55, max(1, len(panel_groups)) - 0.45)
    if panel.style == "percent":
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    elif panel.style == "score":
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=panel.style == "tokens"))
    if not panel_series:
        ax.text(0.5, 0.5, "Not measured", transform=ax.transAxes, ha="center", color="#68788c", fontsize=12)
        ax.set_yticks([])
        return
    members = {group: [key for key in panel_series if (key, group) in cells] for group in panel_groups}
    width = 0.76 / max([len(keys) for keys in members.values()] or [1])
    for j, group in enumerate(panel_groups):
        for i, key in enumerate(members[group]):
            x = j + (i - (len(members[group]) - 1) / 2) * width
            raw = cells[key, group][panel.field]
            if raw is None:
                ax.text(x, ceiling * 0.04, "N/A", ha="center", color="#68788c", fontsize=9)
                continue
            value = raw * panel.scale
            ax.bar(x, value, width=width * 0.86, color=colors[key], zorder=3)
            # Stagger labels on narrow bars so close values stay readable.
            offset = 5 + (10 * (i % 2) if width < 0.3 else 0)
            ax.annotate(bar_label(value, panel.style), (x, value), xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=9)


def draw_matrix(rows, panels, output_dir, *, title, context, footnotes, legend_gap=18) -> list:
    """Write matrix-overview.png and .svg; returns their paths."""
    if not math.isfinite(legend_gap) or legend_gap < 0:
        raise ValueError("legend_gap must be a finite, nonnegative number of points")
    groups = ordered(row["condition_label"] for row in rows)
    series = ordered(series_key(row) for row in rows)
    labels = series_labels(series, rows)
    colors = series_colors(series)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "axes.labelcolor": "#425166",
        "text.color": "#172b42", "xtick.color": "#425166", "ytick.color": "#425166", "svg.fonttype": "none",
        "figure.facecolor": "white",
    })
    columns = 3
    panel_rows = math.ceil(len(panels) / columns)
    width = max(16, len(groups) * len(series) * 1.25)
    legend_rows = math.ceil(len(series) / 2)
    height = 4.7 * panel_rows + 0.3 * (legend_rows - 1) + 0.25 * len(footnotes)
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    axes = fig.subplots(panel_rows, columns, squeeze=False)
    bottom = (0.11 + 0.023 * len(footnotes)) * 9.4 / height
    fig.subplots_adjust(left=0.06, right=0.975, top=0.78, bottom=bottom, wspace=0.28, hspace=0.53)
    fig.text(0.06, 0.956, title, fontsize=22, weight="bold")
    fig.text(0.06, 0.917, textwrap.fill(context, width=int(width * 11)), fontsize=11)
    handles = [Patch(facecolor=colors[key], label=label) for key, label in zip(series, labels, strict=True)]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.88), frameon=False,
                        ncol=min(2, len(series)), fontsize=11)
    for ax, panel in zip(axes.flat, panels):
        draw_panel(ax, panel, rows, groups, series, colors)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for i, note in enumerate(footnotes):
        fig.text(0.06, (0.028 + (len(footnotes) - 1 - i) * 0.023) * 9.4 / height, note, fontsize=9, color="#526379")
    # Measure the rendered legend and panel titles; long legend labels need more room than
    # the row count suggests. A point-based gap stays consistent across resolutions.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bottom = legend.get_window_extent(renderer).y0
    panels_top = max(ax.get_tightbbox(renderer).y1 for ax in axes[0] if ax.get_visible())
    shift = max(0, panels_top + legend_gap * fig.dpi / 72 - legend_bottom)
    if shift:
        top = fig.subplotpars.top - shift / fig.bbox.height
        if top <= fig.subplotpars.bottom:
            raise ValueError("The legend leaves no room for the panels; shorten legend labels or reduce legend_gap")
        fig.subplots_adjust(top=top)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"matrix-overview.{extension}" for extension in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    return paths
