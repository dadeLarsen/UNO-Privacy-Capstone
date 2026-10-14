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
# Panel geometry: margins and gaps as figure fractions, the panel height in inches.
LEFT, RIGHT, WSPACE, PANEL_HEIGHT = 0.06, 0.975, 0.28, 2.45
# Group labels wrap at this many characters a line; the figure then grows to fit the largest label.
GROUP_LABEL_CHARACTERS = 14
# A figure that would grow wider than this, in inches, lays its panels out in two columns, then one.
MAX_WIDTH = 24
# Space between the legend and the top panel titles, in points.
LEGEND_GAP = 18


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


def group_label(label: str) -> str:
    """Wrap each line of a group label so neighbouring labels stay narrow; words split only at hyphens."""
    lines = []
    for line in label.split("\n"):
        lines.extend(textwrap.wrap(line, GROUP_LABEL_CHARACTERS, break_long_words=False) or [""])
    return "\n".join(lines)


def panel_share(columns: int) -> float:
    """The fraction of the figure width that one panel takes in a row of this many panels."""
    return (RIGHT - LEFT) / (columns + (columns - 1) * WSPACE)


def label_extent(labels) -> tuple:
    """Width of the widest and height of the tallest label in inches, at the current tick font size."""
    figure = Figure()
    FigureCanvasAgg(figure)
    renderer = figure.canvas.get_renderer()
    boxes = [figure.text(0, 0, label, fontsize=matplotlib.rcParams["xtick.labelsize"],
                         multialignment="center").get_window_extent(renderer) for label in labels]
    return max(box.width for box in boxes) / figure.dpi, max(box.height for box in boxes) / figure.dpi


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
    ax.set_xticks(range(len(panel_groups)), [group_label(group) for group in panel_groups])
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


def draw_matrix(rows, panels, output_dir, *, title, context, footnotes) -> list:
    """Write matrix-overview.png and .svg; returns their paths."""
    groups = ordered(row["condition_label"] for row in rows)
    series = ordered(series_key(row) for row in rows)
    labels = series_labels(series, rows)
    colors = series_colors(series)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "axes.labelcolor": "#425166",
        "text.color": "#172b42", "xtick.color": "#425166", "ytick.color": "#425166", "svg.fonttype": "none",
        "figure.facecolor": "white",
    })
    label_width, label_height = label_extent([group_label(group) for group in groups])
    # A panel needs room for one bar per run plus a gap per group, and for each group's label. Three
    # panels share a row unless the figure would grow past MAX_WIDTH; then two do, or one.
    axis_width = max((16 * panel_share(3)), (2.0 * len(rows) + 1.0 * len(groups)) * panel_share(3),
                     (label_width + 0.15) * (len(groups) + 0.1))
    columns = next((count for count in (3, 2) if axis_width / panel_share(count) <= MAX_WIDTH), 1)
    width = max(16, axis_width / panel_share(columns))
    panel_rows = math.ceil(len(panels) / columns)
    # Lay out the height in inches: the header, the panel rows, and the footnotes. Group labels sit
    # between rows and above the footnotes, so taller labels make a taller figure.
    legend_rows = math.ceil(len(series) / 2)
    header = 2.4 + 0.3 * (legend_rows - 1)
    row_gap = label_height + 0.78
    footer = label_height + 0.52 + 0.216 * len(footnotes)
    height = header + PANEL_HEIGHT * panel_rows + row_gap * (panel_rows - 1) + footer
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    axes = fig.subplots(panel_rows, columns, squeeze=False)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=1 - header / height, bottom=footer / height, wspace=WSPACE,
                        hspace=row_gap / PANEL_HEIGHT)
    fig.text(LEFT, 1 - 0.48 / height, title, fontsize=22, weight="bold")
    fig.text(LEFT, 1 - 0.9 / height, textwrap.fill(context, width=int(width * 11)), fontsize=11)
    handles = [Patch(facecolor=colors[key], label=label) for key, label in zip(series, labels, strict=True)]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 1 - 1.31 / height), frameon=False,
                        ncol=min(2, len(series)), fontsize=11)
    for ax, panel in zip(axes.flat, panels):
        draw_panel(ax, panel, rows, groups, series, colors)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for i, note in enumerate(footnotes):
        fig.text(LEFT, (0.263 + (len(footnotes) - 1 - i) * 0.216) / height, note, fontsize=9, color="#526379")
    # Measure the rendered legend and panel titles; long legend labels need more room than
    # the row count suggests. A point-based gap stays consistent across resolutions.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bottom = legend.get_window_extent(renderer).y0
    panels_top = max(ax.get_tightbbox(renderer).y1 for ax in axes[0] if ax.get_visible())
    shift = max(0, panels_top + LEGEND_GAP * fig.dpi / 72 - legend_bottom)
    if shift:
        top = fig.subplotpars.top - shift / fig.bbox.height
        if top <= fig.subplotpars.bottom:
            raise ValueError("The legend leaves no room for the panels; shorten legend labels")
        fig.subplots_adjust(top=top)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"matrix-overview.{extension}" for extension in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    return paths
