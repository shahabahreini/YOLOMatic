"""Plotly HTML report generation for benchmark results."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


FONT_FAMILY = "'Inter', system-ui, -apple-system, sans-serif"
PAPER_BG = "#ffffff"
PLOT_BG = "#f9fafb"
GRID = "#f3f4f6"
INK = "#111827"
MUTED = "#6b7280"
BLUE = "#3b82f6"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
PURPLE = "#8b5cf6"
CYAN = "#06b6d4"
PALETTE = [BLUE, GREEN, PURPLE, CYAN, AMBER, RED, "#6366f1", "#14b8a6"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _metric_fill(v: float, is_highlight: bool = False) -> str:
    if is_highlight:
        if v >= 0.85: return "#dbeafe" # Light blue for highlight
        if v >= 0.70: return "#eff6ff"
        return "#f8fafc"
    
    if v >= 0.85:
        return "#dcfce7"
    if v >= 0.70:
        return "#ecfdf5"
    if v >= 0.50:
        return "#fef3c7"
    return "#fee2e2"


def _model_counts(model: Any) -> tuple[int, int, int]:
    return (
        sum(r.tp for r in model.per_image),
        sum(r.fp for r in model.per_image),
        sum(r.fn for r in model.per_image),
    )


def _shorten(value: str, max_chars: int = 72) -> str:
    if len(value) <= max_chars:
        return value
    keep = max_chars - 3
    return f"{value[: keep // 2]}...{value[-(keep - keep // 2):]}"


def _display_names(models: list[Any]) -> dict[Path, str]:
    """Return stable report labels that distinguish repeated best.pt/last.pt files."""
    raw: list[str] = []
    for model in models:
        path = model.weights_path
        if path.name in {"best.pt", "last.pt", "best.pth", "last.pth"} and path.parent.name == "weights":
            run_name = path.parent.parent.name if path.parent.parent != path.parent else path.parent.name
            raw.append(f"{run_name} / {path.name}")
        elif path.name in {"best.pt", "last.pt", "best.pth", "last.pth"}:
            raw.append(f"{path.parent.name} / {path.name}")
        else:
            raw.append(path.stem)

    counts = Counter(raw)
    seen: Counter[str] = Counter()
    names: dict[Path, str] = {}
    for model, label in zip(models, raw):
        if counts[label] > 1:
            seen[label] += 1
            parent = model.weights_path.parent.parent.name if model.weights_path.parent.name == "weights" else model.weights_path.parent.name
            label = f"{label} ({parent or seen[label]})"
        names[model.weights_path] = _shorten(label, 80)
    return names


def _display_name(model: Any, names: dict[Path, str]) -> str:
    return names.get(model.weights_path, model.weights_path.stem)


def _base_layout(fig: go.Figure, *, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": INK},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "size": 12, "color": INK},
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        height=height,
        hoverlabel={"bgcolor": "#111827", "font": {"color": "white", "size": 13}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": MUTED},
        },
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont={"color": MUTED, "size": 11},
        title_font={"size": 12, "color": MUTED},
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont={"color": MUTED, "size": 11},
        title_font={"size": 12, "color": MUTED},
    )
    return fig


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

def _summary_cards_html(result: Any, names: dict[Path, str]) -> str:
    best = max(result.models, key=lambda m: m.map50)
    tp, fp, fn = _model_counts(best)
    cards = [
        ("mAP@50:95", _fmt(best.map50_95), "Stricter IoU sweep (Primary)", "primary-card"),
        ("mAP@50", _fmt(best.map50), "Standard IoU benchmark", ""),
        ("F1 Score", _fmt(best.f1), "Precision/Recall balance", ""),
        ("Best Model", escape(_display_name(best, names)), "Ranked by mAP@50", "model-card"),
    ]
    rendered = []
    for title, value, note, extra_class in cards:
        rendered.append(
            f'<div class="kpi-card {extra_class}">'
            f'<div class="kpi-label">{escape(title)}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{escape(note)}</div>'
            '</div>'
        )
    return '<section class="section kpi-grid">' + "".join(rendered) + "</section>"


def _comparison_table(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    model_names = [_display_name(m, names) for m in models]
    
    # Define metrics and identify which one to highlight
    metrics_data = [
        ("mAP@50:95", [m.map50_95 for m in models], True),
        ("mAP@50", [m.map50 for m in models], False),
        ("mAP@75", [m.map75 for m in models], False),
        ("F1", [m.f1 for m in models], False),
        ("Precision", [m.precision for m in models], False),
        ("Recall", [m.recall for m in models], False),
    ]
    
    headers = ["Rank", "Model", "Task"] + [m[0] for m in metrics_data]
    values = [
        [str(i) for i in range(1, len(models) + 1)],
        model_names,
        [m.task for m in models],
    ]
    fills = [
        ["#f9fafb"] * len(models),
        ["#f9fafb"] * len(models),
        ["#f9fafb"] * len(models),
    ]
    
    for label, vals, highlight in metrics_data:
        values.append([_fmt(v) for v in vals])
        fills.append([_metric_fill(v, is_highlight=highlight) for v in vals])

    fig = go.Figure(go.Table(
        columnwidth=[50, 240, 100, 100, 90, 90, 80, 90, 90],
        header={
            "values": [f"<b>{h}</b>" for h in headers],
            "fill_color": "#111827",
            "font": {"color": "white", "size": 12},
            "align": ["center", "left", "center"] + ["center"] * 6,
            "height": 38,
        },
        cells={
            "values": values,
            "fill_color": fills,
            "font": {"size": 12, "color": INK},
            "align": ["center", "left", "center"] + ["center"] * 6,
            "height": 32,
            "line": {"color": "#f3f4f6", "width": 1},
        },
    ))
    return _base_layout(fig, title="Model Leaderboard (Ranked by mAP@50:95)", height=max(300, 110 + len(models) * 36))


def _ranked_metric_bar(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    y = [_display_name(m, names) for m in models][::-1]
    fig = go.Figure()
    for metric_name, values, color in [
        ("mAP@50:95", [m.map50_95 for m in models][::-1], BLUE),
        ("mAP@50", [m.map50 for m in models][::-1], CYAN),
        ("F1", [m.f1 for m in models][::-1], GREEN),
    ]:
        fig.add_trace(go.Bar(
            name=metric_name,
            y=y,
            x=values,
            orientation="h",
            marker={"color": color, "line": {"width": 0}},
            text=[_fmt(v) for v in values],
            textposition="outside",
            hovertemplate=f"<b>{metric_name}</b>: %{{x:.3f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", xaxis={"range": [0, 1.12], "tickformat": ".0%"}, bargap=0.2, bargroupgap=0.1)
    return _base_layout(fig, title="Ranked Model Quality", height=max(360, 170 + len(models) * 54))


def _precision_recall_chart(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    x = [_display_name(m, names) for m in models]
    fig = go.Figure()
    for metric_name, values, color in [
        ("Precision", [m.precision for m in models], BLUE),
        ("Recall", [m.recall for m in models], AMBER),
        ("F1 Score", [m.f1 for m in models], GREEN),
    ]:
        fig.add_trace(go.Bar(
            name=metric_name,
            x=x,
            y=values,
            marker={"color": color, "line": {"width": 0}},
            text=[_fmt(v) for v in values],
            textposition="outside",
            hovertemplate=f"<b>{metric_name}</b>: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", yaxis={"range": [0, 1.12], "tickformat": ".0%"}, bargap=0.2)
    return _base_layout(fig, title="Detection Balance (P, R, F1)", height=440)


def _size_heatmap(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    x = ["Small<br>&lt;32²", "Medium<br>32²-96²", "Large<br>&gt;96²"]
    y = [_display_name(m, names) for m in models]
    z = [[m.small.map50, m.medium.map50, m.large.map50] for m in models]
    text = [[_fmt(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        x=x,
        y=y,
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#fdf2f2"],
            [0.3, "#fffbeb"],
            [0.6, "#f0fdf4"],
            [1.0, "#dcfce7"],
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}: <b>%{z:.3f}</b><extra></extra>",
    ))
    return _base_layout(fig, title="Object Size Sensitivity (mAP@50)", height=max(320, 160 + len(models) * 40))


def _quality_counts_bar(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    x = [_display_name(m, names) for m in models]
    counts = {
        "TP": [sum(r.tp for r in m.per_image) for m in models],
        "FP": [sum(r.fp for r in m.per_image) for m in models],
        "FN": [sum(r.fn for r in m.per_image) for m in models],
    }
    fig = go.Figure()
    for label, color in [("TP", GREEN), ("FP", RED), ("FN", AMBER)]:
        fig.add_trace(go.Bar(
            name=label,
            x=x,
            y=counts[label],
            marker={"color": color, "line": {"width": 0}},
            hovertemplate=f"<b>{label}</b>: %{{y}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", yaxis_title="Instance Count", bargap=0.3)
    return _base_layout(fig, title="Detection Outcome Counts", height=410)


def _per_image_distribution(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Box(
            name=_display_name(model, names),
            y=[r.f1 for r in model.per_image],
            marker={"color": PALETTE[i % len(PALETTE)], "size": 3},
            line={"width": 1.5},
            boxmean=True,
            jitter=0.4,
            pointpos=-1.5,
            boxpoints="outliers",
            hovertemplate="<b>F1</b>: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(yaxis={"range": [-0.05, 1.05], "tickformat": ".0%"}, showlegend=False)
    return _base_layout(fig, title="Per-Image F1 Stability", height=440)


def _per_image_tables(result: Any, names: dict[Path, str]) -> go.Figure:
    best = max(result.models, key=lambda m: m.map50_95)
    sorted_imgs = sorted(best.per_image, key=lambda r: r.f1)
    worst = sorted_imgs[:20]

    values = [
        [r.image_path.name for r in worst],
        [_fmt(r.f1) for r in worst],
        [_fmt(r.precision) for r in worst],
        [_fmt(r.recall) for r in worst],
        [str(r.tp) for r in worst],
        [str(r.fp) for r in worst],
        [str(r.fn) for r in worst],
        [_fmt(r.mean_iou) for r in worst],
    ]
    headers = ["Image", "F1", "P", "R", "TP", "FP", "FN", "Mean IoU"]
    fig = go.Figure(go.Table(
        columnwidth=[260, 70, 70, 70, 60, 60, 60, 90],
        header={
            "values": [f"<b>{h}</b>" for h in headers],
            "fill_color": "#111827",
            "font": {"color": "white", "size": 12},
            "align": ["left"] + ["center"] * 7,
            "height": 38,
        },
        cells={
            "values": values,
            "fill_color": [["#fff7ed"] * len(worst)] + [["#ffffff"] * len(worst)] * 7,
            "font": {"size": 11, "color": INK},
            "align": ["left"] + ["center"] * 7,
            "height": 28,
            "line": {"color": "#f3f4f6", "width": 1},
        },
    ))
    title = f"Top 20 Difficulty Samples (Model: {_display_name(best, names)})"
    return _base_layout(fig, title=title, height=max(360, 100 + len(worst) * 32))


def _vector_scatter(vector_data: dict, model_name: str) -> go.Figure:
    if not vector_data or not vector_data.get("x"):
        fig = go.Figure()
        fig.add_annotation(text="No vector data available", showarrow=False, font={"size": 14, "color": MUTED})
        return _base_layout(fig, title=f"Visual Semantic Cluster: {model_name}", height=430)

    has_thumbnails = bool(vector_data.get("thumbnails"))
    custom_data = []
    for i in range(len(vector_data["x"])):
        custom_data.append([
            vector_data["image_name"][i],
            vector_data["f1"][i],
            vector_data["precision"][i],
            vector_data["recall"][i],
            vector_data["tp"][i],
            vector_data["fp"][i],
            vector_data["fn"][i],
            vector_data["mean_iou"][i],
            vector_data["thumbnails"][i] if has_thumbnails else "",
        ])

    gt_counts = vector_data["gt_count"]
    max_gt = max(gt_counts) if gt_counts else 1
    marker_sizes = [max(8, min(24, 8 + 16 * (c / max_gt))) for c in gt_counts]
    fig = go.Figure(go.Scatter(
        x=vector_data["x"],
        y=vector_data["y"],
        mode="markers",
        marker={
            "size": marker_sizes,
            "color": vector_data["f1"],
            "colorscale": "Plasma",
            "cmin": 0,
            "cmax": 1,
            "showscale": True,
            "colorbar": {
                "title": "F1",
                "thickness": 15,
                "len": 0.5,
                "y": 0.5,
                "outlinewidth": 0,
            },
            "line": {"width": 0.8, "color": "#ffffff"},
            "opacity": 0.9,
        },
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "F1: <b>%{customdata[1]:.3f}</b><br>"
            "P: %{customdata[2]:.3f} R: %{customdata[3]:.3f}<br>"
            "TP: %{customdata[4]} FP: %{customdata[5]} FN: %{customdata[6]}<br>"
            "Mean IoU: %{customdata[7]:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
    )
    return _base_layout(fig, title=f"Dataset Semantic Landscape ({model_name})", height=560)


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

_GALLERY_JS = """
<script>
(function() {
  var scatter = document.getElementById('scatter-plot');
  var gallery = document.getElementById('gallery-panel');
  var galleryImg = document.getElementById('gallery-img');
  if (!scatter) return;

  function _setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = text;
  }

  scatter.on('plotly_click', function(data) {
    var pt = data.points[0];
    var cd = pt.customdata;
    var thumb = cd[8];
    if (thumb && thumb.length > 0) {
      galleryImg.src = 'data:image/png;base64,' + thumb;
      galleryImg.alt = cd[0];
    } else {
      galleryImg.removeAttribute('src');
      galleryImg.alt = 'No thumbnail available';
    }
    _setText('gc-name', cd[0]);
    _setText('gc-f1', '<strong>F1:</strong> ' + parseFloat(cd[1]).toFixed(3));
    _setText('gc-pr', '<strong>P:</strong> ' + parseFloat(cd[2]).toFixed(3) + ' &nbsp; <strong>R:</strong> ' + parseFloat(cd[3]).toFixed(3));
    _setText('gc-counts', '<strong>TP:</strong> ' + cd[4] + ' &nbsp; <strong>FP:</strong> ' + cd[5] + ' &nbsp; <strong>FN:</strong> ' + cd[6]);
    _setText('gc-iou', '<strong>Mean IoU:</strong> ' + parseFloat(cd[7]).toFixed(3));
    gallery.style.display = 'block';
    gallery.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
})();
</script>
"""

_GALLERY_HTML = """
<section id="gallery-panel" class="gallery-panel" style="display:none;">
  <div class="section-title">Visual Inspector</div>
  <img id="gallery-img" alt="Click a scatter point" />
  <div class="gallery-meta">
    <div id="gc-name" class="gallery-name"></div>
    <div id="gc-f1" class="gallery-item"></div>
    <div id="gc-pr" class="gallery-item"></div>
    <div id="gc-counts" class="gallery-item"></div>
    <div id="gc-iou" class="gallery-item"></div>
  </div>
</section>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_benchmark_report(result: Any, output_dir: Path) -> Path:
    """Generate self-contained HTML report and return its path."""
    from .thumbnails import make_thumbnail_b64
    from .vector_analysis import build_vector_data

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_report_{ts}.html"

    names = _display_names(result.models)
    best = max(result.models, key=lambda m: m.map50)
    best_name = _display_name(best, names)

    def _thumb_fn(r: Any) -> str:
        return make_thumbnail_b64(
            r.image_path,
            r.raw_gts,
            r.raw_preds,
            size=result.config.max_thumbnail_size,
            task=best.task,
        )

    thumbnail_fn = _thumb_fn if result.config.generate_thumbnails else None
    vector_data = build_vector_data(best, thumbnail_fn=thumbnail_fn)

    figs = {
        "leaderboard": _comparison_table(result, names),
        "ranked": _ranked_metric_bar(result, names),
        "prf": _precision_recall_chart(result, names),
        "size": _size_heatmap(result, names),
        "counts": _quality_counts_bar(result, names),
        "distribution": _per_image_distribution(result, names),
        "ranking": _per_image_tables(result, names),
        "scatter": _vector_scatter(vector_data, best_name),
    }

    sections = [_summary_cards_html(result, names)]
    for key, fig in figs.items():
        div_id = "scatter-plot" if key == "scatter" else f"plot-{key}"
        html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=div_id,
            config={"displaylogo": False, "responsive": True},
        )
        sections.append(f'<section class="section chart-section">{html}</section>')

    model_list_items = "".join(
        '<li><span class="model-name">' + escape(_display_name(m, names)) + "</span>"
        + '<span class="model-path">' + escape(str(m.weights_path)) + "</span>"
        + '<span class="model-score">mAP@50 ' + _fmt(m.map50) + " · F1 " + _fmt(m.f1) + "</span></li>"
        for m in sorted(result.models, key=lambda model: model.map50, reverse=True)
    )

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_dir_str = str(result.config.validation_dir)
    conf = _fmt(result.config.conf_threshold, 2)
    iou = _fmt(result.config.iou_threshold, 2)

    full_html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>YOLOMatic Benchmark Report</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">\n'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
        '<style>\n'
        ':root {\n'
        f'  --ink: {INK};\n'
        f'  --muted: {MUTED};\n'
        f'  --blue: {BLUE};\n'
        f'  --green: {GREEN};\n'
        '  --bg: #f3f4f6;\n'
        '  --card-bg: #ffffff;\n'
        '  --border: #e5e7eb;\n'
        '  --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);\n'
        '  --radius: 12px;\n'
        '}\n'
        '*, *::before, *::after { box-sizing: border-box; }\n'
        f'body {{ font-family: {FONT_FAMILY}; background: var(--bg); color: var(--ink); margin: 0; line-height: 1.5; }}\n'
        '.hero { background: #0f172a; color: white; padding: 48px 42px; border-bottom: 1px solid #1e293b; }\n'
        '.hero h1 { margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.025em; }\n'
        '.hero p { margin: 12px 0 0; color: #94a3b8; font-size: 14px; font-weight: 500; }\n'
        '.hero-meta { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 24px; }\n'
        '.pill { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 99px; padding: 6px 14px; font-size: 13px; font-weight: 600; color: #e2e8f0; }\n'
        '.container { max-width: 1400px; margin: 0 auto; padding: 32px 24px 64px; }\n'
        '.section { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 24px; box-shadow: var(--shadow); overflow: hidden; }\n'
        '.chart-section { padding: 20px; }\n'
        '.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; padding: 0; background: transparent; border: none; box-shadow: none; }\n'
        '.kpi-card { border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; background: var(--card-bg); box-shadow: var(--shadow); position: relative; transition: transform 0.2s; }\n'
        '.kpi-card:hover { transform: translateY(-2px); }\n'
        '.primary-card { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-color: #bfdbfe; }\n'
        '.primary-card .kpi-value { color: #1e40af; }\n'
        '.model-card { border-left: 4px solid var(--blue); }\n'
        '.kpi-label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }\n'
        '.kpi-value { margin-top: 8px; font-size: 32px; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; }\n'
        '.kpi-note { margin-top: 12px; color: var(--muted); font-size: 13px; }\n'
        '.model-list { list-style: none; padding: 0; margin: 32px 0 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }\n'
        '.model-list li { display: flex; flex-direction: column; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; transition: background 0.2s; }\n'
        '.model-list li:hover { background: #334155; }\n'
        '.model-name { color: white; font-weight: 700; font-size: 15px; }\n'
        '.model-path { color: #64748b; font-size: 12px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n'
        '.model-score { color: #94a3b8; font-size: 13px; margin-top: 12px; font-weight: 600; }\n'
        '.section-title { font-weight: 800; font-size: 18px; margin-bottom: 20px; color: var(--ink); letter-spacing: -0.01em; }\n'
        '.gallery-panel { max-width: 800px; margin: 0 auto 32px; background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; box-shadow: var(--shadow); }\n'
        '.gallery-panel img { width: 100%; border-radius: 8px; background: #f1f5f9; min-height: 240px; object-fit: contain; border: 1px solid var(--border); }\n'
        '.gallery-meta { margin-top: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; border-top: 1px solid var(--border); padding-top: 20px; }\n'
        '.gallery-name { grid-column: 1 / -1; color: var(--ink); font-weight: 700; font-size: 16px; }\n'
        '.gallery-item { font-size: 13px; color: var(--muted); }\n'
        '.gallery-item strong { color: var(--ink); }\n'
        '.footer { text-align: center; color: var(--muted); font-size: 13px; padding: 48px; border-top: 1px solid var(--border); margin-top: 32px; }\n'
        '@media (max-width: 780px) { .hero { padding: 32px 20px; } .container { padding: 20px 16px; } .kpi-grid { grid-template-columns: 1fr; } }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<header class="hero">\n'
        '<h1>YOLOMatic Benchmark</h1>\n'
        '<p>Detailed Performance Analysis Report</p>\n'
        '<div class="hero-meta">'
        '<span class="pill">Run: ' + escape(timestamp_str) + '</span>'
        '<span class="pill">Validation: ' + escape(val_dir_str) + '</span>'
        '<span class="pill">Models: ' + str(len(result.models)) + '</span>'
        '<span class="pill">Conf: ' + conf + ' / IoU: ' + iou + '</span>'
        '</div>\n'
        '<ul class="model-list">' + model_list_items + '</ul>\n'
        '</header>\n'
        '<main class="container">\n'
        + "\n".join(sections) + "\n"
        + _GALLERY_HTML + "\n"
        '</main>\n'
        '<div class="footer">Generated by <strong>YOLOMatic</strong> &middot; Automated Vision Evaluation</div>\n'
        + _GALLERY_JS
        + '\n</body>\n</html>'
    )

    report_path.write_text(full_html, encoding="utf-8")
    return report_path
