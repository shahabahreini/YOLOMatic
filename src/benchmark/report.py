"""Plotly HTML report generation for benchmark results."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


FONT_FAMILY = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
PAPER_BG = "#ffffff"
PLOT_BG = "#f8fafc"
GRID = "#e5e7eb"
INK = "#111827"
MUTED = "#6b7280"
BLUE = "#2563eb"
GREEN = "#059669"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
CYAN = "#0891b2"
PALETTE = [BLUE, GREEN, PURPLE, CYAN, AMBER, RED, "#4f46e5", "#0f766e"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _metric_fill(v: float) -> str:
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
        title={"text": title, "font": {"size": 18, "color": INK}},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": FONT_FAMILY, "size": 12, "color": INK},
        margin={"l": 64, "r": 28, "t": 72, "b": 56},
        height=height,
        hoverlabel={"bgcolor": "#111827", "font": {"color": "white"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": MUTED})
    return fig


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

def _summary_cards_html(result: Any, names: dict[Path, str]) -> str:
    best = max(result.models, key=lambda m: m.map50)
    tp, fp, fn = _model_counts(best)
    cards = [
        ("Best model", escape(_display_name(best, names)), "Ranked by mAP@50"),
        ("mAP@50", _fmt(best.map50), "Primary benchmark rank"),
        ("mAP@50:95", _fmt(best.map50_95), "COCO-style stricter IoU sweep"),
        ("F1", _fmt(best.f1), "Balance of precision and recall"),
        ("Precision / Recall", f"{_pct(best.precision)} / {_pct(best.recall)}", "At configured confidence + IoU"),
        ("TP / FP / FN", f"{tp} / {fp} / {fn}", "Best model detection counts"),
    ]
    rendered = []
    for title, value, note in cards:
        rendered.append(
            '<div class="kpi-card">'
            f'<div class="kpi-label">{escape(title)}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{escape(note)}</div>'
            '</div>'
        )
    return '<section class="section kpi-grid">' + "".join(rendered) + "</section>"


def _comparison_table(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50, reverse=True)
    model_names = [_display_name(m, names) for m in models]
    metrics = {
        "mAP@50": [m.map50 for m in models],
        "mAP@75": [m.map75 for m in models],
        "mAP@50:95": [m.map50_95 for m in models],
        "F1": [m.f1 for m in models],
        "Precision": [m.precision for m in models],
        "Recall": [m.recall for m in models],
    }
    headers = ["Rank", "Model", "Task"] + list(metrics)
    values = [
        [str(i) for i in range(1, len(models) + 1)],
        model_names,
        [m.task for m in models],
        *[[_fmt(v) for v in vals] for vals in metrics.values()],
    ]
    fills = [
        ["#f8fafc"] * len(models),
        ["#f8fafc"] * len(models),
        ["#f8fafc"] * len(models),
        *[[_metric_fill(v) for v in vals] for vals in metrics.values()],
    ]
    fig = go.Figure(go.Table(
        columnwidth=[48, 260, 100, 90, 90, 100, 80, 90, 90],
        header={
            "values": [f"<b>{h}</b>" for h in headers],
            "fill_color": "#111827",
            "font": {"color": "white", "size": 12},
            "align": ["center", "left", "center"] + ["center"] * 6,
            "height": 34,
        },
        cells={
            "values": values,
            "fill_color": fills,
            "font": {"size": 12, "color": INK},
            "align": ["center", "left", "center"] + ["center"] * 6,
            "height": 30,
        },
    ))
    return _base_layout(fig, title="Model Leaderboard", height=max(280, 92 + len(models) * 34))


def _ranked_metric_bar(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50, reverse=True)
    y = [_display_name(m, names) for m in models][::-1]
    fig = go.Figure()
    for metric_name, values, color in [
        ("mAP@50", [m.map50 for m in models][::-1], BLUE),
        ("mAP@50:95", [m.map50_95 for m in models][::-1], PURPLE),
        ("F1", [m.f1 for m in models][::-1], GREEN),
    ]:
        fig.add_trace(go.Bar(
            name=metric_name,
            y=y,
            x=values,
            orientation="h",
            marker={"color": color},
            text=[_fmt(v) for v in values],
            textposition="outside",
            hovertemplate=f"{metric_name}: %{{x:.3f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", xaxis={"range": [0, 1.08], "tickformat": ".0%"})
    return _base_layout(fig, title="Ranked Model Quality", height=max(360, 170 + len(models) * 46))


def _precision_recall_chart(result: Any, names: dict[Path, str]) -> go.Figure:
    models = result.models
    x = [_display_name(m, names) for m in models]
    fig = go.Figure()
    for metric_name, values, color in [
        ("Precision", [m.precision for m in models], BLUE),
        ("Recall", [m.recall for m in models], AMBER),
        ("F1", [m.f1 for m in models], GREEN),
    ]:
        fig.add_trace(go.Bar(
            name=metric_name,
            x=x,
            y=values,
            marker={"color": color},
            text=[_fmt(v) for v in values],
            textposition="outside",
            hovertemplate=f"{metric_name}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", yaxis={"range": [0, 1.08], "tickformat": ".0%"})
    return _base_layout(fig, title="Precision, Recall, and F1 Balance", height=430)


def _size_heatmap(result: Any, names: dict[Path, str]) -> go.Figure:
    models = result.models
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
            [0.0, "#fee2e2"],
            [0.45, "#fef3c7"],
            [0.70, "#dbeafe"],
            [1.0, "#dcfce7"],
        ],
        zmin=0,
        zmax=1,
        colorbar={"title": "mAP@50"},
        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
    ))
    return _base_layout(fig, title="Object Size Sensitivity", height=max(320, 150 + len(models) * 36))


def _quality_counts_bar(result: Any, names: dict[Path, str]) -> go.Figure:
    models = result.models
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
            marker={"color": color},
            hovertemplate=f"{label}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", yaxis_title="Count")
    return _base_layout(fig, title="Detection Outcome Counts", height=410)


def _per_image_distribution(result: Any, names: dict[Path, str]) -> go.Figure:
    fig = go.Figure()
    for i, model in enumerate(result.models):
        fig.add_trace(go.Box(
            name=_display_name(model, names),
            y=[r.f1 for r in model.per_image],
            marker={"color": PALETTE[i % len(PALETTE)]},
            boxmean=True,
            jitter=0.25,
            pointpos=-1.3,
            boxpoints="outliers",
            hovertemplate="F1: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(yaxis={"range": [0, 1.05], "tickformat": ".0%"})
    return _base_layout(fig, title="Per-Image F1 Distribution", height=430)


def _per_image_tables(result: Any, names: dict[Path, str]) -> go.Figure:
    best = max(result.models, key=lambda m: m.map50)
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
            "height": 34,
        },
        cells={
            "values": values,
            "fill_color": [["#fff7ed"] * len(worst)] + [["#ffffff"] * len(worst)] * 7,
            "font": {"size": 11},
            "align": ["left"] + ["center"] * 7,
            "height": 26,
        },
    ))
    title = f"Worst Images for Best Model: {_display_name(best, names)}"
    return _base_layout(fig, title=title, height=max(360, 100 + len(worst) * 30))


def _vector_scatter(vector_data: dict, model_name: str) -> go.Figure:
    if not vector_data or not vector_data.get("x"):
        fig = go.Figure()
        fig.add_annotation(text="No vector data available", showarrow=False)
        return _base_layout(fig, title=f"Image Embedding Analysis: {model_name}", height=430)

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
    marker_sizes = [max(7, min(22, 7 + 15 * (c / max_gt))) for c in gt_counts]
    fig = go.Figure(go.Scatter(
        x=vector_data["x"],
        y=vector_data["y"],
        mode="markers",
        marker={
            "size": marker_sizes,
            "color": vector_data["f1"],
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": "F1"},
            "line": {"width": 0.6, "color": "#ffffff"},
            "opacity": 0.88,
        },
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "F1: %{customdata[1]:.3f}<br>"
            "Precision: %{customdata[2]:.3f}<br>"
            "Recall: %{customdata[3]:.3f}<br>"
            "TP: %{customdata[4]}  FP: %{customdata[5]}  FN: %{customdata[6]}<br>"
            "Mean IoU: %{customdata[7]:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return _base_layout(fig, title=f"Image Embedding Analysis: {model_name}", height=540)


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
    if (el) el.textContent = text;
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
    _setText('gc-f1', 'F1: ' + parseFloat(cd[1]).toFixed(3)
      + '   P: ' + parseFloat(cd[2]).toFixed(3)
      + '   R: ' + parseFloat(cd[3]).toFixed(3));
    _setText('gc-counts', 'TP: ' + cd[4] + '   FP: ' + cd[5] + '   FN: ' + cd[6]);
    _setText('gc-iou', 'Mean IoU: ' + parseFloat(cd[7]).toFixed(3));
    gallery.style.display = 'block';
  });
})();
</script>
"""

_GALLERY_HTML = """
<section id="gallery-panel" class="gallery-panel" style="display:none;">
  <div class="section-title">Selected Image</div>
  <img id="gallery-img" alt="Click a scatter point" />
  <div class="gallery-meta">
    <div id="gc-name" class="gallery-name"></div>
    <div id="gc-f1"></div>
    <div id="gc-counts"></div>
    <div id="gc-iou"></div>
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
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
        '<style>\n'
        '*, *::before, *::after { box-sizing: border-box; }\n'
        f'body {{ font-family: {FONT_FAMILY}; background:#eef2f7; color:{INK}; margin:0; }}\n'
        '.hero { background:#0f172a; color:white; padding:30px 42px 26px; }\n'
        '.hero h1 { margin:0; font-size:28px; line-height:1.15; letter-spacing:0; }\n'
        '.hero p { margin:8px 0 0; color:#cbd5e1; font-size:13px; }\n'
        '.hero-meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }\n'
        '.pill { background:#1e293b; border:1px solid #334155; border-radius:6px; padding:6px 9px; font-size:12px; color:#e2e8f0; }\n'
        '.container { max-width:1480px; margin:0 auto; padding:24px 20px 42px; }\n'
        '.section { background:white; border:1px solid #e5e7eb; border-radius:8px; margin-bottom:18px; box-shadow:0 1px 2px rgba(15,23,42,.05); }\n'
        '.chart-section { padding:12px 14px; }\n'
        '.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; padding:14px; }\n'
        '.kpi-card { border:1px solid #e5e7eb; border-radius:8px; padding:14px; background:#f8fafc; min-height:112px; }\n'
        f'.kpi-label {{ color:{MUTED}; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }}\n'
        '.kpi-value { margin-top:9px; font-size:24px; font-weight:760; color:#0f172a; overflow-wrap:anywhere; }\n'
        f'.kpi-note {{ margin-top:8px; color:{MUTED}; font-size:12px; line-height:1.35; }}\n'
        '.model-list { list-style:none; padding:0; margin:16px 0 0; display:grid; gap:8px; }\n'
        '.model-list li { display:grid; grid-template-columns:minmax(160px,260px) 1fr auto; gap:12px; align-items:center; background:#111827; border:1px solid #334155; border-radius:6px; padding:8px 10px; font-size:12px; }\n'
        '.model-name { color:white; font-weight:700; }\n'
        '.model-path { color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }\n'
        '.model-score { color:#cbd5e1; white-space:nowrap; }\n'
        '.section-title { font-weight:760; font-size:15px; margin-bottom:10px; }\n'
        '.gallery-panel { max-width:560px; margin:0 auto 18px; background:white; border:1px solid #e5e7eb; border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(15,23,42,.05); }\n'
        '.gallery-panel img { width:100%; border-radius:6px; background:#f1f5f9; min-height:160px; }\n'
        f'.gallery-meta {{ margin-top:10px; font-size:12px; color:{MUTED}; line-height:1.8; }}\n'
        '.gallery-name { color:#111827; font-weight:700; }\n'
        f'.footer {{ text-align:center; color:{MUTED}; font-size:11px; padding:18px; }}\n'
        '@media (max-width: 780px) { .hero { padding:24px 18px; } .model-list li { grid-template-columns:1fr; } .container { padding:16px 10px 30px; } }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<header class="hero">\n'
        '<h1>YOLOMatic Benchmark Report</h1>\n'
        '<p>Generated ' + escape(timestamp_str) + ' · Validation ' + escape(val_dir_str) + '</p>\n'
        '<div class="hero-meta">'
        '<span class="pill">Models: ' + str(len(result.models)) + '</span>'
        '<span class="pill">Confidence: ' + conf + '</span>'
        '<span class="pill">IoU match: ' + iou + '</span>'
        '<span class="pill">Best: ' + escape(best_name) + '</span>'
        '</div>\n'
        '<ul class="model-list">' + model_list_items + '</ul>\n'
        '</header>\n'
        '<main class="container">\n'
        + "\n".join(sections) + "\n"
        + _GALLERY_HTML + "\n"
        '</main>\n'
        '<div class="footer">Generated by <strong>YOLOMatic</strong> benchmark engine</div>\n'
        + _GALLERY_JS
        + '\n</body>\n</html>'
    )

    report_path.write_text(full_html, encoding="utf-8")
    return report_path
