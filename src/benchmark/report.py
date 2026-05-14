"""Plotly HTML report generation for benchmark results."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _metric_color(v: float) -> str:
    r = int(max(0, min(255, 255 * (1 - v) * 2)))
    g = int(max(0, min(255, 255 * v * 2)))
    return f"#{r:02x}{g:02x}40"


def _cell_colors_for_column(values: list[float]) -> list[str]:
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return [_metric_color((v - lo) / span) for v in values]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _summary_cards(result) -> go.Figure:
    best = max(result.models, key=lambda m: m.map50)

    indicators = [
        ("Best Model", best.weights_path.stem, None),
        ("mAP@50", _fmt(best.map50), best.map50),
        ("mAP@50:95", _fmt(best.map50_95), best.map50_95),
        ("F1 Score", _fmt(best.f1), best.f1),
        ("Precision", _pct(best.precision), best.precision),
        ("Recall", _pct(best.recall), best.recall),
    ]

    fig = make_subplots(
        rows=1, cols=len(indicators),
        specs=[[{"type": "indicator"}] * len(indicators)],
    )

    for col, (title, value, gauge_val) in enumerate(indicators, 1):
        kwargs: dict = {
            "title": {"text": title, "font": {"size": 13}},
            "number": {"font": {"size": 22}},
        }
        if gauge_val is not None:
            kwargs["mode"] = "gauge+number"
            kwargs["gauge"] = {
                "axis": {"range": [0, 1], "tickformat": ".0%"},
                "bar": {"color": _metric_color(gauge_val)},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e0e0e0",
            }
            kwargs["value"] = gauge_val
            kwargs["number"] = {"valueformat": ".3f", "font": {"size": 22}}
        else:
            kwargs["mode"] = "number"
            kwargs["value"] = 0
            kwargs["number"] = {"prefix": value, "font": {"size": 18}}

        fig.add_trace(go.Indicator(**kwargs), row=1, col=col)

    fig.update_layout(
        height=200,
        paper_bgcolor="#f8f9fa",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": "Best Model Summary", "font": {"size": 15}},
        font={"family": "Inter, sans-serif"},
    )
    return fig


def _comparison_table(result) -> go.Figure:
    models = result.models
    model_names = [m.weights_path.stem for m in models]

    metrics = {
        "mAP@50": [m.map50 for m in models],
        "mAP@75": [m.map75 for m in models],
        "mAP@50:95": [m.map50_95 for m in models],
        "F1": [m.f1 for m in models],
        "Precision": [m.precision for m in models],
        "Recall": [m.recall for m in models],
    }

    header_vals = ["Model"] + list(metrics.keys())
    cell_vals = [model_names] + [[_fmt(v) for v in vals] for vals in metrics.values()]
    fill_colors = [["#f0f4ff"] * len(models)]
    for vals in metrics.values():
        fill_colors.append(_cell_colors_for_column(vals))

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#2c3e50",
            font={"color": "white", "size": 13},
            align="center",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=fill_colors,
            font={"size": 12},
            align="center",
            height=28,
        ),
    ))
    fig.update_layout(
        title={"text": "Model Comparison", "font": {"size": 15}},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor="#f8f9fa",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def _size_grouped_bar(result) -> go.Figure:
    buckets = ["Small (<32²)", "Medium (32²–96²)", "Large (>96²)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

    fig = go.Figure()
    for i, model in enumerate(result.models):
        fig.add_trace(go.Bar(
            name=model.weights_path.stem,
            x=buckets,
            y=[model.small.map50, model.medium.map50, model.large.map50],
            marker_color=colors[i % len(colors)],
            text=[_fmt(v) for v in [model.small.map50, model.medium.map50, model.large.map50]],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title={"text": "mAP@50 by Object Size", "font": {"size": 15}},
        xaxis_title="Object Size Bucket",
        yaxis_title="mAP@50",
        yaxis={"range": [0, 1.15], "tickformat": ".2f", "gridcolor": "#e0e0e0"},
        xaxis={"gridcolor": "#e0e0e0"},
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 60, "r": 20, "t": 80, "b": 60},
        font={"family": "Inter, sans-serif", "size": 12},
    )
    return fig


def _tp_fp_fn_bar(result) -> go.Figure:
    best = max(result.models, key=lambda m: m.map50)
    total_tp = sum(r.tp for r in best.per_image)
    total_fp = sum(r.fp for r in best.per_image)
    total_fn = sum(r.fn for r in best.per_image)

    fig = go.Figure()
    for val, label, color in [
        (total_tp, "True Positives", "#2ecc71"),
        (total_fp, "False Positives", "#e74c3c"),
        (total_fn, "False Negatives", "#f39c12"),
    ]:
        fig.add_trace(go.Bar(
            y=[label], x=[val], orientation="h",
            marker_color=color, text=[str(val)], textposition="auto",
            name=label,
        ))

    fig.update_layout(
        title={"text": f"Detection Quality — {best.weights_path.stem}", "font": {"size": 15}},
        xaxis_title="Count",
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="white",
        xaxis={"gridcolor": "#e0e0e0"},
        showlegend=False,
        margin={"l": 160, "r": 20, "t": 60, "b": 40},
        font={"family": "Inter, sans-serif", "size": 12},
        height=200,
    )
    return fig


def _per_image_tables(result) -> go.Figure:
    best = max(result.models, key=lambda m: m.map50)
    sorted_imgs = sorted(best.per_image, key=lambda r: r.f1)
    worst = sorted_imgs[:20]
    best_imgs = sorted_imgs[-20:][::-1]

    def _table_vals(imgs: list) -> tuple:
        return (
            [r.image_path.name for r in imgs],
            [_fmt(r.f1) for r in imgs],
            [_fmt(r.precision) for r in imgs],
            [_fmt(r.recall) for r in imgs],
            [str(r.tp) for r in imgs],
            [str(r.fp) for r in imgs],
            [str(r.fn) for r in imgs],
            [_fmt(r.mean_iou) for r in imgs],
        )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Worst 20 Images (lowest F1)", "Best 20 Images (highest F1)"),
        specs=[[{"type": "table"}, {"type": "table"}]],
    )
    headers = ["Image", "F1", "P", "R", "TP", "FP", "FN", "mIoU"]

    for col, imgs in [(1, worst), (2, best_imgs)]:
        fig.add_trace(go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in headers],
                fill_color="#2c3e50",
                font={"color": "white", "size": 11},
                align="center",
            ),
            cells=dict(
                values=list(_table_vals(imgs)),
                fill_color=[["#fff0f0" if col == 1 else "#f0fff0"] * len(imgs)] * len(headers),
                font={"size": 10},
                align=["left"] + ["center"] * 7,
                height=22,
            ),
        ), row=1, col=col)

    fig.update_layout(
        title={"text": "Per-Image Performance Ranking", "font": {"size": 15}},
        paper_bgcolor="#f8f9fa",
        font={"family": "Inter, sans-serif"},
        margin={"l": 10, "r": 10, "t": 80, "b": 10},
        height=600,
    )
    return fig


def _vector_scatter(vector_data: dict, model_name: str) -> go.Figure:
    if not vector_data or not vector_data.get("x"):
        fig = go.Figure()
        fig.add_annotation(text="No vector data available", showarrow=False)
        return fig

    has_thumbnails = bool(vector_data.get("thumbnails"))
    custom_data = []
    for i in range(len(vector_data["x"])):
        row = [
            vector_data["image_name"][i],
            vector_data["f1"][i],
            vector_data["precision"][i],
            vector_data["recall"][i],
            vector_data["tp"][i],
            vector_data["fp"][i],
            vector_data["fn"][i],
            vector_data["mean_iou"][i],
            vector_data["thumbnails"][i] if has_thumbnails else "",
        ]
        custom_data.append(row)

    gt_counts = vector_data["gt_count"]
    max_gt = max(gt_counts) if gt_counts else 1
    marker_sizes = [max(6, min(20, 6 + 14 * (c / max_gt))) for c in gt_counts]

    fig = go.Figure(go.Scatter(
        x=vector_data["x"],
        y=vector_data["y"],
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=vector_data["f1"],
            colorscale="RdYlGn",
            cmin=0, cmax=1,
            colorbar=dict(title="F1", tickformat=".2f"),
            line=dict(width=0.5, color="#333"),
            opacity=0.8,
        ),
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "F1: %{customdata[1]:.3f}<br>"
            "Precision: %{customdata[2]:.3f}<br>"
            "Recall: %{customdata[3]:.3f}<br>"
            "TP: %{customdata[4]}  FP: %{customdata[5]}  FN: %{customdata[6]}<br>"
            "Mean IoU: %{customdata[7]:.3f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title={"text": f"Vector Analysis — {model_name} (colour = F1)", "font": {"size": 15}},
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="white",
        xaxis={"gridcolor": "#e8e8e8", "zeroline": False},
        yaxis={"gridcolor": "#e8e8e8", "zeroline": False},
        margin={"l": 60, "r": 20, "t": 80, "b": 60},
        font={"family": "Inter, sans-serif", "size": 12},
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Gallery — safe DOM-based JS (no innerHTML with user data)
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
<div id="gallery-panel" style="display:none;margin:20px auto;max-width:480px;
     background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;
     box-shadow:0 2px 8px rgba(0,0,0,.10);font-family:Inter,sans-serif;">
  <p style="margin:0 0 10px;font-weight:600;font-size:13px;color:#333;">Selected Image</p>
  <img id="gallery-img" alt="Click a scatter point"
       style="width:100%;border-radius:4px;background:#f0f0f0;min-height:140px;" />
  <div style="margin-top:10px;font-size:12px;color:#555;line-height:1.8;">
    <div id="gc-name"  style="font-weight:600;"></div>
    <div id="gc-f1"></div>
    <div id="gc-counts"></div>
    <div id="gc-iou"></div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_benchmark_report(result, output_dir: Path) -> Path:
    """Generate self-contained HTML report and return its path."""
    from .vector_analysis import build_vector_data
    from .thumbnails import make_thumbnail_b64

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_report_{ts}.html"

    best = max(result.models, key=lambda m: m.map50)

    def _thumb_fn(r):
        return make_thumbnail_b64(
            r.image_path, r.raw_gts, r.raw_preds,
            size=result.config.max_thumbnail_size,
            task=best.task,
        )

    thumbnail_fn = _thumb_fn if result.config.generate_thumbnails else None
    vector_data = build_vector_data(best, thumbnail_fn=thumbnail_fn)

    figs = {
        "summary": _summary_cards(result),
        "comparison": _comparison_table(result),
        "size_bar": _size_grouped_bar(result),
        "tp_fp_fn": _tp_fp_fn_bar(result),
        "ranking": _per_image_tables(result),
        "scatter": _vector_scatter(vector_data, best.weights_path.stem),
    }

    sections: list[str] = []
    for key, fig in figs.items():
        div_id = "scatter-plot" if key == "scatter" else f"plot-{key}"
        html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=div_id,
            config={"displaylogo": False, "responsive": True},
        )
        sections.append(f'<div class="section">{html}</div>')

    model_list_items = "".join(
        "<li><code>" + m.weights_path.name + "</code>"
        + " — " + m.task
        + ", mAP@50=" + _fmt(m.map50)
        + ", F1=" + _fmt(m.f1) + "</li>"
        for m in result.models
    )

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_dir_str = str(result.config.validation_dir)

    # Escape for HTML attribute/text contexts
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

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
        'body { font-family: Inter, system-ui, sans-serif; background: #f0f2f5;'
        ' color: #1a1a2e; margin: 0; padding: 0; }\n'
        '.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);'
        ' color: white; padding: 28px 40px 24px; border-bottom: 3px solid #0f3460; }\n'
        '.header h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }\n'
        '.header p { margin: 6px 0 0; font-size: 13px; opacity: 0.75; }\n'
        '.header ul { margin: 10px 0 0; padding: 0 0 0 18px; font-size: 12px; opacity: 0.8; }\n'
        '.container { max-width: 1400px; margin: 0 auto; padding: 24px 20px 40px; }\n'
        '.section { background: white; border-radius: 10px; padding: 20px;'
        ' margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }\n'
        '.footer { text-align: center; font-size: 11px; color: #888;'
        ' padding: 16px; border-top: 1px solid #e0e0e0; margin-top: 16px; }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<div class="header">\n'
        '  <h1>YOLOMatic Benchmark Report</h1>\n'
        '  <p>Generated: ' + _esc(timestamp_str) + ' &nbsp;|&nbsp; '
        'Validation: ' + _esc(val_dir_str) + '</p>\n'
        '  <ul>' + model_list_items + '</ul>\n'
        '</div>\n'
        '<div class="container">\n'
        + '\n'.join(sections) + '\n'
        + _GALLERY_HTML + '\n'
        '</div>\n'
        '<div class="footer">Generated by <strong>YOLOMatic</strong> benchmark engine</div>\n'
        + _GALLERY_JS
        + '\n</body>\n</html>'
    )

    report_path.write_text(full_html, encoding="utf-8")
    return report_path
