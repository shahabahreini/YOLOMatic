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


def _interpolate_color(v: float, min_v: float, max_v: float, is_highlight: bool = False, inverse: bool = False) -> str:
    """Interpolate between red and green based on relative performance."""
    if max_v == min_v:
        return "#ffffff"
    
    # Normalised score 0.0 to 1.0
    s = (v - min_v) / (max_v - min_v)
    if inverse:
        s = 1.0 - s
    
    if is_highlight:
        # High-contrast Blue scale
        # rgb(240, 249, 255) to rgb(7, 89, 133)
        r = int(240 + s * (7 - 240))
        g = int(249 + s * (89 - 249))
        b = int(255 + s * (133 - 255))
        return f"rgb({r},{g},{b})"

    # Vibrant Red-Yellow-Green scale
    # Red: rgb(254, 226, 226) -> Yellow: rgb(254, 240, 138) -> Green: rgb(187, 247, 208)
    if s < 0.5:
        s2 = s * 2
        r = int(254 + s2 * (254 - 254))
        g = int(226 + s2 * (240 - 226))
        b = int(226 + s2 * (138 - 226))
    else:
        s2 = (s - 0.5) * 2
        r = int(254 + s2 * (187 - 254))
        g = int(240 + s2 * (247 - 240))
        b = int(138 + s2 * (208 - 138))
    
    return f"rgb({r},{g},{b})"


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
    best_50_95 = max(result.models, key=lambda m: m.map50_95)
    best_50 = max(result.models, key=lambda m: m.map50)
    best_f1 = max(result.models, key=lambda m: m.f1)
    best_fps = max(result.models, key=lambda m: m.fps) if any(m.fps > 0 for m in result.models) else None
    
    def render_stat(model: Any, label: str, value: float, suffix: str = ""):
        if model is None: return ""
        m_name = escape(_display_name(model, names))
        return f"""
        <div class="stat-item">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{_fmt(value)}{suffix}</div>
            <div class="stat-model">
                <span class="stat-model-label">Best Model:</span>
                <span class="stat-model-name">{m_name}</span>
            </div>
        </div>
        """

    stat_50_95 = render_stat(best_50_95, "Primary mAP@50:95", best_50_95.map50_95)
    stat_50 = render_stat(best_50, "Secondary mAP@50", best_50.map50)
    stat_f1 = render_stat(best_f1, "Detection F1", best_f1.f1)
    stat_fps = render_stat(best_fps, "Throughput (FPS)", best_fps.fps) if best_fps else ""
    
    return f"""
    <section class="summary-section">
        <div class="section-header">
            <h2 class="section-title">Key Performance Summary</h2>
            <p class="section-subtitle">Validated performance of top models across primary metrics</p>
        </div>
        <div class="summary-grid">
            {stat_50_95}
            {stat_50}
            {stat_f1}
            {stat_fps}
        </div>
    </section>
    """


def _comparison_table(result: Any, names: dict[Path, str]) -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    model_names = [_display_name(m, names) for m in models]
    
    metrics_data = [
        ("mAP@50:95", [m.map50_95 for m in models], True, False),
        ("mAP@50", [m.map50 for m in models], False, False),
        ("mAP@75", [m.map75 for m in models], False, False),
        ("F1", [m.f1 for m in models], False, False),
        ("Precision", [m.precision for m in models], False, False),
        ("Recall", [m.recall for m in models], False, False),
        ("FPS", [m.fps for m in models], True, False),
        ("Latency (ms)", [m.inference_time_ms for m in models], False, True),
    ]
    
    headers = ["Rank", "Model", "Task"] + [m[0] for m in metrics_data]
    values = [
        [str(i) for i in range(1, len(models) + 1)],
        model_names,
        [m.task for m in models],
    ]
    fills = [
        ["#ffffff"] * len(models),
        ["#ffffff"] * len(models),
        ["#ffffff"] * len(models),
    ]
    
    for label, vals, highlight, inverse in metrics_data:
        values.append([_fmt(v) for v in vals])
        if not vals or len(vals) < 2:
            fills.append(["#ffffff"] * len(models))
            continue
            
        # Calculate min/max for THIS column only
        min_v, max_v = min(vals), max(vals)
        fills.append([_interpolate_color(v, min_v, max_v, is_highlight=highlight, inverse=inverse) for v in vals])

    # Header row (40px) + cell rows (32px each) + title/margin overhead (120px)
    HEADER_H = 40
    ROW_H = 32
    OVERHEAD = 120  # title bar + top/bottom margins
    target_height = max(280, OVERHEAD + HEADER_H + len(models) * ROW_H)
    if len(models) > 10:
        target_height = max(target_height, 500)

    fig = go.Figure(go.Table(
        columnwidth=[50, 240, 100, 100, 90, 90, 80, 90, 90, 80, 100],
        header={
            "values": [f"<b>{h}</b>" for h in headers],
            "fill_color": "#0f172a",
            "font": {"color": "white", "size": 12},
            "align": ["center", "left", "center"] + ["center"] * 8,
            "height": 40,
        },
        cells={
            "values": values,
            "fill_color": fills,
            "font": {"size": 12, "color": INK},
            "align": ["center", "left", "center"] + ["center"] * 8,
            "height": ROW_H,
            "line": {"color": "rgba(0,0,0,0.03)", "width": 1},
        },
    ))
    # Use tighter margins so table cells are not obscured
    fig.update_layout(
        title={
            "text": "<b>Relative Performance Leaderboard</b>",
            "font": {"size": 16, "color": INK},
            "x": 0.02,
            "y": 0.98,
            "yanchor": "top",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "size": 12, "color": INK},
        margin={"l": 0, "r": 0, "t": 60, "b": 10},
        height=target_height,
        hoverlabel={"bgcolor": "#111827", "font": {"color": "white", "size": 13}},
    )
    return fig


def _ranked_metric_bar(result: Any, names: dict[Path, str], group_by: str = "model") -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    fig = go.Figure()
    
    if group_by == "model":
        y = [_display_name(m, names) for m in models][::-1]
        metrics = [
            ("mAP@50:95", [m.map50_95 for m in models][::-1], BLUE),
            ("mAP@50", [m.map50 for m in models][::-1], CYAN),
            ("F1", [m.f1 for m in models][::-1], GREEN),
        ]
        for name, vals, color in metrics:
            fig.add_trace(go.Bar(
                name=name,
                y=y,
                x=vals,
                orientation="h",
                marker={"color": color, "line": {"width": 0}},
                text=[_fmt(v) for v in vals],
                textposition="outside",
                hovertemplate=f"<b>{name}</b>: %{{x:.3f}}<extra></extra>",
            ))
        height = max(360, 170 + len(models) * 54)
    else:
        # Group by Metric
        metrics_list = ["mAP@50:95", "mAP@50", "F1"]
        y = metrics_list[::-1]
        for i, model in enumerate(models):
            name = _display_name(model, names)
            vals = [model.map50_95, model.map50, model.f1][::-1]
            fig.add_trace(go.Bar(
                name=name,
                y=y,
                x=vals,
                orientation="h",
                marker={"color": PALETTE[i % len(PALETTE)], "line": {"width": 0}},
                text=[_fmt(v) for v in vals],
                textposition="outside",
                hovertemplate=f"<b>{name}</b><br>Metric: %{{y}}<br>Value: %{{x:.3f}}<extra></extra>",
            ))
        height = max(360, 200 + len(models) * 40)

    fig.update_layout(barmode="group", xaxis={"range": [0, 1.12], "tickformat": ".0%"}, bargap=0.2, bargroupgap=0.1)
    return _base_layout(fig, title="Ranked Model Quality", height=height)


def _precision_recall_chart(result: Any, names: dict[Path, str], group_by: str = "model") -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    fig = go.Figure()
    
    if group_by == "model":
        x = [_display_name(m, names) for m in models]
        metrics = [
            ("Precision", [m.precision for m in models], BLUE),
            ("Recall", [m.recall for m in models], AMBER),
            ("F1 Score", [m.f1 for m in models], GREEN),
        ]
        for name, vals, color in metrics:
            fig.add_trace(go.Bar(
                name=name,
                x=x,
                y=vals,
                marker={"color": color, "line": {"width": 0}},
                text=[_fmt(v) for v in vals],
                textposition="outside",
                hovertemplate=f"<b>{name}</b>: %{{y:.3f}}<extra></extra>",
            ))
    else:
        # Group by Metric
        metrics_list = ["Precision", "Recall", "F1 Score"]
        for i, model in enumerate(models):
            name = _display_name(model, names)
            vals = [model.precision, model.recall, model.f1]
            fig.add_trace(go.Bar(
                name=name,
                x=metrics_list,
                y=vals,
                marker={"color": PALETTE[i % len(PALETTE)], "line": {"width": 0}},
                text=[_fmt(v) for v in vals],
                textposition="outside",
                hovertemplate=f"<b>{name}</b><br>Metric: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>",
            ))
            
    fig.update_layout(barmode="group", yaxis={"range": [0, 1.12], "tickformat": ".0%"}, bargap=0.2)
    return _base_layout(fig, title="Detection Balance (P, R, F1)", height=440)


def _size_sensitivity_bar(result: Any, names: dict[Path, str], group_by: str = "model") -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    fig = go.Figure()
    
    if group_by == "model":
        y = [_display_name(m, names) for m in models][::-1]
        categories = [
            ("Small (<32²)", [m.small.map50_95 for m in models][::-1], BLUE),
            ("Medium (32²-96²)", [m.medium.map50_95 for m in models][::-1], CYAN),
            ("Large (>96²)", [m.large.map50_95 for m in models][::-1], PURPLE),
            ("Overall", [m.map50_95 for m in models][::-1], GREEN),
        ]
        for label, values, color in categories:
            fig.add_trace(go.Bar(
                name=label,
                y=y,
                x=values,
                orientation="h",
                marker={"color": color, "line": {"width": 0}},
                text=[_fmt(v) for v in values],
                textposition="outside",
                hovertemplate=f"<b>{label}</b>: %{{x:.3f}}<extra></extra>",
            ))
        height = max(400, 180 + len(models) * 75)
    else:
        # Group by Metric (Size category)
        categories_list = ["Small (<32²)", "Medium (32²-96²)", "Large (>96²)", "Overall"]
        y = categories_list[::-1]
        for i, model in enumerate(models):
            name = _display_name(model, names)
            vals = [model.small.map50_95, model.medium.map50_95, model.large.map50_95, model.map50_95][::-1]
            fig.add_trace(go.Bar(
                name=name,
                y=y,
                x=vals,
                orientation="h",
                marker={"color": PALETTE[i % len(PALETTE)], "line": {"width": 0}},
                text=[_fmt(v) for v in vals],
                textposition="outside",
                hovertemplate=f"<b>{name}</b><br>Size: %{{y}}<br>Value: %{{x:.3f}}<extra></extra>",
            ))
        height = max(400, 220 + len(models) * 45)
    
    fig.update_layout(
        barmode="group",
        xaxis={"range": [0, 1.15], "tickformat": ".0%"},
        bargap=0.2,
        bargroupgap=0.05
    )
    return _base_layout(
        fig, 
        title="Object Size Sensitivity (mAP@50:95)", 
        height=height
    )


def _size_f1_bar(result: Any, names: dict[Path, str], group_by: str = "model") -> go.Figure:
    models = sorted(result.models, key=lambda m: m.f1, reverse=True)
    fig = go.Figure()
    
    if group_by == "model":
        y = [_display_name(m, names) for m in models][::-1]
        categories = [
            ("Small (<32²)", [m.small.f1 for m in models][::-1], BLUE),
            ("Medium (32²-96²)", [m.medium.f1 for m in models][::-1], CYAN),
            ("Large (>96²)", [m.large.f1 for m in models][::-1], PURPLE),
            ("Overall", [m.f1 for m in models][::-1], GREEN),
        ]
        for label, values, color in categories:
            fig.add_trace(go.Bar(
                name=label,
                y=y,
                x=values,
                orientation="h",
                marker={"color": color, "line": {"width": 0}},
                text=[_fmt(v) for v in values],
                textposition="outside",
                hovertemplate=f"<b>{label}</b>: %{{x:.3f}}<extra></extra>",
            ))
        height = max(400, 180 + len(models) * 75)
    else:
        # Group by Metric (Size category)
        categories_list = ["Small (<32²)", "Medium (32²-96²)", "Large (>96²)", "Overall"]
        y = categories_list[::-1]
        for i, model in enumerate(models):
            name = _display_name(model, names)
            vals = [model.small.f1, model.medium.f1, model.large.f1, model.f1][::-1]
            fig.add_trace(go.Bar(
                name=name,
                y=y,
                x=vals,
                orientation="h",
                marker={"color": PALETTE[i % len(PALETTE)], "line": {"width": 0}},
                text=[_fmt(v) for v in vals],
                textposition="outside",
                hovertemplate=f"<b>{name}</b><br>Size: %{{y}}<br>Value: %{{x:.3f}}<extra></extra>",
            ))
        height = max(400, 220 + len(models) * 45)
    
    fig.update_layout(
        barmode="group",
        xaxis={"range": [0, 1.15], "tickformat": ".0%"},
        bargap=0.2,
        bargroupgap=0.05
    )
    return _base_layout(
        fig, 
        title="Detection Performance by Object Size (F1 Score)", 
        height=height
    )


def _quality_counts_bar(result: Any, names: dict[Path, str], group_by: str = "model") -> go.Figure:
    models = sorted(result.models, key=lambda m: m.map50_95, reverse=True)
    fig = go.Figure()
    
    if group_by == "model":
        x = [_display_name(m, names) for m in models]
        counts = [
            ("TP", [sum(r.tp for r in m.per_image) for m in models], GREEN),
            ("FP", [sum(r.fp for r in m.per_image) for m in models], RED),
            ("FN", [sum(r.fn for r in m.per_image) for m in models], AMBER),
        ]
        for label, vals, color in counts:
            fig.add_trace(go.Bar(
                name=label,
                x=x,
                y=vals,
                marker={"color": color, "line": {"width": 0}},
                hovertemplate=f"<b>{label}</b>: %{{y}}<extra></extra>",
            ))
        barmode = "stack"
    else:
        # Group by Metric (Outcome)
        outcomes = ["TP", "FP", "FN"]
        for i, model in enumerate(models):
            name = _display_name(model, names)
            tp = sum(r.tp for r in model.per_image)
            fp = sum(r.fp for r in model.per_image)
            fn = sum(r.fn for r in model.per_image)
            fig.add_trace(go.Bar(
                name=name,
                x=outcomes,
                y=[tp, fp, fn],
                marker={"color": PALETTE[i % len(PALETTE)], "line": {"width": 0}},
                hovertemplate=f"<b>{name}</b><br>Outcome: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))
        barmode = "group"
        
    fig.update_layout(barmode=barmode, yaxis_title="Instance Count", bargap=0.3)
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
            vector_data["preds"][i] if "preds" in vector_data else [],
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
  var canvas = document.getElementById('gallery-canvas');
  var ctx = canvas.getContext('2d');
  var toggleConf = document.getElementById('toggle-conf');
  
  var currentPreds = [];
  if (!scatter) return;

  function _setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = text;
  }

  function drawConfidences() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!toggleConf.checked || !currentPreds || currentPreds.length === 0) return;

    var iw = galleryImg.naturalWidth;
    var ih = galleryImg.naturalHeight;
    var cw = galleryImg.clientWidth;
    var ch = galleryImg.clientHeight;
    
    // Calculate actual displayed image dimensions (letterbox aware)
    var imgRatio = iw / ih;
    var containerRatio = cw / ch;
    var displayW, displayH, offsetX, offsetY;
    
    if (imgRatio > containerRatio) {
      displayW = cw;
      displayH = cw / imgRatio;
      offsetX = 0;
      offsetY = (ch - displayH) / 2;
    } else {
      displayH = ch;
      displayW = ch * imgRatio;
      offsetY = 0;
      offsetX = (cw - displayW) / 2;
    }

    ctx.font = 'bold 10px sans-serif';
    ctx.textBaseline = 'top';

    currentPreds.forEach(function(p) {
      var x1 = p[0] * displayW + offsetX;
      var y1 = p[1] * displayH + offsetY;
      var conf = p[4];
      
      var text = (conf * 100).toFixed(0) + '%';
      var tw = ctx.measureText(text).width;
      
      ctx.fillStyle = 'rgba(220, 0, 0, 0.8)';
      ctx.fillRect(x1, y1 - 12, tw + 4, 12);
      ctx.fillStyle = 'white';
      ctx.fillText(text, x1 + 2, y1 - 11);
    });
  }

  galleryImg.onload = function() {
    canvas.width = galleryImg.clientWidth;
    canvas.height = galleryImg.clientHeight;
    drawConfidences();
  };

  toggleConf.onchange = drawConfidences;
  window.addEventListener('resize', function() {
    canvas.width = galleryImg.clientWidth;
    canvas.height = galleryImg.clientHeight;
    drawConfidences();
  });

  scatter.on('plotly_click', function(data) {
    var pt = data.points[0];
    var cd = pt.customdata;
    var thumb = cd[8];
    currentPreds = cd[9] || [];

    if (thumb && thumb.length > 0) {
      galleryImg.src = 'data:image/png;base64,' + thumb;
      galleryImg.alt = cd[0];
    } else {
      galleryImg.removeAttribute('src');
      galleryImg.alt = 'No thumbnail available';
      currentPreds = [];
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
<section id="gallery-panel" class="gallery-panel" style="display:none; max-width: 1000px; margin: 0 auto 32px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 32px; box-shadow: var(--shadow);">
  <div class="gallery-header">
    <div class="section-title" style="margin:0;">Visual Inspector</div>
    <div class="gallery-controls">
      <div class="gallery-legend">
        <div class="legend-item"><div class="legend-color gt"></div> Ground Truth</div>
        <div class="legend-item"><div class="legend-color pred"></div> Prediction</div>
      </div>
      <label class="toggle-container">
        <input type="checkbox" id="toggle-conf" checked> Show Confidence
      </label>
    </div>
  </div>
  <div class="gallery-viewer">
    <img id="gallery-img" alt="Click a scatter point" />
    <canvas id="gallery-canvas"></canvas>
  </div>
  <div class="gallery-meta" style="margin-top: 24px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; border-top: 1px solid var(--border); padding-top: 24px;">
    <div id="gc-name" style="grid-column: 1 / -1; color: var(--ink); font-weight: 800; font-size: 18px;"></div>
    <div id="gc-f1" class="gallery-item" style="font-size: 14px; color: var(--muted);"></div>
    <div id="gc-pr" class="gallery-item" style="font-size: 14px; color: var(--muted);"></div>
    <div id="gc-counts" class="gallery-item" style="font-size: 14px; color: var(--muted);"></div>
    <div id="gc-iou" class="gallery-item" style="font-size: 14px; color: var(--muted);"></div>
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

    # Charts that support switchable grouping
    switchable_keys = {"ranked", "prf", "size", "size_f1", "counts"}
    
    figs_model = {
        "leaderboard": _comparison_table(result, names),
        "ranked": _ranked_metric_bar(result, names, group_by="model"),
        "prf": _precision_recall_chart(result, names, group_by="model"),
        "size": _size_sensitivity_bar(result, names, group_by="model"),
        "size_f1": _size_f1_bar(result, names, group_by="model"),
        "counts": _quality_counts_bar(result, names, group_by="model"),
        "distribution": _per_image_distribution(result, names),
        "ranking": _per_image_tables(result, names),
        "scatter": _vector_scatter(vector_data, best_name),
    }
    
    # Map internal keys to their generator functions
    metric_generators = {
        "ranked": _ranked_metric_bar,
        "prf": _precision_recall_chart,
        "size": _size_sensitivity_bar,
        "size_f1": _size_f1_bar,
        "counts": _quality_counts_bar,
    }

    figs_metric = {
        k: metric_generators[k](result, names, group_by="metric")
        for k in switchable_keys
    }

    sections = [_summary_cards_html(result, names)]
    
    all_keys = ["leaderboard", "ranked", "prf", "size", "size_f1", "counts", "distribution", "ranking", "scatter"]
    for key in all_keys:
        div_id_base = "scatter-plot" if key == "scatter" else f"plot-{key}"
        is_table = key in ("leaderboard", "ranking")
        
        if key in switchable_keys:
            # Generate BOTH model and metric versions
            html_model = figs_model[key].to_html(
                full_html=False, include_plotlyjs=False, div_id=f"{div_id_base}-model",
                config={"displaylogo": False, "responsive": True}
            )
            html_metric = figs_metric[key].to_html(
                full_html=False, include_plotlyjs=False, div_id=f"{div_id_base}-metric",
                config={"displaylogo": False, "responsive": True}
            )
            sections.append(f"""
            <section class="section chart-section">
                <div class="group-by-model">{html_model}</div>
                <div class="group-by-metric" style="display:none">{html_metric}</div>
            </section>
            """)
        else:
            fig = figs_model[key]
            html = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=div_id_base,
                config={
                    "displaylogo": False, 
                    "responsive": not is_table,
                    "displayModeBar": not is_table
                },
            )
            sections.append(f'<section class="section chart-section">{html}</section>')

    # Redesigned minimalist header
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    val_dir_str = str(result.config.validation_dir)
    conf = _fmt(result.config.conf_threshold, 2)
    iou = _fmt(result.config.iou_threshold, 2)

    full_html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>YOLOMatic Report</title>\n'
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
        '  --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);\n'
        '  --radius: 12px;\n'
        '  --primary: #0284c7;\n'
        '  --primary-dark: #0369a1;\n'
        '}\n'
        '*, *::before, *::after { box-sizing: border-box; }\n'
        f'body {{ font-family: {FONT_FAMILY}; background: var(--bg); color: var(--ink); margin: 0; line-height: 1.5; -webkit-font-smoothing: antialiased; }}\n'
        '.top-bar { background: #0f172a; color: white; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }\n'
        '.top-bar h1 { margin: 0; font-size: 22px; font-weight: 800; letter-spacing: -0.03em; display: flex; align-items: center; gap: 8px; }\n'
        '.top-bar h1 span { color: #38bdf8; }\n'
        '.top-meta { display: flex; gap: 24px; font-size: 13px; color: #94a3b8; align-items: center; }\n'
        '.top-meta-item { display: flex; align-items: center; gap: 6px; }\n'
        '.top-meta-item strong { color: #f1f5f9; }\n'
        '.container { max-width: 1400px; margin: 0 auto; padding: 32px 24px 64px; }\n'
         '.summary-section { margin-bottom: 32px; }\n'
        '.section-header { margin-bottom: 16px; }\n'
        '.section-title { font-size: 20px; font-weight: 800; color: #1e293b; margin: 0; letter-spacing: -0.02em; }\n'
        '.section-subtitle { font-size: 13px; color: #64748b; margin: 2px 0 0; }\n'
        '.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); }\n'
        '.stat-item { background: white; padding: 24px; display: flex; flex-direction: column; justify-content: center; }\n'
        '.stat-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }\n'
        '.stat-value { font-size: 36px; font-weight: 900; color: var(--primary); line-height: 1; margin: 4px 0; }\n'
        '.stat-model { display: flex; align-items: center; gap: 6px; margin-top: 12px; font-size: 13px; }\n'
        '.stat-model-label { color: var(--muted); font-weight: 500; }\n'
        '.stat-model-name { color: var(--ink); font-weight: 700; }\n'

        '.section { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 32px; box-shadow: var(--shadow); overflow: visible; }\n'
        '.chart-section { padding: 24px; min-height: 200px; overflow: visible; }\n'
        '.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }\n'
        '.kpi-card { border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; background: var(--card-bg); box-shadow: var(--shadow); }\n'
        '.primary-card { border-top: 4px solid #0284c7; }\n'
        '.model-card { border-top: 4px solid #10b981; }\n'
        '.kpi-label { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }\n'
        '.kpi-value { margin-top: 4px; font-size: 24px; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; }\n'
        '.kpi-note { margin-top: 8px; color: var(--muted); font-size: 12px; }\n'
        '.footer strong { color: white; } \n'
        '.group-toggle { display: flex; background: #1e293b; border-radius: 99px; padding: 3px; border: 1px solid #334155; height: 32px; align-items: center; }\n'
        '.group-toggle button { border: 0; background: transparent; color: #94a3b8; padding: 4px 14px; border-radius: 99px; font-size: 11px; font-weight: 700; cursor: pointer; transition: all 0.2s; height: 26px; display: flex; align-items: center; }\n'
        '.group-toggle button.active { background: #38bdf8; color: #0f172a; }\n'
        '.gallery-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }\n'
        '.gallery-controls { display: flex; align-items: center; gap: 24px; }\n'
        '.gallery-legend { display: flex; gap: 16px; font-size: 13px; font-weight: 600; }\n'
        '.legend-color { width: 12px; height: 12px; border-radius: 2px; }\n'
        '.legend-color.gt { border: 2px solid #10b981; }\n'
        '.legend-color.pred { border: 2px solid #ef4444; }\n'
        '.toggle-container { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; cursor: pointer; }\n'
        '.toggle-container input { cursor: pointer; }\n'
        '.gallery-viewer { position: relative; width: 100%; display: flex; justify-content: center; background: #f8fafc; border-radius: 12px; border: 1px solid var(--border); overflow: hidden; }\n'
        '.gallery-viewer img { width: 100%; max-height: 600px; object-fit: contain; }\n'
        '.gallery-viewer canvas { position: absolute; top: 0; left: 0; pointer-events: none; }\n'
        '@media (max-width: 1000px) { .top-bar { flex-direction: column; align-items: flex-start; gap: 16px; } .top-meta { flex-wrap: wrap; gap: 12px; } .summary-grid { grid-template-columns: 1fr; } .group-toggle { align-self: flex-end; } }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<header class="top-bar">\n'
        '  <div style="display: flex; align-items: center; gap: 32px;">\n'
        '    <h1>YOLO<span>Matic</span> Report</h1>\n'
        '    <div class="top-meta">\n'
        f'      <div class="top-meta-item"><strong>Date:</strong> {escape(timestamp_str)}</div>\n'
        f'      <div class="top-meta-item"><strong>Validation:</strong> {escape(_shorten(val_dir_str, 40))}</div>\n'
        f'      <div class="top-meta-item"><strong>Models:</strong> {len(result.models)}</div>\n'
        f'      <div class="top-meta-item"><strong>Best:</strong> {escape(best_name)}</div>\n'
        f'      <div class="top-meta-item"><strong>Thresholds:</strong> {conf} / {iou}</div>\n'
        '    </div>\n'
        '  </div>\n'
        '  <div class="group-toggle">\n'
        '    <button id="btn-group-model" onclick="setGrouping(\'model\')" class="active">BY MODEL</button>\n'
        '    <button id="btn-group-metric" onclick="setGrouping(\'metric\')">BY METRIC</button>\n'
        '  </div>\n'
        '</header>\n'
        '<main class="container">\n'
        + "\n".join(sections) + "\n"
        + _GALLERY_HTML + "\n"
        '</main>\n'
        '<div class="footer">Generated by <strong>YOLOMatic</strong> &middot; Benchmark System</div>\n'
        '<script>\n'
        'function setGrouping(mode) {\n'
        '  const modelEls = document.querySelectorAll(".group-by-model");\n'
        '  const metricEls = document.querySelectorAll(".group-by-metric");\n'
        '  const btnModel = document.getElementById("btn-group-model");\n'
        '  const btnMetric = document.getElementById("btn-group-metric");\n'
        '  if (mode === "model") {\n'
        '    modelEls.forEach(el => el.style.display = "block");\n'
        '    metricEls.forEach(el => el.style.display = "none");\n'
        '    btnModel.classList.add("active");\n'
        '    btnMetric.classList.remove("active");\n'
        '  } else {\n'
        '    modelEls.forEach(el => el.style.display = "none");\n'
        '    metricEls.forEach(el => el.style.display = "block");\n'
        '    btnMetric.classList.add("active");\n'
        '    btnModel.classList.remove("active");\n'
        '  }\n'
        '  setTimeout(() => window.dispatchEvent(new Event("resize")), 50);\n'
        '}\n'
        '</script>\n'
        + _GALLERY_JS
        + '\n</body>\n</html>'
    )

    report_path.write_text(full_html, encoding="utf-8")
    return report_path
