"""
NeuralSentinel-IDS — Real-Time Dashboard (v2)
"""

import os
import json
import numpy as np
from dash import Dash, dcc, html, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="NeuralSentinel-IDS · Live Monitor",
)

BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
BLUE   = "#58a6ff"
RED    = "#f85149"
GREEN  = "#3fb950"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"

STATE_FILE = os.environ.get("ADVERSANET_STATE", "data/dashboard_state.json")


def _demo_state():
    epochs = 100
    t = np.linspace(0, 1, epochs)
    rng = np.random.default_rng(42)
    loss_G   = (0.8  * np.exp(-3 * t) + 0.05 * rng.standard_normal(epochs) * 0.1).tolist()
    loss_D   = (0.5  * np.exp(-2 * t) + 0.03 + 0.05 * rng.standard_normal(epochs) * 0.1).tolist()
    evasion  = np.clip(0.9 * np.exp(-4 * t) + 0.05 + 0.02 * rng.standard_normal(epochs), 0, 1).tolist()
    return {
        "history": {"loss_G": loss_G, "loss_D": loss_D, "evasion_rate": evasion},
        "ids_metrics": {
            "f1_weighted": 0.9831, "roc_auc": 0.9874, "accuracy": 0.9812,
            "precision": 0.9901,  "recall": 0.9763,
            "confusion_matrix": [[9481, 111], [227, 6303]],
        },
        "recent_alerts": [
            {"time": "21:27:03", "category": "DoS",         "src_ip": "192.168.1.47",  "confidence": 0.991},
            {"time": "21:26:58", "category": "Probe",       "src_ip": "10.0.0.112",    "confidence": 0.874},
            {"time": "21:26:44", "category": "U2R",         "src_ip": "172.16.0.5",    "confidence": 0.963},
            {"time": "21:26:31", "category": "R2L",         "src_ip": "192.168.2.88",  "confidence": 0.812},
            {"time": "21:26:19", "category": "DoS",         "src_ip": "10.0.1.33",     "confidence": 0.998},
            {"time": "21:26:07", "category": "Probe",       "src_ip": "192.168.5.21",  "confidence": 0.755},
            {"time": "21:25:52", "category": "Adversarial", "src_ip": "172.16.3.9",    "confidence": 0.931},
        ],
        "attack_counts": {"DoS": 1247, "Probe": 683, "R2L": 214, "U2R": 89, "Adversarial": 312},
        "total_packets": 13405,
        "attacks_detected": 2545,
    }


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return _demo_state()


def _dark(fig, height=300):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color="white", size=12),
        margin=dict(l=10, r=10, t=10, b=30),
        height=height,
        xaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=11)),
    )


def kpi_card(label, val_id, color, icon):
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon + " ", style={"fontSize": "18px"}),
                html.Span(label, style={"color": "#8b949e", "fontSize": "12px", "fontWeight": "500"}),
            ], className="mb-2"),
            html.H2(id=val_id, children="—",
                    style={"color": color, "fontWeight": "800", "fontSize": "1.9rem", "margin": 0}),
        ], style={"padding": "16px 20px"}),
    ], style={"background": CARD, "border": f"1px solid {color}44",
              "borderRadius": "12px", "height": "100%"}), md=3, className="mb-3")


app.layout = dbc.Container(fluid=True,
    style={"background": BG, "minHeight": "100vh", "padding": "24px"}, children=[
    dcc.Interval(id="tick", interval=3000),

    # ── Header ────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("⚡ ", style={"fontSize": "1.9rem"}),
                html.Span("NeuralSentinel",
                          style={"color": BLUE, "fontWeight": "900", "fontSize": "1.9rem"}),
                html.Span("-IDS",
                          style={"color": "#8b949e", "fontWeight": "300", "fontSize": "1.9rem"}),
            ]),
            html.P("Adversarial ML · Generative Attack Simulation · Live Threat Monitor",
                   style={"color": "#8b949e", "marginTop": "4px", "fontSize": "13px"}),
        ], md=8),
        dbc.Col(html.Div([
            dbc.Badge("● LIVE", color="success", pill=True,
                      style={"fontSize": "13px", "padding": "6px 14px", "float": "right"}),
            html.P(id="clock", style={"color": "#8b949e", "textAlign": "right",
                                      "fontSize": "13px", "marginTop": "8px"}),
        ]), md=4),
    ], className="mb-4"),

    # ── KPIs ──────────────────────────────────────────────────────────────
    dbc.Row([
        kpi_card("Total Packets Analyzed", "kpi-packets", BLUE,   "📦"),
        kpi_card("Attacks Detected",       "kpi-attacks", RED,    "🚨"),
        kpi_card("IDS F1 Score",           "kpi-f1",      GREEN,  "🎯"),
        kpi_card("GAN Evasion Rate",       "kpi-evasion", YELLOW, "🔮"),
    ]),

    # ── Row 2 ─────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("GAN Training Dynamics",
                                  style={"color": BLUE, "fontSize": "14px"}),
                           style={"background": CARD, "borderBottom": f"1px solid {BORDER}"}),
            dbc.CardBody(dcc.Graph(id="gan-chart", config={"displayModeBar": False})),
        ], style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "12px"}), md=8),

        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Model Performance",
                                  style={"color": BLUE, "fontSize": "14px"}),
                           style={"background": CARD, "borderBottom": f"1px solid {BORDER}"}),
            dbc.CardBody(dcc.Graph(id="metrics-bar", config={"displayModeBar": False})),
        ], style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "12px"}), md=4),
    ], className="mb-4"),

    # ── Row 3 ─────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Attack Distribution",
                                  style={"color": BLUE, "fontSize": "14px"}),
                           style={"background": CARD, "borderBottom": f"1px solid {BORDER}"}),
            dbc.CardBody(dcc.Graph(id="attack-pie", config={"displayModeBar": False})),
        ], style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "12px"}), md=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Confusion Matrix",
                                  style={"color": BLUE, "fontSize": "14px"}),
                           style={"background": CARD, "borderBottom": f"1px solid {BORDER}"}),
            dbc.CardBody(dcc.Graph(id="confusion-chart", config={"displayModeBar": False})),
        ], style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "12px"}), md=4),

        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("🚨 Live Alerts",
                                  style={"color": RED, "fontSize": "14px"}),
                           style={"background": CARD, "borderBottom": f"1px solid {BORDER}"}),
            dbc.CardBody(html.Div(id="alerts-table",
                                  style={"overflowY": "auto", "maxHeight": "280px"})),
        ], style={"background": CARD, "border": f"1px solid {RED}44", "borderRadius": "12px"}), md=4),
    ]),
])


@callback(Output("clock", "children"), Input("tick", "n_intervals"))
def update_clock(_):
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


@callback(
    Output("kpi-packets",    "children"),
    Output("kpi-attacks",    "children"),
    Output("kpi-f1",         "children"),
    Output("kpi-evasion",    "children"),
    Output("gan-chart",      "figure"),
    Output("metrics-bar",    "figure"),
    Output("attack-pie",     "figure"),
    Output("confusion-chart","figure"),
    Output("alerts-table",   "children"),
    Input("tick", "n_intervals"),
)
def refresh(_):
    s       = load_state()
    hist    = s.get("history", {})
    metrics = s.get("ids_metrics", {})
    counts  = s.get("attack_counts", {})
    alerts  = s.get("recent_alerts", [])
    er_list = hist.get("evasion_rate", [])

    kpi_packets = f"{s.get('total_packets', 0):,}"
    kpi_attacks = f"{s.get('attacks_detected', 0):,}"
    kpi_f1      = f"{metrics.get('f1_weighted', 0):.4f}" if metrics else "N/A"
    # Show post-hardening evasion if available, else last GAN training evasion
    if metrics and "evasion_after_hardening" in metrics:
        kpi_evasion = f"{metrics['evasion_after_hardening']:.2f}%"
    elif er_list:
        kpi_evasion = f"{er_list[-1] * 100:.2f}%"
    else:
        kpi_evasion = "N/A"

    # GAN chart
    epochs = list(range(1, len(hist.get("loss_G", [])) + 1))
    fig_gan = make_subplots(specs=[[{"secondary_y": True}]])
    if epochs:
        fig_gan.add_trace(go.Scatter(
            x=epochs, y=hist["loss_G"], name="Generator Loss",
            line=dict(color=RED, width=2),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.08)"), secondary_y=False)
        fig_gan.add_trace(go.Scatter(
            x=epochs, y=hist["loss_D"], name="Discriminator Loss",
            line=dict(color=BLUE, width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"), secondary_y=False)
        if er_list:
            fig_gan.add_trace(go.Scatter(
                x=epochs[:len(er_list)],
                y=[v * 100 for v in er_list],
                name="Evasion Rate (%)",
                line=dict(color=YELLOW, dash="dot", width=2)), secondary_y=True)
    _dark(fig_gan, 290)
    fig_gan.update_layout(legend=dict(orientation="h", y=-0.18))
    fig_gan.update_yaxes(title_text="Loss", secondary_y=False,
                         title_font=dict(color="#8b949e"), tickfont=dict(color="#8b949e"))
    fig_gan.update_yaxes(title_text="Evasion %", secondary_y=True,
                         title_font=dict(color=YELLOW), tickfont=dict(color=YELLOW))

    # Metrics bar
    mkeys = ["F1 Score", "ROC-AUC", "Accuracy", "Precision", "Recall"]
    mvals = [metrics.get(k, 0) for k in
             ["f1_weighted", "roc_auc", "accuracy", "precision", "recall"]]
    bcolors = [GREEN if v >= 0.95 else YELLOW if v >= 0.85 else RED for v in mvals]
    fig_metrics = go.Figure(go.Bar(
        x=mvals, y=mkeys, orientation="h",
        marker=dict(color=bcolors),
        text=[f"{v:.4f}" for v in mvals],
        textposition="outside", textfont=dict(color="white", size=12),
    ))
    fig_metrics.update_xaxes(range=[0, 1.12], showgrid=False, showticklabels=False)
    fig_metrics.update_yaxes(tickfont=dict(color="white", size=12))
    _dark(fig_metrics, 290)

    # Pie
    PAL = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE]
    if counts:
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.keys()), values=list(counts.values()),
            hole=0.5,
            marker=dict(colors=PAL[:len(counts)], line=dict(color=BG, width=2)),
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} attacks (%{percent})<extra></extra>",
        ))
        total_att = sum(counts.values())
        fig_pie.update_layout(
            annotations=[dict(text=f"<b>{total_att:,}</b><br><span style='font-size:11px'>attacks</span>",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="white", size=14))],
            legend=dict(font=dict(color="white", size=11)),
        )
    else:
        fig_pie = go.Figure()
    _dark(fig_pie, 290)

    # Confusion matrix
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    z  = np.array(cm)
    fig_cm = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: Normal", "Pred: Attack"],
        y=["True: Normal", "True: Attack"],
        colorscale=[[0, "#0d1117"], [0.5, "#1c3a5e"], [1, BLUE]],
        showscale=False,
        text=[[f"<b>{v:,}</b>" for v in row] for row in z.tolist()],
        texttemplate="%{text}", textfont=dict(size=18, color="white"),
    ))
    _dark(fig_cm, 290)

    # Alerts
    CAT_COLOR = {"DoS": RED, "Probe": YELLOW, "R2L": ORANGE,
                 "U2R": PURPLE, "Adversarial": "#ff79c6"}
    if alerts:
        rows = []
        for a in reversed(alerts[-12:]):
            cat   = a.get("category", "?")
            color = CAT_COLOR.get(cat, BLUE)
            rows.append(html.Tr([
                html.Td(a.get("time", ""),
                        style={"color": "#8b949e", "fontSize": "12px", "padding": "5px 8px"}),
                html.Td(html.Span(cat, style={
                    "background": color + "22", "color": color,
                    "padding": "2px 8px", "borderRadius": "20px",
                    "fontSize": "11px", "fontWeight": "600",
                }), style={"padding": "5px 8px"}),
                html.Td(a.get("src_ip", ""),
                        style={"color": "white", "fontSize": "12px",
                               "fontFamily": "monospace", "padding": "5px 8px"}),
                html.Td(f"{a.get('confidence', 0):.1%}",
                        style={"color": GREEN, "fontSize": "12px",
                               "fontWeight": "600", "padding": "5px 8px"}),
            ], style={"borderBottom": f"1px solid {BORDER}"}))
        table = html.Table(
            [html.Thead(html.Tr([
                html.Th(h, style={"color": "#8b949e", "fontSize": "11px",
                                  "padding": "6px 8px", "fontWeight": "500",
                                  "borderBottom": f"1px solid {BORDER}"})
                for h in ["Time", "Category", "Source IP", "Conf."]
            ]))] + [html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        )
    else:
        table = html.P("No alerts yet.", style={"color": "#8b949e", "padding": "20px"})

    return (kpi_packets, kpi_attacks, kpi_f1, kpi_evasion,
            fig_gan, fig_metrics, fig_pie, fig_cm, table)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
