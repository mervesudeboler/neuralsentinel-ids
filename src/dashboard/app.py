"""
NeuralSentinel-IDS — Real-Time Dashboard
Plotly Dash application for live monitoring of:
  • IDS detection metrics
  • GAN training curves (evasion rate, loss)
  • Attack category distribution
  • Adversarial sample visualization (t-SNE / UMAP)
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime
from typing import Optional

# ─── App Bootstrap ────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="NeuralSentinel-IDS · Live Monitor",
)

DARK = "#0d1117"
CARD_BG = "#161b22"
ACCENT = "#58a6ff"
RED = "#f85149"
GREEN = "#3fb950"
YELLOW = "#d29922"

# ─── Shared in-memory state (replace with Redis/Kafka consumer in prod) ───────
STATE_FILE = os.environ.get("NEURALSENTINEL_STATE", "data/dashboard_state.json")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "history": {"loss_G": [], "loss_D": [], "evasion_rate": []},
        "ids_metrics": {},
        "recent_alerts": [],
        "attack_counts": {},
        "total_packets": 0,
        "attacks_detected": 0,
    }


# ─── Layout ───────────────────────────────────────────────────────────────────
def make_stat_card(title: str, value_id: str, color: str = ACCENT) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted small mb-1"),
            html.H3(id=value_id, children="—", style={"color": color, "fontWeight": "bold"}),
        ]),
        style={"background": CARD_BG, "border": f"1px solid {color}33"},
        className="mb-3",
    )


app.layout = dbc.Container(
    fluid=True,
    style={"background": DARK, "minHeight": "100vh", "padding": "20px"},
    children=[
        dcc.Interval(id="interval", interval=3000, n_intervals=0),

        # ── Header ────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.H2("⚡ NeuralSentinel-IDS", style={"color": ACCENT, "fontWeight": "bold"}),
                html.P("Adversarial ML · Intrusion Detection · Live Monitor",
                       className="text-muted"),
            ]), width=8),
            dbc.Col(html.Div([
                html.P(id="clock", className="text-muted text-end small"),
                dbc.Badge("● LIVE", color="success", pill=True, className="float-end"),
            ]), width=4),
        ], className="mb-4"),

        # ── KPI Cards ─────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(make_stat_card("Total Packets Analyzed", "kpi-packets", ACCENT), md=3),
            dbc.Col(make_stat_card("Attacks Detected", "kpi-attacks", RED), md=3),
            dbc.Col(make_stat_card("IDS F1 Score", "kpi-f1", GREEN), md=3),
            dbc.Col(make_stat_card("GAN Evasion Rate", "kpi-evasion", YELLOW), md=3),
        ]),

        # ── GAN Training Chart ────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("GAN Training Dynamics", style={"color": ACCENT}),
                               style={"background": CARD_BG}),
                dbc.CardBody(dcc.Graph(id="gan-chart", style={"height": "320px"})),
            ], style={"background": CARD_BG}), md=6),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("Attack Distribution", style={"color": ACCENT}),
                               style={"background": CARD_BG}),
                dbc.CardBody(dcc.Graph(id="attack-pie", style={"height": "320px"})),
            ], style={"background": CARD_BG}), md=6),
        ], className="mb-4"),

        # ── IDS Confusion & Alerts ────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("IDS Confusion Matrix", style={"color": ACCENT}),
                               style={"background": CARD_BG}),
                dbc.CardBody(dcc.Graph(id="confusion-chart", style={"height": "320px"})),
            ], style={"background": CARD_BG}), md=6),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("Recent Alerts", style={"color": ACCENT}),
                               style={"background": CARD_BG}),
                dbc.CardBody(html.Div(id="alerts-table",
                                      style={"overflowY": "auto", "maxHeight": "280px"})),
            ], style={"background": CARD_BG}), md=6),
        ]),
    ],
)


# ─── Callbacks ────────────────────────────────────────────────────────────────
@callback(
    Output("clock", "children"),
    Input("interval", "n_intervals"),
)
def update_clock(_):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@callback(
    Output("kpi-packets", "children"),
    Output("kpi-attacks", "children"),
    Output("kpi-f1", "children"),
    Output("kpi-evasion", "children"),
    Output("gan-chart", "figure"),
    Output("attack-pie", "figure"),
    Output("confusion-chart", "figure"),
    Output("alerts-table", "children"),
    Input("interval", "n_intervals"),
)
def update_dashboard(_):
    s = load_state()
    history = s.get("history", {})
    metrics = s.get("ids_metrics", {})
    attack_counts = s.get("attack_counts", {})
    alerts = s.get("recent_alerts", [])

    # KPIs
    kpi_packets = f"{s.get('total_packets', 0):,}"
    kpi_attacks = f"{s.get('attacks_detected', 0):,}"
    kpi_f1 = f"{metrics.get('f1_weighted', 0):.4f}" if metrics else "N/A"
    evasion = history.get("evasion_rate", [])
    kpi_evasion = f"{evasion[-1]:.2%}" if evasion else "N/A"

    # ── GAN Chart ─────────────────────────────────────────────────────────────
    fig_gan = make_subplots(specs=[[{"secondary_y": True}]])
    epochs = list(range(1, len(history.get("loss_G", [])) + 1))
    if epochs:
        fig_gan.add_trace(go.Scatter(
            x=epochs, y=history["loss_G"], name="Generator Loss",
            line=dict(color=RED, width=2)), secondary_y=False)
        fig_gan.add_trace(go.Scatter(
            x=epochs, y=history["loss_D"], name="Discriminator Loss",
            line=dict(color=ACCENT, width=2)), secondary_y=False)
        if history.get("evasion_rate"):
            fig_gan.add_trace(go.Scatter(
                x=epochs[:len(history["evasion_rate"])],
                y=[v * 100 for v in history["evasion_rate"]],
                name="Evasion Rate (%)", line=dict(color=YELLOW, dash="dot", width=2)),
                secondary_y=True)
    _apply_dark(fig_gan)
    fig_gan.update_layout(legend=dict(font=dict(color="white")))

    # ── Attack Pie ────────────────────────────────────────────────────────────
    if attack_counts:
        fig_pie = go.Figure(go.Pie(
            labels=list(attack_counts.keys()),
            values=list(attack_counts.values()),
            hole=0.4,
            marker=dict(colors=[ACCENT, RED, GREEN, YELLOW, "#bc8cff"]),
        ))
    else:
        fig_pie = go.Figure(go.Pie(labels=["No data"], values=[1], hole=0.4))
    _apply_dark(fig_pie)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    if cm:
        z = np.array(cm)
        fig_cm = go.Figure(go.Heatmap(
            z=z, x=["Pred: Normal", "Pred: Attack"],
            y=["True: Normal", "True: Attack"],
            colorscale=[[0, CARD_BG], [1, ACCENT]],
            text=z.astype(str), texttemplate="%{text}",
            showscale=False,
        ))
    else:
        fig_cm = go.Figure()
    _apply_dark(fig_cm)

    # ── Alerts Table ──────────────────────────────────────────────────────────
    if alerts:
        rows = [
            html.Tr([
                html.Td(a.get("time", ""), style={"color": "gray", "fontSize": "12px"}),
                html.Td(a.get("category", ""), style={"color": RED}),
                html.Td(a.get("src_ip", ""), style={"color": "white"}),
                html.Td(f"{a.get('confidence', 0):.2%}", style={"color": GREEN}),
            ])
            for a in alerts[-15:]
        ]
        table = html.Table(
            [html.Thead(html.Tr([
                html.Th("Time"), html.Th("Category"), html.Th("Src IP"), html.Th("Conf"),
            ], style={"color": ACCENT}))] + [html.Tbody(rows)],
            className="table table-sm",
            style={"color": "white"},
        )
    else:
        table = html.P("No alerts yet.", className="text-muted")

    return (kpi_packets, kpi_attacks, kpi_f1, kpi_evasion,
            fig_gan, fig_pie, fig_cm, table)


def _apply_dark(fig) -> None:
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
