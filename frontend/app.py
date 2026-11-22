"""
frontend/app.py - Demand Forecast Studio (Streamlit front end)

Run:
    streamlit run frontend/app.py

Environment variables:
    API_BASE_URL         Base URL of the FastAPI server (default http://127.0.0.1:8000)
    API_FORECAST_PATH    Route of the forecast endpoint (default /forecast). The endpoint in
                         the contract was cut off after the host, so set this variable if
                         your route is different.
    API_TIMEOUT_SECONDS  Request timeout in seconds (default 30)

Requirements:
    streamlit>=1.39 (keyed containers), plotly, pandas, requests

Recommended .streamlit/config.toml so native widgets such as the dataframe match the
light editorial theme:
    [theme]
    base = "light"
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
FORECAST_PATH = "/" + os.getenv("API_FORECAST_PATH", "/predict").lstrip("/")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))

NAVY = "#001b44"
NAVY_ACTIVE = "#002f6c"
EMERALD = "#006d43"
MINT = "#75f8b3"
INK = "#1b1c1c"
MUTED = "#434750"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

HISTORY_VIEWS = (("90D", 90), ("180D", 180), ("1Y", 365), ("All", None))
DEFAULT_VIEW_INDEX = 1

LIGHT_PLOT = {
    "text": INK,
    "grid": "rgba(196, 198, 210, 0.25)",
    "line": "rgba(196, 198, 210, 0.55)",
}
DARK_PLOT = {
    "text": "#ffffff",
    "grid": "rgba(255, 255, 255, 0.10)",
    "line": "rgba(255, 255, 255, 0.25)",
}

PLOT_CONFIG = {"displayModeBar": False}


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


NEW_WIDTH_API = _version_tuple(st.__version__) >= (1, 50, 0)


def width_kwargs() -> dict[str, Any]:
    """Full-width argument that works on both older and newer Streamlit releases."""
    return {"width": "stretch"} if NEW_WIDTH_API else {"use_container_width": True}


st.set_page_config(
    page_title="Demand Forecast Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root { color-scheme: light; }

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: #fbf9f8 !important;
}
.stApp { color: #1b1c1c; font-family: 'Inter', sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button {
  font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 3rem 2.5rem 3rem; max-width: 1400px; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(67, 71, 80, 0.25); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
  background: #001b44 !important;
}
[data-testid="stSidebarCollapseButton"] * { color: #d8e2ff !important; }

.brand-name {
  font-family: 'Manrope', sans-serif; color: #ffffff; font-weight: 800;
  font-size: 1.3rem; letter-spacing: -0.02em;
}
.brand-tag { color: #d8e2ff; font-size: 0.82rem; margin-top: 4px; }
.side-divider { height: 1px; background: rgba(216, 226, 255, 0.14); margin: 18px 0 20px; }
.side-label {
  color: #ffffff; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px;
}
.api-note { color: #d8e2ff; font-size: 0.76rem; margin-top: 24px; line-height: 1.7; }
.api-note code {
  color: #75f8b3; background: #002f6c; padding: 3px 8px;
  border-radius: 8px; font-size: 0.74rem; word-break: break-all;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {
  color: #d8e2ff !important; font-size: 0.84rem; font-weight: 500;
}
[data-testid="stSidebar"] div[data-baseweb="input"] {
  background: #00235a !important;
  border: none !important;
  border-radius: 12px !important;
  box-shadow: inset 0 0 0 1px rgba(216, 226, 255, 0.14);
  transition: background 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
  background: #002f6c !important;
  box-shadow: 0 0 0 2px rgba(117, 248, 179, 0.55) !important;
}
[data-testid="stSidebar"] div[data-baseweb="base-input"],
[data-testid="stSidebar"] div[data-baseweb="base-input"] > div {
  background: #002f6c !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] input[type="number"] {
  background: #002f6c !important;
  color: #d8e2ff !important;
  -webkit-text-fill-color: #d8e2ff !important;
  caret-color: #60a5fa !important;
  font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"],
[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] {
  background: #002f6c !important; color: #ffffff !important; border: none !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover,
[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover {
  background: #006d43 !important;
}

/* Streamlit draws slider accents in its default primary color; rotate it to green */
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] {
  filter: hue-rotate(155deg) saturate(1.1);
}
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #d8e2ff !important; }

/* Run Forecast button */
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] .stButton button {
  width: 100%;
  background: #002f6c !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.72rem 1rem !important;
  font-weight: 600 !important;
  margin-top: 8px;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
  transition: background 0.2s ease, transform 0.2s ease;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] .stButton button:hover {
  background: #006d43 !important; color: #ffffff !important; transform: translateY(-1px);
}
[data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible,
[data-testid="stSidebar"] .stButton button:focus-visible {
  outline: none; box-shadow: 0 0 0 2px rgba(117, 248, 179, 0.6) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p,
[data-testid="stSidebar"] .stButton button p { color: #ffffff !important; font-weight: 600; }

/* ---------------- Header bar ---------------- */
.page-header {
  display: flex; justify-content: space-between; align-items: flex-end;
  gap: 20px; flex-wrap: wrap; margin-bottom: 28px;
}
.page-title {
  font-family: 'Manrope', sans-serif; font-size: 2.1rem; font-weight: 800;
  letter-spacing: -0.03em; color: #1b1c1c; line-height: 1.15;
}
.page-sub { color: #434750; font-size: 0.95rem; margin-top: 6px; }
.pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
  background: #d8e2ff; color: #224583; padding: 7px 16px;
  border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}

/* ---------------- KPI cards ---------------- */
.kpi-card {
  background: #ffffff; border-radius: 12px; padding: 22px 24px; min-height: 160px;
  display: flex; flex-direction: column; justify-content: space-between;
  box-shadow: 0 20px 40px rgba(0, 27, 68, 0.06);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 24px 48px rgba(0, 27, 68, 0.10); }
.kpi-label {
  color: #434750; font-size: 0.74rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.kpi-value {
  font-family: 'Manrope', sans-serif; font-size: 2.15rem; font-weight: 800;
  color: #1b1c1c; letter-spacing: -0.03em; line-height: 1.1; margin-top: 14px;
  font-variant-numeric: tabular-nums;
}
.kpi-caption { color: #434750; font-size: 0.82rem; margin-top: 10px; }
.kpi-card.featured { background: #001b44; }
.kpi-card.featured .kpi-label { color: #ffffff; }
.kpi-card.featured .kpi-value { color: #ffffff; }
.kpi-card.featured .kpi-caption { color: #75f8b3; font-weight: 600; }

.growth-pill {
  display: inline-block; font-family: 'Manrope', sans-serif; font-weight: 800;
  font-size: 1.55rem; letter-spacing: -0.02em; padding: 6px 18px;
  border-radius: 999px; margin-top: 12px; font-variant-numeric: tabular-nums;
}
.growth-pill.positive { background: #75f8b3; color: #007147; }
.growth-pill.negative { background: #ffdad6; color: #ba1a1a; }
.growth-pill.neutral { background: #d8e2ff; color: #224583; }

/* ---------------- Cards (keyed containers) ---------------- */
.st-key-card_main, .st-key-card_weekday, .st-key-card_table {
  background: #ffffff;
  border-radius: 12px;
  padding: 1.5rem 1.6rem;
  box-shadow: 0 20px 40px rgba(0, 27, 68, 0.06);
}
.st-key-card_main, .st-key-card_weekday { min-height: 540px; }
.st-key-card_weekday { background: #001b44; }

.card-title {
  font-family: 'Manrope', sans-serif; color: #1b1c1c; font-weight: 700;
  font-size: 1.12rem; letter-spacing: -0.01em;
}
.card-sub { color: #434750; font-size: 0.85rem; margin-top: 4px; }
.on-dark .card-title { color: #ffffff; }
.on-dark .card-sub { color: #75f8b3; font-weight: 500; }
[data-testid="stPlotlyChart"] { background: transparent; }
.spacer { height: 12px; }

/* Download button */
[data-testid="stDownloadButton"] { display: flex; justify-content: flex-end; }
[data-testid="stDownloadButton"] button {
  background: #001b44 !important; color: #ffffff !important; border: none !important;
  border-radius: 12px !important; padding: 0.6rem 1.3rem !important; font-weight: 600 !important;
  transition: background 0.2s ease;
}
[data-testid="stDownloadButton"] button:hover { background: #006d43 !important; }
[data-testid="stDownloadButton"] button p { color: #ffffff !important; font-weight: 600; }

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Alerts, toasts, spinner */
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
  background: #ffdad6 !important; border: none !important; border-radius: 12px !important;
}
[data-testid="stAlert"] { box-shadow: 0 20px 40px rgba(0, 27, 68, 0.06); }
[data-testid="stAlert"] p, [data-testid="stAlert"] li { color: #93000a !important; }
[data-testid="stAlert"] code { color: #93000a; background: rgba(186, 26, 26, 0.10); }
[data-testid="stToast"] {
  background: #001b44 !important; border: none !important; border-radius: 12px !important;
  box-shadow: 0 20px 40px rgba(0, 27, 68, 0.25);
}
[data-testid="stToast"] * { color: #ffffff !important; }
[data-testid="stSpinner"] p { color: #434750 !important; }

@media (max-width: 768px) {
  .block-container { padding: 1.5rem 1rem 2rem; }
  .page-title { font-size: 1.6rem; }
}
"""


# ----------------------------------------------------------------------------
# HTML helpers
# ----------------------------------------------------------------------------
def fmt(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def kpi_card(label: str, value: str, caption: str, featured: bool = False) -> str:
    cls = "kpi-card featured" if featured else "kpi-card"
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-caption">{caption}</div>'
        "</div>"
    )


def growth_card(growth: float | None) -> str:
    if growth is None:
        tone, text = "neutral", "N/A"
    elif growth >= 0:
        tone, text = "positive", f"{growth:+.1f}%"
    else:
        tone, text = "negative", f"{growth:+.1f}%"
    return (
        '<div class="kpi-card">'
        '<div class="kpi-label">Projected Growth</div>'
        f'<div><span class="growth-pill {tone}">{text}</span></div>'
        '<div class="kpi-caption">Forecast average vs. last 60 days</div>'
        "</div>"
    )


def card_header(title: str, subtitle: str, dark: bool = False) -> str:
    cls = "card-head on-dark" if dark else "card-head"
    return (
        f'<div class="{cls}"><div class="card-title">{title}</div>'
        f'<div class="card-sub">{subtitle}</div></div>'
    )


def render_header(params: dict[str, int] | None) -> None:
    pills = ""
    if params:
        pills = (
            '<div class="pills">'
            f'<span class="pill">Store {params["store_id"]}</span>'
            f'<span class="pill">Item {params["item_id"]}</span>'
            f'<span class="pill">{params["days_ahead"]}-day horizon</span>'
            "</div>"
        )
    st.markdown(
        '<div class="page-header"><div>'
        '<div class="page-title">Demand Forecast Studio</div>'
        '<div class="page-sub">Sales history and predicted demand for any store and item.</div>'
        f"</div>{pills}</div>",
        unsafe_allow_html=True,
    )


def show_dataframe(data: pd.DataFrame, height: int) -> None:
    st.dataframe(
        data,
        hide_index=True,
        height=height,
        column_config={
            "Predicted Sales": st.column_config.NumberColumn("Predicted Sales", format="%.2f"),
        },
        **width_kwargs(),
    )


# ----------------------------------------------------------------------------
# API access
# ----------------------------------------------------------------------------
def _extract_detail(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip()
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = [str(item.get("msg", "")) for item in detail if isinstance(item, dict)]
        return "; ".join(m for m in messages if m)
    return ""


def fetch_forecast(
    store_id: int, item_id: int, days_ahead: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the live backend. Returns (data, None) on success or (None, error_message)."""
    url = f"{API_BASE_URL}{FORECAST_PATH}"
    params = {"store_id": store_id, "item_id": item_id, "days_ahead": days_ahead}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            "**The forecast API appears to be offline.**\n\n"
            f"Could not connect to `{url}`. Start your FastAPI server, make sure "
            "`API_BASE_URL` points to it, then select **Run Forecast** again."
        )
    except requests.exceptions.Timeout:
        return None, (
            f"**The forecast API took too long to respond** (more than {REQUEST_TIMEOUT:g} seconds). "
            "Try again in a moment or choose a shorter horizon."
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = _extract_detail(exc.response)
        message = f"**The API returned an error (HTTP {status}).**"
        if detail:
            message += f"\n\n{detail}"
        if status == 404:
            message += (
                f"\n\nCheck that the route `{FORECAST_PATH}` exists on `{API_BASE_URL}`. "
                "You can change it with the `API_FORECAST_PATH` environment variable."
            )
        return None, message
    except ValueError:
        return None, "**The API replied, but the response was not valid JSON.**"
    except requests.exceptions.RequestException as exc:
        return None, f"**Something went wrong while calling the API.**\n\n{exc}"


def parse_payload(data: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the API response and turn it into tidy, sorted DataFrames."""
    if not isinstance(data, dict):
        raise ValueError("The API response had an unexpected shape.")

    hist = pd.DataFrame(data.get("history") or [])
    fc = pd.DataFrame(data.get("forecast") or [])

    if fc.empty or not {"date", "predicted_sales"}.issubset(fc.columns):
        raise ValueError("The API response did not include any forecast rows.")

    if hist.empty:
        hist = pd.DataFrame({"date": pd.to_datetime([]), "sales": pd.Series([], dtype=float)})
    elif not {"date", "sales"}.issubset(hist.columns):
        raise ValueError("The history rows are missing the `date` or `sales` field.")

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist["sales"], errors="coerce")
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    fc["predicted_sales"] = pd.to_numeric(fc["predicted_sales"], errors="coerce")

    hist = hist.dropna(subset=["date", "sales"]).sort_values("date").reset_index(drop=True)
    fc = fc.dropna(subset=["date", "predicted_sales"]).sort_values("date").reset_index(drop=True)
    if fc.empty:
        raise ValueError("The forecast rows in the API response could not be read.")
    return hist, fc


# ----------------------------------------------------------------------------
# Analytics and charts
# ----------------------------------------------------------------------------
def compute_kpis(hist: pd.DataFrame, fc: pd.DataFrame) -> dict[str, float | None]:
    total = float(fc["predicted_sales"].sum())
    avg = float(fc["predicted_sales"].mean())

    hist_avg: float | None = None
    if not hist.empty:
        cutoff = hist["date"].max() - pd.Timedelta(days=59)
        hist_avg = float(hist.loc[hist["date"] >= cutoff, "sales"].mean())

    growth: float | None = None
    if hist_avg is not None and hist_avg > 0:
        growth = (avg - hist_avg) / hist_avg * 100

    return {"total": total, "avg": avg, "hist_avg": hist_avg, "growth": growth}


def style_figure(fig: go.Figure, height: int, palette: dict[str, str]) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=56, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=palette["text"], size=12),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(196,198,210,0.6)",
            font=dict(family="Inter, sans-serif", color=INK, size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=palette["line"],
        tickfont=dict(color=palette["text"]),
    )
    fig.update_yaxes(
        gridcolor=palette["grid"],
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=palette["text"]),
    )
    return fig


def view_window(
    hist: pd.DataFrame, fc: pd.DataFrame, days: int | None
) -> tuple[list[str], list[float]]:
    """X and Y ranges for a history window that always ends at the last forecast date."""
    fc_end = fc["date"].max()
    if hist.empty:
        start = fc["date"].min()
        visible = pd.Series(dtype=float)
    else:
        if days is None:
            start = hist["date"].min()
        else:
            start = max(hist["date"].min(), hist["date"].max() - pd.Timedelta(days=days))
        visible = hist.loc[hist["date"] >= start, "sales"]

    high = float(fc["predicted_sales"].max())
    low = float(fc["predicted_sales"].min())
    if not visible.empty:
        high = max(high, float(visible.max()))
        low = min(low, float(visible.min()))

    top = high * 1.12 if high > 0 else 1.0
    bottom = low * 1.12 if low < 0 else 0.0
    return [start.strftime("%Y-%m-%d"), fc_end.strftime("%Y-%m-%d")], [bottom, top]


def build_main_chart(hist: pd.DataFrame, fc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fc_start = fc["date"].min().strftime("%Y-%m-%d")
    fc_end = fc["date"].max().strftime("%Y-%m-%d")

    # Soft shaded forecast region with a dotted divider where predictions begin
    fig.add_shape(
        type="rect", xref="x", yref="paper", x0=fc_start, x1=fc_end, y0=0, y1=1,
        fillcolor="rgba(0,109,67,0.05)", line=dict(width=0), layer="below",
    )
    fig.add_shape(
        type="line", xref="x", yref="paper", x0=fc_start, x1=fc_start, y0=0, y1=1,
        line=dict(color="rgba(67,71,80,0.45)", width=1, dash="dot"),
    )

    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["date"], y=hist["sales"], mode="lines", name="Historical sales",
                line=dict(color=NAVY, width=2.2),
                fill="tozeroy", fillcolor="rgba(0, 27, 68, 0.08)",
                hovertemplate="%{y:,.1f}<extra>Historical sales</extra>",
            )
        )
        # Bridge segment so the history line flows straight into the forecast
        fig.add_trace(
            go.Scatter(
                x=[hist["date"].iloc[-1], fc["date"].iloc[0]],
                y=[hist["sales"].iloc[-1], fc["predicted_sales"].iloc[0]],
                mode="lines", line=dict(color=EMERALD, width=2.4, dash="dash"),
                hoverinfo="skip", showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=fc["date"], y=fc["predicted_sales"], mode="lines", name="Predicted demand",
            line=dict(color=EMERALD, width=2.4, dash="dash"),
            hovertemplate="%{y:,.1f}<extra>Predicted demand</extra>",
        )
    )
    fig.add_annotation(
        x=fc_start, y=1, xref="x", yref="paper", text="FORECAST", showarrow=False,
        xanchor="left", yanchor="top", xshift=8, yshift=-6,
        font=dict(size=10, color=EMERALD, family="Inter, sans-serif"),
    )

    style_figure(fig, 400, LIGHT_PLOT)

    default_x, default_y = view_window(hist, fc, HISTORY_VIEWS[DEFAULT_VIEW_INDEX][1])
    buttons = []
    for label, days in HISTORY_VIEWS:
        x_range, y_range = view_window(hist, fc, days)
        buttons.append(
            dict(
                label=label,
                method="relayout",
                args=[{"xaxis.range": x_range, "yaxis.range": y_range}],
            )
        )

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(range=default_x),
        yaxis=dict(range=default_y),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=12),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                active=DEFAULT_VIEW_INDEX,
                showactive=True,
                x=0, xanchor="left", y=1.0, yanchor="bottom",
                pad=dict(l=0, r=0, t=0, b=8),
                bgcolor="#d8e2ff",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(family="Inter, sans-serif", size=12, color="#224583"),
            )
        ],
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikedash="dot", spikecolor="rgba(27,28,28,0.30)",
    )
    return fig


def weekday_breakdown(fc: pd.DataFrame) -> pd.Series:
    by_day = fc.assign(day=fc["date"].dt.day_name()).groupby("day")["predicted_sales"].mean()
    return by_day.reindex(WEEKDAYS).fillna(0.0)


def build_weekday_chart(by_day: pd.Series) -> go.Figure:
    values = by_day.tolist()
    peak_index = values.index(max(values))
    colors = [MINT if i == peak_index else EMERALD for i in range(len(values))]

    fig = go.Figure(
        go.Bar(
            x=WEEKDAYS_SHORT,
            y=values,
            customdata=WEEKDAYS,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:,.1f}" for v in values],
            textposition="outside",
            textfont=dict(color="#ffffff", size=11),
            cliponaxis=False,
            hovertemplate="<b>%{customdata}</b><br>%{y:,.1f} units per day<extra></extra>",
        )
    )
    style_figure(fig, 400, DARK_PLOT)
    top = max(values) if values and max(values) > 0 else 1.0
    fig.update_yaxes(range=[0, top * 1.22])
    fig.update_layout(bargap=0.28, showlegend=False)
    return fig


# ----------------------------------------------------------------------------
# App sections
# ----------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("payload", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("has_run", False)


def render_sidebar() -> tuple[int, int, int, bool]:
    with st.sidebar:
        st.markdown(
            '<div class="brand-name">Demand Forecast</div>'
            '<div class="brand-tag">Store and item demand console</div>'
            '<div class="side-divider"></div>'
            '<div class="side-label">Parameters</div>',
            unsafe_allow_html=True,
        )
        store_id = st.number_input("Store ID (1-10)", min_value=1, max_value=10, value=1, step=1)
        item_id = st.number_input("Item ID (1-50)", min_value=1, max_value=50, value=1, step=1)
        days_ahead = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)
        run_clicked = st.button("Run Forecast", type="primary", key="run_forecast")
        st.markdown(
            f'<div class="api-note">API endpoint<br><code>{html.escape(API_BASE_URL)}</code></div>',
            unsafe_allow_html=True,
        )
    return int(store_id), int(item_id), int(days_ahead), run_clicked


def run_forecast(store_id: int, item_id: int, days_ahead: int) -> None:
    with st.spinner("Running demand forecast..."):
        data, error = fetch_forecast(store_id, item_id, days_ahead)

    hist = fc = None
    if error is None:
        try:
            hist, fc = parse_payload(data)
        except ValueError as exc:
            error = f"**The API response could not be used.**\n\n{exc}"

    if error is not None:
        st.session_state.payload = None
        st.session_state.error = error
        return

    st.session_state.payload = {
        "hist": hist,
        "fc": fc,
        "params": {"store_id": store_id, "item_id": item_id, "days_ahead": days_ahead},
    }
    st.session_state.error = None
    st.toast("Forecast updated.")


def render_kpis(hist: pd.DataFrame, fc: pd.DataFrame) -> None:
    kpis = compute_kpis(hist, fc)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(
        kpi_card(
            "Total Predicted Demand", fmt(kpis["total"], 0),
            f"Units over the next {len(fc)} days", featured=True,
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card("Average Daily Forecast", fmt(kpis["avg"], 1), "Predicted units per day"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi_card("Last 60-Day Historical Average", fmt(kpis["hist_avg"], 1), "Actual units per day"),
        unsafe_allow_html=True,
    )
    c4.markdown(growth_card(kpis["growth"]), unsafe_allow_html=True)


def render_charts(hist: pd.DataFrame, fc: pd.DataFrame) -> None:
    by_day = weekday_breakdown(fc)
    peak_day = str(by_day.idxmax())

    main_col, week_col = st.columns([1.9, 1], gap="large")

    with main_col:
        with st.container(key="card_main"):
            st.markdown(
                card_header("Sales Trajectory", "Historical sales flowing into predicted demand"),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                build_main_chart(hist, fc),
                theme=None, config=PLOT_CONFIG, key="chart_main", **width_kwargs(),
            )

    with week_col:
        with st.container(key="card_weekday"):
            st.markdown(
                card_header(
                    "Day-of-Week Demand",
                    f"Average predicted units per day, peak on {peak_day}",
                    dark=True,
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                build_weekday_chart(by_day),
                theme=None, config=PLOT_CONFIG, key="chart_weekday", **width_kwargs(),
            )


def render_table(fc: pd.DataFrame, params: dict[str, int]) -> None:
    export = pd.DataFrame(
        {
            "date": fc["date"].dt.strftime("%Y-%m-%d"),
            "day_of_week": fc["date"].dt.day_name(),
            "predicted_sales": fc["predicted_sales"].round(2),
        }
    )
    table = export.rename(
        columns={"date": "Date", "day_of_week": "Day", "predicted_sales": "Predicted Sales"}
    )
    csv_bytes = export.to_csv(index=False).encode("utf-8")
    file_name = f"forecast_store{params['store_id']}_item{params['item_id']}_{params['days_ahead']}d.csv"

    with st.container(key="card_table"):
        left, right = st.columns([3, 1])
        left.markdown(
            card_header(
                "Forecast Data",
                f"{len(export)} daily predictions from {export['date'].iloc[0]} to {export['date'].iloc[-1]}",
            ),
            unsafe_allow_html=True,
        )
        with right:
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
                key="download_csv",
            )
        show_dataframe(table, height=min(440, 38 + 35 * len(table)))


def main() -> None:
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)
    init_state()

    store_id, item_id, days_ahead, run_clicked = render_sidebar()

    if run_clicked or not st.session_state.has_run:
        st.session_state.has_run = True
        run_forecast(store_id, item_id, days_ahead)

    payload = st.session_state.payload
    render_header(payload["params"] if payload else None)

    if st.session_state.error is not None:
        st.error(st.session_state.error)
        return
    if payload is None:
        return

    hist, fc, params = payload["hist"], payload["fc"], payload["params"]

    render_kpis(hist, fc)
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_charts(hist, fc)
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_table(fc, params)


if __name__ == "__main__":
    main()