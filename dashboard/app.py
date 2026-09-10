"""
BlockWatch -- AI-Powered Bitcoin Transaction Monitoring Dashboard
SIH 2026 -- Problem Statement 26146


Run from the dashboard/ folder:
    streamlit run app.py
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import auth

st.set_page_config(
    page_title="BlockWatch | Bitcoin Transaction Monitor",
    page_icon="\u26D3",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
GRAPH_PATH = os.path.join(BASE_DIR, "..", "data", "graph", "transaction_network.graphml")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "clean")

# ---------------------------------------------------------------------------
# THEME / CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
    --bg-page:      #0E1116;
    --bg-card:      #171B22;
    --bg-card-2:    #1E232C;
    --border-hair:  #262C36;
    --accent-btc:   #F7931A;
    --accent-btc-dim: rgba(247,147,26,0.14);
    --green:        #3ECF8E;
    --green-dim:    rgba(62,207,142,0.14);
    --red:          #FF5C7A;
    --red-dim:      rgba(255,92,122,0.14);
    --amber:        #FFB020;
    --amber-dim:    rgba(255,176,32,0.14);
    --text-primary: #EDEFF3;
    --text-muted:   #8A93A3;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --font-sans: -apple-system, "Segoe UI", system-ui, sans-serif;
}

html, body, [class*="css"] { background-color: var(--bg-page) !important; color: var(--text-primary); font-family: var(--font-sans); }
[data-testid="stAppViewContainer"] { background: var(--bg-page); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background-color: var(--bg-page); border-right: 1px solid var(--border-hair); }
#MainMenu, footer { visibility: hidden; }

/* ---- Sidebar brand ---- */
.brand-row { display:flex; align-items:center; gap:10px; padding: 6px 4px 18px 4px; }
.brand-glyph {
    width:38px; height:38px; border-radius:10px;
    background: linear-gradient(135deg, var(--accent-btc), #C96A00);
    display:flex; align-items:center; justify-content:center;
    font-size:1.3rem; font-weight:800; color:#14100A;
}
.brand-name { font-weight:800; font-size:1.05rem; color:var(--text-primary); line-height:1.1; }
.brand-sub { font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono); }

.nav-section-label {
    font-size: 0.68rem; letter-spacing:0.06em; color: var(--text-muted);
    margin: 14px 4px 6px 4px; font-family: var(--font-mono);
}

/* ---- Cards ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-hair);
    border-radius: 18px;
    padding: 18px 20px;
}
.card-title { font-weight:700; font-size:0.95rem; color:var(--text-primary); margin-bottom:2px; }
.card-sub { font-size:0.75rem; color:var(--text-muted); margin-bottom:10px; }

/* ---- KPI cards ---- */
.kpi-label { color: var(--text-muted); font-size:0.78rem; margin-bottom:6px; }
.kpi-value { font-size:1.7rem; font-weight:800; color:var(--text-primary); margin-bottom:6px; }
.kpi-delta { font-size:0.76rem; font-weight:700; border-radius:20px; padding:2px 9px; display:inline-block; }
.kpi-delta.up   { background: var(--green-dim); color: var(--green); }
.kpi-delta.down { background: var(--red-dim); color: var(--red); }
.kpi-delta.flat { background: var(--accent-btc-dim); color: var(--accent-btc); }

/* ---- Badges ---- */
.badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:0.72rem; font-weight:700; }
.badge-high   { background: var(--red-dim);   color: var(--red); }
.badge-medium { background: var(--amber-dim); color: var(--amber); }
.badge-low    { background: var(--green-dim); color: var(--green); }

.dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:8px; }
.dot-high { background: var(--red); } .dot-medium { background: var(--amber); } .dot-low { background: var(--green); }

.wallet-mono { font-family: var(--font-mono); font-size:0.82rem; color:var(--text-primary); }

/* ---- Activity feed rows ---- */
.feed-row { display:flex; gap:10px; padding: 9px 0; border-bottom: 1px solid var(--border-hair); }
.feed-row:last-child { border-bottom: none; }
.feed-icon { width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:0.85rem; flex-shrink:0; }
.feed-text { font-size:0.83rem; color: var(--text-primary); line-height:1.25; }
.feed-time { font-size:0.7rem; color: var(--text-muted); font-family:var(--font-mono); }

/* ---- CTA card ---- */
.cta-card {
    background: linear-gradient(135deg, #1B2230, #171B22 60%);
    border: 1px solid var(--border-hair);
    border-radius: 18px;
    padding: 20px;
}
.cta-badge { color: var(--accent-btc); font-size:0.72rem; font-weight:700; font-family: var(--font-mono); }
.cta-big { font-size:2.1rem; font-weight:800; color: var(--text-primary); margin: 6px 0 2px 0; }

hr.hr { border:none; border-top:1px solid var(--border-hair); margin: 1.1rem 0; }

.stButton>button {
    background-color: var(--accent-btc); color:#14100A; font-weight:700;
    border:none; border-radius: 10px; padding: 0.5rem 1.1rem;
}
.stButton>button:hover { background-color:#FFA940; color:#14100A; }

.stRadio > div { gap: 2px; }
.stRadio label { padding: 8px 10px; border-radius: 10px; width:100%; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def badge_html(risk):
    r = str(risk).lower()
    cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(r, "badge-low")
    return f"<span class='badge {cls}'>{risk}</span>"


def dot_html(risk):
    r = str(risk).lower()
    cls = {"high": "dot-high", "medium": "dot-medium", "low": "dot-low"}.get(r, "dot-low")
    return f"<span class='dot {cls}'></span>"


def relative_time(ts):
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


@st.cache_data
def load_alerts():
    candidates = ["wallet_alerts_shap.csv", "wallet_alerts.csv", "wallet_scores.csv"]
    for name in candidates:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "rank" not in df.columns:
                sort_col = "suspicion_score" if "suspicion_score" in df.columns else df.columns[-1]
                df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
                df.insert(0, "rank", df.index + 1)
            if "risk_level" not in df.columns and "suspicion_score" in df.columns:
                df["risk_level"] = pd.cut(df["suspicion_score"], bins=[-1, 40, 70, 101], labels=["Low", "Medium", "High"])
            if "confidence_pct" not in df.columns and "suspicion_score" in df.columns:
                df["confidence_pct"] = df["suspicion_score"]
            if "suspicion_score" not in df.columns and "confidence_pct" in df.columns:
                df["suspicion_score"] = df["confidence_pct"]
            return df
    return None


@st.cache_resource
def load_graph():
    try:
        import networkx as nx
    except ImportError:
        return None
    if not os.path.exists(GRAPH_PATH):
        return None
    G = nx.read_graphml(GRAPH_PATH)
    return G


def pipeline_activity_feed():
    """Build a real activity feed from actual output file timestamps."""
    checkpoints = [
        (os.path.join(DATA_CLEAN_DIR, "ledger.csv"), "\U0001F4C4", "Ledger data generated & validated"),
        (os.path.join(DATA_CLEAN_DIR, "network_log.csv"), "\U0001F310", "Network log ingested"),
        (os.path.join(BASE_DIR, "..", "data", "graph", "transaction_network.graphml"), "\U0001F578", "Transaction graph built"),
        (os.path.join(MODELS_DIR, "wallet_features.csv"), "\U0001F9EE", "Wallet features extracted"),
        (os.path.join(MODELS_DIR, "isolation_forest_model.pkl"), "\U0001F916", "Isolation Forest model trained"),
        (os.path.join(MODELS_DIR, "wallet_scores.csv"), "\u26A0\uFE0F", "Anomaly scores generated"),
        (os.path.join(MODELS_DIR, "wallet_alerts_shap.csv"), "\U0001F4A1", "SHAP explanations generated"),
        (os.path.join(MODELS_DIR, "evaluation_results.csv"), "\u2705", "Model evaluated against ground truth"),
    ]
    rows = []
    for path, icon, label in checkpoints:
        if os.path.exists(path):
            rows.append((os.path.getmtime(path), icon, label))
    rows.sort(reverse=True)
    return rows


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.full_name = ""

if not st.session_state.logged_in:
    st.markdown(
        "<div style='max-width:420px;margin:4rem auto 0 auto;text-align:center;'>"
        "<div class='brand-glyph' style='margin:0 auto 14px auto;'>\u20BF</div>"
        "<div style='font-size:1.8rem;font-weight:800;'>BlockWatch</div>"
        "<div style='color:var(--text-muted);font-family:var(--font-mono);font-size:0.85rem;margin-bottom:1.6rem;'>"
        "offline bitcoin transaction intelligence</div></div>",
        unsafe_allow_html=True,
    )
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        t1, t2 = st.tabs(["Log in", "Create account"])
        with t1:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    ok, msg, full_name = auth.login(u, p)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.full_name = full_name
                        st.rerun()
                    else:
                        st.error(msg)
        with t2:
            with st.form("signup_form"):
                name = st.text_input("Full name")
                u2 = st.text_input("Choose a username")
                p2 = st.text_input("Choose a password", type="password")
                if st.form_submit_button("Create account", use_container_width=True):
                    ok, msg = auth.signup(u2, p2, name)
                    (st.success if ok else st.error)(msg)
    st.stop()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='brand-row'>"
        "<div class='brand-glyph'>\u20BF</div>"
        "<div><div class='brand-name'>BlockWatch</div>"
        f"<div class='brand-sub'>{st.session_state.full_name}</div></div></div>",
        unsafe_allow_html=True,
    )
    def _on_search_change():
        if st.session_state.get("global_search"):
            st.session_state.nav_radio = "Wallet Explorer"

    st.text_input(
        "Search wallet...", key="global_search", label_visibility="collapsed",
        placeholder="\U0001F50D Search wallet...", on_change=_on_search_change,
    )
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"
    nav_options = ["Overview", "Alerts", "Network Graph", "Wallet Explorer"]
    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = "Overview"
    st.markdown("<div class='nav-section-label'>DASHBOARDS</div>", unsafe_allow_html=True)
    page = st.radio(
        "nav1", nav_options,
        label_visibility="collapsed", key="nav_radio",
    )
    st.session_state.current_page = page



    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    if st.button("Log out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    st.markdown(
        "<div style='font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-top:10px;'>"
        "PS 26146 &middot; offline &middot; SIH 2026</div>",
        unsafe_allow_html=True,
    )

alerts_df = load_alerts()

if alerts_df is None:
    st.warning("No model output found yet. Run the model pipeline first so this dashboard has data to show.")
    st.stop()

n_total = len(alerts_df)
n_flagged = int((alerts_df["is_anomaly"] == "Yes").sum()) if "is_anomaly" in alerts_df.columns else 0
n_high = int((alerts_df.get("risk_level") == "High").sum())
avg_conf = round(alerts_df.loc[alerts_df["is_anomaly"] == "Yes", "confidence_pct"].mean(), 1) if n_flagged else 0
flag_rate = round(n_flagged / n_total * 100, 1) if n_total else 0


# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        f"<div style='color:var(--text-muted);font-size:0.8rem;'>Dashboards &nbsp;/&nbsp; "
        f"<span style='color:var(--text-primary);font-weight:700;'>{page}</span></div>",
        unsafe_allow_html=True,
    )
with top_r:
    st.markdown(
        "<div style='text-align:right;color:var(--text-muted);font-family:var(--font-mono);font-size:0.78rem;'>Today</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.markdown("<div style='font-size:1.5rem;font-weight:800;margin:6px 0 16px 0;'>Overview</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "Total wallets scanned", f"{n_total:,}", "flat", "in this dataset"),
        (k2, "Flagged as anomalous", f"{n_flagged:,}", "down", f"{flag_rate}% of all wallets"),
        (k3, "High risk alerts", f"{n_high:,}", "down", "needs review"),
        (k4, "Avg. confidence", f"{avg_conf}%", "up", "on flagged wallets"),
    ]
    for col, label, value, trend, sub in kpis:
        with col:
            trend_cls = {"up": "up", "down": "down", "flat": "flat"}[trend]
            arrow = {"up": "\u2191", "down": "\u2193", "flat": "\u2022"}[trend]
            st.markdown(
                f"<div class='card'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{value}</div>"
                f"<span class='kpi-delta {trend_cls}'>{arrow} {sub}</span></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    main_col, side_col = st.columns([2.1, 1])

    with main_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Detection Overview</div><div class='card-sub'>Risk composition of all flagged wallets</div>", unsafe_allow_html=True)

        d1, d2 = st.columns([1.1, 1])
        with d1:
            risk_counts = alerts_df[alerts_df["is_anomaly"] == "Yes"]["risk_level"].value_counts()
            if len(risk_counts):
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values, hole=0.68,
                    marker=dict(colors=[{"High": "#FF5C7A", "Medium": "#FFB020", "Low": "#3ECF8E"}.get(l, "#8A93A3") for l in risk_counts.index]),
                    textinfo="none",
                )])
                fig.update_layout(
                    showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=200,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(text=f"{n_flagged}<br><span style='font-size:11px;color:#8A93A3'>flagged</span>",
                                       x=0.5, y=0.5, font=dict(size=20, color="#EDEFF3"), showarrow=False)],
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with d2:
            st.write("")
            for level, color in [("High", "#FF5C7A"), ("Medium", "#FFB020"), ("Low", "#3ECF8E")]:
                cnt = int(risk_counts.get(level, 0)) if len(risk_counts) else 0
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 0;'>"
                    f"<span><span class='dot' style='background:{color};'></span>{level} risk</span>"
                    f"<span class='wallet-mono'>{cnt}</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Suspicion score curve</div><div class='card-sub'>Top 100 wallets, ranked</div>", unsafe_allow_html=True)
        top100 = alerts_df.sort_values("suspicion_score", ascending=False).head(100).reset_index(drop=True)
        fig2 = go.Figure(go.Scatter(
            x=top100.index, y=top100["suspicion_score"], mode="lines",
            line=dict(color="#F7931A", width=2), fill="tozeroy", fillcolor="rgba(247,147,26,0.12)",
        ))
        fig2.update_layout(
            height=200, margin=dict(l=0, r=0, t=6, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, title="Rank"), yaxis=dict(showgrid=True, gridcolor="#262C36", title="Score"),
            font=dict(color="#8A93A3", size=11),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        wl, cta = st.columns([1.6, 1])
        with wl:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Wallet list</div><div class='card-sub'>Highest suspicion score first</div>", unsafe_allow_html=True)
            for _, row in alerts_df.sort_values("suspicion_score", ascending=False).head(6).iterrows():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--border-hair);'>"
                    f"<span>{dot_html(row.get('risk_level',''))}<span class='wallet-mono'>{str(row['wallet'])[:22]}...</span></span>"
                    f"<span class='wallet-mono' style='color:var(--text-muted);'>{row.get('tx_count','-')} tx</span>"
                    f"<span class='wallet-mono' style='color:var(--accent-btc);'>{row.get('confidence_pct', row.get('suspicion_score',''))}%</span>"
                    f"</div>", unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with cta:
            st.markdown(
                "<div class='cta-card'>"
                "<div class='cta-badge'>MODEL</div>"
                "<div class='cta-big'>Isolation<br>Forest</div>"
                "<div style='color:var(--text-muted);font-size:0.8rem;margin:6px 0 16px 0;'>"
                "Unsupervised anomaly detection &mdash; no labeled fraud data needed. "
                "Evaluated against ground truth.</div>",
                unsafe_allow_html=True,
            )
            if st.button("View full alert list", use_container_width=True):
                st.session_state.current_page = "Alerts"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with side_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Pipeline activity</div><div class='card-sub'>Live, from file timestamps</div>", unsafe_allow_html=True)
        feed = pipeline_activity_feed()
        if feed:
            for ts, icon, label in feed[:8]:
                st.markdown(
                    f"<div class='feed-row'><div class='feed-icon' style='background:var(--accent-btc-dim);'>{icon}</div>"
                    f"<div><div class='feed-text'>{label}</div><div class='feed-time'>{relative_time(ts)}</div></div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='card-sub'>No pipeline output found yet.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Top flagged wallets</div>", unsafe_allow_html=True)
        for _, row in alerts_df[alerts_df["is_anomaly"] == "Yes"].sort_values("suspicion_score", ascending=False).head(5).iterrows():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;padding:7px 0;'>"
                f"<span>{dot_html(row.get('risk_level',''))}<span class='wallet-mono' style='font-size:0.76rem;'>{str(row['wallet'])[:16]}...</span></span>"
                f"{badge_html(row.get('risk_level',''))}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: ALERTS
# ---------------------------------------------------------------------------
elif page == "Alerts":
    st.markdown("<div style='font-size:1.5rem;font-weight:800;margin:6px 0 16px 0;'>Alerts</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        risk_filter = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with c2:
        only_flagged = st.checkbox("Only flagged", value=True)
    with c3:
        search = st.text_input("Search wallet", label_visibility="visible")

    view = alerts_df.copy()
    if only_flagged:
        view = view[view["is_anomaly"] == "Yes"]
    if risk_filter:
        view = view[view["risk_level"].isin(risk_filter)]
    if search:
        view = view[view["wallet"].astype(str).str.contains(search, case=False, na=False)]

    cols = [c for c in ["rank", "wallet", "confidence_pct", "risk_level", "reason"] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True, height=520)
    st.download_button("Download CSV", view.to_csv(index=False).encode(), "filtered_alerts.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: WALLET EXPLORER
# ---------------------------------------------------------------------------
elif page == "Wallet Explorer":
    st.markdown("<div style='font-size:1.5rem;font-weight:800;margin:6px 0 16px 0;'>Wallet Explorer</div>", unsafe_allow_html=True)
    q = st.text_input(
        "Enter a wallet address (or part of it)",
        value=st.session_state.get("global_search", ""),
    )
    if q:
        matches = alerts_df[alerts_df["wallet"].astype(str).str.contains(q, case=False, na=False)]
        if len(matches) == 0:
            st.info("No wallet matched.")
        for _, row in matches.head(20).iterrows():
            st.markdown(
                f"<div class='card' style='margin-bottom:10px;'>"
                f"<div class='wallet-mono' style='font-size:1rem;margin-bottom:8px;'>{row['wallet']}</div>"
                f"{badge_html(row.get('risk_level',''))} "
                f"<span class='wallet-mono' style='color:var(--text-muted);'>confidence: {row.get('confidence_pct','')}%</span>"
                f"<div style='margin-top:10px;'>{row.get('reason','No explanation available.')}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='card-sub'>Type a wallet address above to see its risk profile.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: NETWORK GRAPH
# ---------------------------------------------------------------------------
elif page == "Network Graph":
    st.markdown("<div style='font-size:1.5rem;font-weight:800;margin:6px 0 16px 0;'>Network Graph</div>", unsafe_allow_html=True)
    top_n = st.slider("Top flagged wallets to visualize", 5, 50, 15)
    G = load_graph()
    if G is None:
        st.warning("Graph file not found.")
    else:
        import networkx as nx
        flagged = alerts_df[alerts_df["is_anomaly"] == "Yes"].head(top_n)["wallet"].tolist()
        flagged_in_graph = [w for w in flagged if w in G.nodes]
        if not flagged_in_graph:
            st.info("No flagged wallets found in the sampled graph.")
        else:
            sub_nodes = set(flagged_in_graph)
            for w in flagged_in_graph:
                sub_nodes.update(list(G.neighbors(w))[:8])
            H = G.subgraph(sub_nodes)
            pos = nx.spring_layout(H, seed=42, k=0.6)
            ex, ey = [], []
            for a, b in H.edges():
                ex += [pos[a][0], pos[b][0], None]
                ey += [pos[a][1], pos[b][1], None]
            edge_trace = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#262C36"), hoverinfo="none")
            nx_, ny_, nc_, nt_ = [], [], [], []
            for n in H.nodes():
                nx_.append(pos[n][0]); ny_.append(pos[n][1])
                nc_.append("#FF5C7A" if n in flagged_in_graph else "#3ECF8E")
                nt_.append(str(n))
            node_trace = go.Scatter(x=nx_, y=ny_, mode="markers", marker=dict(size=10, color=nc_, line=dict(width=1, color="#0E1116")), text=nt_, hoverinfo="text")
            fig = go.Figure([edge_trace, node_trace])
            fig.update_layout(showlegend=False, plot_bgcolor="#0E1116", paper_bgcolor="#0E1116",
                               margin=dict(l=10, r=10, t=10, b=10),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=560)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<span style='color:#FF5C7A;'>&#9679;</span> flagged &nbsp;&nbsp;<span style='color:#3ECF8E;'>&#9679;</span> connected", unsafe_allow_html=True)
