from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Yes4All Commerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PALETTE = {
    "navy": "#12304A",
    "blue": "#2563EB",
    "teal": "#0891B2",
    "green": "#16A34A",
    "amber": "#D97706",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "slate": "#64748B",
    "light": "#F6F8FC",
    "grid": "#E2E8F0",
}

CANDIDATE_MODELS = ["Naive", "MA3", "ETS", "SARIMA", "Prophet", "LightGBM_Global", "Ens_ETS_LGBM", "Ens_Median4"]


st.markdown(
    """
    <style>
    :root { --ink:#14213D; --muted:#64748B; --line:#E2E8F0; --panel:#FFFFFF; }
    .stApp { background:#F6F8FC; color:#14213D; }
    .block-container { max-width: 1550px; padding-top: 1.2rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background:rgba(246,248,252,.88); }
    [data-testid="stSidebar"] { display:none; }
    h1,h2,h3,h4,p,span,label,[data-testid="stMarkdownContainer"] { color:#14213D; }
    .app-header {
        background:linear-gradient(118deg,#12304A 0%,#164E63 56%,#2563EB 100%);
        color:white; border-radius:18px; padding:22px 26px; margin-bottom:12px;
        box-shadow:0 14px 30px rgba(18,48,74,.16);
    }
    .app-header h1 { color:white; margin:0; font-size:2rem; letter-spacing:-.02em; }
    .app-header p { color:#DBEAFE; margin:.45rem 0 0; }
    .filter-shell {
        background:white; border:1px solid #E2E8F0; border-radius:16px;
        padding:12px 14px 2px; margin:8px 0 14px; box-shadow:0 6px 18px rgba(15,23,42,.05);
    }
    [data-testid="stMetric"] {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px;
        padding:14px 15px; min-height:118px; box-shadow:0 5px 16px rgba(15,23,42,.045);
    }
    [data-testid="stMetricLabel"] p { color:#64748B !important; font-size:.86rem; }
    [data-testid="stMetricValue"] { color:#12304A !important; font-size:1.75rem; }
    [data-testid="stMetricDelta"] { font-size:.82rem; }
    .callout {
        background:#EFF6FF; border:1px solid #BFDBFE; border-left:5px solid #2563EB;
        border-radius:12px; padding:12px 15px; margin:7px 0 15px; color:#1E3A8A;
    }
    .warning {
        background:#FFF7ED; border:1px solid #FED7AA; border-left:5px solid #D97706;
        border-radius:12px; padding:12px 15px; margin:7px 0 15px; color:#7C2D12;
    }
    .demo {
        display:inline-block; color:#6D28D9; background:#F3E8FF; border:1px solid #DDD6FE;
        border-radius:999px; padding:3px 9px; font-size:.78rem; font-weight:700;
    }
    .status-ok { color:#166534; background:#DCFCE7; border-radius:999px; padding:4px 9px; font-weight:700; }
    .status-risk { color:#991B1B; background:#FEE2E2; border-radius:999px; padding:4px 9px; font-weight:700; }
    div[data-testid="stDataFrame"] { border:1px solid #E2E8F0; border-radius:12px; overflow:hidden; }
    div[data-baseweb="select"] > div, [data-testid="stDateInput"] > div { background:white; }
    .stButton > button, .stLinkButton > a { border-radius:10px; font-weight:700; }
    [data-testid="stPlotlyChart"] { background:white; border:1px solid #E2E8F0; border-radius:14px; padding:5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def bundle_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "app_data_v4.zip",
        Path(__file__).resolve().parent / "data" / "app_data_v4.zip",
        Path(__file__).resolve().parent / "app_data_v3.zip",
        Path(__file__).resolve().parent / "data" / "app_data_v3.zip",
        Path(__file__).resolve().parent / "app_data_v2.zip",
        Path(__file__).resolve().parent / "data" / "app_data_v2.zip",
        Path.cwd() / "app_data_v2.zip",
        Path.cwd() / "data" / "app_data_v2.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy app_data_v4.zip/app_data_v3.zip cạnh streamlit_app.py hoặc trong thư mục data.")


@st.cache_data(show_spinner=False)
def bundle_manifest(path: str, modified_ns: int) -> tuple[list[str], dict]:
    del modified_ns
    with zipfile.ZipFile(path) as archive:
        names = [name.removesuffix(".parquet") for name in archive.namelist() if name.endswith(".parquet")]
        quality = json.loads(archive.read("quality_summary.json").decode("utf-8"))
    return names, quality


@st.cache_resource(show_spinner=False, max_entries=24)
def load_table(path: str, modified_ns: int, table_name: str) -> pd.DataFrame:
    del modified_ns
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_parquet(io.BytesIO(archive.read(f"{table_name}.parquet")))
    for column in ["Day", "month", "Month", "snapshot_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


class LazyTables:
    def __init__(self, path: Path, names: list[str]):
        self.path = path
        self.names = names
        self.modified_ns = path.stat().st_mtime_ns

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.names:
            raise KeyError(name)
        return load_table(str(self.path), self.modified_ns, name)

    def get(self, name: str, default=None):
        return self[name] if name in self.names else default

    def items(self):
        for name in self.names:
            yield name, self[name]


def fmt_number(value: float, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f}"


def fmt_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def safe_divide(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def chart_style(fig: go.Figure, height: int = 390, reverse_y: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=28, r=22, t=56, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#14213D", family="Inter, Arial, sans-serif"),
        title_font=dict(size=16, color="#12304A"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#12304A", font_color="white"),
    )
    fig.update_xaxes(gridcolor="#EEF2F7", linecolor="#CBD5E1", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F7", linecolor="#CBD5E1", zeroline=False, autorange="reversed" if reverse_y else True)
    return fig


def apply_dimensions(frame: pd.DataFrame, product_lines, teams, channels, skus, asins, search_text: str) -> pd.DataFrame:
    result = frame.copy()
    if product_lines and "product_line" in result.columns:
        result = result[result["product_line"].astype(str).isin(product_lines)]
    if teams and "team" in result.columns:
        result = result[result["team"].astype(str).isin(teams)]
    if channels and "channel" in result.columns:
        result = result[result["channel"].astype(str).isin(channels)]
    if skus and "SKU" in result.columns:
        result = result[result["SKU"].astype(str).isin(skus)]
    if asins and "ASIN" in result.columns:
        result = result[result["ASIN"].astype(str).isin(asins)]
    if search_text:
        columns = [column for column in ["SKU", "ASIN", "product_name", "product_line", "campaignName"] if column in result.columns]
        if columns:
            searchable = result[columns].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
            result = result[searchable.str.contains(search_text.strip().lower(), regex=False, na=False)]
    return result


def filter_dates(frame: pd.DataFrame, start_date, end_date, column="Day") -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return frame[frame[column].between(start, end)].copy()


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def chart_help(text: str) -> None:
    safe = str(text).replace('"', '&quot;')
    st.markdown(f'<div style="text-align:right;margin-top:-2.1rem;margin-bottom:.35rem"><span title="{safe}" style="cursor:help;font-size:1.05rem">❓</span></div>', unsafe_allow_html=True)


def build_action_queue(tables: dict[str, pd.DataFrame], acos_limit: float = 0.35) -> pd.DataFrame:
    actions = []
    listing = tables["listing_health_demo"]
    for _, row in listing.iterrows():
        inventory = float(row.get("asin_amz_inventory") or 0) + float(row.get("shared_y4a_inventory") or 0)
        if bool(row.get("listing_blocked_demo", False)):
            actions.append({"Priority": "P0", "Entity": row["ASIN"], "SKU": row["SKU"], "Issue": "Listing blocked (demo)", "Evidence": f"Inventory visible: {inventory:,.0f} units", "Recommended action": "Open Amazon case and verify listing eligibility before increasing traffic", "Data class": "SIMULATED"})
        elif bool(row.get("zip_available_demo", True)) is False:
            actions.append({"Priority": "P1", "Entity": row["ASIN"], "SKU": row["SKU"], "Issue": "ZIP unavailable (demo)", "Evidence": f"ZIP 10001; inventory {inventory:,.0f}", "Recommended action": "Check fulfillment eligibility and open case if stock is available", "Data class": "SIMULATED"})
        elif bool(row.get("long_shipping_demo", False)):
            actions.append({"Priority": "P1", "Entity": row["ASIN"], "SKU": row["SKU"], "Issue": "Long shipping time (demo)", "Evidence": f"Promise: {float(row.get('shipping_days_demo') or 0):.0f} days", "Recommended action": "Rebalance AMZ inventory and check inbound receiving status", "Data class": "SIMULATED"})

    forecast = tables["forecast_supply"]
    risky = forecast[(forecast["war_map_gap_units"] > 0) & forecast["frozen_status"].astype(str).str.contains("Frozen", case=False, na=False)]
    for _, row in risky.iterrows():
        actions.append({"Priority": "P0", "Entity": row["SKU"], "SKU": row["SKU"], "Issue": "Frozen supply below War Map", "Evidence": f"{row['month']:%Y-%m}: shortage {row['war_map_gap_units']:,.0f} units", "Recommended action": "Escalate allocation/PO decision; frozen lead time prevents new buy in this horizon", "Data class": "ACTUAL + CALCULATED"})

    target = tables["target_sku"]
    low_moc = target[target["moc"].between(0.0001, 1.5, inclusive="both")]
    high_moc = target[target["moc"] > 6]
    for _, row in low_moc.iterrows():
        actions.append({"Priority": "P1", "Entity": row["SKU"], "SKU": row["SKU"], "Issue": "MOC below 1.5", "Evidence": f"MOC {row['moc']:.1f}", "Recommended action": "Protect sales, validate incoming and prioritize replenishment", "Data class": "ACTUAL"})
    for _, row in high_moc.head(250).iterrows():
        actions.append({"Priority": "P2", "Entity": row["SKU"], "SKU": row["SKU"], "Issue": "MOC above 6", "Evidence": f"MOC {row['moc']:.1f}", "Recommended action": "Review sell-through plan, promotion and purchase freeze", "Data class": "ACTUAL"})

    ads = tables["ads_campaign"]
    bad_ads = ads[(ads["state"] == "ENABLED") & (ads["ACOS"] > acos_limit) & (ads["SpendUSD"] >= 100)].sort_values("SpendUSD", ascending=False)
    for _, row in bad_ads.iterrows():
        actions.append({"Priority": "P1", "Entity": row["campaignName"], "SKU": "—", "Issue": "Campaign ACOS above threshold", "Evidence": f"ACOS {row['ACOS']:.1%}; spend ${row['SpendUSD']:,.0f}", "Recommended action": "Reduce bid and inspect targets/search terms", "Data class": "ACTUAL"})
    zero_sales = ads[(ads["state"] == "ENABLED") & (ads["SalesUSD"] <= 0) & (ads["SpendUSD"] >= 20)].sort_values("SpendUSD", ascending=False)
    for _, row in zero_sales.iterrows():
        actions.append({"Priority": "P1", "Entity": row["campaignName"], "SKU": "—", "Issue": "Spend with no attributed sales", "Evidence": f"Spend ${row['SpendUSD']:,.0f}; sales $0", "Recommended action": "Pause or add negatives after search-term review", "Data class": "ACTUAL"})
    result = pd.DataFrame(actions)
    if result.empty:
        return result
    order = pd.Categorical(result["Priority"], ["P0", "P1", "P2"], ordered=True)
    return result.assign(_order=order).sort_values(["_order", "Issue", "Entity"]).drop(columns="_order")


DATA_PATH = bundle_path()
TABLE_NAMES, QUALITY = bundle_manifest(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
TABLES = LazyTables(DATA_PATH, TABLE_NAMES)
SALES = TABLES["sales_commercial_monthly"]

st.markdown(
    f"""
    <div class="app-header">
      <h1>Yes4All Commerce Intelligence</h1>
      <p>Commercial Intelligence · Sales history {QUALITY['commercial_sales_min']} → {QUALITY['latest_sales_date']} · Monthly through Aug-2026 + retained Sep-2026 MTD</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "Navigation",
    ["Commercial Intelligence", "Daily Forecast & Supply", "ASIN 360", "Inventory & Forecast", "Sales & Target", "Ads Performance", "Action Center", "Data Quality"],
    default="Commercial Intelligence",
    key="nav_page",
    label_visibility="collapsed",
)

date_min = SALES["Day"].min().date()
date_max = max(SALES["Day"].max().date(), pd.Timestamp(QUALITY["latest_sales_date"]).date())
default_start = max(date_min, pd.Timestamp(year=date_max.year, month=1, day=1).date())
with st.container():
    st.markdown('<div class="filter-shell">', unsafe_allow_html=True)
    row1 = st.columns([1.3, 1.45, 1.15, 1.05, 1.8], gap="small")
    with row1[0]:
        selected_dates = st.date_input("Time range", value=(default_start, date_max), min_value=date_min, max_value=date_max)
    all_product_lines = sorted(SALES["product_line"].dropna().astype(str).unique().tolist())
    with row1[1]:
        selected_product_lines = st.multiselect("Product line", all_product_lines, placeholder="All product lines")
    with row1[2]:
        selected_teams = st.multiselect("Team", sorted(SALES["team"].dropna().astype(str).unique()), placeholder="All teams")
    with row1[3]:
        selected_channels = st.multiselect("Channel", sorted(SALES["channel"].dropna().astype(str).unique()), placeholder="All channels")
    with row1[4]:
        search_text = st.text_input("Fast search", placeholder="Type SKU, ASIN, product or keyword...", help="Searches a compact index and returns at most 50 matches; it does not render a full SKU list.")

    search_source = TABLES.get("search_index", SALES)
    search_columns = [c for c in ["SKU", "ASIN", "product_name", "product_line", "team", "channel"] if c in search_source]
    search_index = search_source[search_columns].fillna("").astype(str).drop_duplicates().reset_index(drop=True)
    search_index["_label"] = search_index.apply(
        lambda r: f"{r.get('SKU', '')} · {r.get('ASIN', '')} · {str(r.get('product_name', ''))[:72]}", axis=1
    )
    matches = search_index.iloc[0:0]
    if len(search_text.strip()) >= 2:
        needle = search_text.strip().lower()
        haystack = search_index[search_columns].agg(" | ".join, axis=1).str.lower()
        matches = search_index[haystack.str.contains(needle, regex=False, na=False)].head(50)

    row2 = st.columns([2.9, 1.2, 1.2, 1.7], gap="small")
    with row2[0]:
        result_options = [None] + matches.index.tolist()
        selected_result = st.selectbox(
            "Matching product",
            result_options,
            format_func=lambda i: "All matching products" if i is None else search_index.loc[i, "_label"],
            disabled=len(matches) == 0,
        )
    selected_skus = [] if selected_result is None else [str(search_index.loc[selected_result, "SKU"])]
    selected_asins = [] if selected_result is None else [str(search_index.loc[selected_result, "ASIN"])]
    effective_search = search_text if selected_result is None else ""
    with row2[1]:
        listing_status_filter = st.multiselect("Listing status", ["Active", "Active with ZIP restriction", "Active with long delivery", "Blocked"], placeholder="All statuses")
    with row2[2]:
        ad_account_filter = st.multiselect("Ads account", sorted(TABLES["ads_campaign"]["Account"].dropna().astype(str).unique()), default=["Yes4All [US]"] if "Yes4All [US]" in set(TABLES["ads_campaign"]["Account"].astype(str)) else [])
    with row2[3]:
        match_note = f"{len(matches)} results (showing max 50)" if len(search_text.strip()) >= 2 else "Type at least 2 characters"
        st.caption(f"Fast index: {match_note} · Daily scope {QUALITY.get('daily_sku', 0):,} SKUs")
    st.markdown("</div>", unsafe_allow_html=True)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates if not isinstance(selected_dates, tuple) else date_max

filtered_sales = apply_dimensions(SALES, selected_product_lines, selected_teams, selected_channels, selected_skus, selected_asins, effective_search)
filtered_sales = filter_dates(filtered_sales, start_date, end_date)

if filtered_sales.empty and page not in {"Ads Performance", "Data Quality"}:
    st.warning("Không có dữ liệu sales trong tổ hợp filter hiện tại. Hãy bỏ bớt filter hoặc đổi khoảng ngày.")


def commercial_intelligence() -> None:
    current = filtered_sales
    gmv = current["Ordered_GMV"].sum()
    nmv = current["Ordered_nmv"].sum()
    units = current["Ordered_units"].sum()
    views = current["Glance_views"].sum()
    ads = current["Total_ADS"].sum()
    promo = current["Total_Promo"].sum()
    ad_units = current["Ad_attributed_units"].sum()
    asp = safe_divide(gmv, units)
    mkt = ads + promo
    mkt_gmv = safe_divide(mkt, gmv)
    ads_gmv = safe_divide(ads, gmv)
    promo_gmv = safe_divide(promo, gmv)
    cpu = safe_divide(mkt, units)
    cm3 = current["CM3"].sum(min_count=1)
    cm3_gmv = safe_divide(cm3, current.loc[current["CM3_costed"], "Ordered_GMV"].sum())
    cm3_coverage = safe_divide(current.loc[current["CM3_costed"], "Ordered_GMV"].sum(), gmv)

    prior_start = pd.Timestamp(start_date) - pd.DateOffset(years=1)
    prior_end = pd.Timestamp(end_date) - pd.DateOffset(years=1)
    prior = filter_dates(apply_dimensions(SALES, selected_product_lines, selected_teams, selected_channels, selected_skus, selected_asins, search_text), prior_start, prior_end)

    def yoy_delta(current_value: float, prior_value: float) -> str | None:
        if prior_value is None or pd.isna(prior_value) or prior_value == 0:
            return None
        return f"{current_value / prior_value - 1:+.1%} YoY"

    prior_gmv = prior["Ordered_GMV"].sum()
    prior_units = prior["Ordered_units"].sum()
    prior_ads = prior["Total_ADS"].sum()
    prior_promo = prior["Total_Promo"].sum()
    prior_cm3 = prior["CM3"].sum(min_count=1)
    prior_asp = safe_divide(prior_gmv, prior_units)
    prior_mkt_gmv = safe_divide(prior_ads + prior_promo, prior_gmv)

    columns = st.columns(5)
    values = [
        ("Ordered GMV", fmt_money(gmv), yoy_delta(gmv, prior_gmv)),
        ("Ordered units", fmt_number(units), yoy_delta(units, prior_units)),
        ("ASP", fmt_money(asp), yoy_delta(asp, prior_asp)),
        ("CM3", fmt_money(cm3), yoy_delta(cm3, prior_cm3)),
        ("CM3 / costed GMV", fmt_pct(cm3_gmv), None),
    ]
    for column, (label, value, delta) in zip(columns, values):
        column.metric(label, value, delta=delta)
    columns = st.columns(5)
    values = [
        ("Ad spend / GMV", fmt_pct(ads_gmv), yoy_delta(ads_gmv, safe_divide(prior_ads, prior_gmv))),
        ("Promo / GMV", fmt_pct(promo_gmv), yoy_delta(promo_gmv, safe_divide(prior_promo, prior_gmv))),
        ("MKT / GMV", fmt_pct(mkt_gmv), yoy_delta(mkt_gmv, prior_mkt_gmv)),
        ("MKT CPU", fmt_money(cpu), None),
        ("CM3 GMV coverage", fmt_pct(cm3_coverage), None),
    ]
    for column, (label, value, delta) in zip(columns, values):
        column.metric(label, value, delta=delta, delta_color="inverse" if label in {"Ad spend / GMV", "Promo / GMV", "MKT / GMV", "MKT CPU"} else "normal")

    scope_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)
    st.markdown(
        f'<div class="callout"><b>Current view:</b> {scope_days} calendar days, {current["SKU"].nunique():,} SKUs and {current["ASIN"].nunique():,} ASINs. '
        f'Paid-attributed units: {ad_units:,.0f}. CM3 covers {fmt_pct(cm3_coverage)} of filtered GMV; uncovered rows are excluded, not treated as zero. September 2026 is partial through {QUALITY["latest_sales_date"]}.</div>',
        unsafe_allow_html=True,
    )

    daily = apply_dimensions(
        TABLES["sales_team_daily"], selected_product_lines, selected_teams, selected_channels,
        selected_skus, selected_asins, effective_search,
    )
    daily = filter_dates(daily, start_date, end_date)
    daily_view = daily.groupby("Day", as_index=False).agg(
        GMV=("Ordered_GMV", "sum"), Units=("Ordered_units", "sum"), Ads=("Total_ADS", "sum"),
        Promo=("Total_Promo", "sum"), CM3=("CM3", lambda s: s.sum(min_count=1)), Costed_GMV=("Ordered_GMV", lambda s: s[daily.loc[s.index, "CM3_costed"]].sum()),
    )
    daily_view["MKT_GMV"] = np.where(daily_view["GMV"] > 0, (daily_view["Ads"] + daily_view["Promo"]) / daily_view["GMV"], np.nan)
    daily_view["CM3_GMV"] = np.where(daily_view["Costed_GMV"] > 0, daily_view["CM3"] / daily_view["Costed_GMV"], np.nan)
    section("Daily overall", "The daily source covers the 163-SKU Tu team scope; portfolio headline metrics above retain the broader commercial mart.")
    chart_help("Bars show daily GMV. The red line is total marketing spend divided by GMV. The green line is CM3 divided by GMV for rows with a valid cost stack. Compare line movements with GMV, but do not interpret co-movement as causality.")
    fig = go.Figure()
    fig.add_bar(x=daily_view["Day"], y=daily_view["GMV"], name="$ GMV", marker_color="#93C5FD")
    fig.add_scatter(x=daily_view["Day"], y=daily_view["MKT_GMV"], name="% MKT / GMV", yaxis="y2", line=dict(color=PALETTE["red"], width=2))
    fig.add_scatter(x=daily_view["Day"], y=daily_view["CM3_GMV"], name="% CM3", yaxis="y2", line=dict(color=PALETTE["green"], width=2))
    fig.update_layout(title="Daily GMV, marketing intensity and CM3", yaxis=dict(title="GMV (USD)", tickprefix="$", tickformat=",.0f"), yaxis2=dict(title="Rate", overlaying="y", side="right", tickformat=".0%", showgrid=False))
    st.plotly_chart(chart_style(fig, 430), width="stretch")

    section("Top Product Line contribution", "Box size = GMV contribution; color = CM3%. Hover to inspect GMV, CPU, MKT/GMV and CM3.")
    chart_help("Large boxes contribute more GMV. Green indicates stronger CM3%; red indicates weaker or negative CM3%. A large red box is a high-priority margin review; a small green box is healthy but not yet material.")
    by_line = current.groupby(["product_line"], as_index=False).agg(
        GMV=("Ordered_GMV", "sum"), Units=("Ordered_units", "sum"), Ads=("Total_ADS", "sum"),
        Promo=("Total_Promo", "sum"), CM3=("CM3", lambda s: s.sum(min_count=1)),
        Costed_GMV=("Ordered_GMV", lambda s: s[current.loc[s.index, "CM3_costed"]].sum()),
    )
    by_line["ASP"] = np.where(by_line["Units"] != 0, by_line["GMV"] / by_line["Units"], np.nan)
    by_line["MKT_GMV"] = np.where(by_line["GMV"] > 0, (by_line["Ads"] + by_line["Promo"]) / by_line["GMV"], np.nan)
    by_line["MKT_CPU"] = np.where(by_line["Units"] > 0, (by_line["Ads"] + by_line["Promo"]) / by_line["Units"], np.nan)
    by_line["CM3_pct_GMV"] = np.where(by_line["Costed_GMV"] > 0, by_line["CM3"] / by_line["Costed_GMV"], np.nan)
    by_line = by_line[by_line["GMV"] > 0].nlargest(20, "GMV")
    fig = px.treemap(by_line, path=["product_line"], values="GMV", color="CM3_pct_GMV", color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"], color_continuous_midpoint=0, custom_data=["GMV", "MKT_CPU", "MKT_GMV", "CM3", "CM3_pct_GMV", "Units"])
    fig.update_traces(hovertemplate="<b>%{label}</b><br>GMV: $%{customdata[0]:,.0f}<br>MKT CPU: $%{customdata[1]:,.0f}<br>MKT / GMV: %{customdata[2]:.1%}<br>CM3: $%{customdata[3]:,.0f}<br>CM3 / GMV: %{customdata[4]:.1%}<br>Units: %{customdata[5]:,.0f}<extra></extra>")
    fig.update_layout(title="Top 20 Product Lines by GMV contribution")
    st.plotly_chart(chart_style(fig, 500), width="stretch")

    section("Marketing relationship views", "Each dot is a SKU in the selected period. These are descriptive relationships, not causal uplift estimates.")
    sku_rel = current.groupby(["SKU", "product_name", "product_line"], as_index=False).agg(
        GMV=("Ordered_GMV", "sum"), Units=("Ordered_units", "sum"), Ads=("Total_ADS", "sum"),
        Promo=("Total_Promo", "sum"), CM3=("CM3", lambda s: s.sum(min_count=1)),
        Costed_GMV=("Ordered_GMV", lambda s: s[current.loc[s.index, "CM3_costed"]].sum()),
    )
    sku_rel["ASP"] = sku_rel["GMV"].div(sku_rel["Units"].replace(0, np.nan))
    sku_rel["MKT_CPU"] = (sku_rel["Ads"] + sku_rel["Promo"]).div(sku_rel["Units"].replace(0, np.nan))
    sku_rel["MKT_CPU_bubble"] = sku_rel["MKT_CPU"].clip(lower=0).fillna(0) + 0.01
    sku_rel["CM3_CPU"] = sku_rel["CM3"].div(sku_rel["Units"].replace(0, np.nan))
    sku_rel["CM3_pct_GMV"] = sku_rel["CM3"].div(sku_rel["Costed_GMV"].replace(0, np.nan))
    sku_rel = sku_rel.replace([np.inf, -np.inf], np.nan).nlargest(500, "GMV")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_help("Dots above the diagonal tendency have relatively more Ads than Promo; dots to the right rely more on Promo. Bubble size is GMV and color is CM3%. Use this to identify expensive marketing mixes, not to claim one spend type caused sales.")
        fig = px.scatter(sku_rel, x="Promo", y="Ads", size="GMV", color="CM3_pct_GMV", hover_name="SKU", hover_data=["product_name", "product_line", "GMV", "Units"], color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"], color_continuous_midpoint=0, title="Promo vs Ads investment by SKU")
        fig.update_xaxes(title="Promo spend (USD)", tickprefix="$", tickformat=",.0f")
        fig.update_yaxes(title="Ads spend (USD)", tickprefix="$", tickformat=",.0f")
        st.plotly_chart(chart_style(fig, 470), width="stretch")
    with c2:
        chart_help("X is average selling price. Y is CM3 per sold unit. Bubble size is marketing cost per unit. A large bubble with low CM3/unit signals that marketing is consuming margin; a higher ASP does not automatically mean better CM3.")
        fig = px.scatter(sku_rel.dropna(subset=["ASP", "CM3_CPU", "MKT_CPU"]), x="ASP", y="CM3_CPU", size="MKT_CPU_bubble", color="CM3_pct_GMV", hover_name="SKU", hover_data={"product_name": True, "product_line": True, "GMV": ":$,.0f", "Units": ":,.0f", "MKT_CPU": ":$,.0f", "MKT_CPU_bubble": False}, color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"], color_continuous_midpoint=0, title="Selling price × marketing cost × CM3")
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["red"])
        fig.update_xaxes(title="ASP (USD)", tickprefix="$", tickformat=",.0f")
        fig.update_yaxes(title="CM3 per unit (USD)", tickprefix="$", tickformat=",.0f")
        st.plotly_chart(chart_style(fig, 470), width="stretch")

    section("PIC scorecard")
    pic = current.groupby("team", as_index=False).agg(
        GMV=("Ordered_GMV", "sum"), Units=("Ordered_units", "sum"), Ads=("Total_ADS", "sum"),
        Promo=("Total_Promo", "sum"), CM3=("CM3", lambda s: s.sum(min_count=1)),
        Costed_GMV=("Ordered_GMV", lambda s: s[current.loc[s.index, "CM3_costed"]].sum()),
        Ad_units=("Ad_attributed_units", "sum"), SKU=("SKU", "nunique"),
    )
    pic["ASP"] = np.where(pic["Units"] != 0, pic["GMV"] / pic["Units"], np.nan)
    pic["Ads_GMV"] = np.where(pic["GMV"] > 0, pic["Ads"] / pic["GMV"], np.nan)
    pic["Promo_GMV"] = np.where(pic["GMV"] > 0, pic["Promo"] / pic["GMV"], np.nan)
    pic["MKT_GMV"] = np.where(pic["GMV"] > 0, (pic["Ads"] + pic["Promo"]) / pic["GMV"], np.nan)
    pic["MKT_CPU"] = np.where(pic["Units"] > 0, (pic["Ads"] + pic["Promo"]) / pic["Units"], np.nan)
    pic["CM3_GMV"] = np.where(pic["Costed_GMV"] > 0, pic["CM3"] / pic["Costed_GMV"], np.nan)
    st.dataframe(pic.sort_values("GMV", ascending=False), width="stretch", hide_index=True, column_config={
        "GMV": st.column_config.NumberColumn(format="$%.0f"), "ASP": st.column_config.NumberColumn(format="$%.0f"),
        "Ads": st.column_config.NumberColumn(format="$%.0f"), "Promo": st.column_config.NumberColumn(format="$%.0f"),
        "CM3": st.column_config.NumberColumn(format="$%.0f"),
        "Ads_GMV": st.column_config.NumberColumn("Ads / GMV", format="percent"),
        "Promo_GMV": st.column_config.NumberColumn("Promo / GMV", format="percent"),
        "MKT_GMV": st.column_config.NumberColumn("MKT / GMV", format="percent"),
        "MKT_CPU": st.column_config.NumberColumn(format="$%.0f"),
        "CM3_GMV": st.column_config.NumberColumn("CM3 / GMV", format="percent"),
    })

    section("Sales investment planner", "Example: choose 10 units and $25 ASP. Budgets use historical ratios in the filtered scope; they are planning benchmarks, not guaranteed causal requirements.")
    calc_cols = st.columns([1, 1, 1.2])
    with calc_cols[0]:
        desired_units = st.number_input("Desired units", min_value=1, value=10, step=1)
    with calc_cols[1]:
        selling_price = st.number_input("Planned ASP (USD)", min_value=0.01, value=25.0, step=1.0)
    with calc_cols[2]:
        benchmark = st.selectbox("Benchmark window", ["Selected scope", "Latest 6 months in selected scope"])
    benchmark_data = current.copy()
    if benchmark.startswith("Latest 6") and not current.empty:
        cutoff = current["Day"].max() - pd.DateOffset(months=5)
        benchmark_data = current[current["Day"].ge(cutoff)]
    bench_gmv = benchmark_data["Ordered_GMV"].sum()
    bench_units = benchmark_data["Ordered_units"].sum()
    bench_ads = benchmark_data["Total_ADS"].sum()
    bench_promo = benchmark_data["Total_Promo"].sum()
    bench_ad_units = benchmark_data["Ad_attributed_units"].sum()
    target_gmv = float(desired_units) * float(selling_price)
    rate_ads = safe_divide(bench_ads, bench_gmv)
    rate_promo = safe_divide(bench_promo, bench_gmv)
    estimated_ads = target_gmv * (0 if pd.isna(rate_ads) else rate_ads)
    estimated_promo = target_gmv * (0 if pd.isna(rate_promo) else rate_promo)
    estimated_mkt = estimated_ads + estimated_promo
    estimated_paid_units = desired_units * (0 if pd.isna(safe_divide(bench_ad_units, bench_units)) else safe_divide(bench_ad_units, bench_units))
    planner_cards = st.columns(6)
    planner_values = [
        ("Target GMV", fmt_money(target_gmv)), ("Ads benchmark", fmt_money(estimated_ads)),
        ("Promo benchmark", fmt_money(estimated_promo)), ("Total MKT", fmt_money(estimated_mkt)),
        ("MKT CPU", fmt_money(safe_divide(estimated_mkt, desired_units))),
        ("Expected paid-attributed units", fmt_number(estimated_paid_units, 1)),
    ]
    for column, (label, value) in zip(planner_cards, planner_values):
        column.metric(label, value)
    st.caption("Promo Units are not calculated because the source has no promotion-attributed unit field. Ads attributed units are source-backed from SB/SD/SP/DSP/Aff.")

    section("SKU detail", "Operational table at the selected time range. Currency is rounded to whole USD; units are rounded to whole units.")
    sku_detail = current.groupby(["SKU", "product_name", "product_line", "team", "channel"], as_index=False).agg(
        Units=("Ordered_units", "sum"), GMV=("Ordered_GMV", "sum"), Ads=("Total_ADS", "sum"),
        Promo=("Total_Promo", "sum"), CM3=("CM3", lambda s: s.sum(min_count=1)),
        Costed_GMV=("Ordered_GMV", lambda s: s[current.loc[s.index, "CM3_costed"]].sum()),
        Ad_units=("Ad_attributed_units", "sum"), Views=("Glance_views", "sum"),
    )
    sku_detail["ASP"] = sku_detail["GMV"].div(sku_detail["Units"].replace(0, np.nan))
    sku_detail["MKT_GMV"] = (sku_detail["Ads"] + sku_detail["Promo"]).div(sku_detail["GMV"].replace(0, np.nan))
    sku_detail["MKT_CPU"] = (sku_detail["Ads"] + sku_detail["Promo"]).div(sku_detail["Units"].replace(0, np.nan))
    sku_detail["CM3_GMV"] = sku_detail["CM3"].div(sku_detail["Costed_GMV"].replace(0, np.nan))
    sku_detail["CM3_coverage"] = sku_detail["Costed_GMV"].div(sku_detail["GMV"].replace(0, np.nan))
    inv = TABLES["inventory_current_sku"][["SKU", "current_inventory", "incoming_inventory", "moc"]].drop_duplicates("SKU")
    sku_detail = sku_detail.merge(inv, on="SKU", how="left")
    sku_detail = sku_detail[["SKU", "product_name", "product_line", "team", "channel", "current_inventory", "incoming_inventory", "moc", "Units", "GMV", "ASP", "Ads", "Promo", "MKT_GMV", "MKT_CPU", "Ad_units", "CM3", "CM3_GMV", "CM3_coverage", "Views"]]
    st.dataframe(
        sku_detail.sort_values("GMV", ascending=False), width="stretch", hide_index=True, height=620,
        column_config={
            "current_inventory": st.column_config.NumberColumn("Inventory", format="%.0f"),
            "incoming_inventory": st.column_config.NumberColumn("Incoming", format="%.0f"),
            "moc": st.column_config.NumberColumn("MOC", format="%.1f"),
            "Units": st.column_config.NumberColumn(format="%.0f"), "Ad_units": st.column_config.NumberColumn(format="%.0f"),
            "Views": st.column_config.NumberColumn(format="%.0f"),
            "GMV": st.column_config.NumberColumn(format="$%.0f"), "ASP": st.column_config.NumberColumn(format="$%.0f"),
            "Ads": st.column_config.NumberColumn(format="$%.0f"), "Promo": st.column_config.NumberColumn(format="$%.0f"),
            "MKT_CPU": st.column_config.NumberColumn(format="$%.0f"), "CM3": st.column_config.NumberColumn(format="$%.0f"),
            "MKT_GMV": st.column_config.NumberColumn("MKT / GMV", format="percent"),
            "CM3_GMV": st.column_config.NumberColumn("CM3 / GMV", format="percent"),
            "CM3_coverage": st.column_config.ProgressColumn("CM3 cost coverage", min_value=0, max_value=1, format="percent"),
        },
    )


def asin_360() -> None:
    available = filtered_sales.groupby(["ASIN", "SKU"], as_index=False).agg(GMV=("Ordered_GMV", "sum")).sort_values("GMV", ascending=False)
    if available.empty:
        return
    default_asin = available.iloc[0]["ASIN"]
    selected = st.selectbox("Select ASIN for 360 review", available["ASIN"].astype(str).tolist(), index=0, key="asin_360_select")
    sales = SALES[SALES["ASIN"].astype(str).eq(str(selected))].copy()
    sku = str(sales["SKU"].mode().iloc[0]) if not sales.empty else ""
    listing = TABLES["listing_health_demo"]
    listing = listing[listing["ASIN"].astype(str).eq(str(selected))]
    inventory = TABLES["inventory_current_asin"]
    inventory = inventory[inventory["ASIN"].astype(str).eq(str(selected))]
    target = TABLES["target_sku"]
    target_row = target[target["SKU"].astype(str).eq(sku)]
    product_name = sales["product_name"].dropna().iloc[0] if not sales.empty and sales["product_name"].notna().any() else sku
    st.markdown(f"### {selected} · {product_name}")
    st.link_button("Open Amazon listing", f"https://www.amazon.com/dp/{selected}", type="primary")

    row = listing.iloc[0] if not listing.empty else pd.Series(dtype=object)
    inv_row = inventory.iloc[0] if not inventory.empty else pd.Series(dtype=object)
    current = filter_dates(sales, start_date, end_date)
    metrics = st.columns(6)
    metrics[0].metric("Ordered units", fmt_number(current["Ordered_units"].sum()))
    metrics[1].metric("Ordered GMV", fmt_money(current["Ordered_GMV"].sum()))
    metrics[2].metric("Ad spend", fmt_money(current["Total_ADS"].sum()))
    metrics[3].metric("AMZ inventory", fmt_number(inv_row.get("asin_amz_inventory", np.nan)))
    metrics[4].metric("Y4A shared pool", fmt_number(inv_row.get("shared_y4a_inventory", np.nan)))
    metrics[5].metric("MOC", fmt_number(target_row["moc"].iloc[0], 1) if not target_row.empty else "—")

    if not listing.empty:
        status = row.get("listing_status_demo", "Unknown")
        st.markdown(f'<div class="warning"><span class="demo">SIMULATED</span> <b>Listing health:</b> {status}. '
                    f'{row.get("listing_issue_demo", "")} · shipping promise {row.get("shipping_days_demo", "—")} days. '
                    'This is demo data seeded from Block ads, target status and inventory; it is not an Amazon listing-health feed.</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Sales & Ads", "Inventory runway", "Ranking & keywords (demo)", "Campaigns", "Recommended actions"])
    with tabs[0]:
        daily = sales.groupby("Day", as_index=False).agg(Units=("Ordered_units", "sum"), GMV=("Ordered_GMV", "sum"), Ads=("Total_ADS", "sum"))
        c1, c2 = st.columns(2)
        fig = px.line(daily, x="Day", y=["Units", "GMV"], markers=True, title="Sales trend")
        c1.plotly_chart(chart_style(fig, 370), width="stretch")
        fig = px.bar(daily, x="Day", y="Ads", title="Ads trend", color_discrete_sequence=[PALETTE["purple"]])
        c2.plotly_chart(chart_style(fig, 370), width="stretch")
    with tabs[1]:
        forecast = TABLES["forecast_supply"]
        forecast = forecast[forecast["SKU"].astype(str).eq(sku)].sort_values("month")
        if forecast.empty:
            st.info("No six-month forecast found for this SKU.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["baseline_units"], name="Baseline forecast", line=dict(color=PALETTE["slate"], dash="dot")))
            fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["war_map_units"], name="War Map forecast", line=dict(color=PALETTE["blue"], width=3)))
            fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["inventory_constrained_target_units"], name="Inventory-constrained target", fill="tozeroy", line=dict(color=PALETTE["green"], width=3)))
            fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["available_supply_units"], name="Available supply", line=dict(color=PALETTE["amber"], dash="dash", width=3)))
            fig.update_layout(title="Demand versus available supply", yaxis_title="Units")
            st.plotly_chart(chart_style(fig, 430), width="stretch")
            st.dataframe(forecast[["month", "baseline_units", "war_map_units", "inventory_constrained_target_units", "available_supply_units", "recalc_ending_units", "war_map_gap_units", "frozen_status"]], width="stretch", hide_index=True)
    with tabs[2]:
        ranking = TABLES["ranking_keyword_demo"]
        ranking = ranking[ranking["ASIN"].astype(str).eq(str(selected))]
        if ranking.empty:
            st.info("No simulated ranking rows for this ASIN.")
        else:
            keyword = st.selectbox("Keyword", sorted(ranking["keyword_demo"].unique()), key="keyword_select")
            rank_view = ranking[ranking["keyword_demo"].eq(keyword)]
            fig = px.line(rank_view, x="Day", y=["organic_rank_demo", "sponsored_rank_demo"], markers=True, title=f"Keyword rank trend · {keyword}")
            st.plotly_chart(chart_style(fig, 400, reverse_y=True), width="stretch")
            st.caption("SIMULATED: rank and search volume are deterministic demo values. Lower rank is better.")
    with tabs[3]:
        campaigns = TABLES["ads_asin_campaign"]
        campaigns = campaigns[campaigns["ASIN"].astype(str).eq(str(selected))].sort_values("SpendUSD", ascending=False)
        if campaigns.empty:
            st.info("No campaign-to-ASIN mapping found in the audit workbook.")
        else:
            st.dataframe(campaigns[["campaignName", "Account", "programType", "state", "SpendUSD", "SalesUSD", "ACOS", "ROAS", "campaign_action", "AuditFlags"]], width="stretch", hide_index=True, column_config={"ACOS": st.column_config.NumberColumn(format="percent"), "ROAS": st.column_config.NumberColumn(format="%.2f")})
    with tabs[4]:
        actions = build_action_queue(TABLES)
        entity_actions = actions[(actions["Entity"].astype(str).eq(str(selected))) | (actions["SKU"].astype(str).eq(sku))]
        st.dataframe(entity_actions, width="stretch", hide_index=True)


def inventory_forecast() -> None:
    inv = apply_dimensions(TABLES["inventory_current_sku"], selected_product_lines, selected_teams, selected_channels, selected_skus, [], search_text)
    target = apply_dimensions(TABLES["target_sku"], selected_product_lines, selected_teams, selected_channels, selected_skus, [], search_text)
    forecast = apply_dimensions(TABLES["forecast_supply"], selected_product_lines, selected_teams, selected_channels, selected_skus, [], search_text)
    history = apply_dimensions(TABLES["inventory_history"], selected_product_lines, selected_teams, selected_channels, selected_skus, [], search_text)
    cards = st.columns(5)
    cards[0].metric("Current inventory", fmt_number(inv["current_inventory"].sum()))
    cards[1].metric("Incoming", fmt_number(inv["incoming_inventory"].sum()))
    cards[2].metric("MOC ≤ 1.5", f"{target['moc'].between(0.0001, 1.5).sum():,} SKUs")
    cards[3].metric("MOC > 6", f"{(target['moc'] > 6).sum():,} SKUs")
    frozen_gap = forecast[(forecast["war_map_gap_units"] > 0) & forecast["frozen_status"].astype(str).str.contains("Frozen", case=False, na=False)]["SKU"].nunique()
    cards[4].metric("Frozen supply risk", f"{frozen_gap:,} SKUs")

    c1, c2 = st.columns([1, 1.7], gap="large")
    with c1:
        moc_counts = target.groupby("moc_band", as_index=False)["SKU"].nunique().sort_values("SKU")
        fig = px.bar(moc_counts, x="SKU", y="moc_band", orientation="h", title="SKU count by MOC band", color="SKU", color_continuous_scale=["#FEE2E2", "#F59E0B", "#16A34A"])
        st.plotly_chart(chart_style(fig, 390), width="stretch")
    with c2:
        hist = history.groupby("month", as_index=False).agg(Opening_inventory=("opening_total", "sum"), SKU_count=("SKU", "nunique"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["month"], y=hist["Opening_inventory"], mode="lines+markers", name="Opening inventory", line=dict(color=PALETTE["blue"], width=3), fill="tozeroy"))
        fig.update_layout(title="Opening inventory history", yaxis_title="Units")
        st.plotly_chart(chart_style(fig, 390), width="stretch")
        st.caption("The supplied history has one point per month (not weekly) from Sep 2025 to Sep 2026.")

    section("Six-month demand and supply", "Final target is recalculated as min(War Map forecast, opening inventory + usable incoming), then rolled forward monthly.")
    monthly = forecast.groupby("month", as_index=False).agg(Baseline=("baseline_units", "sum"), War_Map=("war_map_units", "sum"), Constrained=("inventory_constrained_target_units", "sum"), Available=("available_supply_units", "sum"), Gap=("war_map_gap_units", "sum"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["Baseline"], name="Baseline forecast", line=dict(color=PALETTE["slate"], dash="dot", width=3)))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["War_Map"], name="War Map forecast", line=dict(color=PALETTE["blue"], width=3)))
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["Constrained"], name="Inventory-constrained target", marker_color=PALETTE["green"], opacity=.72))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["Available"], name="Available supply", line=dict(color=PALETTE["amber"], dash="dash", width=3)))
    fig.update_layout(title="Portfolio demand versus supply", yaxis_title="Units", barmode="group")
    st.plotly_chart(chart_style(fig, 450), width="stretch")

    risk = forecast.groupby(["SKU", "product_name", "product_line", "frozen_status"], as_index=False).agg(War_map=("war_map_units", "sum"), Constrained=("inventory_constrained_target_units", "sum"), Supply_gap=("war_map_gap_units", "sum"), Ending=("recalc_ending_units", "last")).sort_values("Supply_gap", ascending=False)
    st.dataframe(risk.head(100), width="stretch", hide_index=True)


def sales_target() -> None:
    performance = apply_dimensions(TABLES["performance_mtd"], selected_product_lines, selected_teams, selected_channels, selected_skus, [], search_text)
    actual_units = performance["actual_units_mtd"].sum()
    unit_target = performance["unit_target_oct"].sum()
    actual_gmv = performance["actual_gmv_mtd"].sum()
    gmv_target = performance["gmv_target_oct"].sum()
    opening = performance["opening_total"].sum()
    sell_through = safe_divide(actual_units, opening)
    days_observed = int(QUALITY["latest_sales_date"][-2:])
    pace_units = actual_units / max(days_observed, 1) * 30
    cards = st.columns(6)
    cards[0].metric("Actual units", fmt_number(actual_units))
    cards[1].metric("October unit target", fmt_number(unit_target))
    cards[2].metric("Unit attainment", fmt_pct(safe_divide(actual_units, unit_target)))
    cards[3].metric("Actual GMV", fmt_money(actual_gmv))
    cards[4].metric("GMV target", fmt_money(gmv_target))
    cards[5].metric("Sell-through proxy", fmt_pct(sell_through))
    st.markdown('<div class="warning"><b>Timing note:</b> current actuals are Sep 1–22 while the supplied target is October. The pace comparison is directional until October actuals are loaded. Sell-through proxy = Sep ordered units ÷ Sep opening inventory; receipts-to-date are not available.</div>', unsafe_allow_html=True)

    by_line = performance.groupby("product_line", as_index=False).agg(Actual_units=("actual_units_mtd", "sum"), Target_units=("unit_target_oct", "sum"), Actual_GMV=("actual_gmv_mtd", "sum"), Target_GMV=("gmv_target_oct", "sum"))
    by_line["Attainment"] = np.where(by_line["Target_units"] > 0, by_line["Actual_units"] / by_line["Target_units"], np.nan)
    by_line = by_line.sort_values("Target_units", ascending=False).head(20)
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        long = by_line.melt(id_vars="product_line", value_vars=["Actual_units", "Target_units"], var_name="Series", value_name="Units")
        fig = px.bar(long, x="product_line", y="Units", color="Series", barmode="group", title="Actual units versus October target", color_discrete_map={"Actual_units": PALETTE["blue"], "Target_units": PALETTE["slate"]})
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(chart_style(fig, 440), width="stretch")
    with right:
        top = by_line.sort_values("Attainment").tail(15)
        fig = px.bar(top, x="Attainment", y="product_line", orientation="h", title="Unit target attainment", color="Attainment", color_continuous_scale=["#FEE2E2", "#F59E0B", "#16A34A"])
        fig.add_vline(x=1, line_dash="dash", line_color=PALETTE["navy"])
        st.plotly_chart(chart_style(fig, 440), width="stretch")

    st.metric("30-day pace estimate", fmt_number(pace_units), delta=f"{safe_divide(pace_units, unit_target)-1:+.1%} vs October target" if unit_target else None)
    st.dataframe(performance.sort_values("unit_attainment").head(100)[["SKU", "product_name", "product_line", "actual_units_mtd", "unit_target_oct", "unit_attainment", "actual_gmv_mtd", "gmv_target_oct", "sell_through_proxy", "actual_acos_proxy"]], width="stretch", hide_index=True, column_config={"unit_attainment": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"), "sell_through_proxy": st.column_config.NumberColumn(format="percent"), "actual_acos_proxy": st.column_config.NumberColumn(format="percent")})


def ads_performance() -> None:
    ads = TABLES["ads_campaign"].copy()
    if ad_account_filter:
        ads = ads[ads["Account"].astype(str).isin(ad_account_filter)]
    spend = ads["SpendUSD"].sum()
    sales = ads["SalesUSD"].sum()
    clicks = ads["clicks"].sum()
    orders = ads["orders"].sum()
    cards = st.columns(6)
    cards[0].metric("Campaign spend", fmt_money(spend))
    cards[1].metric("Attributed sales", fmt_money(sales))
    cards[2].metric("ACOS", fmt_pct(safe_divide(spend, sales)))
    cards[3].metric("ROAS", f"{safe_divide(sales, spend):.2f}x" if spend else "—")
    cards[4].metric("Clicks", fmt_number(clicks))
    cards[5].metric("Orders", fmt_number(orders))
    st.caption("RAW_CAMPAIGN is a snapshot, not a daily campaign history. The daily Ads Trend below comes from the hourly sales workbook at SKU/ASIN level.")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        plot_ads = ads[(ads["SpendUSD"] > 0) | (ads["SalesUSD"] > 0)].copy()
        fig = px.scatter(plot_ads, x="SpendUSD", y="SalesUSD", size="orders", color="campaign_action", hover_name="campaignName", log_x=True, log_y=True, title="Campaign efficiency map", color_discrete_sequence=[PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["amber"]])
        st.plotly_chart(chart_style(fig, 430), width="stretch")
    with right:
        daily = filtered_sales.groupby("Day", as_index=False).agg(Ad_spend=("Total_ADS", "sum"), Ordered_NMV=("Ordered_nmv", "sum"))
        daily["ACOS_proxy"] = np.where(daily["Ordered_NMV"] > 0, daily["Ad_spend"] / daily["Ordered_NMV"], np.nan)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["Day"], y=daily["Ad_spend"], name="Ad spend", marker_color=PALETTE["purple"]))
        fig.add_trace(go.Scatter(x=daily["Day"], y=daily["ACOS_proxy"], name="ACOS proxy", yaxis="y2", line=dict(color=PALETTE["red"], width=3)))
        fig.update_layout(title="Monthly ads trend", yaxis=dict(title="Spend (USD)"), yaxis2=dict(title="ACOS", overlaying="y", side="right", tickformat=".0%", showgrid=False))
        st.plotly_chart(chart_style(fig, 430), width="stretch")
    section("Campaign action queue")
    action_ads = ads[ads["campaign_action"] != "Monitor"].sort_values("SpendUSD", ascending=False)
    st.dataframe(action_ads[["campaignName", "Account", "programType", "state", "SpendUSD", "SalesUSD", "ACOS", "ROAS", "campaign_action", "AuditFlags"]].head(200), width="stretch", hide_index=True, column_config={"ACOS": st.column_config.NumberColumn(format="percent"), "ROAS": st.column_config.NumberColumn(format="%.2f")})


def action_center() -> None:
    threshold_pct = st.slider("Campaign ACOS alert threshold (%)", min_value=10, max_value=100, value=35, step=5)
    actions = build_action_queue(TABLES, threshold_pct / 100)
    if selected_skus:
        actions = actions[actions["SKU"].astype(str).isin(selected_skus)]
    if selected_asins:
        actions = actions[actions["Entity"].astype(str).isin(selected_asins)]
    if search_text:
        text = search_text.lower().strip()
        actions = actions[actions.astype(str).agg(" | ".join, axis=1).str.lower().str.contains(text, regex=False, na=False)]
    cards = st.columns(4)
    cards[0].metric("Total actions", f"{len(actions):,}")
    cards[1].metric("P0", f"{(actions['Priority'] == 'P0').sum():,}" if not actions.empty else "0")
    cards[2].metric("P1", f"{(actions['Priority'] == 'P1').sum():,}" if not actions.empty else "0")
    cards[3].metric("Simulated signals", f"{(actions['Data class'] == 'SIMULATED').sum():,}" if not actions.empty else "0")
    st.markdown('<div class="callout"><b>Rule engine:</b> Listing/ZIP/ranking flags are demo data. Inventory, frozen supply gap and campaign metrics are source-backed. Recommended actions are rules, not autonomous changes.</div>', unsafe_allow_html=True)
    priority_filter = st.multiselect("Priority", ["P0", "P1", "P2"], default=["P0", "P1"])
    if priority_filter:
        actions = actions[actions["Priority"].isin(priority_filter)]
    st.dataframe(actions, width="stretch", hide_index=True, height=640)


def daily_forecast_supply() -> None:
    daily = apply_dimensions(
        TABLES["sales_team_daily"], selected_product_lines, selected_teams, selected_channels,
        selected_skus, [], effective_search,
    )
    daily = filter_dates(daily, start_date, end_date)
    scope_skus = sorted(daily["SKU"].dropna().astype(str).unique())
    selection = TABLES["forecast_auto_selection"]
    selection = selection[selection["SKU"].astype(str).isin(scope_skus)].copy()
    future = TABLES["forecast_future_auto"]
    future = future[future["SKU"].astype(str).isin(scope_skus)].copy()
    status = TABLES["forecast_sku_status"]
    status = status[status["SKU"].astype(str).isin(scope_skus)].copy()

    section("Daily Forecast & Supply", "Daily demand history + automatic model selection by SKU + inventory-constrained fulfillment")
    st.markdown(
        '<div class="callout"><b>Forecast gate:</b> select the lowest-WAPE model per SKU among models with |Bias| ≤ 20% and at least 18 valid backtest points. OOS-suspect months are excluded. If evidence is insufficient, fall back to the product-group/workbook rule. Inventory never changes demand forecast.</div>',
        unsafe_allow_html=True,
    )
    if not scope_skus:
        st.warning("No SKU matches the current filters.")
        return

    if len(scope_skus) > 1 and not selected_skus:
        cards = st.columns(5)
        cards[0].metric("SKUs in daily scope", f"{len(scope_skus):,}")
        cards[1].metric("Daily rows", f"{len(daily):,}")
        cards[2].metric("Median selected WAPE", fmt_pct(selection["WAPE"].median()))
        cards[3].metric("Median |Bias|", fmt_pct(selection["Bias"].abs().median()))
        cards[4].metric("6M demand forecast", fmt_number(future["Forecast_unconstrained"].sum()))
        left, right = st.columns([1, 1.45], gap="large")
        with left:
            counts = selection["Selected_model"].fillna("Status rule").value_counts().rename_axis("Model").reset_index(name="SKU")
            fig = px.bar(counts, x="SKU", y="Model", orientation="h", color="SKU", title="Selected model distribution", color_continuous_scale="Blues")
            st.plotly_chart(chart_style(fig, 430), width="stretch")
        with right:
            plot = selection.dropna(subset=["WAPE", "Bias"]).merge(status[["SKU", "Demand_class", "Main_PL"]], on="SKU", how="left")
            fig = px.scatter(plot, x="Bias", y="WAPE", color="Demand_class", hover_name="SKU", hover_data=["Selected_model", "Main_PL", "N"], title="Reliability map — WAPE vs Bias")
            fig.add_vline(x=-0.20, line_dash="dash", line_color=PALETTE["amber"])
            fig.add_vline(x=0.20, line_dash="dash", line_color=PALETTE["amber"])
            st.plotly_chart(chart_style(fig, 430), width="stretch")
        st.caption("Use Fast search above and select one result to open its forecast and inventory runway.")
        st.dataframe(selection.sort_values("WAPE", ascending=False).head(40), width="stretch", hide_index=True)
        return

    sku = selected_skus[0] if selected_skus else scope_skus[0]
    daily = daily[daily["SKU"].astype(str).eq(sku)].copy()
    future = future[future["SKU"].astype(str).eq(sku)].sort_values("Month").copy()
    selection = selection[selection["SKU"].astype(str).eq(sku)]
    status = status[status["SKU"].astype(str).eq(sku)]
    product_mode = daily["product_name"].dropna().astype(str).mode()
    product = product_mode.iloc[0] if not product_mode.empty else ""
    asin_text = ", ".join(sorted(daily["ASIN"].dropna().astype(str).unique())[:5])
    st.markdown(f"### {sku} · {product}")
    st.caption(f"ASIN: {asin_text or '—'}")

    last_day = daily["Day"].max()
    recent = daily[daily["Day"] > last_day - pd.Timedelta(days=28)]
    prior = daily[(daily["Day"] <= last_day - pd.Timedelta(days=28)) & (daily["Day"] > last_day - pd.Timedelta(days=56))]
    recent_units, prior_units = recent["Ordered_units"].sum(), prior["Ordered_units"].sum()
    sel = selection.iloc[0] if not selection.empty else pd.Series(dtype=object)
    stat = status.iloc[0] if not status.empty else pd.Series(dtype=object)
    cards = st.columns(7)
    cards[0].metric("Units · last 28D", fmt_number(recent_units), None if prior_units == 0 else f"{recent_units / prior_units - 1:+.1%} vs prior 28D")
    cards[1].metric("GMV · last 28D", fmt_money(recent["Ordered_GMV"].sum()))
    cards[2].metric("ASP", fmt_money(safe_divide(recent["Ordered_GMV"].sum(), recent_units)))
    cards[3].metric("Selected model", str(sel.get("Selected_model", "Status rule")))
    cards[4].metric("WAPE", fmt_pct(sel.get("WAPE", np.nan)))
    cards[5].metric("Bias", fmt_pct(sel.get("Bias", np.nan)))
    cards[6].metric("MAPE*", fmt_pct(sel.get("MAPE", np.nan)))
    st.caption("* MAPE excludes zero-actual observations and is diagnostic only; WAPE is the primary accuracy metric.")

    day = daily.groupby("Day", as_index=False)[["Ordered_units", "Ordered_GMV", "Total_ADS", "Total_Promo"]].sum().sort_values("Day")
    day["Units · 7D MA"] = day["Ordered_units"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(x=day["Day"], y=day["Ordered_units"], name="Daily units", marker_color="#BFDBFE")
    fig.add_scatter(x=day["Day"], y=day["Units · 7D MA"], name="7-day moving average", line=dict(color=PALETTE["blue"], width=3))
    fig.update_layout(title="Daily demand and 7-day signal", barmode="overlay")
    st.plotly_chart(chart_style(fig, 400), width="stretch")

    actual_month = daily.assign(Month=daily["Day"].dt.to_period("M").dt.to_timestamp()).groupby("Month", as_index=False)["Ordered_units"].sum()
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_scatter(x=actual_month["Month"], y=actual_month["Ordered_units"], name="Actual units", line=dict(color=PALETTE["navy"], width=3))
        fig.add_scatter(x=future["Month"], y=future["Forecast_unconstrained"], name="Auto demand forecast", mode="lines+markers", line=dict(color=PALETTE["teal"], width=3, dash="dash"))
        if future["P80"].notna().any():
            fig.add_scatter(x=future["Month"], y=future["P80"], name="P80 planning band", line=dict(color=PALETTE["amber"], dash="dot"))
        fig.update_layout(title="Monthly actual vs unconstrained demand forecast", yaxis_title="Units")
        st.plotly_chart(chart_style(fig, 430), width="stretch")
    with right:
        models = TABLES["forecast_all_models"]
        models = models[models["SKU"].astype(str).eq(sku)].copy()
        available = [m for m in CANDIDATE_MODELS if m in models and models[m].notna().any()]
        if available:
            long = models.melt(id_vars=["Month"], value_vars=available, var_name="Model", value_name="Forecast")
            fig = px.line(long, x="Month", y="Forecast", color="Model", markers=True, title="Candidate model forecasts")
            st.plotly_chart(chart_style(fig, 430), width="stretch")
        else:
            st.info("This SKU uses a lifecycle/status rule rather than a trained model.")

    section("Inventory-constrained fulfillment", "Demand stays unchanged; inventory only limits the fulfillable target.")
    controls = st.columns([1.35, 1, 2.2])
    with controls[0]:
        include_incoming = st.toggle("Block incoming into 3-month plan", value=True, help="Include aggregate incoming in the planning block.")
    with controls[1]:
        receipt_month = st.selectbox("Assumed receipt month", [1, 2, 3], index=2, disabled=not include_incoming)
    with controls[2]:
        st.caption("The source has no PO-level ETA. Conservative default: all incoming becomes usable in month 3.")

    on_hand, incoming = float(stat.get("Salable_inv", 0) or 0), float(stat.get("Incoming", 0) or 0)
    opening, rows = on_hand, []
    for i, row in future.reset_index(drop=True).iterrows():
        inbound = incoming if include_incoming and i + 1 == int(receipt_month) else 0.0
        available_units, demand = opening + inbound, float(row["Forecast_unconstrained"] or 0)
        fulfillable, ending = min(demand, available_units), max(available_units - demand, 0.0)
        rows.append({"Month": row["Month"], "Opening": opening, "Incoming": inbound, "Demand forecast": demand, "Fulfillable target": fulfillable, "Lost / delayed units": max(demand - available_units, 0.0), "Ending inventory": ending})
        opening = ending
    plan = pd.DataFrame(rows)
    plan_cards = st.columns(4)
    plan_cards[0].metric("Current salable", fmt_number(on_hand))
    plan_cards[1].metric("Incoming in source", fmt_number(incoming))
    plan_cards[2].metric("Fulfillable · 6M", fmt_number(plan["Fulfillable target"].sum() if not plan.empty else 0))
    plan_cards[3].metric("Supply gap · 6M", fmt_number(plan["Lost / delayed units"].sum() if not plan.empty else 0))
    if not plan.empty:
        fig = go.Figure()
        fig.add_bar(x=plan["Month"], y=plan["Demand forecast"], name="Unconstrained demand", marker_color="#93C5FD")
        fig.add_bar(x=plan["Month"], y=plan["Fulfillable target"], name="Fulfillable", marker_color=PALETTE["green"])
        fig.add_scatter(x=plan["Month"], y=plan["Ending inventory"], name="Ending inventory", mode="lines+markers", line=dict(color=PALETTE["red"], width=3, dash="dash"))
        fig.update_layout(title="Demand, fulfillable target and projected inventory", barmode="group", yaxis_title="Units")
        st.plotly_chart(chart_style(fig, 430), width="stretch")
        st.dataframe(plan.round(0), width="stretch", hide_index=True)
    with st.expander("Model evidence and selection audit"):
        metrics = TABLES["forecast_backtest_sku_model"]
        st.dataframe(metrics[metrics["SKU"].astype(str).eq(sku)].sort_values("WAPE"), width="stretch", hide_index=True)


def data_quality() -> None:
    st.subheader("Data freshness and coverage")
    cards = st.columns(5)
    cards[0].metric("Sales freshness", QUALITY["latest_sales_date"])
    cards[1].metric("Commercial SKUs", f"{QUALITY['commercial_sku']:,}")
    cards[2].metric("Target SKUs", f"{QUALITY['target_sku']:,}")
    cards[3].metric("PIC mapped SKUs", f"{QUALITY['commercial_pic_mapped_sku']:,}")
    cards[4].metric("Inventory coverage", fmt_pct(QUALITY["inventory_target_sku_coverage"]))
    if "daily_rows" in QUALITY:
        st.caption(f"Daily mart: {QUALITY['daily_rows']:,} rows · {QUALITY['daily_sku']:,} SKUs · {QUALITY['daily_min_date']} → {QUALITY['daily_max_date']} · {QUALITY['forecast_method']}")
    if "cm3_gmv_coverage" in QUALITY:
        st.markdown(
            f'<div class="callout"><b>CM3 engine:</b> {QUALITY["cm3_source"]}<br>'
            f'{QUALITY["cm3_formula"]}<br><b>Historical GMV coverage:</b> {fmt_pct(QUALITY["cm3_gmv_coverage"])}. '
            f'{QUALITY["cm3_limitations"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="warning"><b>Scope note:</b> Commercial history contains {QUALITY["commercial_sku"]:,} SKUs; {QUALITY["commercial_pic_unmapped_sku"]:,} are marked N/A because they are absent from the current 1,200-SKU Target mapping. September 2026 is partial. Listing health and ranking/keyword remain visibly labeled simulated.</div>', unsafe_allow_html=True)
    definitions = pd.DataFrame([{"Metric": key, "Definition": value} for key, value in QUALITY["metric_definitions"].items()])
    st.dataframe(definitions, width="stretch", hide_index=True)
    st.subheader("How to update the latest numbers")
    st.markdown(
        """
        1. Export the same five source files with the same sheet/column names.
        2. Run `prepare_app_data.py`, then `prepare_daily_forecast.py`, then `prepare_cm3_mart.py` to rebuild `app_data_v4.zip`.
        3. In GitHub, replace only `app_data_v4.zip` and commit. Streamlit Cloud redeploys automatically.
        4. Use the quality page to confirm freshness, SKU coverage and the simulated-data labels.

        A runtime file uploader is intentionally not used for the production source because uploads disappear when the Streamlit session restarts.
        """
    )
    with st.expander("Source inventory"):
        st.caption("Tables are listed from the ZIP manifest without loading every Parquet file into RAM.")
        st.dataframe(pd.DataFrame({"Table": TABLE_NAMES, "Loading": "On demand"}), width="stretch", hide_index=True)


if page == "Commercial Intelligence":
    commercial_intelligence()
elif page == "Daily Forecast & Supply":
    daily_forecast_supply()
elif page == "ASIN 360":
    asin_360()
elif page == "Inventory & Forecast":
    inventory_forecast()
elif page == "Sales & Target":
    sales_target()
elif page == "Ads Performance":
    ads_performance()
elif page == "Action Center":
    action_center()
else:
    data_quality()
