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
        Path(__file__).resolve().parent / "app_data_v2.zip",
        Path(__file__).resolve().parent / "data" / "app_data_v2.zip",
        Path.cwd() / "app_data_v2.zip",
        Path.cwd() / "data" / "app_data_v2.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy app_data_v2.zip cạnh streamlit_app.py hoặc trong thư mục data.")


@st.cache_data(show_spinner="Đang nạp Data Mart V2...")
def load_bundle(path: str, modified_ns: int) -> tuple[dict[str, pd.DataFrame], dict]:
    del modified_ns
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".parquet"):
                tables[name.removesuffix(".parquet")] = pd.read_parquet(io.BytesIO(archive.read(name)))
        quality = json.loads(archive.read("quality_summary.json").decode("utf-8"))
    for frame in tables.values():
        for column in ["Day", "month", "snapshot_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return tables, quality


def fmt_number(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{digits}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{digits}f}K"
    return f"{value:,.0f}"


def fmt_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${fmt_number(value)}"


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
TABLES, QUALITY = load_bundle(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
SALES = TABLES["sales_current"]

st.markdown(
    f"""
    <div class="app-header">
      <h1>Yes4All Commerce Intelligence</h1>
      <p>Sales, inventory, target, ads and action center · Actuals through {QUALITY['latest_sales_date']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "Navigation",
    ["Executive Overview", "ASIN 360", "Inventory & Forecast", "Sales & Target", "Ads Performance", "Action Center", "Data Quality"],
    default="Executive Overview",
    label_visibility="collapsed",
)

date_min = SALES["Day"].min().date()
date_max = SALES["Day"].max().date()
with st.container():
    st.markdown('<div class="filter-shell">', unsafe_allow_html=True)
    row1 = st.columns([1.3, 1.6, 1.2, 1.15, 1.55], gap="small")
    with row1[0]:
        selected_dates = st.date_input("Time range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    all_product_lines = sorted(SALES["product_line"].dropna().astype(str).unique().tolist())
    with row1[1]:
        selected_product_lines = st.multiselect("Product line", all_product_lines, placeholder="All product lines")
    with row1[2]:
        selected_teams = st.multiselect("Team", sorted(SALES["team"].dropna().astype(str).unique()), placeholder="All teams")
    with row1[3]:
        selected_channels = st.multiselect("Channel", sorted(SALES["channel"].dropna().astype(str).unique()), placeholder="All channels")
    with row1[4]:
        search_text = st.text_input("Search", placeholder="Product name, SKU, ASIN...")

    prefiltered = apply_dimensions(SALES, selected_product_lines, selected_teams, selected_channels, [], [], search_text)
    row2 = st.columns([1.3, 1.6, 1.2, 1.15, 1.55], gap="small")
    with row2[0]:
        selected_skus = st.multiselect("SKU", sorted(prefiltered["SKU"].dropna().astype(str).unique()), placeholder="All SKUs")
    asin_pool = prefiltered[prefiltered["SKU"].astype(str).isin(selected_skus)] if selected_skus else prefiltered
    with row2[1]:
        selected_asins = st.multiselect("ASIN", sorted(asin_pool["ASIN"].dropna().astype(str).unique()), placeholder="All ASINs")
    with row2[2]:
        listing_status_filter = st.multiselect("Listing status", ["Active", "Active with ZIP restriction", "Active with long delivery", "Blocked"], placeholder="All statuses")
    with row2[3]:
        ad_account_filter = st.multiselect("Ads account", sorted(TABLES["ads_campaign"]["Account"].dropna().astype(str).unique()), default=["Yes4All [US]"] if "Yes4All [US]" in set(TABLES["ads_campaign"]["Account"].astype(str)) else [])
    with row2[4]:
        st.caption(f"Coverage: {QUALITY['sales_sku']:,}/{QUALITY['target_sku']:,} target SKUs · Inventory map {QUALITY['inventory_target_sku_coverage']:.1%}")
    st.markdown("</div>", unsafe_allow_html=True)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates if not isinstance(selected_dates, tuple) else date_max

filtered_sales = apply_dimensions(SALES, selected_product_lines, selected_teams, selected_channels, selected_skus, selected_asins, search_text)
filtered_sales = filter_dates(filtered_sales, start_date, end_date)

if filtered_sales.empty and page not in {"Ads Performance", "Data Quality"}:
    st.warning("Không có dữ liệu sales trong tổ hợp filter hiện tại. Hãy bỏ bớt filter hoặc đổi khoảng ngày.")


def executive_overview() -> None:
    current = filtered_sales
    gmv = current["Ordered_GMV"].sum()
    nmv = current["Ordered_nmv"].sum()
    units = current["Ordered_units"].sum()
    ads = current["Total_ADS"].sum()
    promo = current["Total_Promo"].sum()
    acos = safe_divide(ads, nmv)
    columns = st.columns(6)
    values = [
        ("Ordered GMV", fmt_money(gmv)), ("Ordered NMV", fmt_money(nmv)),
        ("Total units", fmt_number(units)), ("Ad spend", fmt_money(ads)),
        ("Promo spend", fmt_money(promo)), ("ACOS proxy", fmt_pct(acos)),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)

    scope_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)
    st.markdown(
        f'<div class="callout"><b>Current view:</b> {scope_days} days, {current["SKU"].nunique():,} SKUs and {current["ASIN"].nunique():,} ASINs. '
        f'ACOS proxy uses total ad spend ÷ ordered NMV, not campaign-attributed sales.</div>',
        unsafe_allow_html=True,
    )

    daily = current.groupby("Day", as_index=False).agg(GMV=("Ordered_GMV", "sum"), Units=("Ordered_units", "sum"), Ads=("Total_ADS", "sum"))
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["Day"], y=daily["GMV"], name="GMV", marker_color=PALETTE["blue"]))
        fig.add_trace(go.Scatter(x=daily["Day"], y=daily["Units"], name="Units", yaxis="y2", mode="lines+markers", line=dict(color=PALETTE["teal"], width=3)))
        fig.update_layout(title="Daily sales trajectory", yaxis=dict(title="GMV (USD)"), yaxis2=dict(title="Units", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(chart_style(fig, 410), width="stretch")
    with right:
        by_line = current.groupby("product_line", as_index=False).agg(GMV=("Ordered_GMV", "sum"), Units=("Ordered_units", "sum")).nlargest(12, "GMV").sort_values("GMV")
        fig = px.bar(by_line, x="GMV", y="product_line", orientation="h", color="Units", color_continuous_scale=["#DBEAFE", "#2563EB"], title="Top product lines by GMV")
        st.plotly_chart(chart_style(fig, 410), width="stretch")

    performance = apply_dimensions(TABLES["performance_mtd"], selected_product_lines, selected_teams, selected_channels, selected_skus, [], search_text)
    performance = performance[performance["unit_target_oct"] > 0].copy()
    if not performance.empty:
        performance["pace_gap"] = performance["unit_attainment"] - (pd.Timestamp(end_date).day / 31)
        risk = performance.nsmallest(15, "pace_gap")
        section("Target pace exceptions", "MTD actual units versus a linear October target pace. September actuals are used only as the latest available operating signal.")
        st.dataframe(
            risk[["SKU", "product_name", "product_line", "actual_units_mtd", "unit_target_oct", "unit_attainment", "sell_through_proxy"]],
            width="stretch",
            hide_index=True,
            column_config={
                "actual_units_mtd": st.column_config.NumberColumn("Actual units"),
                "unit_target_oct": st.column_config.NumberColumn("Oct target"),
                "unit_attainment": st.column_config.ProgressColumn("Attainment", min_value=0, max_value=1, format="percent"),
                "sell_through_proxy": st.column_config.NumberColumn("Sell-through proxy", format="percent"),
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
        campaigns = campaigns[campaigns["ASIN"].astype(str).eq(str(selected))].sort_values("spend", ascending=False)
        if campaigns.empty:
            st.info("No campaign-to-ASIN mapping found in the audit workbook.")
        else:
            st.dataframe(campaigns[["campaignName", "Account", "programType", "state", "spend", "sales", "orders", "ACOS", "campaign_action", "AuditFlags"]], width="stretch", hide_index=True, column_config={"ACOS": st.column_config.NumberColumn(format="percent")})
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
        fig.update_layout(title="Daily ads trend", yaxis=dict(title="Spend (USD)"), yaxis2=dict(title="ACOS", overlaying="y", side="right", tickformat=".0%", showgrid=False))
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


def data_quality() -> None:
    st.subheader("Data freshness and coverage")
    cards = st.columns(5)
    cards[0].metric("Sales freshness", QUALITY["latest_sales_date"])
    cards[1].metric("Current sales SKUs", f"{QUALITY['sales_sku']:,}")
    cards[2].metric("Target SKUs", f"{QUALITY['target_sku']:,}")
    cards[3].metric("Sales coverage", fmt_pct(QUALITY["sales_target_sku_coverage"]))
    cards[4].metric("Inventory coverage", fmt_pct(QUALITY["inventory_target_sku_coverage"]))
    st.markdown('<div class="warning"><b>Coverage gap:</b> The latest hourly file covers about 65% of target SKUs. Portfolio totals should be read as “covered SKUs,” not the complete 1,200-SKU target portfolio. Listing health and ranking/keyword are visibly labeled simulated.</div>', unsafe_allow_html=True)
    definitions = pd.DataFrame([{"Metric": key, "Definition": value} for key, value in QUALITY["metric_definitions"].items()])
    st.dataframe(definitions, width="stretch", hide_index=True)
    st.subheader("How to update the latest numbers")
    st.markdown(
        """
        1. Export the same five source files with the same sheet/column names.
        2. Run the supplied Colab preparation script to rebuild `app_data_v2.zip`.
        3. In GitHub, replace only `app_data_v2.zip` and commit. Streamlit Cloud redeploys automatically.
        4. Use the quality page to confirm freshness, SKU coverage and the simulated-data labels.

        A runtime file uploader is intentionally not used for the production source because uploads disappear when the Streamlit session restarts.
        """
    )
    with st.expander("Source inventory"):
        rows = []
        for name, frame in TABLES.items():
            rows.append({"Table": name, "Rows": len(frame), "Columns": len(frame.columns)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


if page == "Executive Overview":
    executive_overview()
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
