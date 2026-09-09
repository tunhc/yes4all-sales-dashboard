from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Yes4All Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "cyan": "#2DD4BF",
    "blue": "#38BDF8",
    "purple": "#A78BFA",
    "pink": "#F472B6",
    "orange": "#FB923C",
    "red": "#FB7185",
    "green": "#4ADE80",
    "slate": "#94A3B8",
}

POSITION_COLORS = {
    "Leader": COLORS["green"],
    "Emerging": COLORS["cyan"],
    "Core at Risk": COLORS["orange"],
    "Tail / Review": COLORS["slate"],
    "New / Reactivated": COLORS["purple"],
    "No recent sales": "#475569",
    "Insufficient comparison": COLORS["pink"],
}

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #07111f 0%, #0b1730 52%, #10152d 100%);
        color: #e5eefb;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #111936 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.20);
    }
    [data-testid="stMetric"] {
        background: rgba(15, 30, 55, 0.86);
        border: 1px solid rgba(45, 212, 191, 0.24);
        border-radius: 16px;
        padding: 15px 17px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    [data-testid="stMetricLabel"] { color: #9fb3cc; }
    [data-testid="stMetricValue"] { color: #f8fbff; }
    .hero {
        padding: 22px 25px;
        border-radius: 20px;
        background:
          radial-gradient(circle at 82% 25%, rgba(167,139,250,.25), transparent 34%),
          linear-gradient(120deg, rgba(8,47,73,.92), rgba(30,27,75,.92));
        border: 1px solid rgba(56,189,248,.28);
        margin-bottom: 16px;
    }
    .hero h1 { margin: 0; font-size: 2.05rem; color: #f8fbff; }
    .hero p { margin: 7px 0 0; color: #b9c9dc; }
    .insight {
        padding: 14px 17px;
        border-left: 4px solid #2dd4bf;
        border-radius: 10px;
        background: rgba(13, 34, 55, .88);
        margin: 8px 0 16px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_data_dir() -> Path:
    candidates = [
        Path("/content/yes4all_bigdata/app_data"),
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
        Path.cwd() / "app_data",
    ]
    for candidate in candidates:
        if (candidate / "sales_daily_sku_app.parquet").exists():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy sales_daily_sku_app.parquet. "
        "Hãy đặt dữ liệu trong /content/yes4all_bigdata/app_data hoặc thư mục data."
    )


@st.cache_data(show_spinner="Đang nạp App Data Mart...")
def load_data(data_dir: str):
    base = Path(data_dir)
    sku = pd.read_parquet(base / "sales_daily_sku_app.parquet")
    asin = pd.read_parquet(base / "sales_daily_asin_app.parquet")
    for frame in (sku, asin):
        frame["Day"] = pd.to_datetime(frame["Day"]).dt.normalize()
        for col in [
            "Glance_views", "Ordered_units", "Ordered GMV", "Ordered_nmv",
            "Shipped_units", "Shipped_nmv", "Total Promo", "Total ADS",
        ]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return sku, asin


def safe_ratio(numerator, denominator):
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def money(value):
    return f"${value:,.0f}"


def compact_number(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def delta_text(current_value, previous_value):
    if previous_value == 0:
        return "N/A" if current_value == 0 else "New"
    return f"{(current_value / previous_value - 1) * 100:+.1f}%"


def aggregate_summary(frame):
    return {
        "gmv": float(frame["Ordered GMV"].sum()),
        "nmv": float(frame["Ordered_nmv"].sum()),
        "units": float(frame["Ordered_units"].sum()),
        "views": float(frame["Glance_views"].sum()),
        "ads": float(frame["Total ADS"].sum()),
        "promo": float(frame["Total Promo"].sum()),
    }


def filter_dimensions(frame, product_lines, skus, asins, search_text):
    result = frame
    if product_lines:
        result = result[result["product_line"].isin(product_lines)]
    if skus:
        result = result[result["SKU"].astype(str).isin(skus)]
    if asins and "ASIN" in result.columns:
        result = result[result["ASIN"].astype(str).isin(asins)]
    if search_text:
        text = search_text.strip().lower()
        searchable = (
            result["SKU"].astype(str).str.lower().fillna("") + " | " +
            result["product_name"].astype(str).str.lower().fillna("") + " | " +
            result["product_line"].astype(str).str.lower().fillna("")
        )
        if "ASIN" in result.columns:
            searchable = searchable + " | " + result["ASIN"].astype(str).str.lower().fillna("")
        result = result[searchable.str.contains(text, regex=False, na=False)]
    return result


def comparison_dates(start_date, end_date, mode):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if mode == "Cùng kỳ năm trước (YoY)":
        return start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
    days = (end - start).days + 1
    prior_end = start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=days - 1)
    return prior_start, prior_end


def build_period_data(frame, start, end, prior_start, prior_end):
    current = frame[frame["Day"].between(start, end)].copy()
    previous = frame[frame["Day"].between(prior_start, prior_end)].copy()
    return current, previous


def plot_layout(fig, height=430):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,14,28,0.58)",
        font=dict(color="#dce8f8"),
        hoverlabel=dict(bgcolor="#101c33", font_color="#ffffff"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    return fig


def trend_frame(current, previous, current_start, prior_start, grain, metric):
    frequency = {"Ngày": "D", "Tuần": "W-MON", "Tháng": "MS"}[grain]

    def aggregate(frame, period, anchor):
        if frame.empty:
            return pd.DataFrame(columns=["Period", "Value", "Series"])
        temp = frame.groupby(pd.Grouper(key="Day", freq=frequency))[metric].sum().reset_index()
        temp["Period"] = (temp["Day"] - pd.Timestamp(anchor)).dt.days
        temp["Series"] = period
        return temp[["Period", metric, "Series"]].rename(columns={metric: "Value"})

    return pd.concat(
        [
            aggregate(current, "Kỳ được chọn", current_start),
            aggregate(previous, "Kỳ so sánh", prior_start),
        ],
        ignore_index=True,
    )


def group_performance(current, previous, group_col):
    measures = [
        "Glance_views", "Ordered_units", "Ordered GMV", "Ordered_nmv",
        "Total Promo", "Total ADS",
    ]
    current_group = current.groupby(group_col, as_index=False)[measures].sum()
    previous_group = previous.groupby(group_col, as_index=False)[measures].sum()
    current_group = current_group.rename(columns={col: f"current_{col}" for col in measures})
    previous_group = previous_group.rename(columns={col: f"prior_{col}" for col in measures})
    result = current_group.merge(previous_group, on=group_col, how="outer").fillna(0)
    result["gmv_change"] = result["current_Ordered GMV"] - result["prior_Ordered GMV"]
    result["gmv_yoy_pct"] = np.where(
        result["prior_Ordered GMV"] != 0,
        (result["current_Ordered GMV"] / result["prior_Ordered GMV"] - 1) * 100,
        np.nan,
    )
    result["conversion_proxy_pct"] = np.where(
        result["current_Glance_views"] != 0,
        result["current_Ordered_units"] / result["current_Glance_views"] * 100,
        np.nan,
    )
    result["ad_to_gmv_pct"] = np.where(
        result["current_Ordered GMV"] != 0,
        result["current_Total ADS"] / result["current_Ordered GMV"] * 100,
        np.nan,
    )
    result["promo_to_gmv_pct"] = np.where(
        result["current_Ordered GMV"] != 0,
        result["current_Total Promo"] / result["current_Ordered GMV"] * 100,
        np.nan,
    )
    return result


def classify_sku_table(table):
    if table.empty:
        table["Position"] = []
        return table
    table = table.copy()
    table["line_median_gmv"] = table.groupby("product_line")["current_Ordered GMV"].transform("median")

    def classify(row):
        current_gmv = row["current_Ordered GMV"]
        previous_gmv = row["prior_Ordered GMV"]
        growth = row["gmv_yoy_pct"]
        if current_gmv <= 0 and previous_gmv <= 0:
            return "No recent sales"
        if current_gmv > 0 and previous_gmv <= 0:
            return "New / Reactivated"
        if pd.isna(growth):
            return "Insufficient comparison"
        if current_gmv >= row["line_median_gmv"] and growth >= 0:
            return "Leader"
        if current_gmv < row["line_median_gmv"] and growth >= 10:
            return "Emerging"
        if current_gmv >= row["line_median_gmv"] and growth < 0:
            return "Core at Risk"
        return "Tail / Review"

    table["Position"] = table.apply(classify, axis=1)
    return table


def make_csv(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


try:
    DATA_DIR = find_data_dir()
    sku_daily, asin_daily = load_data(str(DATA_DIR))
except Exception as exc:
    st.error(f"Không thể đọc dữ liệu: {exc}")
    st.stop()


min_date = sku_daily["Day"].min().date()
max_date = sku_daily["Day"].max().date()
default_start = max(pd.Timestamp(min_date), pd.Timestamp(max_date) - pd.Timedelta(days=89)).date()

st.sidebar.markdown("## YES4ALL")
st.sidebar.caption("Sales Intelligence Control Center")

page = st.sidebar.radio(
    "Không gian phân tích",
    [
        "Portfolio Overview",
        "Product Line Map",
        "SKU Deep Dive",
        "Ads & Promo",
        "Anomaly & Quality",
    ],
)

date_selection = st.sidebar.date_input(
    "Khoảng thời gian",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    start_date = end_date = date_selection

comparison_mode = st.sidebar.selectbox(
    "So sánh với",
    ["Cùng kỳ năm trước (YoY)", "Kỳ liền trước"],
)

all_product_lines = sorted(sku_daily["product_line"].dropna().astype(str).unique())
selected_lines = st.sidebar.multiselect("Product Line", all_product_lines)
search_text = st.sidebar.text_input(
    "Tìm sản phẩm",
    placeholder="Tên, SKU hoặc ASIN...",
)

sku_options_frame = filter_dimensions(sku_daily, selected_lines, [], [], search_text)
sku_options = sorted(sku_options_frame["SKU"].dropna().astype(str).unique())
selected_skus = st.sidebar.multiselect("SKU", sku_options)

asin_options_frame = filter_dimensions(asin_daily, selected_lines, selected_skus, [], search_text)
asin_options = sorted(asin_options_frame["ASIN"].dropna().astype(str).unique())
selected_asins = st.sidebar.multiselect("ASIN", asin_options)

grain = st.sidebar.segmented_control(
    "Độ chi tiết biểu đồ",
    options=["Ngày", "Tuần", "Tháng"],
    default="Tuần",
)
if grain is None:
    grain = "Tuần"

source = asin_daily if selected_asins else sku_daily
filtered_source = filter_dimensions(
    source, selected_lines, selected_skus, selected_asins, search_text
)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
prior_start, prior_end = comparison_dates(start_ts, end_ts, comparison_mode)
current, previous = build_period_data(
    filtered_source, start_ts, end_ts, prior_start, prior_end
)

if current.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
    st.stop()

current_summary = aggregate_summary(current)
previous_summary = aggregate_summary(previous)

st.markdown(
    f"""
    <div class="hero">
      <h1>Yes4All Sales Intelligence</h1>
      <p>{start_ts.date()} → {end_ts.date()} · {comparison_mode} · Cập nhật đến {max_date}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_kpis():
    kpis = [
        ("Ordered GMV", money(current_summary["gmv"]), delta_text(current_summary["gmv"], previous_summary["gmv"])),
        ("Ordered NMV", money(current_summary["nmv"]), delta_text(current_summary["nmv"], previous_summary["nmv"])),
        ("Ordered Units", compact_number(current_summary["units"]), delta_text(current_summary["units"], previous_summary["units"])),
        ("Glance Views", compact_number(current_summary["views"]), delta_text(current_summary["views"], previous_summary["views"])),
        ("Ad Spend", money(current_summary["ads"]), delta_text(current_summary["ads"], previous_summary["ads"])),
        ("Promo Spend", money(current_summary["promo"]), delta_text(current_summary["promo"], previous_summary["promo"])),
    ]
    columns = st.columns(len(kpis))
    for column, (label, value, delta) in zip(columns, kpis):
        column.metric(label, value, delta)


render_kpis()

growth = safe_ratio(current_summary["gmv"], previous_summary["gmv"])
growth_text = "chưa đủ dữ liệu so sánh"
if not pd.isna(growth):
    growth_text = f"{'tăng' if growth >= 1 else 'giảm'} {abs(growth - 1) * 100:.1f}%"

top_line_dynamic = group_performance(current, previous, "product_line").sort_values(
    "current_Ordered GMV", ascending=False
)
top_line_name = top_line_dynamic.iloc[0]["product_line"] if not top_line_dynamic.empty else "N/A"
st.markdown(
    f"""
    <div class="insight"><b>Insight động:</b> GMV kỳ được chọn {growth_text} so với kỳ đối chiếu.
    Product Line đóng góp GMV lớn nhất là <b>{top_line_name}</b>. Đây là tín hiệu mô tả từ dữ liệu;
    không được diễn giải Ads hoặc Promo là nguyên nhân nếu chưa có thiết kế nhân quả.</div>
    """,
    unsafe_allow_html=True,
)


if page == "Portfolio Overview":
    left, right = st.columns([1.65, 1])
    with left:
        metric_choice = st.selectbox(
            "Chỉ số xu hướng",
            ["Ordered GMV", "Ordered_units", "Ordered_nmv", "Glance_views"],
            index=0,
        )
        trend = trend_frame(current, previous, start_ts, prior_start, grain, metric_choice)
        fig = px.line(
            trend,
            x="Period",
            y="Value",
            color="Series",
            markers=True,
            color_discrete_sequence=[COLORS["cyan"], COLORS["purple"]],
            title=f"{metric_choice} — kỳ hiện tại và kỳ so sánh",
            labels={"Period": "Số ngày từ đầu kỳ", "Value": metric_choice},
        )
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    with right:
        share = top_line_dynamic.sort_values("current_Ordered GMV", ascending=False).head(10)
        fig = px.bar(
            share.sort_values("current_Ordered GMV"),
            x="current_Ordered GMV",
            y="product_line",
            orientation="h",
            title="Top Product Line theo GMV",
            color="current_Ordered GMV",
            color_continuous_scale=["#172554", COLORS["blue"], COLORS["cyan"]],
            labels={"current_Ordered GMV": "GMV", "product_line": "Product Line"},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    st.subheader("Cơ cấu GMV theo Product Line")
    tree = top_line_dynamic[top_line_dynamic["current_Ordered GMV"] > 0].copy()
    fig = px.treemap(
        tree,
        path=[px.Constant("Portfolio"), "product_line"],
        values="current_Ordered GMV",
        color="gmv_yoy_pct",
        color_continuous_scale=[COLORS["red"], "#24324b", COLORS["cyan"]],
        color_continuous_midpoint=0,
        hover_data={"gmv_yoy_pct": ":.1f"},
    )
    st.plotly_chart(plot_layout(fig, 500), use_container_width=True)

elif page == "Product Line Map":
    pl = group_performance(current, previous, "product_line")
    pl["bubble_size"] = pl["current_Ordered_units"].clip(lower=0) + 1
    fig = px.scatter(
        pl,
        x="gmv_yoy_pct",
        y="current_Ordered GMV",
        size="bubble_size",
        color="conversion_proxy_pct",
        text="product_line",
        size_max=62,
        color_continuous_scale=["#312e81", COLORS["blue"], COLORS["cyan"]],
        title="Product Line Map — tăng trưởng và quy mô",
        labels={
            "gmv_yoy_pct": "GMV growth (%)",
            "current_Ordered GMV": "Current GMV",
            "conversion_proxy_pct": "Units / Views (%)",
        },
        hover_data={
            "current_Ordered_units": ":,.0f",
            "current_Total ADS": ":,.0f",
            "current_Total Promo": ":,.0f",
        },
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["slate"])
    fig.update_traces(textposition="top center")
    st.plotly_chart(plot_layout(fig, 600), use_container_width=True)

    display_pl = pl.sort_values("current_Ordered GMV", ascending=False).rename(
        columns={
            "product_line": "Product Line",
            "current_Ordered GMV": "Current GMV",
            "prior_Ordered GMV": "Comparison GMV",
            "gmv_yoy_pct": "GMV Growth %",
            "current_Ordered_units": "Units",
            "conversion_proxy_pct": "Units / Views %",
            "ad_to_gmv_pct": "Ads / GMV %",
            "promo_to_gmv_pct": "Promo / GMV %",
        }
    )
    visible = [
        "Product Line", "Current GMV", "Comparison GMV", "GMV Growth %",
        "Units", "Units / Views %", "Ads / GMV %", "Promo / GMV %",
    ]
    st.dataframe(display_pl[visible], use_container_width=True, hide_index=True)
    st.download_button("Tải bảng Product Line", make_csv(display_pl[visible]), "product_line_filtered.csv", "text/csv")

elif page == "SKU Deep Dive":
    current_sku = current.groupby(["SKU", "product_line"], as_index=False)[
        ["Glance_views", "Ordered_units", "Ordered GMV", "Ordered_nmv", "Total Promo", "Total ADS"]
    ].sum()
    previous_sku = previous.groupby(["SKU", "product_line"], as_index=False)[
        ["Glance_views", "Ordered_units", "Ordered GMV", "Ordered_nmv", "Total Promo", "Total ADS"]
    ].sum()
    current_sku = current_sku.rename(columns={col: f"current_{col}" for col in current_sku.columns if col not in ["SKU", "product_line"]})
    previous_sku = previous_sku.rename(columns={col: f"prior_{col}" for col in previous_sku.columns if col not in ["SKU", "product_line"]})
    sku_perf = current_sku.merge(previous_sku, on=["SKU", "product_line"], how="outer").fillna(0)
    sku_perf["gmv_yoy_pct"] = np.where(
        sku_perf["prior_Ordered GMV"] != 0,
        (sku_perf["current_Ordered GMV"] / sku_perf["prior_Ordered GMV"] - 1) * 100,
        np.nan,
    )
    sku_perf = classify_sku_table(sku_perf)
    sku_perf["bubble_size"] = sku_perf["current_Ordered_units"].clip(lower=0) + 1
    fig = px.scatter(
        sku_perf,
        x="gmv_yoy_pct",
        y="current_Ordered GMV",
        size="bubble_size",
        color="Position",
        hover_name="SKU",
        hover_data=["product_line", "current_Ordered_units", "prior_Ordered GMV"],
        color_discrete_map=POSITION_COLORS,
        size_max=45,
        title="SKU Positioning Matrix",
        labels={"gmv_yoy_pct": "GMV growth (%)", "current_Ordered GMV": "Current GMV"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["slate"])
    st.plotly_chart(plot_layout(fig, 590), use_container_width=True)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        position_count = sku_perf["Position"].value_counts().reset_index()
        position_count.columns = ["Position", "SKU Count"]
        fig = px.bar(
            position_count,
            x="Position",
            y="SKU Count",
            color="Position",
            color_discrete_map=POSITION_COLORS,
            title="Phân bố vị trí SKU",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(plot_layout(fig), use_container_width=True)
    with col_b:
        focus_sku = selected_skus[0] if selected_skus else str(
            sku_perf.sort_values("current_Ordered GMV", ascending=False).iloc[0]["SKU"]
        )
        sku_series = current[current["SKU"].astype(str) == focus_sku]
        sku_trend = sku_series.groupby(pd.Grouper(key="Day", freq="W-MON"))["Ordered GMV"].sum().reset_index()
        fig = px.area(
            sku_trend,
            x="Day",
            y="Ordered GMV",
            title=f"GMV trend — SKU {focus_sku}",
            color_discrete_sequence=[COLORS["purple"]],
        )
        st.plotly_chart(plot_layout(fig), use_container_width=True)

    sku_table = sku_perf.sort_values("current_Ordered GMV", ascending=False).rename(
        columns={
            "product_line": "Product Line",
            "current_Ordered GMV": "Current GMV",
            "prior_Ordered GMV": "Comparison GMV",
            "gmv_yoy_pct": "GMV Growth %",
            "current_Ordered_units": "Units",
            "current_Total ADS": "Ad Spend",
            "current_Total Promo": "Promo Spend",
        }
    )
    sku_visible = ["SKU", "Product Line", "Position", "Current GMV", "Comparison GMV", "GMV Growth %", "Units", "Ad Spend", "Promo Spend"]
    st.dataframe(sku_table[sku_visible], use_container_width=True, hide_index=True)
    st.download_button("Tải bảng SKU", make_csv(sku_table[sku_visible]), "sku_positioning_filtered.csv", "text/csv")

elif page == "Ads & Promo":
    st.info(
        "Ads/Promo được trình bày dưới dạng liên hệ mô tả. "
        "ROAS proxy dùng tổng NMV chia Ad Spend, không phải attributed ROAS và không chứng minh quan hệ nhân quả."
    )
    pl = group_performance(current, previous, "product_line")
    left, right = st.columns(2)
    with left:
        spend = pl[["product_line", "current_Total ADS", "current_Total Promo"]].melt(
            id_vars="product_line", var_name="Spend Type", value_name="Spend"
        )
        spend["Spend Type"] = spend["Spend Type"].replace(
            {"current_Total ADS": "Ads", "current_Total Promo": "Promo"}
        )
        top_names = pl.nlargest(12, "current_Ordered GMV")["product_line"]
        spend = spend[spend["product_line"].isin(top_names)]
        fig = px.bar(
            spend,
            x="Spend",
            y="product_line",
            color="Spend Type",
            orientation="h",
            barmode="group",
            title="Ads và Promo theo Product Line",
            color_discrete_map={"Ads": COLORS["cyan"], "Promo": COLORS["purple"]},
        )
        st.plotly_chart(plot_layout(fig, 520), use_container_width=True)
    with right:
        pl["bubble_size"] = pl["current_Ordered_units"].clip(lower=0) + 1
        fig = px.scatter(
            pl,
            x="current_Total ADS",
            y="current_Ordered GMV",
            size="bubble_size",
            color="promo_to_gmv_pct",
            hover_name="product_line",
            size_max=55,
            title="Ad Spend và GMV",
            color_continuous_scale=["#312e81", COLORS["pink"], COLORS["orange"]],
            labels={"current_Total ADS": "Ad Spend", "current_Ordered GMV": "GMV"},
        )
        st.plotly_chart(plot_layout(fig, 520), use_container_width=True)

    efficiency = pl.copy()
    efficiency["NMV / Ads proxy"] = np.where(
        efficiency["current_Total ADS"] != 0,
        efficiency["current_Ordered_nmv"] / efficiency["current_Total ADS"],
        np.nan,
    )
    efficiency = efficiency.sort_values("current_Ordered GMV", ascending=False).rename(
        columns={
            "product_line": "Product Line",
            "current_Ordered GMV": "GMV",
            "current_Total ADS": "Ads",
            "current_Total Promo": "Promo",
            "ad_to_gmv_pct": "Ads / GMV %",
            "promo_to_gmv_pct": "Promo / GMV %",
        }
    )
    columns = ["Product Line", "GMV", "Ads", "Promo", "Ads / GMV %", "Promo / GMV %", "NMV / Ads proxy"]
    st.dataframe(efficiency[columns], use_container_width=True, hide_index=True)

else:
    left, right, third = st.columns(3)
    left.metric("Dòng sau lọc", f"{len(current):,}")
    right.metric("SKU sau lọc", f"{current['SKU'].nunique():,}")
    third.metric("Ngày có dữ liệu", f"{current['Day'].nunique():,}")

    daily = current.groupby("Day", as_index=False)["Ordered GMV"].sum()
    median = daily["Ordered GMV"].median()
    mad = (daily["Ordered GMV"] - median).abs().median()
    if mad > 0:
        daily["Robust Z"] = 0.6745 * (daily["Ordered GMV"] - median) / mad
    else:
        daily["Robust Z"] = 0.0
    daily["Review Flag"] = np.where(daily["Robust Z"].abs() > 3.5, "Review", "Normal")

    fig = px.scatter(
        daily,
        x="Day",
        y="Ordered GMV",
        color="Review Flag",
        color_discrete_map={"Normal": COLORS["blue"], "Review": COLORS["red"]},
        title="Daily GMV anomaly monitor",
        hover_data={"Robust Z": ":.2f"},
    )
    st.plotly_chart(plot_layout(fig, 500), use_container_width=True)
    st.caption(
        "Review Flag chỉ đánh dấu ngày khác biệt thống kê bằng Robust Z; "
        "Flash Sale hoặc sự kiện hợp lệ vẫn có thể được đánh dấu và không nên tự động xóa."
    )

    negative = current[
        (current["Ordered_units"] < 0) | (current["Ordered GMV"] < 0)
    ].copy()
    st.subheader("Negative adjustments cần rà soát")
    st.dataframe(
        negative.sort_values("Day", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    if not negative.empty:
        st.download_button(
            "Tải negative adjustments",
            make_csv(negative),
            "negative_adjustments_filtered.csv",
            "text/csv",
        )


st.divider()
st.caption(
    f"Source: App Data Mart · Grain: {'Day × ASIN' if selected_asins else 'Day × SKU'} · "
    f"Coverage: {min_date} → {max_date} · Latest incomplete day 2026-09-09 đã được loại khỏi phân tích."
)
