from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("YES4ALL_PROJECT_ROOT", "/workspace/scratch/5886bde9948a"))
UPLOAD = Path(os.environ.get("YES4ALL_SOURCE_DIR", ROOT / "upload"))
WORK = Path(os.environ.get("YES4ALL_WORK_DIR", ROOT / "work"))
OUT = Path(os.environ.get("YES4ALL_OUTPUT_DIR", WORK / "yes4all_dashboard_v2"))
TMP = OUT / "_bundle"


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA})
    )


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def stable_fraction(key: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(lambda value: pd.NA if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)).astype("string")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    return buffer.getvalue()


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    target_path = UPLOAD / "SSO_US_Oct_Target_20260922.xlsx"
    inventory_path = UPLOAD / "USA Inventory Y4A-AMZ.xlsx"
    inventory_history_path = UPLOAD / "Inv đầu kỳ Weekly.xlsx"
    hourly_path = UPLOAD / "SSO Data Extraction Hourly -- hourly (21).xlsx"
    ads_path = UPLOAD / "Y4A_Advertising_Audit_Report_WORKING_CURRENT.xlsx"
    commercial_sales_path = UPLOAD / "Yes4All_Data sales 2023 up to Aug312026.xlsx"

    # Target and six-month supply plan.
    target = pd.read_excel(target_path, sheet_name="SSO_US_Oct_Target_SKU_Monthly", header=1, dtype={"SKU": str, "ASIN": str})
    target = target.rename(
        columns={
            "Product name": "product_name",
            "Category": "category",
            "Product line": "product_line",
            "Team": "team",
            "Channel": "channel",
            "MOC": "moc",
            "MOC band": "moc_band",
            "Keep / Kill": "keep_kill",
            "Block ads": "block_ads",
            "War Plan": "war_plan",
            "PO treatment": "po_treatment",
            "Unit sold Target": "unit_target_oct",
            "GMV Target USD": "gmv_target_oct",
            "Beginning inventory units": "beginning_inventory_oct",
            "Incoming units": "incoming_oct",
            "Ending inventory units": "ending_inventory_oct",
            "Promotion": "promotion_budget_oct",
            "Ads": "ads_budget_oct",
            "MKT": "mkt_budget_oct",
            "MKT / GMV": "mkt_to_gmv_oct",
            "V1 status": "target_status",
            "Normal ASP USD": "normal_asp_usd",
            "Giá tính GMV": "gmv_price_usd",
        }
    )
    target["SKU"] = clean_text(target["SKU"])
    target["ASIN"] = clean_text(target["ASIN"])
    target = target[target["SKU"].notna()].drop_duplicates("SKU", keep="first")
    numeric(
        target,
        [
            "moc", "unit_target_oct", "gmv_target_oct", "beginning_inventory_oct",
            "incoming_oct", "ending_inventory_oct", "promotion_budget_oct",
            "ads_budget_oct", "mkt_budget_oct", "mkt_to_gmv_oct", "normal_asp_usd",
            "gmv_price_usd",
        ],
    )

    asin_map = target[["SKU", "ASIN", "product_name", "product_line", "category", "team", "channel"]].copy()
    asin_map["ASIN"] = asin_map["ASIN"].fillna("").str.split(";")
    asin_map = asin_map.explode("ASIN")
    asin_map["ASIN"] = clean_text(asin_map["ASIN"])
    asin_map = asin_map[asin_map["ASIN"].notna()].drop_duplicates(["SKU", "ASIN"])

    forecast = pd.read_excel(target_path, sheet_name="6 months demand", header=0, dtype={"SKU": str, "ASIN": str})
    forecast = forecast.rename(
        columns={
            "Product name": "product_name",
            "Product line": "product_line",
            "Channel": "channel",
            "Team": "team",
            "Portfolio": "portfolio",
            "MOC": "moc",
            "MOC band": "moc_band",
            "War group": "war_group",
            "Month": "month",
            "Baseline units": "baseline_units",
            "Potential units": "war_map_units",
            "Target demand units": "source_target_units",
            "Opening units": "source_opening_units",
            "Confirmed incoming": "confirmed_incoming",
            "Usable incoming": "usable_incoming",
            "Deferred incoming": "deferred_incoming",
            "Ending units": "source_ending_units",
            "Frozen status": "frozen_status",
        }
    )
    forecast["SKU"] = clean_text(forecast["SKU"])
    forecast["month"] = pd.to_datetime(forecast["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    forecast = forecast[forecast["SKU"].notna() & forecast["month"].notna()].copy()
    numeric(
        forecast,
        [
            "baseline_units", "war_map_units", "source_target_units", "source_opening_units",
            "confirmed_incoming", "usable_incoming", "deferred_incoming", "source_ending_units", "moc",
        ],
    )
    constrained_rows = []
    for sku, group in forecast.sort_values(["SKU", "month"]).groupby("SKU", sort=False):
        previous_ending = None
        for _, row in group.iterrows():
            opening = float(row["source_opening_units"]) if previous_ending is None else float(previous_ending)
            usable = max(float(row["usable_incoming"]), 0.0)
            available = max(opening, 0.0) + usable
            desired = max(float(row["war_map_units"]), 0.0)
            constrained = min(desired, available)
            ending = max(available - constrained, 0.0)
            copy = row.to_dict()
            copy.update(
                {
                    "recalc_opening_units": opening,
                    "available_supply_units": available,
                    "inventory_constrained_target_units": constrained,
                    "recalc_ending_units": ending,
                    "war_map_gap_units": max(desired - constrained, 0.0),
                    "baseline_gap_units": max(float(row["baseline_units"]) - constrained, 0.0),
                }
            )
            constrained_rows.append(copy)
            previous_ending = ending
    forecast = pd.DataFrame(constrained_rows)

    # Latest daily sales snapshot (the workbook name says hourly, but the available grain is Day x SKU x ASIN).
    sales = pd.read_excel(hourly_path, sheet_name="hourly", dtype={"sku": str, "asin": str})
    sales = sales.rename(
        columns={
            "date": "Day", "sku": "SKU", "asin": "ASIN", "main_category": "main_category",
            "ordered_units": "Ordered_units", "ordered_nmv": "Ordered_nmv", "ordered_gmv": "Ordered_GMV",
            "total_promo": "Total_Promo", "total_ads": "Total_ADS",
        }
    )
    sales["Day"] = pd.to_datetime(sales["Day"], errors="coerce").dt.normalize()
    sales["SKU"] = clean_text(sales["SKU"])
    sales["ASIN"] = clean_text(sales["ASIN"])
    numeric(sales, [c for c in sales.columns if c not in {"Day", "department", "SKU", "ASIN", "main_category"}])
    sales = sales[sales["Day"].notna() & sales["SKU"].notna()].copy()
    sales = sales.merge(
        target[["SKU", "product_name", "product_line", "category", "team", "channel"]],
        on="SKU",
        how="left",
        validate="many_to_one",
    )
    sales["product_line"] = sales["product_line"].fillna(sales["main_category"]).fillna("Unmapped")
    sales["product_name"] = sales["product_name"].fillna(sales["SKU"])
    sales["category"] = sales["category"].fillna(sales["main_category"]).fillna("Unmapped")
    sales["team"] = sales["team"].fillna("Unmapped")
    sales["channel"] = sales["channel"].fillna("Unmapped")
    sales["ACOS_proxy"] = np.where(sales["Ordered_nmv"] > 0, sales["Total_ADS"] / sales["Ordered_nmv"], np.nan)
    ad_clicks = sum((sales.get(c, 0) for c in ["sb_clicks", "sd_clicks", "sp_clicks"]))
    ad_impressions = sum((sales.get(c, 0) for c in ["sb_impressions", "sd_impressions", "sp_impressions"]))
    ad_orders = sum((sales.get(c, 0) for c in ["sb_orders", "sd_orders", "sp_orders"]))
    sales["Ad_clicks"] = ad_clicks
    sales["Ad_impressions"] = ad_impressions
    sales["Ad_orders"] = ad_orders
    sales["CTR"] = np.where(ad_impressions > 0, ad_clicks / ad_impressions, np.nan)
    sales["Ad_CVR"] = np.where(ad_clicks > 0, ad_orders / ad_clicks, np.nan)

    # Commercial sales history: authoritative monthly data through Aug-2026,
    # then retain the existing hourly source for Sep-2026 and roll it to month.
    # Team/PIC is mapped strictly by SKU from the target workbook; unmatched = N/A.
    commercial_raw = pd.read_excel(
        commercial_sales_path,
        sheet_name="Export",
        dtype={"SKU": str, "ASIN": str},
    )
    commercial_raw = commercial_raw.rename(
        columns={
            "Month": "Day",
            "Ordered GMV": "Ordered_GMV",
            "Total Promo": "Total_Promo",
            "Total ADS": "Total_ADS",
        }
    )
    commercial_raw["Day"] = pd.to_datetime(commercial_raw["Day"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    commercial_raw["SKU"] = clean_text(commercial_raw["SKU"])
    commercial_raw["ASIN"] = clean_text(commercial_raw["ASIN"])
    commercial_raw = commercial_raw[
        commercial_raw["Day"].notna()
        & commercial_raw["SKU"].notna()
        & commercial_raw["Dept"].astype("string").str.strip().eq("SSO")
        & commercial_raw["Country"].astype("string").str.strip().eq("USA")
        & commercial_raw["Day"].lt(pd.Timestamp("2026-09-01"))
    ].copy()

    commercial_numeric = [
        "Glance_views", "Ordered_units", "Ordered_revenue", "Ordered_nmv",
        "Shipped_units", "Shipped_revenue", "Shipped_nmv", "Ordered_GMV",
        "Total_Promo", "Price_discount_spend", "Best_deal_spend",
        "Lightning_deal_spend", "VM_promo_spend", "Coupon_spend", "Total_ADS",
        "sb_spend", "sd_spend", "sp_spend", "dsp_spend", "aff_spend",
        "Sb_clicks", "Sd_clicks", "Sp_clicks", "Dsp_clicks", "Aff_clicks",
        "Sb_impressions", "Sd_impressions", "Sp_impressions", "Dsp_impressions", "Aff_impressions",
        "Aff_ordered_nmv", "Dsp_ordered_nmv", "Sb_ordered_nmv", "Sd_ordered_nmv", "Sp_ordered_nmv",
        "Aff_ordered_units", "Dsp_ordered_units", "Sb_ordered_units", "Sd_ordered_units", "Sp_ordered_units",
        "Aff_orders", "Dsp_orders", "Sb_orders", "Sd_orders", "Sp_orders",
    ]
    numeric(commercial_raw, commercial_numeric)

    dimension_map = target[
        ["SKU", "product_name", "product_line", "category", "team", "channel"]
    ].drop_duplicates("SKU")
    source_dimensions = commercial_raw[["SKU", "product_name", "product_line"]].rename(
        columns={"product_name": "source_product_name", "product_line": "source_product_line"}
    )
    commercial_raw = commercial_raw.drop(columns=["product_name", "product_line"]).merge(
        dimension_map,
        on="SKU",
        how="left",
        validate="many_to_one",
    ).merge(
        source_dimensions.drop_duplicates("SKU", keep="last"),
        on="SKU",
        how="left",
        validate="many_to_one",
    )
    commercial_raw["product_name"] = commercial_raw["product_name"].fillna(commercial_raw["source_product_name"]).fillna(commercial_raw["SKU"])
    commercial_raw["product_line"] = commercial_raw["product_line"].fillna(commercial_raw["source_product_line"]).fillna("Unmapped")
    commercial_raw["category"] = commercial_raw["category"].fillna("Unmapped")
    commercial_raw["team"] = commercial_raw["team"].fillna("N/A")
    commercial_raw["channel"] = commercial_raw["channel"].fillna("N/A")
    commercial_raw["data_source"] = "Monthly sales through Aug-2026"

    # Preserve the existing September data, aggregated from daily to monthly.
    september = sales[sales["Day"].ge(pd.Timestamp("2026-09-01"))].copy()
    september["Day"] = september["Day"].dt.to_period("M").dt.to_timestamp()
    september["Glance_views"] = 0.0
    september["Ordered_revenue"] = september["Ordered_GMV"]
    september["Shipped_units"] = 0.0
    september["Shipped_revenue"] = 0.0
    september["Shipped_nmv"] = 0.0
    september["Price_discount_spend"] = september.get("price_discount_spend", 0.0)
    september["Best_deal_spend"] = september.get("best_deal_spend", 0.0)
    september["Lightning_deal_spend"] = september.get("lightning_deal_spend", 0.0)
    september["VM_promo_spend"] = september.get("vm_promo_spend", 0.0)
    september["Coupon_spend"] = september.get("coupon_spend", 0.0)
    september["dsp_spend"] = 0.0
    september["aff_spend"] = 0.0
    for prefix in ["Dsp", "Aff"]:
        for suffix in ["clicks", "impressions", "ordered_nmv", "ordered_units", "orders"]:
            september[f"{prefix}_{suffix}"] = 0.0
    september["Sb_clicks"] = september.get("sb_clicks", 0.0)
    september["Sd_clicks"] = september.get("sd_clicks", 0.0)
    september["Sp_clicks"] = september.get("sp_clicks", 0.0)
    september["Sb_impressions"] = september.get("sb_impressions", 0.0)
    september["Sd_impressions"] = september.get("sd_impressions", 0.0)
    september["Sp_impressions"] = september.get("sp_impressions", 0.0)
    september["Sb_ordered_nmv"] = september.get("sb_ordered_nmv", 0.0)
    september["Sd_ordered_nmv"] = september.get("sd_ordered_nmv", 0.0)
    september["Sp_ordered_nmv"] = september.get("sp_ordered_nmv", 0.0)
    september["Sb_ordered_units"] = september.get("sb_ordered_units", 0.0)
    september["Sd_ordered_units"] = september.get("sd_ordered_units", 0.0)
    september["Sp_ordered_units"] = september.get("sp_ordered_units", 0.0)
    september["Sb_orders"] = september.get("sb_orders", 0.0)
    september["Sd_orders"] = september.get("sd_orders", 0.0)
    september["Sp_orders"] = september.get("sp_orders", 0.0)
    september["data_source"] = "Existing Sep-2026 hourly data (monthly roll-up)"

    commercial_columns = [
        "Day", "SKU", "ASIN", "product_name", "product_line", "category", "team", "channel",
        "Glance_views", "Ordered_units", "Ordered_revenue", "Ordered_nmv", "Shipped_units",
        "Shipped_revenue", "Shipped_nmv", "Ordered_GMV", "Total_Promo", "Price_discount_spend",
        "Best_deal_spend", "Lightning_deal_spend", "VM_promo_spend", "Coupon_spend", "Total_ADS",
        "sb_spend", "sd_spend", "sp_spend", "dsp_spend", "aff_spend",
        "Sb_clicks", "Sd_clicks", "Sp_clicks", "Dsp_clicks", "Aff_clicks",
        "Sb_impressions", "Sd_impressions", "Sp_impressions", "Dsp_impressions", "Aff_impressions",
        "Aff_ordered_nmv", "Dsp_ordered_nmv", "Sb_ordered_nmv", "Sd_ordered_nmv", "Sp_ordered_nmv",
        "Aff_ordered_units", "Dsp_ordered_units", "Sb_ordered_units", "Sd_ordered_units", "Sp_ordered_units",
        "Aff_orders", "Dsp_orders", "Sb_orders", "Sd_orders", "Sp_orders", "data_source",
    ]
    for column in commercial_columns:
        if column not in september.columns:
            september[column] = pd.NA if column in {"product_name", "product_line", "category", "team", "channel", "data_source"} else 0.0
        if column not in commercial_raw.columns:
            commercial_raw[column] = pd.NA if column in {"product_name", "product_line", "category", "team", "channel", "data_source"} else 0.0
    commercial = pd.concat(
        [commercial_raw[commercial_columns], september[commercial_columns]],
        ignore_index=True,
    )
    commercial["team"] = commercial["team"].replace({"Unmapped": "N/A"}).fillna("N/A")
    commercial["channel"] = commercial["channel"].replace({"Unmapped": "N/A"}).fillna("N/A")

    group_dimensions = ["Day", "SKU", "ASIN", "product_name", "product_line", "category", "team", "channel", "data_source"]
    group_metrics = [column for column in commercial_columns if column not in group_dimensions]
    commercial = commercial.groupby(group_dimensions, as_index=False, dropna=False)[group_metrics].sum()
    commercial["Ad_clicks"] = commercial[["Sb_clicks", "Sd_clicks", "Sp_clicks", "Dsp_clicks", "Aff_clicks"]].sum(axis=1)
    commercial["Ad_impressions"] = commercial[["Sb_impressions", "Sd_impressions", "Sp_impressions", "Dsp_impressions", "Aff_impressions"]].sum(axis=1)
    commercial["Ad_orders"] = commercial[["Sb_orders", "Sd_orders", "Sp_orders", "Dsp_orders", "Aff_orders"]].sum(axis=1)
    commercial["Ad_attributed_units"] = commercial[["Sb_ordered_units", "Sd_ordered_units", "Sp_ordered_units", "Dsp_ordered_units", "Aff_ordered_units"]].sum(axis=1)
    commercial["Ad_attributed_nmv"] = commercial[["Sb_ordered_nmv", "Sd_ordered_nmv", "Sp_ordered_nmv", "Dsp_ordered_nmv", "Aff_ordered_nmv"]].sum(axis=1)
    commercial["MKT_spend"] = commercial["Total_ADS"] + commercial["Total_Promo"]
    commercial["ASP"] = np.where(commercial["Ordered_units"] != 0, commercial["Ordered_GMV"] / commercial["Ordered_units"], np.nan)
    commercial["Ads_pct_GMV"] = np.where(commercial["Ordered_GMV"] > 0, commercial["Total_ADS"] / commercial["Ordered_GMV"], np.nan)
    commercial["Promo_pct_GMV"] = np.where(commercial["Ordered_GMV"] > 0, commercial["Total_Promo"] / commercial["Ordered_GMV"], np.nan)
    commercial["MKT_pct_GMV"] = np.where(commercial["Ordered_GMV"] > 0, commercial["MKT_spend"] / commercial["Ordered_GMV"], np.nan)
    commercial["Ads_CPU"] = np.where(commercial["Ordered_units"] > 0, commercial["Total_ADS"] / commercial["Ordered_units"], np.nan)
    commercial["Promo_CPU"] = np.where(commercial["Ordered_units"] > 0, commercial["Total_Promo"] / commercial["Ordered_units"], np.nan)
    commercial["MKT_CPU"] = np.where(commercial["Ordered_units"] > 0, commercial["MKT_spend"] / commercial["Ordered_units"], np.nan)
    commercial["Ad_attributed_CPU"] = np.where(commercial["Ad_attributed_units"] > 0, commercial["Total_ADS"] / commercial["Ad_attributed_units"], np.nan)
    commercial["Ad_units_share"] = np.where(commercial["Ordered_units"] > 0, commercial["Ad_attributed_units"] / commercial["Ordered_units"], np.nan)
    commercial["ROAS_attributed"] = np.where(commercial["Total_ADS"] > 0, commercial["Ad_attributed_nmv"] / commercial["Total_ADS"], np.nan)
    commercial["ACOS_attributed"] = np.where(commercial["Ad_attributed_nmv"] > 0, commercial["Total_ADS"] / commercial["Ad_attributed_nmv"], np.nan)
    commercial["Conversion_proxy"] = np.where(commercial["Glance_views"] > 0, commercial["Ordered_units"] / commercial["Glance_views"], np.nan)
    commercial["CTR"] = np.where(commercial["Ad_impressions"] > 0, commercial["Ad_clicks"] / commercial["Ad_impressions"], np.nan)
    commercial["Ad_CVR"] = np.where(commercial["Ad_clicks"] > 0, commercial["Ad_orders"] / commercial["Ad_clicks"], np.nan)
    commercial["Year"] = commercial["Day"].dt.year
    commercial["Month"] = commercial["Day"].dt.month
    commercial["YearMonth"] = commercial["Day"].dt.to_period("M").astype(str)
    commercial["Promo_units"] = np.nan
    commercial["promo_units_definition"] = "Unavailable in source"

    # Current inventory. Y4A stock is a shared SKU pool repeated over ASIN rows: use MAX at SKU grain.
    inv = pd.read_excel(inventory_path, sheet_name="report", header=3, dtype={"SKU": str, "ASIN": str})
    inv = inv.loc[:, ~inv.columns.astype(str).str.startswith("Unnamed")]
    inv["SKU"] = clean_text(inv["SKU"])
    inv["ASIN"] = clean_text(inv["ASIN"])
    inv = inv[(inv["DEP."].astype("string").str.strip() == "SSO") & inv["SKU"].notna()].copy()
    numeric(inv, ["SALABLE Y4A", "SALABLE AMZ", "Incoming Y4A", "Incoming AMZ"])
    inv_asin = inv[["SKU", "ASIN", "ASIN STATUS", "SALABLE Y4A", "SALABLE AMZ", "Incoming Y4A", "Incoming AMZ"]].copy()
    inv_asin = inv_asin.rename(
        columns={
            "ASIN STATUS": "asin_status", "SALABLE Y4A": "shared_y4a_inventory",
            "SALABLE AMZ": "asin_amz_inventory", "Incoming Y4A": "shared_y4a_incoming",
            "Incoming AMZ": "asin_amz_incoming",
        }
    )
    inv_asin = inv_asin[inv_asin["ASIN"].notna()].drop_duplicates(["SKU", "ASIN"], keep="first")

    inv_sku = (
        inv.groupby("SKU", as_index=False)
        .agg(
            y4a_inventory=("SALABLE Y4A", "max"),
            amz_inventory=("SALABLE AMZ", "sum"),
            y4a_incoming=("Incoming Y4A", "max"),
            amz_incoming=("Incoming AMZ", "sum"),
            asin_count=("ASIN", "nunique"),
            active_asin=("ASIN STATUS", lambda s: int((s.astype("string").str.lower() == "active").sum())),
        )
    )
    inv_sku["current_inventory"] = inv_sku["y4a_inventory"] + inv_sku["amz_inventory"]
    inv_sku["incoming_inventory"] = inv_sku["y4a_incoming"] + inv_sku["amz_incoming"]
    inv_sku = inv_sku.merge(target[["SKU", "product_name", "product_line", "category", "team", "channel", "moc", "moc_band", "keep_kill"]], on="SKU", how="left")

    # Repair one known crosswalk mismatch through ASIN: target SKU '9' -> inventory SKU '9E00'.
    if "9" in set(target["SKU"].dropna()) and "9E00" in set(inv_sku["SKU"].dropna()):
        missing_target = ~inv_sku["SKU"].isin(target["SKU"])
        inv_sku.loc[missing_target & inv_sku["SKU"].eq("9E00"), "SKU"] = "9"
        inv_asin.loc[inv_asin["SKU"].eq("9E00"), "SKU"] = "9"

    # Opening-inventory history. Actual available grain is monthly.
    inv_history = pd.read_excel(inventory_history_path, sheet_name="Export", dtype={"sku": str})
    inv_history = inv_history.rename(
        columns={
            "sku": "SKU", "first_date_of_month": "month", "inv_amz_đầu kỳ": "opening_amz",
            "inv_y4a_đầu kỳ": "opening_y4a", "inv_wf_đầu kỳ": "opening_wf", "inv_total_đầu kỳ": "opening_total",
        }
    )
    inv_history["SKU"] = clean_text(inv_history["SKU"])
    inv_history["month"] = pd.to_datetime(inv_history["month"], errors="coerce").dt.normalize()
    inv_history = inv_history[(inv_history["department"] == "SSO") & (inv_history["country"] == "USA") & inv_history["SKU"].notna() & inv_history["month"].notna()].copy()
    numeric(inv_history, ["opening_amz", "opening_y4a", "opening_wf", "opening_total"])
    inv_history = inv_history.drop_duplicates(["month", "SKU"], keep="last")
    inv_history = inv_history.merge(target[["SKU", "product_line", "product_name", "team", "channel"]], on="SKU", how="left")

    # Campaign audit: keep non-test US campaigns; all US accounts remain filterable.
    ads = pd.read_excel(ads_path, sheet_name="RAW_CAMPAIGN")
    ads = ads[(ads["marketplaceId"] == "ATVPDKIKX0DER") & (~ads["IsTestBrandAccount"].fillna(False).astype(bool))].copy()
    numeric(ads, ["SpendUSD", "SalesUSD", "impressions", "clicks", "orders", "BudgetUSD", "effectiveDailyCoverage", "topOfSearchImpressionShare"])
    ads["ACOS"] = np.where(ads["SalesUSD"] > 0, ads["SpendUSD"] / ads["SalesUSD"], np.nan)
    ads["ROAS"] = np.where(ads["SpendUSD"] > 0, ads["SalesUSD"] / ads["SpendUSD"], np.nan)
    ads["CTR"] = np.where(ads["impressions"] > 0, ads["clicks"] / ads["impressions"], np.nan)
    ads["CVR"] = np.where(ads["clicks"] > 0, ads["orders"] / ads["clicks"], np.nan)
    ads["CPC"] = np.where(ads["clicks"] > 0, ads["SpendUSD"] / ads["clicks"], np.nan)
    ads["campaign_action"] = "Monitor"
    ads.loc[(ads["state"] == "ENABLED") & (ads["SalesUSD"] <= 0) & (ads["SpendUSD"] >= 20), "campaign_action"] = "Pause or add negatives"
    ads.loc[(ads["state"] == "ENABLED") & (ads["ACOS"] > 0.35) & (ads["SpendUSD"] >= 100), "campaign_action"] = "Reduce bid and inspect targets"
    ads.loc[(ads["state"] == "ENABLED") & (ads["ROAS"] >= 4) & (ads["effectiveDailyCoverage"] < 80), "campaign_action"] = "Consider budget increase"
    ads.loc[(ads["state"] == "PAUSED") & (ads["ROAS"] >= 5) & (ads["SalesUSD"] >= 500), "campaign_action"] = "Review for reactivation"

    asin_campaign = pd.read_excel(ads_path, sheet_name="RAW_ASIN_CAMP", dtype={"adAsin": str, "campaignId": str})
    asin_campaign = asin_campaign.rename(columns={"adAsin": "ASIN"})
    asin_campaign["ASIN"] = clean_text(asin_campaign["ASIN"])
    asin_campaign = asin_campaign.merge(
        ads[["campaignId", "campaignName", "Account", "programType", "state", "SpendUSD", "SalesUSD", "ACOS", "ROAS", "campaign_action", "AuditFlags"]],
        on="campaignId",
        how="inner",
        validate="many_to_one",
    )

    # Deterministic demo listing health, explicitly classified as simulated.
    listing = asin_map.merge(target[["SKU", "block_ads", "target_status", "moc", "moc_band", "keep_kill"]], on="SKU", how="left")
    listing = listing.merge(inv_asin, on=["SKU", "ASIN"], how="left")
    listing_rows = []
    as_of = pd.Timestamp(sales["Day"].max())
    for _, row in listing.iterrows():
        key = f"{row['SKU']}|{row['ASIN']}"
        blocked_basis = str(row.get("block_ads", "")).lower()
        status_basis = str(row.get("target_status", "")).lower()
        inv_amz = float(row.get("asin_amz_inventory") or 0)
        p_block = 0.32 if "blocked" in blocked_basis else 0.04
        listing_blocked = stable_fraction(key, "listing-block") < p_block
        zip_blocked = stable_fraction(key, "zip-block") < (0.22 if listing_blocked else 0.035)
        shipping_days = int(2 + math.floor(stable_fraction(key, "ship-days") * 10))
        if inv_amz <= 0:
            shipping_days += 4
        long_shipping = shipping_days >= 8
        if listing_blocked:
            listing_status = "Blocked"
            issue = "Listing eligibility review"
        elif zip_blocked:
            listing_status = "Active with ZIP restriction"
            issue = "ZIP code unavailable"
        elif long_shipping:
            listing_status = "Active with long delivery"
            issue = "Long shipping promise"
        else:
            listing_status = "Active"
            issue = "None"
        listing_rows.append(
            {
                **row.to_dict(),
                "snapshot_date": as_of,
                "listing_status_demo": listing_status,
                "listing_issue_demo": issue,
                "zip_code_tested_demo": "10001",
                "zip_available_demo": not zip_blocked,
                "shipping_days_demo": shipping_days,
                "long_shipping_demo": long_shipping,
                "listing_blocked_demo": listing_blocked,
                "is_simulated": True,
                "simulation_basis": "Deterministic demo seeded by Block ads, target status, and inventory",
            }
        )
    listing = pd.DataFrame(listing_rows)

    # Deterministic 30-day demo ranking and keyword history.
    asin_sales = sales.groupby(["SKU", "ASIN"], as_index=False).agg(units=("Ordered_units", "sum"), gmv=("Ordered_GMV", "sum"))
    ranking_base = asin_map.merge(asin_sales, on=["SKU", "ASIN"], how="left").fillna({"units": 0.0, "gmv": 0.0})
    ranking_rows = []
    dates = pd.date_range(as_of - pd.Timedelta(days=29), as_of, freq="D")
    for _, row in ranking_base.iterrows():
        key = f"{row['SKU']}|{row['ASIN']}"
        nonnegative_units = max(float(row["units"]), 0.0)
        base_rank = max(3, min(180, int(125 - 18 * np.log1p(nonnegative_units) + 35 * stable_fraction(key, "rank-base"))))
        keywords = [
            str(row.get("product_line") or "fitness equipment"),
            f"{str(row.get('category') or 'fitness')} equipment",
            "home gym accessory",
        ]
        for keyword_index, keyword in enumerate(keywords):
            keyword_offset = keyword_index * 11
            for day_index, day in enumerate(dates):
                wave = 6 * math.sin((day_index + stable_fraction(key, "wave") * 5) / 4)
                noise = (stable_fraction(f"{key}|{day.date()}|{keyword_index}", "rank-noise") - 0.5) * 8
                organic = int(max(1, min(200, round(base_rank + keyword_offset + wave + noise - day_index * 0.18))))
                sponsored = int(max(1, min(200, organic + round((stable_fraction(key, f"sponsored-{keyword_index}") - 0.55) * 24))))
                ranking_rows.append(
                    {
                        "Day": day,
                        "SKU": row["SKU"],
                        "ASIN": row["ASIN"],
                        "product_name": row.get("product_name"),
                        "product_line": row.get("product_line"),
                        "keyword_demo": keyword,
                        "organic_rank_demo": organic,
                        "sponsored_rank_demo": sponsored,
                        "search_volume_demo": int(300 + 9700 * stable_fraction(keyword, "volume")),
                        "is_simulated": True,
                    }
                )
    ranking = pd.DataFrame(ranking_rows)

    # Keep historical sales in a separate legacy-scope table. Do not blend it with the new broader hourly scope.
    history_sku_path = WORK / "data" / "sales_daily_sku_app.parquet"
    history_asin_path = WORK / "data" / "sales_daily_asin_app.parquet"
    existing_bundle = os.environ.get("YES4ALL_EXISTING_BUNDLE")
    if history_sku_path.exists() and history_asin_path.exists():
        history_sku = pd.read_parquet(history_sku_path)
        history_asin = pd.read_parquet(history_asin_path)
    elif existing_bundle and Path(existing_bundle).exists():
        with zipfile.ZipFile(existing_bundle) as archive:
            history_sku = pd.read_parquet(io.BytesIO(archive.read("sales_history_sku.parquet")))
            history_asin = pd.read_parquet(io.BytesIO(archive.read("sales_history_asin.parquet")))
    else:
        history_sku = pd.DataFrame(columns=["Day", "SKU", "product_line", "product_name", "Ordered_units", "Ordered GMV", "Ordered_nmv", "Total ADS", "Total Promo"])
        history_asin = pd.DataFrame(columns=["Day", "SKU", "ASIN", "product_line", "product_name", "Ordered_units", "Ordered GMV", "Ordered_nmv", "Total ADS", "Total Promo"])

    latest_day = pd.Timestamp(sales["Day"].max())
    latest_opening = inv_history[inv_history["month"] == inv_history["month"].max()][["SKU", "opening_total"]]
    actual_mtd = sales.groupby("SKU", as_index=False).agg(actual_units_mtd=("Ordered_units", "sum"), actual_gmv_mtd=("Ordered_GMV", "sum"), actual_nmv_mtd=("Ordered_nmv", "sum"), actual_ads_mtd=("Total_ADS", "sum"))
    performance = target.merge(actual_mtd, on="SKU", how="left").merge(latest_opening, on="SKU", how="left")
    numeric(performance, ["actual_units_mtd", "actual_gmv_mtd", "actual_nmv_mtd", "actual_ads_mtd", "opening_total"])
    performance["unit_attainment"] = np.where(performance["unit_target_oct"] > 0, performance["actual_units_mtd"] / performance["unit_target_oct"], np.nan)
    performance["gmv_attainment"] = np.where(performance["gmv_target_oct"] > 0, performance["actual_gmv_mtd"] / performance["gmv_target_oct"], np.nan)
    performance["sell_through_proxy"] = np.where(performance["opening_total"] > 0, performance["actual_units_mtd"] / performance["opening_total"], np.nan)
    performance["actual_acos_proxy"] = np.where(performance["actual_nmv_mtd"] > 0, performance["actual_ads_mtd"] / performance["actual_nmv_mtd"], np.nan)

    quality = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "latest_sales_date": latest_day.date().isoformat(),
        "sales_date_min": pd.Timestamp(sales["Day"].min()).date().isoformat(),
        "sales_rows": int(len(sales)),
        "sales_sku": int(sales["SKU"].nunique()),
        "sales_asin": int(sales["ASIN"].nunique()),
        "target_sku": int(target["SKU"].nunique()),
        "sales_target_sku_coverage": float(target["SKU"].isin(sales["SKU"]).mean()),
        "inventory_target_sku_coverage": float(target["SKU"].isin(inv_sku["SKU"]).mean()),
        "inventory_history_frequency": "monthly",
        "inventory_history_min": pd.Timestamp(inv_history["month"].min()).date().isoformat(),
        "inventory_history_max": pd.Timestamp(inv_history["month"].max()).date().isoformat(),
        "campaign_rows_us_non_test": int(len(ads)),
        "listing_health_classification": "simulated_demo",
        "ranking_keyword_classification": "simulated_demo",
        "commercial_sales_min": pd.Timestamp(commercial["Day"].min()).date().isoformat(),
        "commercial_sales_max": pd.Timestamp(commercial["Day"].max()).date().isoformat(),
        "commercial_rows": int(len(commercial)),
        "commercial_sku": int(commercial["SKU"].nunique()),
        "commercial_asin": int(commercial["ASIN"].nunique()),
        "commercial_pic_mapped_sku": int(commercial.loc[commercial["team"] != "N/A", "SKU"].nunique()),
        "commercial_pic_unmapped_sku": int(commercial.loc[commercial["team"] == "N/A", "SKU"].nunique()),
        "commercial_team_distribution": {
            str(key): int(value) for key, value in commercial.groupby("team")["SKU"].nunique().items()
        },
        "metric_definitions": {
            "baseline_units": "Baseline forecast supplied in target workbook",
            "war_map_units": "Potential units supplied in target workbook",
            "inventory_constrained_target_units": "min(War Map units, recalculated opening inventory + usable incoming), rolled forward monthly",
            "frozen_status": "Procurement flexibility flag: Frozen means no additional purchasing is assumed beyond currently usable incoming",
            "sell_through_proxy": "September ordered units divided by September opening inventory; receipts-to-date are unavailable",
            "ASP": "Ordered GMV divided by ordered units",
            "Ads_pct_GMV": "Total ADS divided by ordered GMV",
            "Promo_pct_GMV": "Total Promo divided by ordered GMV",
            "MKT_pct_GMV": "(Total ADS + Total Promo) divided by ordered GMV",
            "MKT_CPU": "(Total ADS + Total Promo) divided by ordered units",
            "Ad_attributed_units": "Sum of SB, SD, SP, DSP and affiliate attributed ordered units",
            "Promo_units": "Not available in the supplied source; no proxy is presented as actual",
            "planning_calculator": "Historical spend-rate benchmark for planning; descriptive/correlational, not a causal sales guarantee",
        },
    }

    tables = {
        "sales_current.parquet": sales,
        "performance_mtd.parquet": performance,
        "target_sku.parquet": target,
        "forecast_supply.parquet": forecast,
        "inventory_current_sku.parquet": inv_sku,
        "inventory_current_asin.parquet": inv_asin,
        "inventory_history.parquet": inv_history,
        "ads_campaign.parquet": ads,
        "ads_asin_campaign.parquet": asin_campaign,
        "listing_health_demo.parquet": listing,
        "ranking_keyword_demo.parquet": ranking,
        "sales_history_sku.parquet": history_sku,
        "sales_history_asin.parquet": history_asin,
        "sales_commercial_monthly.parquet": commercial,
    }

    bundle_path = OUT / "app_data_v2.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, frame in tables.items():
            bundle.writestr(name, to_parquet_bytes(frame))
        bundle.writestr("quality_summary.json", json.dumps(quality, indent=2, ensure_ascii=False))

    manifest = {
        "bundle": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "tables": {name: {"rows": int(len(frame)), "columns": list(frame.columns)} for name, frame in tables.items()},
        "quality": quality,
    }
    (OUT / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"bundle": str(bundle_path), "size_mb": round(bundle_path.stat().st_size / 1024 / 1024, 2), "tables": {k: len(v) for k, v in tables.items()}, "quality": quality}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    build()
