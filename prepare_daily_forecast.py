from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_MODELS = [
    "Naive", "MA3", "ETS", "SARIMA", "Prophet", "LightGBM_Global",
    "Ens_ETS_LGBM", "Ens_Median4",
]


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def read_bundle(path: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".parquet"):
                tables[name.removesuffix(".parquet")] = pd.read_parquet(io.BytesIO(archive.read(name)))
        quality = json.loads(archive.read("quality_summary.json").decode("utf-8"))
    return tables, quality


def build_daily(history_dir: Path, current: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    files = sorted(history_dir.glob("Yes4All data tusteam * daily.xlsx"))
    if len(files) != 3:
        raise ValueError(f"Expected 3 daily workbooks, found {len(files)} in {history_dir}")

    raw = pd.concat((pd.read_excel(path) for path in files), ignore_index=True)
    raw["Day"] = pd.to_datetime(raw["Day"], errors="coerce")
    raw["SKU"] = clean_text(raw["SKU"])
    raw["ASIN"] = clean_text(raw["ASIN"])
    raw = raw[
        raw["Day"].notna()
        & raw["SKU"].ne("")
        & clean_text(raw["Dept"]).str.upper().eq("SSO")
        & clean_text(raw["Country"]).str.upper().eq("USA")
    ].copy()

    rename = {
        "Ordered GMV": "Ordered_GMV", "Total Promo": "Total_Promo", "Total ADS": "Total_ADS",
        "Price_discount_spend": "price_discount_spend", "Best_deal_spend": "best_deal_spend",
        "Lightning_deal_spend": "lightning_deal_spend", "VM_promo_spend": "vm_promo_spend",
        "Coupon_spend": "coupon_spend",
    }
    raw = raw.rename(columns=rename)
    numeric = [
        "Glance_views", "Ordered_units", "Ordered_nmv", "Ordered_GMV", "Total_Promo", "Total_ADS",
        "price_discount_spend", "best_deal_spend", "lightning_deal_spend", "vm_promo_spend",
        "coupon_spend", "sb_spend", "sd_spend", "sp_spend", "dsp_spend", "aff_spend",
    ]
    for prefix in ["Sb", "Sd", "Sp", "Dsp", "Aff"]:
        numeric += [f"{prefix}_clicks", f"{prefix}_impressions", f"{prefix}_ordered_nmv", f"{prefix}_ordered_units", f"{prefix}_orders"]
    for column in numeric:
        if column not in raw:
            raw[column] = 0.0
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)

    dimensions = ["Day", "SKU", "ASIN", "product_name", "product_line"]
    daily = raw.groupby(dimensions, dropna=False, as_index=False)[numeric].sum()
    scope_skus = set(daily["SKU"])

    # Retain September MTD from the previously validated mart, but only for the 163-SKU scope.
    cur = current[current["SKU"].astype(str).isin(scope_skus)].copy()
    cur["Day"] = pd.to_datetime(cur["Day"], errors="coerce")
    cur = cur[cur["Day"] > daily["Day"].max()]
    if not cur.empty:
        normalized = pd.DataFrame({
            "Day": cur["Day"], "SKU": clean_text(cur["SKU"]), "ASIN": clean_text(cur["ASIN"]),
            "product_name": cur["product_name"], "product_line": cur["product_line"],
        })
        for column in numeric:
            if column in cur:
                normalized[column] = pd.to_numeric(cur[column], errors="coerce").fillna(0.0)
            else:
                lower = column[:1].lower() + column[1:]
                normalized[column] = pd.to_numeric(cur.get(lower, 0.0), errors="coerce").fillna(0.0) if hasattr(cur.get(lower, 0.0), "fillna") else 0.0
        daily = pd.concat([daily, normalized], ignore_index=True)

    mapping = target[["SKU", "team", "channel", "category"]].drop_duplicates("SKU").copy()
    mapping["SKU"] = clean_text(mapping["SKU"])
    daily = daily.merge(mapping, on="SKU", how="left")
    daily["team"] = daily["team"].fillna("N/A")
    daily["channel"] = daily["channel"].fillna("N/A")
    daily["category"] = daily["category"].fillna("N/A")

    daily["Ad_clicks"] = sum(daily.get(f"{p}_clicks", 0) for p in ["Sb", "Sd", "Sp", "Dsp", "Aff"])
    daily["Ad_impressions"] = sum(daily.get(f"{p}_impressions", 0) for p in ["Sb", "Sd", "Sp", "Dsp", "Aff"])
    daily["Ad_orders"] = sum(daily.get(f"{p}_orders", 0) for p in ["Sb", "Sd", "Sp", "Dsp", "Aff"])
    daily["Ad_attributed_units"] = sum(daily.get(f"{p}_ordered_units", 0) for p in ["Sb", "Sd", "Sp", "Dsp", "Aff"])
    daily["Ad_attributed_nmv"] = sum(daily.get(f"{p}_ordered_nmv", 0) for p in ["Sb", "Sd", "Sp", "Dsp", "Aff"])
    daily["ASP"] = daily["Ordered_GMV"].div(daily["Ordered_units"].replace(0, np.nan))
    daily["Ads_GMV"] = daily["Total_ADS"].div(daily["Ordered_GMV"].replace(0, np.nan))
    daily["Promo_GMV"] = daily["Total_Promo"].div(daily["Ordered_GMV"].replace(0, np.nan))
    daily["MKT_GMV"] = (daily["Total_ADS"] + daily["Total_Promo"]).div(daily["Ordered_GMV"].replace(0, np.nan))
    daily["CPU"] = (daily["Total_ADS"] + daily["Total_Promo"]).div(daily["Ordered_units"].replace(0, np.nan))
    return daily.sort_values(["Day", "SKU", "ASIN"]).reset_index(drop=True)


def build_forecast(workbook: Path) -> dict[str, pd.DataFrame]:
    status = pd.read_excel(workbook, sheet_name="2_SKU_Status")
    status = status[status["SKU"].notna()].copy()
    status["SKU"] = clean_text(status["SKU"])
    group_selection = pd.read_excel(workbook, sheet_name="7_Model_Selection")
    group_selection = group_selection[group_selection["Demand_class"].notna()].copy()
    summary = pd.read_excel(workbook, sheet_name="8_Forecast_Sep26_Feb27")
    summary = summary[summary["SKU"].notna()].copy()
    summary["SKU"] = clean_text(summary["SKU"])
    all_models = pd.read_excel(workbook, sheet_name="9_Forecast_All_Models")
    all_models = all_models[all_models["SKU"].notna() & all_models["Month"].notna()].copy()
    all_models["SKU"] = clean_text(all_models["SKU"])
    all_models["Month"] = pd.to_datetime(all_models["Month"].astype(str), errors="coerce")
    backtest = pd.read_excel(workbook, sheet_name="12_Backtest_Detail")
    backtest = backtest[(backtest["SKU"].notna()) & (backtest["Model"].isin(CANDIDATE_MODELS))].copy()
    backtest["SKU"] = clean_text(backtest["SKU"])
    backtest["Forecast"] = pd.to_numeric(backtest["Forecast"], errors="coerce")
    backtest["Ordered_units"] = pd.to_numeric(backtest["Ordered_units"], errors="coerce")
    backtest = backtest[(backtest["OOS_suspect"].fillna(0) == 0) & backtest["Forecast"].notna() & backtest["Ordered_units"].notna()]
    backtest["abs_error"] = (backtest["Forecast"] - backtest["Ordered_units"]).abs()
    backtest["error"] = backtest["Forecast"] - backtest["Ordered_units"]
    backtest["ape"] = np.where(backtest["Ordered_units"] > 0, backtest["abs_error"] / backtest["Ordered_units"], np.nan)
    metrics = backtest.groupby(["SKU", "Model"], as_index=False).agg(
        N=("Ordered_units", "size"), Actual=("Ordered_units", "sum"), Forecast=("Forecast", "sum"),
        Abs_error=("abs_error", "sum"), Error=("error", "sum"), MAPE=("ape", "mean"),
    )
    metrics["WAPE"] = metrics["Abs_error"].div(metrics["Actual"].replace(0, np.nan))
    metrics["Bias"] = metrics["Error"].div(metrics["Actual"].replace(0, np.nan))

    group_map = group_selection.set_index(["Demand_class", "Volume_tier"])["Recommended_model"].to_dict()
    workbook_rec = all_models.groupby("SKU")["Recommended_model"].first().to_dict()
    status_lookup = status.set_index("SKU")
    choices = []
    for sku in sorted(set(summary["SKU"]) | set(all_models["SKU"])):
        candidates = metrics[metrics["SKU"].eq(sku)].sort_values("WAPE")
        eligible = candidates[(candidates["N"] >= 18) & candidates["WAPE"].notna()]
        bias_ok = eligible[eligible["Bias"].abs() <= 0.20]
        if not bias_ok.empty:
            row = bias_ok.iloc[0]
            model, source, gate = row["Model"], "SKU backtest", "PASS"
        elif not eligible.empty:
            row = eligible.iloc[0]
            model, source, gate = row["Model"], "SKU backtest", "REVIEW — bias >20%"
        else:
            demand = status_lookup.at[sku, "Demand_class"] if sku in status_lookup.index else np.nan
            tier = summary.loc[summary["SKU"].eq(sku), "Volume_tier"].iloc[0] if sku in set(summary["SKU"]) else np.nan
            model = group_map.get((demand, tier), workbook_rec.get(sku, "P50"))
            if model not in CANDIDATE_MODELS:
                model = workbook_rec.get(sku, "P50")
            row = candidates[candidates["Model"].eq(model)].head(1)
            row = row.iloc[0] if not row.empty else None
            source, gate = "Group/workbook fallback", "FALLBACK"
        choices.append({
            "SKU": sku, "Selected_model": model, "Selection_source": source, "Bias_gate": gate,
            "N": np.nan if row is None else row["N"], "MAPE": np.nan if row is None else row["MAPE"],
            "WAPE": np.nan if row is None else row["WAPE"], "Bias": np.nan if row is None else row["Bias"],
        })
    selection = pd.DataFrame(choices)

    future = all_models.merge(selection, on="SKU", how="left")
    future["Forecast_unconstrained"] = [
        max(0.0, float(row.get(row["Selected_model"], row.get("Final_forecast", 0)) or 0))
        for _, row in future.iterrows()
    ]

    month_cols = [column for column in summary.columns if isinstance(column, str) and column[:4].isdigit()]
    fallback = summary.melt(
        id_vars=[c for c in ["SKU", "PIC", "Product", "Main_PL", "Status", "Demand_class", "Volume_tier", "Model_used", "Basis"] if c in summary],
        value_vars=month_cols, var_name="Month", value_name="Forecast_unconstrained",
    )
    fallback["Month"] = pd.to_datetime(fallback["Month"], errors="coerce")
    fallback = fallback[~fallback["SKU"].isin(set(future["SKU"]))]
    fallback = fallback.rename(columns={"Model_used": "Selected_model"})
    fallback["Selection_source"] = "Workbook status rule"
    fallback["Bias_gate"] = "N/A"
    fallback["Forecast_unconstrained"] = pd.to_numeric(fallback["Forecast_unconstrained"], errors="coerce").fillna(0).clip(lower=0)

    keep = ["SKU", "Month", "Forecast_unconstrained", "Selected_model", "Selection_source", "Bias_gate", "N", "MAPE", "WAPE", "Bias", "P50", "P80", "P90"]
    for column in keep:
        if column not in fallback:
            fallback[column] = np.nan
        if column not in future:
            future[column] = np.nan
    final = pd.concat([future[keep], fallback[keep]], ignore_index=True)
    final = final.merge(status, on="SKU", how="left")
    return {
        "forecast_sku_status": status,
        "forecast_group_selection": group_selection,
        "forecast_backtest_sku_model": metrics,
        "forecast_auto_selection": selection,
        "forecast_all_models": all_models,
        "forecast_future_auto": final,
    }


def write_bundle(output: Path, tables: dict[str, pd.DataFrame], quality: dict) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, frame in tables.items():
            buffer = io.BytesIO()
            frame.to_parquet(buffer, index=False)
            archive.writestr(f"{name}.parquet", buffer.getvalue())
        archive.writestr("quality_summary.json", json.dumps(quality, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("app_data_v2.zip"))
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("app_data_v2.zip"))
    args = parser.parse_args()

    tables, quality = read_bundle(args.base)
    daily = build_daily(args.history_dir, tables["sales_current"], tables["target_sku"])
    tables["sales_team_daily"] = daily
    tables.update(build_forecast(args.forecast))
    quality.update({
        "daily_min_date": str(daily["Day"].min().date()),
        "daily_max_date": str(daily["Day"].max().date()),
        "daily_rows": int(len(daily)),
        "daily_sku": int(daily["SKU"].nunique()),
        "daily_asin": int(daily["ASIN"].nunique()),
        "forecast_method": "Per-SKU WAPE minimum with |Bias| <= 20%; group/workbook fallback when backtest is insufficient",
        "incoming_note": "Aggregate incoming has no monthly ETA; dashboard timing is an explicit scenario assumption.",
    })
    write_bundle(args.output, tables, quality)
    print(f"PASS: {args.output} | daily={len(daily):,} | {daily['Day'].min():%Y-%m-%d} → {daily['Day'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
