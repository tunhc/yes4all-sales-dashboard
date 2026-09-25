from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


R_LB = {"DI": 0.00617589, "SPT": 0.00617589, "DS": 0.11579795}
R_CBM = {"DI": 6.207626, "SPT": 6.207626, "DS": 116.392987}
CU_LB, CU_CBM = 0.00634563, 6.378239
ARD = {"DI": 135, "DS": 168, "SPT": 35}
FRATE = 0.115 / 365


def read_bundle(path: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".parquet"):
                tables[name.removesuffix(".parquet")] = pd.read_parquet(io.BytesIO(archive.read(name)))
        quality = json.loads(archive.read("quality_summary.json").decode("utf-8"))
    return tables, quality


def read_cost_stack(path: Path) -> tuple[pd.DataFrame, dict]:
    html = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="DATA" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise ValueError("CM3 DATA block not found in HTML")
    payload = json.loads(match.group(1))
    products = pd.DataFrame(payload["products"])
    products = products.rename(columns={
        "s": "SKU", "a": "cost_ASIN", "n": "cost_product_name", "pl": "cost_product_line",
        "gmv": "canonical_gmv", "rdi": "canonical_rdi", "rds": "canonical_rds",
        "sptp": "canonical_sptp", "cur": "canonical_lane", "ok": "cost_stack_ok",
    })
    products["SKU"] = products["SKU"].fillna("").astype(str).str.strip()
    return products, payload


def resolve_lane(row: pd.Series) -> str:
    channel = str(row.get("channel", "")).upper().strip()
    if channel.startswith("SPT"):
        return "SPT"
    if channel.startswith("DS"):
        return "DS"
    if channel.startswith("DI"):
        return "DI"
    current = str(row.get("canonical_lane", "")).upper()
    if "SPT" in current:
        return "SPT"
    if "DS" in current:
        return "DS"
    return "DI"


def cm3_row(row: pd.Series) -> tuple[float, float, str, str]:
    units = float(row.get("Ordered_units", 0) or 0)
    gmv_total = float(row.get("Ordered_GMV", 0) or 0)
    if units <= 0 or not bool(row.get("cost_stack_ok", 0)):
        return np.nan, np.nan, "UNCOSTED", "Missing cost stack or zero units"

    lane = resolve_lane(row)
    asp = gmv_total / units if gmv_total > 0 else float(row.get("canonical_gmv", 0) or 0)
    promo_rate = max(float(row.get("Total_Promo", 0) or 0) / gmv_total, 0) if gmv_total > 0 else 0
    ads_rate = max(float(row.get("Total_ADS", 0) or 0) / gmv_total, 0) if gmv_total > 0 else 0
    ordered_revenue = float(row.get("Ordered_revenue", 0) or 0)
    if ordered_revenue > 0:
        revenue = ordered_revenue / units
        revenue_source = "Actual Ordered_revenue"
    else:
        revenue = float(row.get({"DI": "canonical_rdi", "DS": "canonical_rds", "SPT": "canonical_sptp"}[lane], 0) or 0)
        revenue_source = "Canonical sell-in fallback"

    fob = float(row.get("fob", 0) or 0)
    duty = float(row.get("duty", 0) or 0)
    wt = float(row.get("wt", 0) or 0)
    cbm = float(row.get("cbm", 0) or 0)
    c = {k: 0.0 for k in ["fob", "tariff", "inb", "cirro", "vt", "perf", "ref", "ffee", "stor", "pp", "lm", "promo", "ads", "amex", "fin", "ret"]}
    c["fob"] = -fob
    c["tariff"] = -fob * duty
    c["inb"] = -max(wt * R_LB[lane], cbm * R_CBM[lane])

    if lane == "SPT":
        nmv = revenue
    else:
        nmv = asp * (1 - 0.75 * promo_rate)
        c["promo"] = -asp * promo_rate
        c["ads"] = -asp * ads_rate
        c["amex"] = asp * ads_rate * 0.015

    if lane == "DI":
        c["vt"] = -revenue * 0.01
        c["perf"] = -revenue * 0.005
    elif lane == "DS":
        c["cirro"] = -max(wt * CU_LB, cbm * CU_CBM)
        c["vt"] = -revenue * 0.1375
        c["perf"] = -revenue * 0.005
        c["stor"] = -cbm * 20.10
        c["pp"] = -(1.50 + max(wt - 20, 0) * 0.10)

    ar = ARD[lane]
    io = 0 if lane in {"DI", "SPT"} else 60
    c["fin"] = (
        c["fob"] * (ar - 120)
        + c["tariff"] * (ar - 68)
        + (c["inb"] + c["cirro"]) * (ar - io)
        + (c["pp"] + c["lm"] + c["ads"]) * (ar - 103)
    ) * FRATE
    pre_tu = revenue + sum(c.values())
    true_up = 0.0
    if lane in {"DI", "DS"} and nmv > 0:
        csa = 0.505 if lane == "DI" else 0.40
        vm = asp * 0.25 * promo_rate
        net_ppm = (nmv - revenue + abs(c["vt"]) + abs(c["perf"]) + vm) / nmv
        true_up = max(0.0, csa - net_ppm) * nmv
    cm3_unit = pre_tu - true_up
    return cm3_unit * units, cm3_unit, lane, revenue_source


def add_cm3(frame: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["SKU"] = frame["SKU"].fillna("").astype(str).str.strip()
    keep = [
        "SKU", "cost_ASIN", "cost_product_name", "cost_product_line", "cost_stack_ok",
        "canonical_lane", "canonical_gmv", "canonical_rdi", "canonical_rds", "canonical_sptp",
        "fob", "duty", "wt", "cbm", "ffee", "fsto", "ftier",
    ]
    frame = frame.merge(products[keep].drop_duplicates("SKU"), on="SKU", how="left")
    results = frame.apply(cm3_row, axis=1, result_type="expand")
    results.columns = ["CM3", "CM3_per_unit", "CM3_lane", "CM3_revenue_source"]
    frame[results.columns] = results
    frame["CM3_pct_GMV"] = frame["CM3"].div(pd.to_numeric(frame["Ordered_GMV"], errors="coerce").replace(0, np.nan))
    frame["CM3_costed"] = frame["CM3"].notna()
    # Keep the full unit-cost stack once in cm3_cost_stack.parquet. Repeating the
    # same helper fields on every monthly/daily sales row adds several MB without
    # helping the dashboard, so the fact tables retain only calculated CM3 fields.
    frame = frame.drop(columns=[column for column in keep if column != "SKU"], errors="ignore")
    return frame


def write_bundle(path: Path, tables: dict[str, pd.DataFrame], quality: dict) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, frame in tables.items():
            buffer = io.BytesIO()
            frame.to_parquet(buffer, index=False)
            archive.writestr(f"{name}.parquet", buffer.getvalue())
        archive.writestr("quality_summary.json", json.dumps(quality, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cm3-html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables, quality = read_bundle(args.base)
    products, payload = read_cost_stack(args.cm3_html)
    # V4 uses sales_team_daily as the canonical daily fact. These two legacy
    # daily rollups duplicate the same history and are no longer referenced by
    # streamlit_app.py; removing them keeps the GitHub-upload bundle under 25 MB.
    tables.pop("sales_history_sku", None)
    tables.pop("sales_history_asin", None)
    tables["cm3_cost_stack"] = products
    for table in ["sales_commercial_monthly", "sales_team_daily"]:
        tables[table] = add_cm3(tables[table], products)

    sales = tables["sales_commercial_monthly"]
    search_cols = ["SKU", "ASIN", "product_name", "product_line", "team", "channel"]
    tables["search_index"] = sales[search_cols].fillna("").astype(str).drop_duplicates().reset_index(drop=True)
    gmv = float(sales["Ordered_GMV"].sum())
    covered_gmv = float(sales.loc[sales["CM3_costed"], "Ordered_GMV"].sum())
    quality.update({
        "cm3_source": f"{payload.get('v')} · {payload.get('canon')} · built {payload.get('built')}",
        "cm3_formula": "CM3 $ = CM3 per unit by lane × actual units; actual Ads% and Promo% by period; True-Up applied for DI/DS",
        "cm3_cost_stack_sku": int(products["cost_stack_ok"].fillna(0).sum()),
        "cm3_sales_sku_covered": int(sales.loc[sales["CM3_costed"], "SKU"].nunique()),
        "cm3_gmv_coverage": covered_gmv / gmv if gmv else 0,
        "cm3_limitations": "Rows without cost stack are excluded. Ordered_revenue is used when available; otherwise canonical sell-in is used. SPT uses the canonical no-MKT rule from the supplied HTML.",
    })
    quality.setdefault("metric_definitions", {}).update({
        "CM3": "Lane-specific CM3 per unit from the supplied V9.8 HTML cost stack multiplied by actual ordered units; actual Ads% and Promo% are applied by period and DI/DS True-Up is included",
        "CM3_pct_GMV": "CM3 divided by Ordered GMV for rows with a valid CM3 cost stack; uncovered rows are excluded rather than treated as zero",
    })
    write_bundle(args.output, tables, quality)
    print(f"PASS: {args.output} | CM3 GMV coverage={quality['cm3_gmv_coverage']:.1%} | costed SKUs={quality['cm3_sales_sku_covered']:,}")


if __name__ == "__main__":
    main()
