# Yes4All Commerce Intelligence V3

## Repository files

- `streamlit_app.py`: dashboard application.
- `app_data_v3.zip`: prepared data mart with daily sales and forecast audit tables.
- `requirements.txt`: Streamlit Cloud dependencies.
- `prepare_app_data.py`: preparation script for Google Colab.
- `prepare_daily_forecast.py`: adds the 2023–2026 daily mart and the automatic forecast/model-selection layer.

Keep all three files in the repository root. In Streamlit Cloud, set the main file path to `streamlit_app.py`.

## Refresh

In Google Colab, upload the six source workbooks to `/content/source` and the current `app_data_v2.zip` to `/content`. The required sources are:

1. `Yes4All_Data sales 2023 up to Aug312026.xlsx`
2. `SSO Data Extraction Hourly -- hourly (21).xlsx`
3. `SSO_US_Oct_Target_20260922.xlsx`
4. `USA Inventory Y4A-AMZ.xlsx`
5. `Inv đầu kỳ Weekly.xlsx`
6. `Y4A_Advertising_Audit_Report_WORKING_CURRENT.xlsx`

Then run:

```python
%env YES4ALL_PROJECT_ROOT=/content
%env YES4ALL_SOURCE_DIR=/content/source
%env YES4ALL_WORK_DIR=/content/work
%env YES4ALL_OUTPUT_DIR=/content/output
%env YES4ALL_EXISTING_BUNDLE=/content/app_data_v2.zip
!python /content/prepare_app_data.py
```

Download `/content/output/app_data_v2.zip`, replace the existing zip in GitHub, and commit. Streamlit Cloud will redeploy the app automatically.

## Daily forecast refresh

After the base mart has been prepared, upload the three daily sales workbooks and
`Yes4All_US_Amazon_Forecast_Model.xlsx`, then run:

```bash
python prepare_daily_forecast.py \
  --base app_data_v2.zip \
  --history-dir /content/source \
  --forecast /content/source/Yes4All_US_Amazon_Forecast_Model.xlsx \
  --output app_data_v3.zip
```

Commit `app_data_v3.zip` with `streamlit_app.py`. The app checks V3 first and falls back to V2 only for backward compatibility.

Forecast selection logic:

- demand forecast is independent of inventory;
- per SKU, choose the lowest-WAPE candidate with at least 18 valid backtest points and absolute Bias no higher than 20%;
- OOS-suspect backtest months are excluded;
- insufficient-history SKUs use the demand-class/volume-tier or workbook lifecycle rule;
- MAPE excludes zero-actual observations and is displayed as a diagnostic, while WAPE is the primary accuracy measure;
- aggregate incoming has no monthly ETA, so the dashboard exposes receipt month 1–3 as an explicit scenario assumption and defaults to month 3.

Commercial sales logic:

- January 2023–August 2026 comes from the finalized monthly sales workbook.
- September 2026 remains sourced from the existing hourly workbook and is rolled up to month.
- PIC is mapped strictly by SKU from the Target workbook into `Team Cẩm Tú`, `Team Đồng Dinh`, `Spreetail`, or `N/A`.

## Data classifications

- Sales, inventory, target, supply forecast and campaign audit: supplied source data.
- ASP = Ordered GMV / Ordered Units.
- Ads / GMV, Promo / GMV and MKT / GMV use Ordered GMV as denominator; MKT = Ads + Promo.
- MKT CPU = (Ads + Promo) / Ordered Units.
- Ads attributed units = SB + SD + SP + DSP + Affiliate attributed units.
- Promo Units are unavailable in the supplied sources and are not simulated.
- The Sales Investment Planner is a historical-ratio planning benchmark, not a causal sales guarantee.
- Inventory-constrained target: calculated as `min(War Map units, recalculated opening units + usable incoming)` and rolled forward by month.
- Listing health and ranking/keyword history: deterministic simulated demo data, visibly labeled in the app.
- Sell-through proxy: September ordered units divided by September opening inventory because receipts-to-date were not supplied.
