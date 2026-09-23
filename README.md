# Yes4All Commerce Intelligence V2

## Repository files

- `streamlit_app.py`: dashboard application.
- `app_data_v2.zip`: prepared data mart.
- `requirements.txt`: Streamlit Cloud dependencies.
- `prepare_app_data.py`: preparation script for Google Colab.

Keep all three files in the repository root. In Streamlit Cloud, set the main file path to `streamlit_app.py`.

## Refresh

In Google Colab, upload the five source workbooks to `/content/source` and the current `app_data_v2.zip` to `/content`. Then run:

```python
%env YES4ALL_PROJECT_ROOT=/content
%env YES4ALL_SOURCE_DIR=/content/source
%env YES4ALL_WORK_DIR=/content/work
%env YES4ALL_OUTPUT_DIR=/content/output
%env YES4ALL_EXISTING_BUNDLE=/content/app_data_v2.zip
!python /content/prepare_app_data.py
```

Download `/content/output/app_data_v2.zip`, replace the existing zip in GitHub, and commit. Streamlit Cloud will redeploy the app automatically. The existing bundle is used only to preserve the legacy 2023–2026 history tables; current-month pages are rebuilt from the new source files.

## Data classifications

- Sales, inventory, target, supply forecast and campaign audit: supplied source data.
- Inventory-constrained target: calculated as `min(War Map units, recalculated opening units + usable incoming)` and rolled forward by month.
- Listing health and ranking/keyword history: deterministic simulated demo data, visibly labeled in the app.
- Sell-through proxy: September ordered units divided by September opening inventory because receipts-to-date were not supplied.
