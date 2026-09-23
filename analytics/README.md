# Analytics

Run `01_eda.ipynb` before `02_modeling.ipynb`. Both currently run as starters.
Set `DOWNLOAD_TITANIC = True` in EDA only when ready for the first download.
This saves the raw `titanic.csv` here; subsequent runs reuse the same file.
The modeling notebook never calls sns.load_dataset.

Use an EDA copy for full-data descriptive cleaning. For modeling, use the same
raw CSV lineage, apply only deterministic row/column decisions, split first,
and fit imputers/encoders/scalers only on training data. Do not reuse values
imputed from the full EDA dataset. Explain this separation in your write-up.

`data/` is for derived tables, `models/` for the complete fitted pipeline and
`outputs/` for real metrics/interpretations. All remaining tasks are listed in
the root checklist and notebook sections. No models or results are claimed yet.
