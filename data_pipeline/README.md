# Data pipeline

Open `01_data_pipeline.ipynb`. The starter defines the fixed conversion rate,
cleaning function and normalized SQLite schema; scraping/loading/query results
remain TODO. Store data in `data/` and actual query outputs in `outputs/`.

The required conversion is **1 GBP = 105.50 INR**. The starter cleaning strategy
drops rows whose required fields cannot be parsed; explain any losses after
running it on your real scraped data. Do not substitute invented rows.
