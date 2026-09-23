# Data pipeline

Run `python pipeline.py` here or Run All in `01_data_pipeline.ipynb`.
The public `books.toscrape.com` site is a scraping-practice source and needs
no key. The recorded run yielded **77 books**: Mystery 32, Historical Fiction
26 and Classics 19.

The cleaner captures title, listed GBP price, text rating, availability and
category; parses price as float and One-Five as integer 1-5; and maps stock
to Boolean. Rows with any invalid required field are dropped rather than
imputed, since an unknown price or category makes comparison unreliable.
**Zero** rows were dropped in this run. The conversion is the fixed assignment
constant **1 GBP = 105.50 INR**, not a live exchange rate.

`data/books.db` is rebuilt on each run. `categories` has a primary key and
unique name; `books` has a category foreign key. `outputs/query_results.md`
contains six executed SQL queries with actual outputs. The JOIN result is
also read with `pd.read_sql`, reproduced with `pd.merge`, and checked for
equality. Raw and cleaned CSVs, query CSVs and a summary JSON are included.
