# Book data pipeline

This part collects book details from `books.toscrape.com`, a website made for
scraping practice. Run `01_data_pipeline.ipynb`, or run `pipeline.py` using the
project Python environment. A new scrape needs internet.

The last run collected 77 books: 32 Mystery, 26 Historical Fiction and
19 Classics. Every listing page in these three categories is read. The script
collects title, price, text rating, availability and category.

Price becomes a decimal number, ratings One to Five become integers 1 to 5,
and stock status becomes True or False. A row with an invalid required value
is dropped and counted, because guessing its price or category could mislead
the comparison. The recorded run dropped zero rows. Missing HTML fields are
sent through the same cleaning rule instead of causing the scraper to stop.

The required price conversion is **1 GBP = 105.50 INR**. It is a fixed
assignment value, not today's exchange rate. INR prices are rounded to two
decimal places.

SQLite stores categories once in a `categories` table. The `books` table points
to each category using its ID. This primary-key/foreign-key link avoids repeating
category names in every database row.

`data/books.db` is rebuilt when the script runs. Raw and cleaned CSV files are
also saved. `outputs/query_results.md` contains six SQL queries with their
answers. It covers WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN and JOIN.
The JOIN is also done using pandas merge; both results are shown side by side
and checked for equality.
