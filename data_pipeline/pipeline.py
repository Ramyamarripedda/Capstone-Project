"""Scrape practice-book categories, clean them, and query a normalized SQLite DB."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50  # Fixed value defined by the assignment.
CATEGORY_NAMES = ("Mystery", "Historical Fiction", "Classics")
RATING = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL CHECK(price_gbp >= 0),
    price_inr REAL NOT NULL CHECK(price_inr >= 0),
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""

QUERIES = {
    "01_in_stock_where_limit": """
        SELECT title, price_gbp, rating FROM books
        WHERE in_stock = 1 ORDER BY price_gbp DESC LIMIT 10
    """,
    "02_distinct_categories": """
        SELECT DISTINCT category_name FROM categories ORDER BY category_name
    """,
    "03_rating_between": """
        SELECT title, rating, price_inr FROM books
        WHERE rating BETWEEN 4 AND 5 ORDER BY rating DESC, title LIMIT 10
    """,
    "04_category_in": """
        SELECT c.category_name, COUNT(*) AS book_count
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Mystery', 'Classics')
        GROUP BY c.category_name ORDER BY c.category_name
    """,
    "05_join_top_rated": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10
    """,
    "06_category_average": """
        SELECT c.category_name, COUNT(*) AS book_count,
               ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        GROUP BY c.category_name ORDER BY average_price_gbp DESC
    """,
}


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    # Let BeautifulSoup detect the page's charset; requests may guess ISO-8859-1
    # and turn the pound sign into the Unicode replacement character.
    return BeautifulSoup(response.content, "html.parser")


def scrape_books() -> pd.DataFrame:
    """Read all listing pages of three named categories."""
    rows = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Student capstone practice scraper/1.0"})
        home = fetch_soup(session, BASE_URL)
        links = {}
        for anchor in home.select("ul.nav-list ul a"):
            name = anchor.get_text(" ", strip=True)
            links[name] = urljoin(BASE_URL, anchor["href"])
        for category in CATEGORY_NAMES:
            if category not in links:
                raise ValueError(f"Category not found: {category}")
            page_url = links[category]
            while page_url:
                soup = fetch_soup(session, page_url)
                cards = soup.select("article.product_pod")
                if not cards:
                    raise ValueError(f"No book cards at {page_url}")
                for card in cards:
                    rows.append({
                        "title": (card.select_one("h3 a").get("title", "").strip()
                                  if card.select_one("h3 a") else ""),
                        "price": (card.select_one("p.price_color").get_text(strip=True)
                                  if card.select_one("p.price_color") else ""),
                        "star_rating": next(
                            (name for name in RATING if card.select_one("p.star-rating")
                             and name in card.select_one("p.star-rating").get("class", [])),
                            ""),
                        "availability": (card.select_one("p.availability").get_text(" ", strip=True)
                                         if card.select_one("p.availability") else ""),
                        "category": category,
                    })
                next_link = soup.select_one("li.next a")
                page_url = urljoin(page_url, next_link["href"]) if next_link else None
    return pd.DataFrame(rows)


def clean_books(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    books = raw.copy()
    books["price_gbp"] = pd.to_numeric(
        books["price"].astype(str).str.replace("£", "", regex=False).str.strip(),
        errors="coerce",
    )
    books["rating"] = books["star_rating"].map(RATING)
    availability = books["availability"].astype(str).str.lower().str.strip()
    books["in_stock"] = availability.map(
        lambda value: True if value.startswith("in stock") else
        False if value.startswith("out of stock") else None
    )
    for column in ("title", "category"):
        books[column] = books[column].astype("string").str.strip().replace("", pd.NA)
    books = books.dropna(subset=["title", "category", "price_gbp", "rating", "in_stock"])
    books = books.loc[books["price_gbp"] >= 0].copy()
    dropped = len(raw) - len(books)
    books["price_gbp"] = books["price_gbp"].astype(float)
    books["rating"] = books["rating"].astype(int)
    books["in_stock"] = books["in_stock"].astype(bool)
    books["price_inr"] = (books["price_gbp"] * GBP_TO_INR).round(2)
    return books.reset_index(drop=True), dropped


def load_database(books: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # Rebuild this generated artifact from scratch on each run.
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(SCHEMA)
        categories = sorted(books["category"].unique())
        connection.executemany(
            "INSERT INTO categories(category_name) VALUES (?)",
            [(name,) for name in categories],
        )
        category_id = dict(connection.execute("SELECT category_name, category_id FROM categories"))
        records = [
            (row.title, row.price_gbp, row.price_inr, row.rating,
             int(row.in_stock), category_id[row.category])
            for row in books.itertuples(index=False)
        ]
        connection.executemany(
            """INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            records,
        )
        connection.commit()
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def query_and_compare(db_path: Path) -> tuple[dict[str, pd.DataFrame], bool]:
    results = {}
    with closing(sqlite3.connect(db_path)) as connection:
        for name, query in QUERIES.items():
            results[name] = pd.read_sql(query, connection)
        books = pd.read_sql("SELECT * FROM books", connection)
        categories = pd.read_sql("SELECT * FROM categories", connection)
    merged = books.merge(categories, on="category_id")
    pandas_join = (merged[["category_name", "title", "rating", "price_gbp"]]
                   .sort_values(["rating", "category_name", "title"],
                                ascending=[False, True, True])
                   .head(10).reset_index(drop=True))
    sql_join = results["05_join_top_rated"].reset_index(drop=True)
    pd.testing.assert_frame_equal(sql_join, pandas_join, check_dtype=False)
    results["05_pandas_merge_same_result"] = pandas_join
    results["05_side_by_side"] = pd.concat(
        [sql_join.add_prefix("SQL "), pandas_join.add_prefix("pandas ")], axis=1)
    return results, True


def run() -> dict:
    raw = scrape_books()
    cleaned, dropped = clean_books(raw)
    if len(cleaned) < 60 or cleaned["category"].nunique() < 3:
        raise ValueError("Required 60 books from three categories not reached")
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    raw.to_csv(ROOT / "data" / "books_raw.csv", index=False)
    cleaned.to_csv(ROOT / "data" / "books_clean.csv", index=False)
    db_path = ROOT / "data" / "books.db"
    load_database(cleaned, db_path)
    results, same = query_and_compare(db_path)
    for name, table in results.items():
        table.to_csv(out / f"{name}.csv", index=False)
    with (out / "query_results.md").open("w", encoding="utf-8") as report:
        report.write("# Executed SQL queries\n\n")
        for name, query in QUERIES.items():
            report.write(f"## {name}\n\n```sql\n{query.strip()}\n```\n\n")
            report.write(results[name].to_markdown(index=False) + "\n\n")
        report.write("## pandas merge of query 05\n\n")
        report.write(results["05_pandas_merge_same_result"].to_markdown(index=False) + "\n\n")
        report.write("## SQL and pandas side by side\n\n")
        report.write(results["05_side_by_side"].to_markdown(index=False) + "\n\n")
        report.write(f"SQL JOIN and pandas merge identical: **{same}**.\n")
    summary = {"raw_rows": len(raw), "clean_rows": len(cleaned),
               "dropped_rows": dropped, "categories": cleaned["category"].value_counts().to_dict(),
               "join_matches": same, "fixed_gbp_to_inr": GBP_TO_INR}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run())
