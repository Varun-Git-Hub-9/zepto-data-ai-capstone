import re
import requests
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def scrape_books():
    books = []

    # Scrape the first 5 catalogue pages
    for page in range(1, 6):
        url = f"{BASE_URL}catalogue/page-{page}.html"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for item in soup.select("article.product_pod"):

            # Book title
            title = item.h3.a["title"]

            # Price
            price_text = item.select_one(".price_color").get_text(strip=True)

            # Remove characters such as £ or Â and keep only numbers/decimal point
            cleaned_price = re.sub(r"[^\d.]", "", price_text)

            try:
                price_gbp = float(cleaned_price)
            except ValueError:
                price_gbp = None

            # Rating
            rating_text = item.p["class"][1]
            rating = rating_map.get(rating_text)

            # Availability
            availability_text = item.select_one(".availability").get_text(
                " ", strip=True
            )

            in_stock = "In stock" in availability_text

            # Open the individual book page to get its category
            relative_link = item.h3.a["href"]
            detail_url = BASE_URL + "catalogue/" + relative_link

            detail_response = requests.get(detail_url, timeout=10)
            detail_response.raise_for_status()

            detail_soup = BeautifulSoup(
                detail_response.text,
                "html.parser"
            )

            breadcrumb = detail_soup.select("ul.breadcrumb li a")
            category = breadcrumb[-1].get_text(strip=True)

            books.append(
                {
                    "title": title,
                    "price_gbp": price_gbp,
                    "rating": rating,
                    "availability": availability_text,
                    "in_stock": in_stock,
                    "category": category
                }
            )

    return pd.DataFrame(books)


# Scrape the books
df = scrape_books()


# -------------------------
# Data cleaning
# -------------------------

# Convert price to numeric
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

# If a price failed to parse, fill it using the median price
if df["price_gbp"].isna().any():
    median_price = df["price_gbp"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)


# Convert rating to numeric
df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

# If a rating failed to parse, use the median rating
if df["rating"].isna().any():
    median_rating = df["rating"].median()
    df["rating"] = df["rating"].fillna(median_rating)

df["rating"] = df["rating"].round().astype(int)


# Make sure in_stock is boolean
df["in_stock"] = df["in_stock"].astype(bool)


# -------------------------
# Currency conversion
# -------------------------

# Project-defined fixed conversion rate:
# 1 GBP = 105.50 INR
df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
).round(2)


# -------------------------
# Basic validation
# -------------------------

print("\nScraping completed")
print("------------------")

print("Total books:", len(df))
print("Categories:", df["category"].nunique())

print("\nColumn types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------
# SQLite database
# ---------------------------------------------------------

conn = sqlite3.connect("books.db")
cursor = conn.cursor()

# Recreate the tables so the script can be run multiple times
cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    availability TEXT,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

# Insert categories
categories = sorted(df["category"].unique())

cursor.executemany(
    "INSERT INTO categories (category_name) VALUES (?)",
    [(category,) for category in categories]
)

conn.commit()

# Get category IDs
category_df = pd.read_sql(
    "SELECT category_id, category_name FROM categories",
    conn
)

category_map = dict(
    zip(category_df["category_name"], category_df["category_id"])
)

# Insert books
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            availability,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["price_gbp"],
        row["price_inr"],
        int(row["rating"]),
        row["availability"],
        int(row["in_stock"]),
        category_map[row["category"]]
    ))

conn.commit()

print("\nDatabase created successfully.")


# ---------------------------------------------------------
# SQL queries
# ---------------------------------------------------------

query1 = """
SELECT title, price_gbp, price_inr
FROM books
WHERE in_stock = 1
ORDER BY price_gbp DESC
LIMIT 10;
"""

query2 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
"""

query3 = """
SELECT title, rating, price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp ASC
LIMIT 15;
"""

query4 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp ASC;
"""

query5 = """
SELECT
    b.title,
    c.category_name,
    b.rating,
    b.price_gbp,
    b.price_inr
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
WHERE b.rating = 5
ORDER BY b.price_gbp DESC
LIMIT 10;
"""

queries = [query1, query2, query3, query4, query5]

for number, query in enumerate(queries, start=1):
    result = pd.read_sql(query, conn)

    print(f"\nSQL Query {number}")
    print(query)
    print(result)


# ---------------------------------------------------------
# Read SQL results into pandas
# ---------------------------------------------------------

sql_result_1 = pd.read_sql(query1, conn)
sql_join_result = pd.read_sql(query5, conn)

print("\nQuery 1 loaded into pandas:")
print(sql_result_1)

print("\nJOIN query loaded into pandas:")
print(sql_join_result)


# ---------------------------------------------------------
# Reproduce the JOIN using pd.merge
# ---------------------------------------------------------

books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

pandas_join_result = (
    merged_df[merged_df["rating"] == 5]
    .sort_values("price_gbp", ascending=False)
    .head(10)[
        [
            "title",
            "category_name",
            "rating",
            "price_gbp",
            "price_inr"
        ]
    ]
    .reset_index(drop=True)
)

sql_join_result = sql_join_result.reset_index(drop=True)

print("\nSQL JOIN result:")
print(sql_join_result)

print("\npd.merge result:")
print(pandas_join_result)

print(
    "\nDo SQL JOIN and pd.merge match?",
    sql_join_result.equals(pandas_join_result)
)

conn.close()
