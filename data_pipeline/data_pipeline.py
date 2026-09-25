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
