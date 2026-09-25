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

    for page in range(1, 6):
        url = f"{BASE_URL}catalogue/page-{page}.html"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for item in soup.select("article.product_pod"):
            title = item.h3.a["title"]

            price_text = item.select_one(".price_color").get_text(strip=True)
            price_gbp = float(price_text.replace("£", ""))

            rating_text = item.p["class"][1]
            rating = rating_map.get(rating_text)

            availability_text = item.select_one(".availability").get_text(
                " ", strip=True
            )
            in_stock = "In stock" in availability_text

            relative_link = item.h3.a["href"]
            detail_url = BASE_URL + "catalogue/" + relative_link

            detail_response = requests.get(detail_url, timeout=10)
            detail_response.raise_for_status()

            detail_soup = BeautifulSoup(detail_response.text, "html.parser")

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


df = scrape_books()

df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

print("Total books:", len(df))
print("Categories:", df["category"].nunique())
print(df.head())
