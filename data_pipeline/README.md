# Module 1 - Data Pipeline

This module collects book data from Books to Scrape, cleans the scraped values, converts the prices from GBP to INR, stores the data in SQLite, and runs SQL queries on the stored data.

## Data Source

The data is collected from:

https://books.toscrape.com/

I used the first 5 catalogue pages. The script currently collects 100 books across 29 categories.

## What the script does

The pipeline performs these steps:

1. Scrapes book information using `requests` and `BeautifulSoup`.
2. Collects:
   - title
   - price
   - star rating
   - availability
   - category
3. Cleans the scraped values.
4. Converts the text star rating into an integer from 1 to 5.
5. Converts availability into a boolean `in_stock` column.
6. Converts GBP price into INR.
7. Stores the cleaned data in a normalized SQLite database.
8. Runs SQL queries using different SQL clauses.
9. Reads SQL query results into pandas.
10. Recreates the JOIN result using `pd.merge()` and compares both outputs.

## Price Conversion

For this project I used the required fixed conversion rate:

`1 GBP = 105.50 INR`

The INR price is calculated as:

```python
price_inr = price_gbp * 105.50
