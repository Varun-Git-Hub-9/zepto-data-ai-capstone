# Zepto Data & AI Capstone

This repository contains my capstone project for the Certificate Program in Artificial Intelligence and Machine Learning.

The project is divided into three modules:

- `data_pipeline` - web scraping, data cleaning, currency conversion, SQLite storage, SQL queries, and pandas validation
- `analytics` - Titanic EDA, preprocessing, classification, imbalance handling, hyperparameter tuning, regression, and model persistence
- `support_assistant` - RAG-based support assistant using embeddings, ChromaDB, LangGraph, FastAPI, Pydantic, and Docker

## Repository Structure

```text
zepto-data-ai-capstone/
├── README.md
├── data_pipeline/
├── analytics/
└── support_assistant/




## Module 1 - Data Pipeline

The data pipeline scrapes book data from Books to Scrape.

It collects 100 books from the first five catalogue pages and extracts title, price, rating, availability, and category.

The data is cleaned, converted from GBP to INR using the fixed project rate of:

```text
1 GBP = 105.50 INR
