# Market Basket Analysis of AO3 Story Tropes

An end-to-end data mining pipeline that extracts fanfiction metadata from Archive of Our Own (AO3), transforms unstructured tag arrays into binary transaction matrices, and applies the **Apriori Algorithm** to discover statistical co-occurrences of narrative tropes.

## 📌 Project Overview
- **Data Source:** Scraped tag arrays from AO3 top-rated works via `BeautifulSoup`.
- **Target Domain:** Mining creative writing metadata to analyze author tagging behavior.
- **Core Methodology:** Market Basket Analysis / Association Rule Mining.
- **Primary Tech Stack:** Python, Pandas, BeautifulSoup, Mlxtend.

---

## ⚙️ Technical Pipeline

```text
[ Web Scraper ] ──> [ Tag Filtering ] ──> [ TransactionEncoder ] ──> [ Apriori Mining ]
 (AO3 Blurbs)       (No Ships/Names)      (Binary Matrix 1s/0s)       (Rules & Metrics)
