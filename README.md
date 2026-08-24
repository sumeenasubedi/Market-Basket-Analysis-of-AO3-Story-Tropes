# Market Basket Analysis of AO3 Story Tropes

An end-to-end data mining pipeline that extracts fanfiction metadata from Archive of Our Own (AO3), transforms unstructured tag arrays into binary transaction matrices, and applies the **Apriori Algorithm** to discover statistical co-occurrences of narrative tropes.

## 📌 Project Overview
- **Data Source:** Scraped tag arrays from AO3 top-rated works via `BeautifulSoup`.
- **Target Domain:** Mining creative writing metadata to analyze author tagging behavior.
- **Core Methodology:** Market Basket Analysis / Association Rule Mining.
- **Primary Tech Stack:** Python, Pandas, BeautifulSoup, Mlxtend.

---

## ⚙️ Technical Pipeline
1. **Scraping & Fault Tolerance:** Extracts story tags while incorporating a 3-second network timeout with pre-buffered dataset fallbacks.
2. **Preprocessing:** Filters out specific character entities and relationship pairings (`/`) to isolate universal narrative tropes (e.g., *Slow Burn*, *Hurt/Comfort*).
3. **Encoding:** Converts text lists into a high-dimensional binary boolean matrix ($1$ = tag present, $0$ = tag absent).
4. **Association Mining:** Runs the **Apriori Algorithm** to uncover high-confidence trope pairs using Support, Confidence, and Lift metrics.

---

## 📊 Mining Metrics & Thresholds

- **Minimum Support ($\text{min\_support}$):** `0.10` (10% minimum transaction frequency)
- **Minimum Lift ($\text{min\_threshold}$):** `1.0` (Positive correlation only)
- **Max Lift Achieved:** `5.0` (Proves intentional author pairing 5x over random distribution)

---

## 🚀 Key Applications

- **Content Recommendation Engines:** Suggesting new stories to readers based on co-occurring narrative tropes.
- **Metadata Auto-Completion:** Assisting creators with smart tag auto-suggestions during story publishing.
- **Digital Humanities:** Quantifying literary tropes and community writing patterns through data analytics.

---

## 🛠️ How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ao3-trope-mining.git](https://github.com/YOUR_USERNAME/ao3-trope-mining.git)
   cd ao3-trope-mining
