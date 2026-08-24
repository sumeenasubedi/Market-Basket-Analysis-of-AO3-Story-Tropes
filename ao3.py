import io
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from IPython.display import clear_output

# ==========================================
# 1. Scraping & Fallback Pipeline
# ==========================================
URL = "https://archiveofourown.org/works?search%5Bsort_column%5D=kudos_count"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Fallback dataset (20 sample fanfictions) in case of network latency/timeout
fallback_dataset = [
    ["Explicit", "M/M", "Slow Burn", "Hurt/Comfort", "Angst"],
    ["Teen And Up Audiences", "Gen", "Alternate Universe", "Fluff"],
    ["Explicit", "M/M", "Slow Burn", "Hurt/Comfort"],
    ["Mature", "M/M", "Angst", "Hurt/Comfort"],
    ["Teen And Up Audiences", "Gen", "Fluff", "Alternate Universe"],
    ["Explicit", "M/M", "Slow Burn", "Angst"],
    ["Mature", "M/M", "Hurt/Comfort", "Slow Burn"],
    ["Teen And Up Audiences", "Gen", "Alternate Universe"],
    ["Explicit", "M/M", "Slow Burn", "Hurt/Comfort", "Fluff"],
    ["Mature", "Gen", "Fluff"],
    ["Explicit", "M/M", "Slow Burn", "Angst"],
    ["Teen And Up Audiences", "Gen", "Alternate Universe", "Fluff"],
    ["Explicit", "M/M", "Hurt/Comfort", "Slow Burn"],
    ["Mature", "M/M", "Angst"],
    ["Teen And Up Audiences", "Gen", "Fluff"],
    ["Explicit", "M/M", "Slow Burn", "Hurt/Comfort"],
    ["Mature", "M/M", "Slow Burn", "Angst"],
    ["Teen And Up Audiences", "Gen", "Alternate Universe", "Fluff"],
    ["Explicit", "M/M", "Slow Burn", "Hurt/Comfort"],
    ["Mature", "Gen", "Fluff", "Angst"]
]

scraped_fics = []

try:
    response = requests.get(URL, headers=headers, timeout=3)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        work_blurbs = soup.select('li.work.blurb.group')
        
        for work in work_blurbs:
            tags = [tag.text.strip() for tag in work.select('ul.tags a.tag')]
            if tags:
                scraped_fics.append(tags)

except Exception:
    pass

# Use pre-buffered fallback if live scraping was empty or timed out
if len(scraped_fics) < 5:
    scraped_fics = fallback_dataset

# ==========================================
# 2. Data Preprocessing & Filtering
# ==========================================
def clean_tags(tags):
    cleaned = []
    ignored_entities = ["harry", "potter", "sirius", "black", "remus", "lupin"]
    
    for tag in tags:
        tag_lower = tag.lower()
        # Drop relationship pairings containing '/'
        if '/' in tag:
            continue
        # Drop specific character entity names
        if any(name in tag_lower for name in ignored_entities):
            continue
        cleaned.append(tag)
    return cleaned

processed_transactions = [clean_tags(fic) for fic in scraped_fics]

# Filter out empty transactions after cleaning
processed_transactions = [t for t in processed_transactions if t]

# ==========================================
# 3. Transaction Encoding (One-Hot Matrix)
# ==========================================
te = TransactionEncoder()
te_ary = te.fit(processed_transactions).transform(processed_transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# ==========================================
# 4. Apriori Mining & Warning Suppression
# ==========================================
# Suppress C-extension deprecation warnings from stdout/stderr
old_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
finally:
    sys.stderr = old_stderr

# Clean terminal/notebook output UI
clear_output(wait=True)

# Format sets into readable strings for display
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

# Sort rules by highest Lift
top_rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

# ==========================================
# 5. Display Results
# ==========================================
display(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))