# Crawler Documentation

This folder contains tools for collecting and crawling news article URLs from Fox News and NBC News.

## Files Overview

### 1. `crawler.py` - Main Article Crawler
Extracts titles from news URLs.

### 2. `collect_extra_urls.py` - **NEW!** Extra URL Collector
Collects additional article URLs from Fox News and NBC News websites for expanded training data.

### 3. `sample_urls.py` - URL Sampler
Creates random samples for testing.

### 4. `test_crawler.py` - Test Script
Validates crawler functionality.

---

## Features

### Main Crawler (`crawler.py`)
- Extracts titles from multiple HTML formats:
  - `<title>` tags
  - `<h1 class="headline">` tags (NBC News format)
  - Meta tags (og:title, twitter:title)
  - Article-specific title elements
- Labels each article as 'fox' or 'nbc'
- Handles various title formats and cleans up suffixes
- Respectful crawling with configurable delays
- Comprehensive error handling
- **NEW:** Command-line arguments support

### Extra URL Collector (`collect_extra_urls.py`)
- Scrapes section pages (politics, us-news, world, etc.)
- Validates article URLs (excludes video, live, archives, etc.)
- Removes duplicates automatically
- Balances Fox/NBC URL collection
- Target: ~500 URLs per source

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Crawl Original URLs (Basic Usage)

Run the crawler with default settings:

```bash
python crawler/crawler.py
```

This will:
1. Read URLs from `data/raw/urls_initial.csv`
2. Crawl each URL with a 1-second delay between requests
3. Save results to `data/processed/crawled_data.csv`

### 2. **NEW:** Collect Extra URLs for Expanded Training

To expand your training dataset, first collect additional URLs from news websites:

```bash
python crawler/collect_extra_urls.py
```

This will:
- Scrape Fox News and NBC News section pages
- Collect ~500 URLs from each source (~1000 total)
- Remove duplicates automatically
- Save to `data/raw/urls_collected_extra.csv`
- Show detailed progress and statistics

**Sections crawled:**
- **Fox News:** Politics, US, World, Opinion, Entertainment, Lifestyle, Media, Tech, Sports
- **NBC News:** Politics, US News, World, Business, Health, Tech, Latino, Asian America, Investigations

### 3. Crawl the Extra URLs

After collecting URLs, crawl them to extract titles:

```bash
python crawler/crawler.py \
    --input data/raw/urls_collected_extra.csv \
    --output data/processed/extra_crawled_data.csv
```

### 4. Merge Datasets (Optional)

To combine original and extra crawled data:

```python
import pandas as pd

# Load both datasets
original = pd.read_csv('data/processed/crawled_data.csv')
extra = pd.read_csv('data/processed/extra_crawled_data.csv')

# Merge
combined = pd.concat([original, extra], ignore_index=True)

# Remove duplicates
combined = combined.drop_duplicates(subset=['url'], keep='first')

# Save
combined.to_csv('data/processed/crawled_data_merged.csv', index=False)

print(f"Original: {len(original)}, Extra: {len(extra)}, Merged: {len(combined)}")
```

### 5. Advanced Options

The crawler now supports command-line arguments:

```bash
# Custom input/output paths
python crawler/crawler.py --input my_urls.csv --output my_results.csv

# Adjust delay between requests (be respectful to servers!)
python crawler/crawler.py --delay 2.0

# Combine all options
python crawler/crawler.py \
    --input data/raw/urls_collected_extra.csv \
    --output data/processed/extra_crawled_data.csv \
    --delay 1.5
```

**Arguments:**
- `--input`: Input CSV file with URLs (default: `data/raw/urls_initial.csv`)
- `--output`: Output CSV file for results (default: `data/processed/crawled_data.csv`)
- `--delay`: Delay between requests in seconds (default: 1.0)

### Custom Configuration

You can modify the script to change settings:

```python
from crawler.crawler import NewsCrawler

crawler = NewsCrawler(
    input_csv='path/to/urls.csv',
    output_csv='path/to/output.csv',
    delay=2.0  # 2 second delay between requests
)
crawler.run()
```

## Output Format

The output CSV contains:
- `url`: The original URL
- `title`: Extracted article title
- `label`: Either 'fox' or 'nbc'
- `status`: 'success', 'failed', or 'no_title'

## Title Extraction Examples

The crawler handles various title formats:

**Fox News:**
```html
<title>Michael Moore warns Kamala Harris to not go 'centrist' | Fox News</title>
```
Extracted: "Michael Moore warns Kamala Harris to not go 'centrist'"

**NBC News:**
```html
<h1 class="headline speakable">How Stairwell B became the site of a miracle</h1>
```
Extracted: "How Stairwell B became the site of a miracle"

## Notes

- The crawler includes a 1-second delay between requests by default to be respectful to the websites
- Failed requests and missing titles are logged in the output CSV
- The crawler uses multiple strategies to find titles for maximum coverage
