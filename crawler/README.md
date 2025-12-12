# News Crawler

This crawler extracts article titles from Fox News and NBC News URLs.

## Features

- Extracts titles from multiple HTML formats:
  - `<title>` tags
  - `<h1 class="headline">` tags (NBC News format)
  - Meta tags (og:title, twitter:title)
  - Article-specific title elements
- Labels each article as 'fox' or 'nbc'
- Handles various title formats and cleans up suffixes
- Respectful crawling with configurable delays
- Comprehensive error handling

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the crawler with default settings:

```bash
python crawler/crawler.py
```

This will:
1. Read URLs from `data/raw/urls_initial.csv`
2. Crawl each URL and extract the title
3. Save results to `data/raw/crawled_titles.csv`

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
