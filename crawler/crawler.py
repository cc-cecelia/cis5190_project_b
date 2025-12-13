"""
Web Crawler for Fox News and NBC News Articles
Extracts titles from news URLs and labels them by source
"""

import csv
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
from urllib.parse import urlparse


class NewsCrawler:
    def __init__(self, input_csv: str, output_csv: str, delay: float = 1.0):
        """
        Initialize the news crawler
        
        Args:
            input_csv: Path to input CSV file containing URLs
            output_csv: Path to output CSV file for results
            delay: Delay between requests in seconds (to be respectful)
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def get_source_label(self, url: str) -> Optional[str]:
        """
        Determine the news source from URL
        
        Args:
            url: The URL to check
            
        Returns:
            'fox' for Fox News, 'nbc' for NBC News, None otherwise
        """
        domain = urlparse(url).netloc.lower()
        if 'foxnews.com' in domain:
            return 'fox'
        elif 'nbcnews.com' in domain:
            return 'nbc'
        return None
    
    def extract_title(self, html: str, source: str) -> Optional[str]:
        """
        Extract title from HTML content based on source
        
        Args:
            html: HTML content
            source: News source ('fox' or 'nbc')
            
        Returns:
            Extracted title or None if not found
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try multiple strategies to find the title
        title = None
        
        # Strategy 1: Look for h1 tags with specific classes (ACTUAL HEADLINES)
        # This is the most reliable way to get real headlines
        h1_candidates = [
            soup.find('h1', class_=re.compile(r'headline.*speakable|speakable.*headline', re.I)),  # Both headline and speakable
            soup.find('h1', class_='headline'),
            soup.find('h1', class_=re.compile(r'headline', re.I)),
            soup.find('h1', class_=re.compile(r'article.*title', re.I)),
            soup.find('h1', class_=re.compile(r'article.*headline', re.I)),
            soup.find('h1'),  # Any h1 as last resort
        ]
        
        for h1 in h1_candidates:
            if h1 and h1.get_text(strip=True):
                title = h1.get_text(strip=True)
                break
        
        # Strategy 2: Look for meta tags (backup)
        if not title or len(title) < 10:
            meta_candidates = [
                soup.find('meta', property='og:title'),
                soup.find('meta', attrs={'name': 'title'}),
                soup.find('meta', attrs={'name': 'twitter:title'}),
            ]
            
            for meta in meta_candidates:
                if meta and meta.get('content'):
                    title = meta['content'].strip()
                    # Clean up suffixes from meta tags
                    if source == 'fox':
                        title = re.sub(r'\s*\|\s*Fox News\s*$', '', title)
                    elif source == 'nbc':
                        title = re.sub(r'\s*-\s*NBC News\s*$', '', title)
                        title = re.sub(r'\s*\|\s*NBC News\s*$', '', title)
                    break
        
        # Strategy 3: Look for article title classes
        if not title or len(title) < 10:
            article_title_candidates = [
                soup.find('h1', class_='title'),
                soup.find('div', class_='headline'),
                soup.find('h1', attrs={'data-testid': 'article-title'}),
            ]
            
            for element in article_title_candidates:
                if element and element.get_text(strip=True):
                    title = element.get_text(strip=True)
                    break
        
        # Strategy 4: <title> tag (last resort - may have SEO text)
        if not title or len(title) < 10:
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title = title_tag.string.strip()
                # Clean up suffixes
                if source == 'fox':
                    title = re.sub(r'\s*\|\s*Fox News\s*$', '', title)
                elif source == 'nbc':
                    title = re.sub(r'\s*-\s*NBC News\s*$', '', title)
                    title = re.sub(r'\s*\|\s*NBC News\s*$', '', title)
        
        # Clean up the title
        if title:
            # Remove extra whitespace
            title = ' '.join(title.split())
            
        return title if title else None
    
    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from URL
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def crawl_urls(self) -> List[Dict[str, str]]:
        """
        Crawl all URLs from input CSV
        
        Returns:
            List of dictionaries containing url, title, and label
        """
        results = []
        
        # Read URLs from CSV
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            urls = [row[0] for row in reader if row]  # Get first column
        
        print(f"Found {len(urls)} URLs to crawl")
        
        for idx, url in enumerate(urls, 1):
            # Determine source
            source = self.get_source_label(url)
            
            if source is None:
                print(f"[{idx}/{len(urls)}] Skipping {url} (not Fox News or NBC News)")
                continue
            
            print(f"[{idx}/{len(urls)}] Crawling {url}")
            
            # Fetch content
            html = self.fetch_url(url)
            
            if html is None:
                results.append({
                    'url': url,
                    'title': None,
                    'label': source,
                    'status': 'failed'
                })
                continue
            
            # Extract title
            title = self.extract_title(html, source)
            
            if title:
                print(f"  → Title: {title[:80]}...")
                results.append({
                    'url': url,
                    'title': title,
                    'label': source,
                    'status': 'success'
                })
            else:
                print(f"  → No title found")
                results.append({
                    'url': url,
                    'title': None,
                    'label': source,
                    'status': 'no_title'
                })
            
            # Be respectful with delays
            time.sleep(self.delay)
        
        return results
    
    def save_results(self, results: List[Dict[str, str]]):
        """
        Save results to CSV
        
        Args:
            results: List of crawl results
        """
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'title', 'label', 'status'])
            writer.writeheader()
            writer.writerows(results)
        
        # Print summary
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        no_title = sum(1 for r in results if r['status'] == 'no_title')
        
        fox_count = sum(1 for r in results if r['label'] == 'fox' and r['status'] == 'success')
        nbc_count = sum(1 for r in results if r['label'] == 'nbc' and r['status'] == 'success')
        
        print("\n" + "="*60)
        print("CRAWL SUMMARY")
        print("="*60)
        print(f"Total URLs processed: {total}")
        print(f"Successfully extracted: {success}")
        print(f"  - Fox News: {fox_count}")
        print(f"  - NBC News: {nbc_count}")
        print(f"Failed to fetch: {failed}")
        print(f"No title found: {no_title}")
        print(f"\nResults saved to: {self.output_csv}")
    
    def run(self):
        """Run the crawler"""
        print("Starting news crawler...")
        results = self.crawl_urls()
        self.save_results(results)
        print("Crawling complete!")


def main():
    """Main entry point"""
    # Configuration
    INPUT_CSV = 'data/raw/urls_initial.csv'
    OUTPUT_CSV = 'data/processed/crawled_data.csv'
    DELAY = 1.0  # seconds between requests
    
    # Create and run crawler
    crawler = NewsCrawler(INPUT_CSV, OUTPUT_CSV, delay=DELAY)
    crawler.run()


if __name__ == '__main__':
    main()
