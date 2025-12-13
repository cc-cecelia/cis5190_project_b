"""
Extra URL Collector for Fox News and NBC News
Collects additional URLs from news websites for expanded training data
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import random
from urllib.parse import urljoin, urlparse
from typing import List, Set, Dict
import re


class ExtraURLCollector:
    """Collects additional URLs from Fox News and NBC News websites"""
    
    def __init__(self, existing_urls_files: List[str] = None):
        """
        Initialize collector with duplicate tracking
        
        Args:
            existing_urls_files: List of CSV files to check for existing URLs
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.fox_base_url = "https://www.foxnews.com"
        self.nbc_base_url = "https://www.nbcnews.com"
        
        # Store collected URLs to avoid duplicates (normalized)
        self.collected_urls: Set[str] = set()
        
        # Load existing URLs from previous collections
        self.existing_urls: Set[str] = set()
        if existing_urls_files:
            self._load_existing_urls(existing_urls_files)
    
    def _load_existing_urls(self, files: List[str]):
        """Load URLs from existing CSV files to avoid duplicates"""
        import os
        for filepath in files:
            if not os.path.exists(filepath):
                print(f"Note: {filepath} not found, skipping duplicate check for this file")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'url' in row:
                            normalized = self._normalize_url(row['url'])
                            self.existing_urls.add(normalized)
                print(f"Loaded {len(self.existing_urls)} existing URLs from {filepath}")
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for duplicate detection
        - Remove query parameters
        - Remove fragments
        - Remove trailing slashes
        - Convert to lowercase
        """
        if not url:
            return ""
        
        # Remove query params and fragments
        clean_url = url.split('?')[0].split('#')[0]
        
        # Remove trailing slash
        clean_url = clean_url.rstrip('/')
        
        # Lowercase for comparison
        clean_url = clean_url.lower()
        
        return clean_url
        
    def _auto_save_checkpoint(self, articles: List[Dict[str, str]], source: str):
        """
        Auto-save every 500 articles to prevent data loss
        
        Args:
            articles: Current list of collected articles
            source: 'fox' or 'nbc'
        """
        import os
        os.makedirs("data/raw", exist_ok=True)
        
        output_file = "data/raw/extra_crawled_data.csv"
        
        print(f"\n💾 Auto-saving: {len(articles)} articles to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['url', 'title', 'label', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(articles)
        
        successful_count = sum(1 for a in articles if a.get('status') == 'success')
        print(f"✓ Saved: {len(articles)} articles ({successful_count} successful)")
        
    def is_valid_article_url(self, url: str, source: str) -> bool:
        """
        Check if URL is a valid article URL (not a category/hub page)
        
        Args:
            url: URL to check
            source: 'fox' or 'nbc'
            
        Returns:
            True if valid article URL
        """
        if not url:
            return False
        
        # Normalize URL for duplicate checking
        normalized = self._normalize_url(url)
        
        # Check if already collected in this session
        if normalized in self.collected_urls:
            return False
        
        # Check if exists in previous collections
        if normalized in self.existing_urls:
            return False
        
        
        if source == 'fox':
            # Fox News article patterns
            if not 'foxnews.com' in url:
                return False
            
            # EXCLUDE category/hub pages and non-article content
            if any(x in url.lower() for x in [
                '/category/',      # Category hub pages
                '/video/',         # Video content
                '/live/',          # Live streams
                '/person/',        # Author pages
                '/shows/',         # TV shows
                '/tv/',            # TV section
                '/about',          # About pages
                '/contact',        # Contact pages
                '/privacy',        # Privacy policy
                '/terms',          # Terms of service
                'foxnews.com/v/',  # Video subdomain
                '/games/',         # Games section
                '/newsletters/',   # Newsletters
                '/apps/',          # Apps pages
                '/radio/',         # Radio content
            ]):
                return False
            
            # EXCLUDE section pages without article titles (just the base section)
            # e.g., /politics, /us, /world should be excluded
            path = urlparse(url).path.strip('/')
            sections = ['politics', 'us', 'world', 'opinion', 'entertainment', 
                       'lifestyle', 'media', 'tech', 'sports', 'health', 'science']
            if path in sections:
                return False
            
            # EXCLUDE year-based archive pages
            if re.search(r'/\d{4}/', url):
                return False
            
            # REQUIRE: Must have at least 3 path segments (e.g., /politics/article-title)
            # AND the last segment should be a meaningful article slug
            path_parts = [p for p in urlparse(url).path.split('/') if p]
            if len(path_parts) < 2:
                return False
            
            # Last part should be article slug (has dashes, reasonable length)
            article_slug = path_parts[-1]
            if len(article_slug) < 10:  # Too short to be an article
                return False
            
            # Should have multiple words (indicated by dashes)
            if article_slug.count('-') < 2:
                return False
            
            return True
            
        elif source == 'nbc':
            # NBC News article patterns
            if not 'nbcnews.com' in url:
                return False
            
            # EXCLUDE non-article pages
            if any(x in url.lower() for x in [
                '/video/',         # Video content
                '/live/',          # Live streams
                '/live-blog/',     # Live blogs
                '/archives/',      # Archive pages
                '/sitemap/',       # Sitemap
                '/newsletters/',   # Newsletter signup
                '/think/',         # Opinion section hub
                '/select/',        # Shopping section
                '/better/',        # Lifestyle hub
                '/know-your-value/', # KYV section hub
                '/today/',         # Today show
                '/dateline/',      # Dateline show
                'link.nbcnews.com', # External link redirects
                '/join/',          # Sign-up pages
            ]):
                return False
            
            # NBC articles typically have article-title-n123456 or article-title-rcna123456 format
            # This is the BEST indicator of a real article
            if re.search(r'-n\d{6,}$', url) or re.search(r'-rcna\d{6,}$', url):
                return True
            
            # EXCLUDE section hubs (just base sections without article IDs)
            path = urlparse(url).path.strip('/')
            sections = ['politics', 'us-news', 'world', 'business', 'health', 
                       'tech', 'news/latino', 'news/asian-america', 'news/investigations']
            if path in sections:
                return False
            
            # If no article ID, must have meaningful slug
            path_parts = [p for p in urlparse(url).path.split('/') if p]
            if len(path_parts) < 2:
                return False
            
            article_slug = path_parts[-1]
            if len(article_slug) < 15:  # NBC slugs are usually longer
                return False
            
            # Should have multiple words
            if article_slug.count('-') < 3:
                return False
                
            return True
        
        return False
    
    def extract_article_urls(self, html: str, base_url: str, source: str) -> List[str]:
        """
        Extract article URLs from HTML page with smart targeting
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            source: 'fox' or 'nbc'
            
        Returns:
            List of valid article URLs
        """
        soup = BeautifulSoup(html, 'lxml')
        urls = []
        
        # Target article containers for better precision
        if source == 'fox':
            # Look for article containers with class patterns
            article_containers = soup.find_all(['article', 'div'], 
                                              class_=re.compile(r'(article|story|content|item)', re.I))
        elif source == 'nbc':
            # NBC uses specific article markup
            article_containers = soup.find_all(['article', 'div'],
                                              class_=re.compile(r'(article|story|tease|package)', re.I))
        else:
            article_containers = [soup]  # Fallback to whole page
        
        # Extract links from article containers first (higher quality)
        for container in article_containers:
            for link in container.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    full_url = urljoin(base_url, href)
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                # Clean URL
                full_url = full_url.split('#')[0]
                
                # Validate and add
                if self.is_valid_article_url(full_url, source):
                    if full_url not in urls:
                        urls.append(full_url)
        
        # If we didn't find enough, do a broader search
        if len(urls) < 10:
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if href.startswith('/'):
                    full_url = urljoin(base_url, href)
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                full_url = full_url.split('#')[0]
                
                if self.is_valid_article_url(full_url, source):
                    if full_url not in urls:
                        urls.append(full_url)
        
        return urls
    
    def extract_title_from_url(self, url: str, source: str) -> str:
        """
        Extract title from a news article URL
        Uses the same robust extraction logic as crawler.py
        
        Args:
            url: Article URL
            source: 'fox' or 'nbc'
            
        Returns:
            Extracted title or empty string if failed
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
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
            
            return title if title else ""
            
        except Exception as e:
            return ""
    
    def collect_from_section(self, section_url: str, source: str, max_urls: int = 50) -> List[Dict[str, str]]:
        """
        Collect URLs and titles from a specific news section
        
        Args:
            section_url: URL of the section page
            source: 'fox' or 'nbc'
            max_urls: Maximum number of URLs to collect from this section
            
        Returns:
            List of dicts with 'url', 'title', 'label', and 'status' keys
        """
        print(f"\nCollecting from: {section_url}")
        
        try:
            response = requests.get(section_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Extract article URLs
            article_urls = self.extract_article_urls(response.text, section_url, source)
            article_urls = article_urls[:max_urls]
            
            print(f"  Found {len(article_urls)} URLs, crawling titles...")
            
            collected = []
            for i, url in enumerate(article_urls, 1):
                # Check if already collected
                normalized = self._normalize_url(url)
                if normalized in self.collected_urls:
                    continue
                
                # Extract title
                print(f"    [{i}/{len(article_urls)}] {url[:60]}...")
                title = self.extract_title_from_url(url, source)
                
                if title:
                    collected.append({
                        'url': url,
                        'title': title,
                        'label': source,
                        'status': 'success'
                    })
                    self.collected_urls.add(normalized)
                    print(f"      ✓ {title[:70]}...")
                else:
                    collected.append({
                        'url': url,
                        'title': '',
                        'label': source,
                        'status': 'failed'
                    })
                    print(f"      ✗ Failed")
                
                # Be polite - delay between title extractions
                time.sleep(random.uniform(1.0, 1.5))
            
            success = sum(1 for x in collected if x['status'] == 'success')
            print(f"  Crawled {success}/{len(collected)} titles successfully")
            return collected
            
        except Exception as e:
            print(f"  Error collecting from {section_url}: {e}")
            return []
    
    def collect_fox_news_urls(self, target_count: int = 500) -> List[Dict[str, str]]:
        """
        Collect URLs and titles from Fox News
        
        Args:
            target_count: Target number of URLs to collect
            
        Returns:
            List of dicts with 'url', 'title', 'label', and 'status' keys
        """
        print("\n" + "="*80)
        print("COLLECTING FOX NEWS URLs AND TITLES")
        print("="*80)
        
        # Comprehensive list of Fox News sections
        fox_sections = [
            # Main news sections
            "https://www.foxnews.com/politics",
            "https://www.foxnews.com/us",
            "https://www.foxnews.com/world",
            "https://www.foxnews.com/opinion",
            
            # Topical sections
            "https://www.foxnews.com/us/crime",
            "https://www.foxnews.com/us/immigration",
            "https://www.foxnews.com/us/military",
            "https://www.foxnews.com/us/education",
            "https://www.foxnews.com/us/economy",
            
            # Lifestyle & Culture
            "https://www.foxnews.com/entertainment",
            "https://www.foxnews.com/lifestyle",
            "https://www.foxnews.com/media",
            "https://www.foxnews.com/health",
            "https://www.foxnews.com/food-drink",
            
            # Science & Tech
            "https://www.foxnews.com/tech",
            "https://www.foxnews.com/science",
            "https://www.foxnews.com/science/air-space",
            
            # Sports
            "https://www.foxnews.com/sports",
            "https://www.foxnews.com/sports/nfl",
            "https://www.foxnews.com/sports/nba",
            
            # Travel & Outdoors
            "https://www.foxnews.com/travel",
            "https://www.foxnews.com/outdoors",
        ]
        
        collected = []
        
        for section_url in fox_sections:
            if len(collected) >= target_count:
                break
            
            print(f"\nSection: {section_url.split('foxnews.com/')[-1]}")
            articles = self.collect_from_section(section_url, 'fox', max_urls=50)
            
            for article in articles:
                if len(collected) >= target_count:
                    break
                collected.append(article)  # Already has url, title, label, status
                
                # Auto-save every 500 articles
                if len(collected) % 500 == 0:
                    self._auto_save_checkpoint(collected, 'fox')
            
            # Be polite - delay between requests
            time.sleep(random.uniform(1.5, 2.5))
        
        print(f"\nTotal Fox News URLs collected: {len(collected)}")
        return collected
    
    def collect_nbc_news_urls(self, target_count: int = 500) -> List[Dict[str, str]]:
        """
        Collect URLs and titles from NBC News
        
        Args:
            target_count: Target number of URLs to collect
            
        Returns:
            List of dicts with 'url', 'title', 'label', and 'status' keys
        """
        print("\n" + "="*80)
        print("COLLECTING NBC NEWS URLs AND TITLES")
        print("="*80)
        
        # Comprehensive list of NBC News sections
        nbc_sections = [
            # Main news sections
            "https://www.nbcnews.com/politics",
            "https://www.nbcnews.com/us-news",
            "https://www.nbcnews.com/world",
            
            # Topical sections
            "https://www.nbcnews.com/us-news/crime-courts",
            "https://www.nbcnews.com/politics/immigration",
            "https://www.nbcnews.com/politics/congress",
            "https://www.nbcnews.com/politics/white-house",
            "https://www.nbcnews.com/politics/2024-election",
            
            # Business & Economy
            "https://www.nbcnews.com/business",
            "https://www.nbcnews.com/business/economy",
            "https://www.nbcnews.com/business/personal-finance",
            
            # Health & Science
            "https://www.nbcnews.com/health",
            "https://www.nbcnews.com/science",
            "https://www.nbcnews.com/health/mental-health",
            
            # Tech & Innovation
            "https://www.nbcnews.com/tech",
            "https://www.nbcnews.com/tech/tech-news",
            "https://www.nbcnews.com/tech/security",
            
            # Social & Culture
            "https://www.nbcnews.com/news/latino",
            "https://www.nbcnews.com/news/asian-america",
            "https://www.nbcnews.com/news/investigations",
            "https://www.nbcnews.com/news/education",
            "https://www.nbcnews.com/news/military",
            
            # Environment & Climate
            "https://www.nbcnews.com/science/environment",
            "https://www.nbcnews.com/science/climate",
        ]
        
        collected = []
        
        for section_url in nbc_sections:
            if len(collected) >= target_count:
                break
            
            print(f"\nSection: {section_url.split('nbcnews.com/')[-1]}")
            articles = self.collect_from_section(section_url, 'nbc', max_urls=50)
            
            for article in articles:
                if len(collected) >= target_count:
                    break
                collected.append(article)  # Already has url, title, label, status
                
                # Auto-save every 500 articles
                if len(collected) % 500 == 0:
                    self._auto_save_checkpoint(collected, 'nbc')
            
            # Be polite - delay between requests
            time.sleep(random.uniform(1.5, 2.5))
        
        print(f"\nTotal NBC News URLs collected: {len(collected)}")
        return collected
    
    def save_to_csv(self, urls: List[Dict[str, str]], output_file: str):
        """
        Save collected URLs to CSV file with final deduplication
        
        Args:
            urls: List of URL dictionaries
            output_file: Output CSV file path
        """
        print(f"\nFinal deduplication check...")
        
        # Final deduplication based on normalized URLs
        seen = set()
        unique_urls = []
        duplicates_removed = 0
        
        for item in urls:
            normalized = self._normalize_url(item['url'])
            if normalized not in seen:
                seen.add(normalized)
                unique_urls.append(item)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate(s)")
        
        print(f"\nSaving {len(unique_urls)} unique URLs to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['url', 'label']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(unique_urls)
        
        print(f"✓ Saved successfully")

    def save_crawled_data(self, articles: List[Dict[str, str]], output_file: str):
        """
        Save full article data (with titles) to CSV file
        Matches the format of crawled_data.csv: url, title, label, status
        
        Args:
            articles: List of article dictionaries with url, title, label, status
            output_file: Output CSV file path
        """
        print(f"\nSaving {len(articles)} articles to {output_file}")
        
        successful_count = sum(1 for a in articles if a.get('status') == 'success')
        failed_count = len(articles) - successful_count
        
        if failed_count > 0:
            print(f"  Note: {failed_count} article(s) failed title extraction")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['url', 'title', 'label', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(articles)
        
        print(f"✓ Saved {len(articles)} articles ({successful_count} successful, {failed_count} failed)")
        

def main():
    """Main execution function"""
    import sys
    
    # Parse command line arguments
    fox_target = 10  # default
    nbc_target = 10  # default
    
    if len(sys.argv) >= 3:
        try:
            fox_target = int(sys.argv[1])
            nbc_target = int(sys.argv[2])
        except ValueError:
            print("Usage: python collect_extra_urls.py [fox_count] [nbc_count]")
            print("Example: python collect_extra_urls.py 500 500")
            sys.exit(1)
    
    # Files to check for existing URLs (to avoid duplicates)
    existing_files = [
        'data/raw/urls_initial.csv',           # Original URLs from teacher
        'data/raw/urls_collected_extra.csv',   # Previous extra collection
        'data/processed/crawled_data.csv',     # Already crawled data
    ]
    
    print("="*80)
    print("INITIALIZING URL COLLECTOR WITH DUPLICATE DETECTION")
    print("="*80)
    
    collector = ExtraURLCollector(existing_urls_files=existing_files)
    
    if collector.existing_urls:
        print(f"\n✓ Loaded {len(collector.existing_urls)} existing URLs for duplicate detection")
    else:
        print("\nNote: No existing URLs loaded (first run or files not found)")
    
    # Collect URLs and titles
    print("\n" + "="*80)
    print(f"STARTING URL AND TITLE COLLECTION")
    print(f"Target: {fox_target} Fox News + {nbc_target} NBC News")
    print("="*80)
    
    fox_articles = collector.collect_fox_news_urls(target_count=fox_target)
    nbc_articles = collector.collect_nbc_news_urls(target_count=nbc_target)
    
    # Combine all articles
    all_articles = fox_articles + nbc_articles
    
    # Count successes and failures
    successful = [a for a in all_articles if a.get('status') == 'success']
    failed = [a for a in all_articles if a.get('status') == 'failed']
    
    # Summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    print(f"Fox News articles: {len(fox_articles)}")
    print(f"NBC News articles: {len(nbc_articles)}")
    print(f"Total articles collected: {len(all_articles)}")
    print(f"Successful title extractions: {len(successful)}")
    print(f"Failed title extractions: {len(failed)}")
    if len(nbc_articles) > 0:
        print(f"Balance ratio (Fox:NBC): {len(fox_articles)/len(nbc_articles):.2f}:1")
    
    # Duplicate statistics
    print(f"\nDuplicate Detection:")
    print(f"  Existing URLs checked: {len(collector.existing_urls)}")
    print(f"  New unique URLs found: {len(all_articles)}")
    print(f"  Duplicates avoided: {len(collector.existing_urls)} (from previous collections)")
    
    # Save both files
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    
    # 1. Save URLs only (for compatibility)
    urls_file = "data/raw/urls_collected_extra.csv"
    collector.save_to_csv(all_articles, urls_file)
    
    # 2. Save full crawled data with titles (matches format of crawled_data.csv)
    crawled_file = "data/raw/extra_crawled_data.csv"
    collector.save_crawled_data(all_articles, crawled_file)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"✓ URLs saved to: {urls_file}")
    print(f"✓ Full data saved to: {crawled_file}")
    print(f"\nYou can now:")
    print(f"1. Merge {crawled_file} with data/processed/crawled_data.csv")
    print(f"2. Re-run preprocessing on the combined dataset")
    print("="*80)


if __name__ == "__main__":
    main()
