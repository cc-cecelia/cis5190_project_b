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
        
    def _auto_save_checkpoint(self, all_articles: List[Dict[str, str]]):
        """
        Auto-save all collected articles (Fox + NBC) to prevent data loss
        
        Args:
            all_articles: All collected articles from both Fox and NBC
        """
        import os
        os.makedirs("data/raw", exist_ok=True)
        
        output_file = "data/raw/batch4_checkpoint.csv"
        
        fox_count = sum(1 for a in all_articles if a.get('label') == 'fox')
        nbc_count = sum(1 for a in all_articles if a.get('label') == 'nbc')
        successful_count = sum(1 for a in all_articles if a.get('status') == 'success')
        
        print(f"\n💾 Auto-saving checkpoint: {len(all_articles)} articles to {output_file}")
        print(f"   Fox: {fox_count} | NBC: {nbc_count} | Success: {successful_count}")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['url', 'title', 'label', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_articles)
        
        print(f"✓ Checkpoint saved successfully")
        
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
    
    def collect_from_section(self, section_url: str, source: str, max_urls: int = 200) -> List[Dict[str, str]]:
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
    
    def collect_fox_news_urls(self, target_count: int = 500, all_articles: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Collect URLs and titles from Fox News sitemap
        
        Args:
            target_count: Target number of articles to collect
            all_articles: Global list to track all articles for periodic checkpointing
        """
        print("\n" + "="*80)
        print("COLLECTING FOX NEWS URLs FROM SITEMAP")
        print("="*80)
        
        # Generate Fox News sitemap URLs for 2023
        # Format: https://www.foxnews.com/html-sitemap/2023/january/4
        sitemap_urls = []
        
        # 2023 months (all 12 months)
        months_2023 = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december']
        
        # Generate daily sitemaps for 2023 (1-31 days per month)
        for month in months_2023:
            for day in range(1, 32):  # 1 to 31
                sitemap_url = f"https://www.foxnews.com/html-sitemap/2023/{month}/{day}"
                sitemap_urls.append(sitemap_url)
        
        print(f"Generated {len(sitemap_urls)} sitemap URLs to crawl")
        
        # Shuffle to get diverse content
        random.shuffle(sitemap_urls)
        
        collected = []
        
        for sitemap_url in sitemap_urls:
            if len(collected) >= target_count:
                break
            
            sitemap_name = sitemap_url.split('html-sitemap/')[-1]
            print(f"\nSitemap: {sitemap_name}")
            
            # Collect from sitemap (no max_urls limit per sitemap, collect all available)
            articles = self.collect_from_section(sitemap_url, 'fox', max_urls=500)
            
            for article in articles:
                if len(collected) >= target_count:
                    break
                collected.append(article)
                
                # Add to global list and checkpoint every 500 articles
                if all_articles is not None:
                    all_articles.append(article)
                    if len(all_articles) % 500 == 0:
                        self._auto_save_checkpoint(all_articles)
            
            time.sleep(random.uniform(2.0, 3.5))
        
        print(f"\nTotal Fox News URLs collected: {len(collected)}")
        return collected
    
    def collect_nbc_news_urls(self, target_count: int = 500, all_articles: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Collect URLs and titles from NBC News archive
        
        Args:
            target_count: Target number of articles to collect
            all_articles: Global list to track all articles for periodic checkpointing
        """
        print("\n" + "="*80)
        print("COLLECTING NBC NEWS URLs FROM ARCHIVE")
        print("="*80)
        
        # Generate NBC News archive URLs for 2023
        # Format: https://www.nbcnews.com/archive/articles/2023/november
        # NBC has page 1 (base URL) and page 2 (/2)
        archive_urls = []
        
        # 2023 months (all 12 months)
        months_2023 = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december']
        
        for month in months_2023:
            # Page 1 (base URL)
            archive_urls.append(f"https://www.nbcnews.com/archive/articles/2023/{month}")
            # Page 2
            archive_urls.append(f"https://www.nbcnews.com/archive/articles/2023/{month}/2")
        
        print(f"Generated {len(archive_urls)} archive URLs to crawl")
        
        # Shuffle for diversity
        random.shuffle(archive_urls)
        
        collected = []
        
        for archive_url in archive_urls:
            if len(collected) >= target_count:
                break
            
            archive_name = archive_url.split('archive/articles/')[-1]
            print(f"\nArchive: {archive_name}")
            
            # Collect from archive (no max_urls limit per page, collect all available)
            articles = self.collect_from_section(archive_url, 'nbc', max_urls=500)
            
            for article in articles:
                if len(collected) >= target_count:
                    break
                collected.append(article)
                
                # Add to global list and checkpoint every 500 articles
                if all_articles is not None:
                    all_articles.append(article)
                    if len(all_articles) % 500 == 0:
                        self._auto_save_checkpoint(all_articles)
            
            time.sleep(random.uniform(2.0, 3.5))
        
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
        'data/raw/batch1_urls_collected_extra.csv',   # Previous extra collection 
        'data/processed/crawled_data.csv',   # Previous extra collection
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
    
    # Track all articles globally for proper checkpointing (every 500 articles)
    all_articles = []
    
    # Collect Fox News articles
    print("\n--- Collecting Fox News articles ---")
    fox_articles = collector.collect_fox_news_urls(target_count=fox_target, all_articles=all_articles)
    
    # Save checkpoint after Fox collection
    if all_articles:
        collector._auto_save_checkpoint(all_articles)
        print(f"✓ Checkpoint saved after Fox collection: {len(all_articles)} total articles")
    
    # Collect NBC News articles
    print("\n--- Collecting NBC News articles ---")
    nbc_articles = collector.collect_nbc_news_urls(target_count=nbc_target, all_articles=all_articles)
    
    # Save final checkpoint after NBC collection
    if all_articles:
        collector._auto_save_checkpoint(all_articles)
        print(f"✓ Final checkpoint saved: {len(all_articles)} total articles")
    
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
    urls_file = "data/raw/batch4_urls_collected_extra.csv"
    collector.save_to_csv(all_articles, urls_file)
    
    # 2. Save full crawled data with titles (matches format of crawled_data.csv)
    crawled_file = "data/processed/batch4_extra_crawled_data.csv"
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
