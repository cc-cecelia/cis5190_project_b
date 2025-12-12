"""
Create a random sample of URLs for testing
"""

import csv
import random


def sample_urls(input_csv: str, output_csv: str, sample_size: int = 200):
    """
    Randomly sample URLs from input CSV and save to output CSV
    
    Args:
        input_csv: Path to input CSV file with URLs
        output_csv: Path to output CSV file for sampled URLs
        sample_size: Number of URLs to sample
    """
    # Read all URLs
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        urls = [row[0] for row in reader if row and row[0].strip()]
    
    print(f"Total URLs in {input_csv}: {len(urls)}")
    
    # Filter for Fox News and NBC News only
    fox_urls = [url for url in urls if 'foxnews.com' in url.lower()]
    nbc_urls = [url for url in urls if 'nbcnews.com' in url.lower()]
    
    print(f"Fox News URLs: {len(fox_urls)}")
    print(f"NBC News URLs: {len(nbc_urls)}")
    
    # Calculate how many to sample from each source
    # Try to keep it balanced
    total_available = len(fox_urls) + len(nbc_urls)
    
    if total_available < sample_size:
        print(f"\nWarning: Only {total_available} Fox/NBC URLs available.")
        print(f"Sampling all available URLs instead of {sample_size}.")
        sample_size = total_available
    
    # Sample proportionally or 50/50 if both have enough
    if len(fox_urls) >= sample_size // 2 and len(nbc_urls) >= sample_size // 2:
        # Balanced sampling
        fox_sample_size = sample_size // 2
        nbc_sample_size = sample_size - fox_sample_size
    else:
        # Proportional sampling
        fox_ratio = len(fox_urls) / total_available
        fox_sample_size = min(int(sample_size * fox_ratio), len(fox_urls))
        nbc_sample_size = min(sample_size - fox_sample_size, len(nbc_urls))
    
    print(f"\nSampling:")
    print(f"  - {fox_sample_size} Fox News URLs")
    print(f"  - {nbc_sample_size} NBC News URLs")
    print(f"  - Total: {fox_sample_size + nbc_sample_size} URLs")
    
    # Random sampling
    random.seed(42)  # For reproducibility
    fox_sample = random.sample(fox_urls, fox_sample_size)
    nbc_sample = random.sample(nbc_urls, nbc_sample_size)
    
    # Combine and shuffle
    all_samples = fox_sample + nbc_sample
    random.shuffle(all_samples)
    
    # Write to output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for url in all_samples:
            writer.writerow([url])
    
    print(f"\n✓ Saved {len(all_samples)} URLs to {output_csv}")


if __name__ == '__main__':
    sample_urls(
        input_csv='data/raw/urls_initial.csv',
        output_csv='data/raw/test_urls.csv',
        sample_size=200
    )
