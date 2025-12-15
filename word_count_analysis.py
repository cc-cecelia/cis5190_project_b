"""
Word Count Analysis for Fox News and NBC News
Analyzes word frequency distribution and creates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import sys
from pathlib import Path

# NLTK stop words (common English words to exclude)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
    'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', 
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


def clean_text(text):
    """
    Clean and tokenize text
    - Convert to lowercase
    - Remove punctuation
    - Split into words
    - Remove stop words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and keep only letters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove stop words and short words (less than 3 characters)
    words = [word for word in words if word not in STOP_WORDS and len(word) >= 3]
    
    return words


def analyze_word_counts(csv_file, max_lines=None):
    """
    Analyze word counts for Fox and NBC articles
    
    Args:
        csv_file: Path to the processed_data.csv file
        max_lines: Maximum number of lines to read (None = all lines)
    """
    print("="*80)
    print("WORD COUNT ANALYSIS FOR FOX NEWS AND NBC NEWS")
    print("="*80)
    
    # Read CSV file
    print(f"\nReading data from: {csv_file}")
    if max_lines:
        df = pd.read_csv(csv_file, nrows=max_lines, names=['title', 'label'])
        print(f"Analyzing first {max_lines} lines")
    else:
        df = pd.read_csv(csv_file, names=['title', 'label'])
        print(f"Analyzing all {len(df)} lines")
    
    # Separate by source
    fox_titles = df[df['label'] == 'fox']['title'].tolist()
    nbc_titles = df[df['label'] == 'nbc']['title'].tolist()
    
    print(f"\nArticle counts:")
    print(f"  Fox News: {len(fox_titles)} articles")
    print(f"  NBC News: {len(nbc_titles)} articles")
    
    # Count words for Fox News
    print("\nProcessing Fox News articles...")
    fox_words = []
    for title in fox_titles:
        if pd.notna(title):
            fox_words.extend(clean_text(str(title)))
    
    # Count words for NBC News
    print("Processing NBC News articles...")
    nbc_words = []
    for title in nbc_titles:
        if pd.notna(title):
            nbc_words.extend(clean_text(str(title)))
    
    # Get word counts
    fox_counter = Counter(fox_words)
    nbc_counter = Counter(nbc_words)
    
    print(f"\nTotal words (excluding stop words):")
    print(f"  Fox News: {len(fox_words)} words ({len(fox_counter)} unique)")
    print(f"  NBC News: {len(nbc_words)} words ({len(nbc_counter)} unique)")
    
    return fox_counter, nbc_counter, len(fox_titles), len(nbc_titles)


def create_visualizations(fox_counter, nbc_counter, fox_count, nbc_count, top_n=20):
    """
    Create visualizations comparing word frequencies
    
    Args:
        fox_counter: Counter object with Fox word frequencies
        nbc_counter: Counter object with NBC word frequencies
        fox_count: Number of Fox articles
        nbc_count: Number of NBC articles
        top_n: Number of top words to display
    """
    # Get top words
    fox_top = fox_counter.most_common(top_n)
    nbc_top = nbc_counter.most_common(top_n)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Word Frequency Analysis: Fox News vs NBC News', fontsize=16, fontweight='bold')
    
    # 1. Top words for Fox News
    ax1 = axes[0, 0]
    fox_words = [word for word, _ in fox_top]
    fox_counts = [count for _, count in fox_top]
    ax1.barh(range(len(fox_words)), fox_counts, color='#C41E3A')
    ax1.set_yticks(range(len(fox_words)))
    ax1.set_yticklabels(fox_words)
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency')
    ax1.set_title(f'Top {top_n} Words - Fox News ({fox_count} articles)')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Top words for NBC News
    ax2 = axes[0, 1]
    nbc_words = [word for word, _ in nbc_top]
    nbc_counts = [count for _, count in nbc_top]
    ax2.barh(range(len(nbc_words)), nbc_counts, color='#0089CF')
    ax2.set_yticks(range(len(nbc_words)))
    ax2.set_yticklabels(nbc_words)
    ax2.invert_yaxis()
    ax2.set_xlabel('Frequency')
    ax2.set_title(f'Top {top_n} Words - NBC News ({nbc_count} articles)')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Comparison of common top words
    ax3 = axes[1, 0]
    # Find common words in top 30 of both
    fox_top_words = set([word for word, _ in fox_counter.most_common(30)])
    nbc_top_words = set([word for word, _ in nbc_counter.most_common(30)])
    common_words = list(fox_top_words & nbc_top_words)[:15]
    
    if common_words:
        fox_common = [fox_counter[word] for word in common_words]
        nbc_common = [nbc_counter[word] for word in common_words]
        
        x = range(len(common_words))
        width = 0.35
        
        ax3.bar([i - width/2 for i in x], fox_common, width, label='Fox News', color='#C41E3A')
        ax3.bar([i + width/2 for i in x], nbc_common, width, label='NBC News', color='#0089CF')
        ax3.set_xlabel('Words')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Common Top Words Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(common_words, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No common words in top 30', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Common Top Words Comparison')
    
    # 4. Word count distribution statistics
    ax4 = axes[1, 1]
    stats_text = f"""
    STATISTICS
    
    Fox News:
    • Total articles: {fox_count:,}
    • Total words: {sum(fox_counter.values()):,}
    • Unique words: {len(fox_counter):,}
    • Avg words/article: {sum(fox_counter.values())/fox_count:.1f}
    • Most common: "{fox_counter.most_common(1)[0][0]}" ({fox_counter.most_common(1)[0][1]} times)
    
    NBC News:
    • Total articles: {nbc_count:,}
    • Total words: {sum(nbc_counter.values()):,}
    • Unique words: {len(nbc_counter):,}
    • Avg words/article: {sum(nbc_counter.values())/nbc_count:.1f}
    • Most common: "{nbc_counter.most_common(1)[0][0]}" ({nbc_counter.most_common(1)[0][1]} times)
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'word_count_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    plt.show()


def print_top_words(fox_counter, nbc_counter, top_n=30):
    """Print top words for both sources"""
    print("\n" + "="*80)
    print(f"TOP {top_n} WORDS")
    print("="*80)
    
    print(f"\nFOX NEWS - Top {top_n} Words:")
    print("-" * 50)
    for i, (word, count) in enumerate(fox_counter.most_common(top_n), 1):
        print(f"{i:2d}. {word:20s} {count:5d}")
    
    print(f"\nNBC NEWS - Top {top_n} Words:")
    print("-" * 50)
    for i, (word, count) in enumerate(nbc_counter.most_common(top_n), 1):
        print(f"{i:2d}. {word:20s} {count:5d}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze word counts in Fox and NBC news titles')
    parser.add_argument('--file', type=str, 
                       default='data/processed/processed_data_merged.csv',
                       help='Path to processed_data.csv file')
    parser.add_argument('--lines', type=int, default=None,
                       help='Maximum number of lines to analyze (default: all)')
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top words to display (default: 20)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Analyze word counts
    fox_counter, nbc_counter, fox_count, nbc_count = analyze_word_counts(
        args.file, max_lines=args.lines
    )
    
    # Print top words
    print_top_words(fox_counter, nbc_counter, top_n=args.top)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(fox_counter, nbc_counter, fox_count, nbc_count, top_n=args.top)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
