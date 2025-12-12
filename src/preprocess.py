"""
Comprehensive Data Preprocessing for News Headlines
Combines data cleaning and text normalization
"""

import csv
import re
import unicodedata
import torch
from typing import Any, List, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only needs to run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """Handles all text cleaning and normalization"""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        return re.sub(r'<[^>]+>', '', text)
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters but keep basic punctuation"""
        text = unicodedata.normalize('NFKD', text)
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\'\"\-\,\.\!\?\:\;\(\)\/]', '', text, flags=re.UNICODE)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: remove extra spaces, tabs, newlines"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def handle_punctuation(self, text: str) -> str:
        """Handle punctuation marks within headlines"""
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Fix spacing before punctuation
        text = re.sub(r'\s+([,\.!?;:])', r'\1', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([,\.!?;:])\s*', r'\1 ', text)
        
        # Remove space before closing parenthesis
        text = re.sub(r'\s+\)', ')', text)
        
        # Remove space after opening parenthesis
        text = re.sub(r'\(\s+', '(', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def convert_to_lowercase(self, text: str) -> str:
        """Convert text to lowercase for normalization"""
        return text.lower()
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to reduce words to base forms"""
        if not self.lemmatize:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def clean_and_normalize(self, text: str) -> Optional[str]:
        """
        Apply all cleaning and normalization steps
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned and normalized text, or None if invalid
        """
        if not text or text.strip() == '':
            return None
        
        # Step 1: Data Cleaning - Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 2: Data Cleaning - Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 3: Data Cleaning - Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 4: Data Cleaning - Handle punctuation
        text = self.handle_punctuation(text)
        
        # Validation after cleaning
        if len(text.strip()) < 10:
            return None
        
        # Step 5: Text Normalization - Convert to lowercase
        text = self.convert_to_lowercase(text)
        
        # Step 6: Text Normalization - Tokenization
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split if tokenization fails
            tokens = text.split()
        
        # Step 7: Text Normalization - Remove stopwords
        tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Step 8: Text Normalization - Lemmatization
        tokens = self.lemmatize_tokens(tokens)
        
        # Remove empty tokens and rejoin
        tokens = [token for token in tokens if token.strip()]
        
        if not tokens:
            return None
        
        normalized_text = ' '.join(tokens)
        
        return normalized_text


def prepare_data(path: str, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess data from CSV
    
    Args:
        path: Path to CSV file with 'title' and 'label' columns
        normalize: Whether to apply full text normalization (lowercase, stopwords, lemmatization)
                  If False, only basic cleaning is applied
    
    Returns:
        X: torch.Tensor of dtype=torch.object
           each element is {"text": preprocessed_title_str}
        y: torch.Tensor of dtype=torch.long
           each element is int(0/1) where 0=fox, 1=nbc
    """
    
    preprocessor = TextPreprocessor(
        remove_stopwords=normalize,
        lemmatize=normalize
    )
    
    X_list = []
    y_list = []
    
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"].strip()
            label_str = row["label"].strip().lower()
            
            # Convert label: fox=0, nbc=1
            if label_str == "fox":
                label = 0
            elif label_str in ["nbc", "nbc news"]:
                label = 1
            else:
                print(f"Warning: Unknown label '{label_str}' - skipping")
                skipped += 1
                continue
            
            # Apply preprocessing
            processed_title = preprocessor.clean_and_normalize(title)
            
            if not processed_title:
                skipped += 1
                continue
            
            # model.predict expects dict {"text": ...}
            X_list.append({"text": processed_title})
            y_list.append(label)

    if skipped > 0:
        print(f"Skipped {skipped} invalid records during preprocessing")
    
    print(f"Loaded {len(X_list)} valid records")

    # Convert to torch Tensors (object dtype allowed)
    X_tensor = torch.tensor(X_list, dtype=torch.object)
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    return X_tensor, y_tensor


def preprocess_and_save(input_csv: str, output_csv: str, normalize: bool = True) -> None:
    """
    Preprocess crawled data and save to a new CSV file
    
    Args:
        input_csv: Path to input CSV file (crawled data with 'url', 'title', 'label', 'status')
        output_csv: Path to output CSV file (preprocessed data with 'title', 'label')
        normalize: Whether to apply full text normalization (lowercase, stopwords, lemmatization)
    """
    from collections import Counter
    
    preprocessor = TextPreprocessor(
        remove_stopwords=normalize,
        lemmatize=normalize
    )
    
    processed_data = []
    title_counter = Counter()
    
    total_count = 0
    skipped_failed = 0
    skipped_invalid = 0
    skipped_duplicates = 0
    
    print(f"Reading data from: {input_csv}")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    total_count = len(records)
    print(f"Processing {total_count} records...")
    
    for idx, row in enumerate(records, 1):
        # Skip failed crawls
        if 'status' in row and row['status'] != 'success':
            skipped_failed += 1
            continue
        
        title = row.get('title', '').strip()
        label = row.get('label', '').strip().lower()
        
        # Validate label
        if label not in ['fox', 'nbc', 'nbc news']:
            print(f"[{idx}/{total_count}] Warning: Unknown label '{label}' - skipping")
            skipped_invalid += 1
            continue
        
        # Normalize label
        if label in ['nbc', 'nbc news']:
            label = 'nbc'
        
        # Apply preprocessing
        processed_title = preprocessor.clean_and_normalize(title)
        
        if not processed_title:
            skipped_invalid += 1
            continue
        
        # Check for duplicates
        title_counter[processed_title] += 1
        if title_counter[processed_title] > 1:
            skipped_duplicates += 1
            continue
        
        processed_data.append({
            'title': processed_title,
            'label': label
        })
        
        # Progress indicator
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{total_count} ({idx*100//total_count}%)")
    
    # Save to CSV
    print(f"\nSaving preprocessed data to: {output_csv}")
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['title', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_data)
    
    # Summary statistics
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Total records processed: {total_count}")
    print(f"Successfully preprocessed: {len(processed_data)}")
    print(f"Skipped (failed crawl): {skipped_failed}")
    print(f"Skipped (invalid/empty): {skipped_invalid}")
    print(f"Skipped (duplicates): {skipped_duplicates}")
    print(f"Success rate: {len(processed_data)*100/total_count:.1f}%")
    
    # Label distribution
    fox_count = sum(1 for d in processed_data if d['label'] == 'fox')
    nbc_count = sum(1 for d in processed_data if d['label'] == 'nbc')
    
    print(f"\nLabel distribution:")
    print(f"  Fox: {fox_count} ({fox_count*100/len(processed_data):.1f}%)")
    print(f"  NBC: {nbc_count} ({nbc_count*100/len(processed_data):.1f}%)")
    print(f"  Balance ratio (Fox:NBC): {fox_count/nbc_count:.2f}:1")
    
    # Title length statistics
    lengths = [len(d['title']) for d in processed_data]
    word_counts = [len(d['title'].split()) for d in processed_data]
    
    print(f"\nText statistics (after preprocessing):")
    print(f"  Average length: {sum(lengths)/len(lengths):.1f} characters")
    print(f"  Min length: {min(lengths)} characters")
    print(f"  Max length: {max(lengths)} characters")
    print(f"  Average words: {sum(word_counts)/len(word_counts):.1f} words")
    print(f"  Min words: {min(word_counts)} words")
    print(f"  Max words: {max(word_counts)} words")
    
    print("\n" + "="*80)
    print(f"Preprocessing complete! Output saved to: {output_csv}")
    print("="*80)


def main():
    """
    Main function to preprocess crawled data
    Run with: python -m src.preprocess
    """
    import os
    
    # Default paths
    input_csv = "data/processed/crawled_data.csv"
    output_csv = "data/processed/processed_data.csv"
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        print("\nPlease run the crawler first to generate the data.")
        return
    
    print("="*80)
    print("NEWS HEADLINES PREPROCESSING")
    print("="*80)
    print(f"\nInput:  {input_csv}")
    print(f"Output: {output_csv}")
    print("\nPreprocessing steps:")
    print("  1. Data Cleaning:")
    print("     - Remove HTML tags")
    print("     - Remove special characters")
    print("     - Normalize whitespace")
    print("     - Handle punctuation")
    print("  2. Text Normalization:")
    print("     - Convert to lowercase")
    print("     - Tokenization")
    print("     - Remove stopwords")
    print("     - Lemmatization (reduce words to base forms)")
    print("\n" + "="*80 + "\n")
    
    # Run preprocessing with full normalization
    preprocess_and_save(input_csv, output_csv, normalize=True)


if __name__ == "__main__":
    main()
