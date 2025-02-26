from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
from functools import lru_cache
import multiprocessing
import os
from collections import Counter

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'

@lru_cache(maxsize=None)
def get_alphabet(alphabet_str: str):
    """Get alphabet and corresponding dictionary mapping."""
    alphabet = alphabet_str.upper()
    alphabet_dict = {char: i for i, char in enumerate(alphabet)}
    return alphabet, alphabet_dict

def decrypt_vigenere(ciphertext: str, key: str, alphabet_str: str) -> str:
    """Decrypt Vigenere cipher using the provided key and alphabet."""
    alphabet, alphabet_dict = get_alphabet(alphabet_str)
    alphabet_length = len(alphabet)
    
    key_shifts = [alphabet_dict.get(k, 0) for k in key.upper() if k in alphabet_dict]
    if not key_shifts:
        return ciphertext

    plaintext = []
    key_index = 0
    
    for char in ciphertext:
        if char in alphabet_dict:
            char_index = alphabet_dict[char]
            shift = key_shifts[key_index % len(key_shifts)]
            decrypted_char = alphabet[(char_index - shift) % alphabet_length]
            plaintext.append(decrypted_char.lower())
            key_index += 1
        else:
            plaintext.append(char.lower())
    
    return ''.join(plaintext)

def load_dictionary(file_path: str, alphabet_str: str, min_length: int = 3, max_length: int = 15) -> List[str]:
    """Load dictionary words that only contain characters from the given alphabet."""
    alphabet, _ = get_alphabet(alphabet_str)
    alphabet_set = set(alphabet)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [word.strip().upper() for word in file 
                    if set(word.strip().upper()).issubset(alphabet_set) 
                    and min_length <= len(word.strip()) <= max_length]
    except FileNotFoundError:
        print(f"{RED}Error: {file_path} not found{RESET}")
        return []

def try_key_directly(key: str, ciphertext: str, alphabet_str: str) -> str:
    """Try a specific key directly."""
    return decrypt_vigenere(ciphertext, key, alphabet_str)

def contains_all_phrases(text: str, phrases: List[str]) -> bool:
    """Check if plaintext contains all target phrases."""
    if not phrases:  # If no phrases provided, return True
        return True
        
    return all(phrase.upper() in text.upper().replace(" ", "") for phrase in phrases)

def highlight_match(text: str, phrases: List[str]) -> str:
    """Highlight all matched phrases in the plaintext."""
    result = text
    for phrase in phrases:
        phrase_upper = phrase.upper().replace(" ", "")
        text_upper = result.upper().replace(" ", "")
        
        if phrase_upper in text_upper:
            start = text_upper.find(phrase_upper)
            end = start + len(phrase_upper)
            
            # Create highlighted version
            highlighted = f"{RED}{result[start:end]}{RESET}"
            result = result[:start] + highlighted + result[end:]
            
    return result

def process_batch(args: Tuple) -> List[Dict]:
    """Process a batch of dictionary words as potential keys."""
    word_batch, ciphertext, target_phrases, alphabet_str = args
    results = []
    
    for word in word_batch:
        plaintext = decrypt_vigenere(ciphertext, word, alphabet_str)
        
        # Check if plaintext contains all target phrases
        if contains_all_phrases(plaintext, target_phrases):
            results.append({
                'key': word,
                'plaintext': plaintext,
                'matched_phrases': target_phrases
            })
                
    return results

def batch_words(words: List[str], batch_size: int = 500) -> List[List[str]]:
    """Split word list into batches for parallel processing."""
    return [words[i:i + batch_size] for i in range(0, len(words), batch_size)]

def crack_vigenere(ciphertext: str, alphabet_str: str, target_phrases: List[str], dictionary_path: str, 
                  specific_keys: List[str] = None) -> List[Dict]:
    """
    Attempt to crack Vigenere cipher by trying dictionary words as keys.
    Returns list of possible solutions that contain target phrases.
    """
    all_results = []
    
    # First try specific keys if provided
    if specific_keys:
        print(f"\n{YELLOW}Trying specific keys...{RESET}")
        for key in specific_keys:
            plaintext = try_key_directly(key, ciphertext, alphabet_str)
            if contains_all_phrases(plaintext, target_phrases):
                all_results.append({
                    'key': key,
                    'plaintext': plaintext,
                    'matched_phrases': target_phrases
                })
                print(f"Match found with key: {RED}{key}{RESET}")
    
    # Load dictionary words for brute force
    print(f"\n{YELLOW}Loading dictionary...{RESET}")
    dictionary = load_dictionary(dictionary_path, alphabet_str)
    print(f"Loaded {YELLOW}{len(dictionary)}{RESET} valid words from dictionary")
    
    if dictionary:
        # Create batches for parallel processing
        word_batches = batch_words(dictionary)
        num_processes = max(1, os.cpu_count() - 1)
        
        total_batches = len(word_batches)
        
        print(f"\n{YELLOW}Trying potential keys from dictionary...{RESET}")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_batch, (batch, ciphertext, target_phrases, alphabet_str))
                for batch in word_batches
            ]
            
            with tqdm(total=total_batches, desc="Processing batches", colour="yellow") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        print(f"{RED}Batch processing error: {e}{RESET}")
                        continue
        
        end_time = time.time()
        print(f"\n{YELLOW}Cracking complete in {end_time - start_time:.2f} seconds{RESET}")
    
    print(f"Found {RED}{len(all_results)}{RESET} possible solutions")
    return all_results

def save_results_to_file(results: List[Dict], filename: str):
    """Save results to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write("-" * 50 + "\n")
                f.write(f"Key: {result['key']}\n")
                f.write(f"Matched phrases: {', '.join(result['matched_phrases'])}\n")
                f.write(f"Plaintext: {result['plaintext']}\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"{RED}Error saving results to file: {e}{RESET}")

def analyze_frequency(text):
    """
    Analyze character frequency in the plaintext and display results.
    
    Args:
        text (str): The plaintext to analyze
    """
    print(f"\n{YELLOW}Frequency Analysis{RESET}")
    print(f"{GREY}-{RESET}" * 50)
    
    # Ensure text is uppercase for consistency
    text = text.upper()
    
    # Count letter frequencies
    letter_count = {}
    total_letters = 0
    
    for char in text:
        if char.isalpha():
            letter_count[char] = letter_count.get(char, 0) + 1
            total_letters += 1
    
    # Calculate frequencies and sort by frequency (descending)
    frequencies = [(char, count, count/total_letters*100) for char, count in letter_count.items()]
    frequencies.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"{'Character':<10}{'Count':<10}{'Frequency %':<15}{'Bar Chart'}")
    print(f"{GREY}-{RESET}" * 50)
    
    for char, count, percentage in frequencies:
        bar_length = int(percentage) * 2  # Scale for better visualization
        bar = "█" * bar_length
        print(f"{char:<10}{count:<10}{percentage:.2f}%{'':<10}{RED}{bar}{RESET}")
    
    # Add some statistical analysis
    print(f"{GREY}-{RESET}" * 50)
    print(f"Total letters analyzed: {YELLOW}{total_letters}{RESET}")
    
    # Compare with English language frequency
    english_freq = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 7.31, 'N': 6.95,
        'S': 6.28, 'R': 6.02, 'H': 5.92, 'D': 4.32, 'L': 3.98, 'U': 2.88,
        'C': 2.71, 'M': 2.61, 'F': 2.30, 'Y': 2.11, 'W': 2.09, 'G': 2.03,
        'P': 1.82, 'B': 1.49, 'V': 1.11, 'K': 0.69, 'X': 0.17, 'Q': 0.11,
        'J': 0.10, 'Z': 0.07
    }
    
    # Calculate deviation from English frequency
    print(f"\n{YELLOW}Deviation from Standard English{RESET}")
    print(f"{'Character':<10}{'Text %':<15}{'English %':<15}{'Deviation'}")
    print(f"{GREY}-{RESET}" * 50)
    
    # Convert frequencies to a dict for easier lookup
    text_freq = {char: percentage for char, _, percentage in frequencies}
    
    for char in sorted(english_freq.keys()):
        text_percentage = text_freq.get(char, 0)
        eng_percentage = english_freq[char]
        deviation = text_percentage - eng_percentage
        
        # Highlight significant deviations
        if abs(deviation) > 3:
            color = RED
        elif abs(deviation) > 1.5:
            color = YELLOW
        else:
            color = RESET
            
        print(f"{char:<10}{text_percentage:.2f}%{'':<10}{eng_percentage:.2f}%{'':<10}{color}{deviation:+.2f}%{RESET}")
    
    # Look for recurring patterns (potential key length indicators)
    print(f"\n{YELLOW}Common Bigrams and Trigrams{RESET}")
    
    # Analyze bigrams
    bigrams = {}
    for i in range(len(text) - 1):
        if text[i].isalpha() and text[i+1].isalpha():
            bigram = text[i:i+2]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
    
    # Analyze trigrams
    trigrams = {}
    for i in range(len(text) - 2):
        if text[i].isalpha() and text[i+1].isalpha() and text[i+2].isalpha():
            trigram = text[i:i+3]
            trigrams[trigram] = trigrams.get(trigram, 0) + 1
    
    # Show top bigrams
    top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"Top Bigrams: ", end="")
    print(", ".join([f"{RED}{b}{RESET}({c})" for b, c in top_bigrams]))
    
    # Show top trigrams
    top_trigrams = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"Top Trigrams: ", end="")
    print(", ".join([f"{RED}{t}{RESET}({c})" for t, c in top_trigrams]))
    
    print(f"\n{GREY}Analysis complete.{RESET}")

def run():
    print(f"""{GREY} 
██    ██ ██  ██████  ███████ ███    ██ ███████ ██████  ███████ 
██    ██ ██ ██       ██      ████   ██ ██      ██   ██ ██      
██    ██ ██ ██   ███ █████   ██ ██  ██ █████   ██████  █████   
 ██  ██  ██ ██    ██ ██      ██  ██ ██ ██      ██   ██ ██      
  ████   ██  ██████  ███████ ██   ████ ███████ ██   ██ ███████ 
                                                              {RESET}""")
    print(f"{RED}V{RESET}igenere {RED}B{RESET}rute {RED}F{RESET}orcer")
    print(f"{GREY}-{RESET}" * 50)
    
    use_test = input(f"Use test case? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
    
    if use_test:
        ciphertext = "EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJYQTQUXQBQVYUVLLTREVJYQTMKYRDMFD"
        alphabet_str = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
        target_phrases = ["BETWEEN", "SUBTLE"]
        expected_key = "PALIMPSEST"
        specific_keys = ["PALIMPSEST"]  # Try this key directly
        
        print(f"\n{GREY}----------------------")
        print(f"Running a test case...")
        print(f"Ciphertext: {ciphertext}")
        print(f"Target phrases: {', '.join(target_phrases)}")
        print(f"Expected key: {expected_key}")
        print(f"----------------------{RESET}")
    else:
        ciphertext = input("Enter ciphertext: ").upper()
        
        alphabet_input = input(f"Enter alphabet (press Enter for default {RED}ABCDEFGHIJKLMNOPQRSTUVWXYZ{RESET}): ").upper()
        alphabet_str = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        target_phrases_input = input(f"Enter known plaintext words/phrases (comma-separated): ").upper()
        target_phrases = [phrase.strip() for phrase in target_phrases_input.split(",")]
        
        specific_keys_input = input(f"Enter specific keys to try first (comma-separated, or press Enter to skip): ").upper()
        specific_keys = [key.strip() for key in specific_keys_input.split(",")] if specific_keys_input else []
    
    dictionary_path = "words_alpha.txt"
    
    # Run the cracking process
    results = crack_vigenere(ciphertext, alphabet_str, target_phrases, dictionary_path, specific_keys)
    
    # Display results
    if results:
        for i, result in enumerate(results):
            print(f"{GREY}-{RESET}" * 50)
            print(f"Solution #{i+1}:")
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"Matched phrases: {YELLOW}{', '.join(result['matched_phrases'])}{RESET}")
            
            highlighted = highlight_match(result['plaintext'], result['matched_phrases'])
            print(f"Plaintext: {highlighted}")
        
        # Check if test case was successful
        if use_test:
            test_passed = any(r['key'] == expected_key for r in results)
            print(f"\n{YELLOW}TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")
        
        # Option to save results
        save_filename = input("Enter filename to save results (or press Enter to skip): ")
        if save_filename:
            save_filename = save_filename + ".txt" if not save_filename.endswith(".txt") else save_filename
            save_results_to_file(results, save_filename)
        
        # Option to run frequency analysis
        if results:
            analyze_option = input(f"Run frequency analysis on best match? ({YELLOW}Y/N{RESET}): ").upper()
            if analyze_option == 'Y':
                analyze_frequency(results[0]['plaintext'])
    else:
        print(f"\n{RED}NO SOLUTIONS FOUND{RESET}")
        
    print(f"\n{GREY}Program complete.{RESET}")

if __name__ == "__main__":
    run()