import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Dict
from functools import lru_cache

# --- DUMMY UTILS CLASS (to make script standalone) ---
class DummyUtils:
    def calculate_ioc(self, text: str) -> float:
        text = text.upper()
        text = ''.join(filter(str.isalpha, text))
        n = len(text)
        if n < 2: return 0.0
        freqs = {}
        for char in text:
            freqs[char] = freqs.get(char, 0) + 1
        numerator = sum(count * (count - 1) for count in freqs.values())
        denominator = n * (n - 1)
        return numerator / denominator if denominator > 0 else 0.0

    def save_results_to_file(self, results: List[Dict], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Gronsfeld Cipher Brute-Force Results\n")
                f.write("=====================================\n\n")
                for result in results:
                    f.write(f"Key: {result['key']}\n")
                    f.write(f"IoC Score: {result.get('ioc', 0):.4f}\n")
                    f.write(f"Bigram Score: {result.get('bigram_score', 0):.2f}\n")
                    f.write(f"Matched Phrases: {', '.join(result['matched_phrases'])}\n")
                    f.write(f"Decrypted: {result['decrypted']}\n")
                    f.write("-" * 20 + "\n")
            print(f"{GREEN}Results successfully saved to {filename}{RESET}")
        except IOError as e:
            print(f"{RED}Error saving file: {e}{RESET}")

    def analyze_frequency_vg(self, text: str):
        print("\n--- Frequency Analysis ---")
        freqs = {}
        text = ''.join(filter(str.isalpha, text.upper()))
        total = len(text)
        if total == 0:
            print("No alphabetic characters to analyze.")
            return
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            count = text.count(char)
            freqs[char] = count
        sorted_freqs = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
        print("Character Frequencies:")
        for char, count in sorted_freqs:
            percentage = (count / total) * 100
            print(f"{char}: {count:<4} ({percentage:.2f}%)")
        print("-" * 25)

utils = DummyUtils()

# --- COLOR CODES AND PATHS ---
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
GREEN = '\033[38;5;2m'
RESET = '\033[0m'

data_dir = os.path.join(os.path.dirname(__file__), "data")
bigram_freq_path = os.path.join(data_dir, "english_bigrams.txt")

# --- SCORING FUNCTIONS ---

@lru_cache(maxsize=None)
def load_bigram_frequencies(file_path: str) -> Dict[str, float]:
    freqs = {}
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    freqs[parts[0].upper()] = float(parts[1])
        total = sum(freqs.values())
        if total > 0:
            for bigram in freqs:
                freqs[bigram] /= total
        return freqs
    except FileNotFoundError:
        return {}

def calculate_bigram_score(text: str, expected_freqs: Dict[str, float]) -> float:
    if not expected_freqs: return float('inf')
    text = "".join(char for char in text.upper() if char.isalpha())
    if len(text) < 2: return float('inf')
    observed_counts = {}
    total_bigrams = 0
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        observed_counts[bigram] = observed_counts.get(bigram, 0) + 1
        total_bigrams += 1
    if total_bigrams == 0: return float('inf')
    chi_squared_score = 0
    floor = 0.01
    for bigram, expected_prob in expected_freqs.items():
        observed_count = observed_counts.get(bigram, 0)
        expected_count = expected_prob * total_bigrams
        difference = observed_count - expected_count
        chi_squared_score += (difference * difference) / max(expected_count, floor)
    return chi_squared_score

# --- GRONSFELD CORE FUNCTIONS ---

def highlight_phrases(text: str, phrases: list) -> str:
    """Highlight all matched phrases in the plaintext."""
    highlighted_text = text
    for phrase in phrases:
        # Decrypted text is lowercase, so we match lowercase phrases
        phrase_lower = phrase.lower()
        highlighted_text = highlighted_text.replace(phrase_lower, f"{RED}{phrase_lower}{RESET}")
    return highlighted_text

def gronsfeld_decrypt(ciphertext, key, alphabet):
    """Decrypt text using Gronsfeld cipher"""
    decrypted_text = []
    key_length = len(key)
    alphabet_length = len(alphabet)
    
    for i, char in enumerate(ciphertext):
        if char in alphabet:
            char_index = alphabet.index(char)
            shift = int(key[i % key_length])
            decrypted_index = (char_index - shift) % alphabet_length
            decrypted_char = alphabet[decrypted_index]
            decrypted_text.append(decrypted_char.lower())
        else:
            decrypted_text.append(char.lower())
    return ''.join(decrypted_text)

def try_decrypt(args):
    """Try decryption with a given key batch and score the results."""
    ciphertext, alphabet, keys, known_plaintexts, min_ioc, max_ioc, expected_freqs = args
    results = []
    
    for key in keys:
        try:
            decrypted = gronsfeld_decrypt(ciphertext, key, alphabet)
            
            # Primary filter: must contain at least one known plaintext word
            matched_phrases = [word for word in known_plaintexts if word.lower() in decrypted.lower()]
            if matched_phrases:
                ioc = utils.calculate_ioc(decrypted)
                bigram_score = calculate_bigram_score(decrypted, expected_freqs)
                
                results.append({
                    'key': key,
                    'decrypted': decrypted,
                    'ioc': ioc,
                    'bigram_score': bigram_score,
                    'matched_phrases': matched_phrases
                })
        except (ValueError, IndexError):
            continue
    
    return results

def run():
    print(f"{RED}Gronsfeld {RED}Cipher")
    print(f"{GREY}-{RESET}" * 50)
    
    # Get ciphertext, convert to uppercase, and remove spaces
    ciphertext = input("Enter the ciphertext: ").upper().replace(" ", "")
    
    alphabet_input = input(f"Enter custom alphabet (default: {RED}A-Z{RESET}): ").upper()
    alphabet = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"Using alphabet: {RED}{alphabet}{RESET}")
    
    key_length = int(input("Enter the key length: "))
    
    known_text = input(f"Enter known plaintext words (comma-separated, default: {RED}THE, AND, THAT{RESET}): ").upper()
    known_plaintexts = [word.strip() for word in known_text.split(',')] if known_text.strip() else ["THE", "AND", "THAT"]
    print(f"Using known plaintext words: {RED}{', '.join(known_plaintexts)}{RESET}")
    
    min_ioc_str = input(f"Enter minimum IoC (default: {YELLOW}0.062{RESET}): ") or "0.062"
    max_ioc_str = input(f"Enter maximum IoC (default: {YELLOW}0.071{RESET}): ") or "0.071"
    min_ioc, max_ioc = float(min_ioc_str), float(max_ioc_str)
    print(f"Using IoC range: {YELLOW}{min_ioc}-{max_ioc}{RESET}")

    # Load bigram frequencies once
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found. Scores will be less accurate.{RESET}")
    
    batch_size = 1000
    total_combinations = 10 ** key_length
    total_batches = (total_combinations + batch_size - 1) // batch_size
    
    print(f"\n{YELLOW}Trying all {total_combinations} possible {key_length}-digit keys...{RESET}")
    print(f"Processing {YELLOW}{total_batches}{RESET} batches of up to {YELLOW}{batch_size}{RESET} keys each...")
    
    results = []
    processed_keys = 0
    with ProcessPoolExecutor() as executor:
        futures = []
        for start in range(0, total_combinations, batch_size):
            end = min(start + batch_size, total_combinations)
            batch = [str(i).zfill(key_length) for i in range(start, end)]
            args = (ciphertext, alphabet, batch, known_plaintexts, min_ioc, max_ioc, expected_freqs)
            futures.append(executor.submit(try_decrypt, args))
        
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                processed_keys += batch_size
                progress_percent = (min(processed_keys, total_combinations) / total_combinations) * 100
                print(f"Progress: {progress_percent:.1f}%", end='\r')
            except Exception as e:
                print(f"\n{RED}Error processing batch: {str(e)}{RESET}")
    
    print(f"\n\n{YELLOW}Processing complete!{RESET}")
    
    if results:
        # Sort results based on the specified criteria
        results.sort(key=lambda x: (
            0 if min_ioc <= x['ioc'] <= max_ioc else 1, # Priority 1: In IoC range
            x['bigram_score'],                          # Priority 2: Bigram score (lower is better)
            -x['ioc']                                   # Priority 3: IoC (higher is better)
        ))
        
        print(f"\n{YELLOW}--- RANKED SOLUTIONS FOUND ---{RESET}")
        print(f"{GREY}Ranked by: 1. IoC in Range, 2. Bigram Score, 3. IoC Score{RESET}")
        
        for i, result in enumerate(results):
            is_in_range = min_ioc <= result['ioc'] <= max_ioc
            range_marker = GREEN + " (In Range)" + RESET if is_in_range else ""
            
            highlighted = highlight_phrases(result['decrypted'], result['matched_phrases'])

            print(f"{GREY}-{RESET}" * 50)
            print(f"Rank #{i+1}: Key: {YELLOW}{result['key']}{RESET}")
            print(f"Scores: IoC: {YELLOW}{result['ioc']:.4f}{RESET}{range_marker} | Bigram: {YELLOW}{result['bigram_score']:.2f}{RESET} (Lower is better)")
            print(f"Matched phrases: {YELLOW}{', '.join(result['matched_phrases'])}{RESET}")
            print(f"Decrypted text: {highlighted}")
        
        save_results = input(f"\nSave results to file? ({YELLOW}Y/N{RESET}): ").upper()
        if save_results == 'Y':
            filename = input("Enter filename for results: ") or "gronsfeld_results.txt"
            utils.save_results_to_file(results, filename)
        
        analyze_option = input(f"Run frequency analysis on best match? ({YELLOW}Y/N{RESET}): ").upper()
        if analyze_option == 'Y':
            utils.analyze_frequency_vg(results[0]['decrypted'])
    else:
        print(f"\n{RED}NO SOLUTIONS FOUND WITH GIVEN PARAMETERS{RESET}")
    
    print(f"\n{GREY}Program complete.{RESET}")

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file at '{bigram_freq_path}'...{RESET}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\n")
    run()