from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
from functools import lru_cache
import os

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

    def save_results_to_file(self, results: List[Dict], filename: str, include_phrases: bool = True):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Vigenere Cipher Brute-Force Results\n")
                f.write("=====================================\n\n")
                for result in results:
                    f.write(f"Key: {result['key']}\n")
                    if 'ioc' in result:
                        f.write(f"IoC Score: {result['ioc']:.4f}\n")
                    if 'bigram_score' in result:
                        f.write(f"Bigram Score: {result['bigram_score']:.2f}\n")
                    if include_phrases and result.get('is_phrase_match'):
                        f.write(f"Matched Phrases: {', '.join(result['matched_phrases'])}\n")
                    f.write(f"Plaintext: {result['plaintext']}\n")
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
dictionary_path = os.path.join(data_dir, "words_alpha.txt")
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

# --- VIGENERE CORE FUNCTIONS ---

@lru_cache(maxsize=None)
def get_alphabet(alphabet_str: str):
    alphabet = alphabet_str.upper()
    alphabet_dict = {char: i for i, char in enumerate(alphabet)}
    return alphabet, alphabet_dict

def decrypt_vigenere(ciphertext: str, key: str, alphabet_str: str) -> str:
    alphabet, alphabet_dict = get_alphabet(alphabet_str)
    alphabet_length = len(alphabet)
    key_shifts = [alphabet_dict.get(k, 0) for k in key.upper() if k in alphabet_dict]
    if not key_shifts: return ciphertext
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

def contains_all_phrases(text: str, phrases: List[str]) -> bool:
    if not phrases: return False
    return all(phrase.upper() in text.upper().replace(" ", "") for phrase in phrases)

def highlight_phrases(text: str, phrases: List[str]) -> str:
    highlighted_text = text
    if not phrases: return highlighted_text
    for phrase in phrases:
        # As decrypted text is lowercase, we search for lowercase phrases
        phrase_lower = phrase.lower().replace(" ", "")
        highlighted_text = highlighted_text.replace(phrase_lower, f"{RED}{phrase_lower}{RESET}")
    return highlighted_text

def process_batch(args: Tuple) -> List[Dict]:
    word_batch, ciphertext, target_phrases, alphabet_str, expected_freqs = args
    results = []
    
    for word in word_batch:
        plaintext = decrypt_vigenere(ciphertext, word, alphabet_str)
        is_phrase_match = contains_all_phrases(plaintext, target_phrases)
        
        # We process every word for scores, as a good statistical match might be the key
        # even if it doesn't contain the specific crib words.
        ioc = utils.calculate_ioc(plaintext)
        bigram_score = calculate_bigram_score(plaintext, expected_freqs)
        
        # Add to results if it's either a phrase match or a good IoC match (will be filtered later)
        # This reduces redundant calculations
        results.append({
            'key': word,
            'plaintext': plaintext,
            'ioc': ioc,
            'bigram_score': bigram_score,
            'is_phrase_match': is_phrase_match,
            'matched_phrases': target_phrases if is_phrase_match else []
        })
    return results

def batch_words(words: List[str], batch_size: int = 500) -> List[List[str]]:
    return [words[i:i + batch_size] for i in range(0, len(words), batch_size)]

def crack_vigenere(ciphertext: str, alphabet_str: str, target_phrases: List[str], dictionary_path: str,
                   specific_keys: List[str], expected_freqs: Dict) -> List[Dict]:
    all_results = []
    
    if specific_keys:
        print(f"\n{YELLOW}Trying specific keys...{RESET}")
        for key in specific_keys:
            plaintext = decrypt_vigenere(ciphertext, key, alphabet_str)
            is_phrase_match = contains_all_phrases(plaintext, target_phrases)
            ioc = utils.calculate_ioc(plaintext)
            bigram_score = calculate_bigram_score(plaintext, expected_freqs)
            all_results.append({
                'key': key, 'plaintext': plaintext, 'ioc': ioc, 'bigram_score': bigram_score,
                'is_phrase_match': is_phrase_match, 'matched_phrases': target_phrases if is_phrase_match else []
            })
            if is_phrase_match: print(f"Phrase match found with specific key: {RED}{key}{RESET}")
            
    print(f"\n{YELLOW}Loading dictionary...{RESET}")
    dictionary = load_dictionary(dictionary_path, alphabet_str)
    print(f"Loaded {YELLOW}{len(dictionary)}{RESET} valid words from dictionary")
    
    if dictionary:
        word_batches = batch_words(dictionary)
        num_processes = max(1, os.cpu_count() - 1)
        total_batches = len(word_batches)
        
        print(f"\n{YELLOW}Trying {len(dictionary)} potential keys from dictionary...{RESET}")
        print(f"Processing {YELLOW}{total_batches}{RESET} batches with {YELLOW}{num_processes}{RESET} processes")
        start_time = time.time()
        processed_batches = 0
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_batch, (batch, ciphertext, target_phrases, alphabet_str, expected_freqs))
                for batch in word_batches
            ]
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    processed_batches += 1
                    progress_percent = (processed_batches / total_batches) * 100
                    if processed_batches % max(1, total_batches // 20) == 0 or processed_batches == total_batches:
                        print(f"Progress: {processed_batches}/{total_batches} batches ({progress_percent:.1f}%)", end='\r')
                except Exception as e:
                    print(f"{RED}Batch processing error: {e}{RESET}")
                    processed_batches += 1
                    continue
        print("\n")
        end_time = time.time()
        print(f"{YELLOW}Dictionary search complete in {end_time - start_time:.2f} seconds{RESET}")
    
    return all_results

def run():
    print(f"""{GREY} 
██    ██ ██  ██████  ███████ ███    ██ ███████ ██████  ███████ 
██    ██ ██ ██      ██      ████   ██ ██      ██   ██ ██      
██    ██ ██ ██  ███ █████   ██ ██  ██ █████   ██████  █████   
 ██  ██  ██ ██   ██ ██      ██  ██ ██ ██      ██   ██ ██      
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
        specific_keys = ["PALIMPSEST"]
        min_ioc = 0.065
        max_ioc = 0.07
        
        print(f"\n{GREY}--- Running Test Case ---{RESET}")
        print(f"Ciphertext: {ciphertext}")
        print(f"Target phrases: {', '.join(target_phrases)}")
        print(f"Expected key: {expected_key}")
        print(f"-------------------------{RESET}")
    else:
        ciphertext = input("Enter ciphertext: ").upper()
        alphabet_input = input(f"Enter alphabet (default: {RED}ABCDEFGHIJKLMNOPQRSTUVWXYZ{RESET}): ").upper()
        alphabet_str = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        target_phrases_input = input("Enter known plaintext words/phrases (comma-separated, or press Enter for none): ").upper()
        target_phrases = [phrase.strip() for phrase in target_phrases_input.split(",")] if target_phrases_input else []
        specific_keys_input = input("Enter specific keys to try first (comma-separated, or press Enter to skip): ").upper()
        specific_keys = [key.strip() for key in specific_keys_input.split(",")] if specific_keys_input else []
        
        min_ioc_str = input(f"Enter minimum IoC (default: {YELLOW}0.065{RESET}): ") or "0.065"
        max_ioc_str = input(f"Enter maximum IoC (default: {YELLOW}0.070{RESET}): ") or "0.070"
        min_ioc, max_ioc = float(min_ioc_str), float(max_ioc_str)

    # Load bigram frequencies once
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found. Scores will be less accurate.{RESET}")

    all_results = crack_vigenere(
        ciphertext, alphabet_str, target_phrases, dictionary_path, 
        specific_keys, expected_freqs
    )
    
    # Filter out results that are not a phrase match and not in IoC range
    filtered_results = [
        r for r in all_results 
        if r['is_phrase_match'] or (min_ioc <= r['ioc'] <= max_ioc)
    ]
    
    # Sort results based on the specified criteria
    filtered_results.sort(key=lambda x: (
        0 if x['is_phrase_match'] else 1,            # Priority 1: Phrase matches first
        0 if min_ioc <= x['ioc'] <= max_ioc else 1, # Priority 2: In IoC range
        x['bigram_score'],                          # Priority 3: Bigram score (lower is better)
        -x['ioc']                                   # Priority 4: IoC (higher is better)
    ))

    if filtered_results:
        print(f"\n{YELLOW}--- TOP 10 RANKED SOLUTIONS ---{RESET}")
        print(f"{GREY}Ranked by: 1. Phrase Match, 2. IoC in Range, 3. Bigram Score, 4. IoC Score{RESET}")
        
        for i, result in enumerate(filtered_results[:10]):
            is_in_range = min_ioc <= result['ioc'] <= max_ioc
            range_marker = GREEN + " (In Range)" + RESET if is_in_range else ""
            phrase_marker = YELLOW + " (Phrase Match)" + RESET if result['is_phrase_match'] else ""
            
            highlighted = highlight_phrases(result['plaintext'], result['matched_phrases'])

            print(f"{GREY}-{RESET}" * 50)
            print(f"Rank #{i+1}: Key: {YELLOW}{result['key']}{RESET}{phrase_marker}")
            print(f"Scores: IoC: {YELLOW}{result['ioc']:.4f}{RESET}{range_marker} | Bigram: {YELLOW}{result['bigram_score']:.2f}{RESET} (Lower is better)")
            print(f"Plaintext: {highlighted}")
        
        if use_test:
            test_passed = any(r['key'] == expected_key for r in filtered_results if r['is_phrase_match'])
            print(f"\n{YELLOW}PHRASE-MATCH TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")

        save_choice = input(f"\nSave all {len(filtered_results)} found results to file? ({YELLOW}Y/N{RESET}): ").upper()
        if save_choice == 'Y':
            filename = input("Enter filename for results: ") or "vigenere_results.txt"
            utils.save_results_to_file(filtered_results, filename)

        analyze_choice = input(f"Run frequency analysis on best match? ({YELLOW}Y/N{RESET}): ").upper()
        if analyze_choice == 'Y':
            utils.analyze_frequency_vg(filtered_results[0]['plaintext'])
    else:
        print(f"\n{RED}NO SOLUTIONS FOUND{RESET}")
        
    print(f"\n{GREY}Program complete.{RESET}")

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file at '{bigram_freq_path}'...{RESET}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\n")
    run()