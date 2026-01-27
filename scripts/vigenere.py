from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import time
from functools import lru_cache
import os
import random
import math
from utils.utils import get_input_ciphertexts  # Imported from your utils

# --- DUMMY UTILS CLASS (Maintained for standalone scoring compatibility if needed) ---
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
                        f.write(f"IoC Score: {result.get('ioc', 0):.4f}\n")
                    if 'bigram_score' in result:
                        f.write(f"Bigram Score: {result.get('bigram_score', 0):.2f}\n")
                    if include_phrases and result.get('is_phrase_match'):
                        f.write(f"Matched Phrases: {', '.join(result.get('matched_phrases', []))}\n")
                    f.write(f"Plaintext: {result['plaintext']}\n")
                    f.write("-" * 20 + "\n")
            print(f"{GREEN}Results successfully saved to {filename}{RESET}")
        except IOError as e:
            print(f"{RED}Error saving file: {e}{RESET}")

utils = DummyUtils()

# --- COLOR CODES AND PATHS ---
RESET = '\033[0m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
BLUE = '\033[34m'
CYAN = '\033[36m'
WHITE = '\033[37m'
GREY = '\033[90m'

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
        char_upper = char.upper()
        if char_upper in alphabet_dict:
            char_index = alphabet_dict[char_upper]
            shift = key_shifts[key_index % len(key_shifts)]
            decrypted_char = alphabet[(char_index - shift + alphabet_length) % alphabet_length]
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
    highlighted_text = text.lower()
    if not phrases: return highlighted_text
    for phrase in phrases:
        phrase_lower = phrase.lower().replace(" ", "")
        highlighted_text = highlighted_text.replace(phrase_lower, f"{RED}{phrase_lower}{RESET}")
    return highlighted_text

# --- ATTACK FUNCTIONS ---

def process_batch(args: Tuple) -> List[Dict]:
    word_batch, ciphertext, target_phrases, alphabet_str, expected_freqs = args
    results = []
    for word in word_batch:
        plaintext = decrypt_vigenere(ciphertext, word, alphabet_str)
        is_phrase_match = contains_all_phrases(plaintext, target_phrases)
        ioc = utils.calculate_ioc(plaintext)
        bigram_score = calculate_bigram_score(plaintext, expected_freqs)
        results.append({
            'key': word, 'plaintext': plaintext, 'ioc': ioc, 'bigram_score': bigram_score,
            'is_phrase_match': is_phrase_match,
            'matched_phrases': target_phrases if is_phrase_match else []
        })
    return results

def batch_words(words: List[str], batch_size: int = 500) -> List[List[str]]:
    return [words[i:i + batch_size] for i in range(0, len(words), batch_size)]

def run_dictionary_attack(ciphertext: str, alphabet_str: str, expected_freqs: Dict, config: Dict):
    """
    Executes dictionary attack using configuration passed from main run loop.
    """
    # Unpack config
    target_phrases = config.get('target_phrases', [])
    specific_keys = config.get('specific_keys', [])
    min_ioc = config.get('min_ioc', 0.065)
    max_ioc = config.get('max_ioc', 0.070)
    
    all_results = []
    
    # 1. Try Specific Keys first
    if specific_keys:
        print(f"\n{YELLOW}Checking specific keys...{RESET}")
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
            
    # 2. Load Dictionary
    print(f"{YELLOW}Loading dictionary...{RESET}")
    dictionary = load_dictionary(dictionary_path, alphabet_str)
    
    if dictionary:
        word_batches = batch_words(dictionary)
        num_processes = max(1, os.cpu_count() - 1)
        total_batches = len(word_batches)
        
        print(f"Processing {YELLOW}{len(dictionary):,}{RESET} words with {YELLOW}{num_processes}{RESET} processes...")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_batch, (b, ciphertext, target_phrases, alphabet_str, expected_freqs)) for b in word_batches]
            for i, future in enumerate(as_completed(futures)):
                all_results.extend(future.result())
                if total_batches > 10: # Only print update if meaningful
                    print(f"Progress: {(i + 1) / total_batches:.1%}", end='\r')
        print(f"\nDictionary search complete in {time.time() - start_time:.2f} seconds.{RESET}")
    
    # 3. Filter and Rank
    filtered_results = [r for r in all_results if r['is_phrase_match'] or (min_ioc <= r['ioc'] <= max_ioc)]
    
    filtered_results.sort(key=lambda x: (
        0 if x['is_phrase_match'] else 1, 
        0 if min_ioc <= x['ioc'] <= max_ioc else 1,
        x['bigram_score'], 
        -x['ioc']
    ))

    # 4. Display
    if filtered_results:
        print(f"\n{YELLOW}--- TOP 5 MATCHES ---{RESET}")
        for i, result in enumerate(filtered_results[:5]):
            range_marker = GREEN + " (IoC Match)" + RESET if min_ioc <= result['ioc'] <= max_ioc else ""
            phrase_marker = YELLOW + " (Phrase Match)" + RESET if result['is_phrase_match'] else ""
            highlighted = highlight_phrases(result['plaintext'], result['matched_phrases'])
            print(f"Rank #{i+1}: Key: {YELLOW}{result['key']}{RESET}{phrase_marker}")
            print(f"Scores: IoC: {YELLOW}{result['ioc']:.4f}{RESET}{range_marker} | Bigram: {YELLOW}{result['bigram_score']:.2f}{RESET}")
            print(f"Text: {highlighted[:100]}...") # Truncate for display
            print(f"{GREY}-{'':-^30}{RESET}")
    else:
        print(f"{RED}NO SOLUTIONS FOUND in specified IoC range.{RESET}")
        
    return filtered_results

def generate_neighbor_key(key: str, alphabet: str) -> str:
    key_list = list(key)
    pos = random.randint(0, len(key_list) - 1)
    key_list[pos] = random.choice(alphabet)
    return "".join(key_list)

def run_simulated_annealing_attack(ciphertext: str, alphabet_str: str, expected_freqs: Dict, config: Dict):
    """
    Executes SA attack using configuration passed from main run loop.
    """
    key_length = config.get('key_length', 5)
    iterations = config.get('iterations', 200000)
    initial_temp = 1000.0

    print(f"\n{BLUE}--- Simulated Annealing (Key Len: {key_length}) ---{RESET}")
    
    alphabet, _ = get_alphabet(alphabet_str)
    
    # Initial State
    current_key = "".join(random.choice(alphabet) for _ in range(key_length))
    plaintext = decrypt_vigenere(ciphertext, current_key, alphabet_str)
    current_score = calculate_bigram_score(plaintext, expected_freqs)

    best_key, best_score = current_key, current_score
    
    try:
        for i in range(iterations):
            temp = initial_temp * (1.0 - (i + 1) / iterations)
            if temp <= 0: break

            neighbor_key = generate_neighbor_key(current_key, alphabet)
            plaintext = decrypt_vigenere(ciphertext, neighbor_key, alphabet_str)
            neighbor_score = calculate_bigram_score(plaintext, expected_freqs)
            
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_key, current_score = neighbor_key, neighbor_score
                if current_score < best_score:
                    best_key, best_score = current_key, current_score
                    # Print update only when improvement found (to avoid console spam in batch)
                    best_text = decrypt_vigenere(ciphertext, best_key, alphabet_str).upper()
                    print(f"Score: {best_score:8.2f} | Key: {best_key} | Text: {best_text[:30]}...", end='\r')

    except KeyboardInterrupt:
        print("\nSkipping to next result...")
    
    print("\n" + "="*60)
    best_plaintext = decrypt_vigenere(ciphertext, best_key, alphabet_str).upper()
    print(f"Final Best Key: {YELLOW}{best_key}{RESET}")
    print(f"Bigram Score: {YELLOW}{best_score:.2f}{RESET}")
    print(f"IoC: {YELLOW}{utils.calculate_ioc(best_plaintext):.4f}{RESET}")
    print(f"Text: {best_plaintext}")
    
    return [{'key': best_key, 'plaintext': best_plaintext, 'bigram_score': best_score}]

def run_direct_decrypt(ciphertext: str, alphabet_str: str, config: Dict):
    key = config.get('key', '')
    if not key:
        print(f"{RED}Key not provided.{RESET}")
        return
    
    plaintext = decrypt_vigenere(ciphertext, key, alphabet_str)
    print(f"\nKey: {YELLOW}{key}{RESET}")
    print(f"Plaintext: {plaintext}")

# --- MAIN APPLICATION ---

def run():
    print(f"{GREY}================================{RESET}")
    print(f"{RED}VIGENERE SOLVER{RESET}")
    print(f"{GREY}================================{RESET}")
    
    # 1. Get Inputs (Manual or File)
    ciphertexts = get_input_ciphertexts(prompt="Enter ciphertext")
    if not ciphertexts:
        print(f"{RED}No input provided.{RESET}")
        return

    # 2. Get Alphabet (Global Setting)
    alphabet_input = input(f"Enter alphabet (default: {RED}A-Z{RESET}): ").upper()
    alphabet_str = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # 3. Load Data
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found.{RESET}")

    # 4. Select Mode (Global Setting)
    print(f"\n{GREY}Select Attack Mode for {len(ciphertexts)} input(s):{RESET}")
    print(f"  ({YELLOW}1{RESET}) Direct Decryption (Known Key)")
    print(f"  ({YELLOW}2{RESET}) Dictionary Attack (Key is an English word)")
    print(f"  ({YELLOW}3{RESET}) Simulated Annealing (Unknown Key of known length)")
    mode = input(">> ").strip()

    # 5. Gather Configuration ONCE
    config = {}
    
    if mode == '1':
        config['key'] = input("Enter the decryption key: ").upper().strip()
        
    elif mode == '2':
        # Config for Dictionary Attack
        phrases_in = input("Enter required plaintext words/phrases (comma-separated, optional): ").upper()
        config['target_phrases'] = [p.strip() for p in phrases_in.split(",")] if phrases_in else []
        
        keys_in = input("Enter specific keys to prioritize (comma-separated, optional): ").upper()
        config['specific_keys'] = [k.strip() for k in keys_in.split(",")] if keys_in else []
        
        min_ioc = input(f"Enter min IoC (default: {YELLOW}0.065{RESET}): ")
        config['min_ioc'] = float(min_ioc) if min_ioc else 0.065
        
        max_ioc = input(f"Enter max IoC (default: {YELLOW}0.070{RESET}): ")
        config['max_ioc'] = float(max_ioc) if max_ioc else 0.070
        
    elif mode == '3':
        # Config for Annealing
        try:
            kl = input("Enter exact key length: ")
            config['key_length'] = int(kl)
            iters = input(f"Enter iterations (default: {YELLOW}200,000{RESET}): ")
            config['iterations'] = int(iters) if iters else 200000
        except ValueError:
            print(f"{RED}Invalid integer input.{RESET}")
            return
            
    else:
        print(f"{RED}Invalid mode selected.{RESET}")
        return

    # 6. Execute Batch
    print(f"\n{BLUE}=== Starting Batch Processing ({len(ciphertexts)} items) ==={RESET}")
    
    for i, ciphertext in enumerate(ciphertexts):
        # Display Header
        display_sample = ciphertext.replace("\n", "")[:40]
        print(f"\n{GREY}Input #{i+1}: {display_sample}...{RESET}")
        
        # Run selected mode
        if mode == '1':
            run_direct_decrypt(ciphertext, alphabet_str, config)
        elif mode == '2':
            results = run_dictionary_attack(ciphertext, alphabet_str, expected_freqs, config)
            # Optional: Save individual results logic could be added here
        elif mode == '3':
            run_simulated_annealing_attack(ciphertext, alphabet_str, expected_freqs, config)
            
    print(f"\n{GREY}Batch processing complete.{RESET}")
    input(f"{YELLOW}Press Enter to return to menu...{RESET}")

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file...{RESET}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\n")
    run()