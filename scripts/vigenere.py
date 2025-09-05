from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
from functools import lru_cache
import os
import random
import math

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

# --- DICTIONARY ATTACK FUNCTIONS ---

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

def run_dictionary_attack(ciphertext: str, alphabet_str: str, expected_freqs: Dict):
    print(f"\n{BLUE}--- Dictionary Attack ---{RESET}")
    target_phrases_input = input("Enter known plaintext words/phrases (comma-separated, or press Enter for none): ").upper()
    target_phrases = [phrase.strip() for phrase in target_phrases_input.split(",")] if target_phrases_input else []
    specific_keys_input = input("Enter specific keys to try first (comma-separated, or press Enter to skip): ").upper()
    specific_keys = [key.strip() for key in specific_keys_input.split(",")] if specific_keys_input else []
    
    min_ioc_str = input(f"Enter minimum IoC (default: {YELLOW}0.065{RESET}): ") or "0.065"
    max_ioc_str = input(f"Enter maximum IoC (default: {YELLOW}0.070{RESET}): ") or "0.070"
    min_ioc, max_ioc = float(min_ioc_str), float(max_ioc_str)

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
        
        print(f"\n{YELLOW}Trying {len(dictionary):,} potential keys...{RESET}")
        print(f"Processing with {YELLOW}{num_processes}{RESET} processes...")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_batch, (b, ciphertext, target_phrases, alphabet_str, expected_freqs)) for b in word_batches]
            for i, future in enumerate(as_completed(futures)):
                all_results.extend(future.result())
                print(f"Progress: {(i + 1) / total_batches:.1%}", end='\r')
        print(f"\nDictionary search complete in {time.time() - start_time:.2f} seconds.{RESET}")
    
    filtered_results = [r for r in all_results if r['is_phrase_match'] or (min_ioc <= r['ioc'] <= max_ioc)]
    
    filtered_results.sort(key=lambda x: (
        0 if x['is_phrase_match'] else 1, 0 if min_ioc <= x['ioc'] <= max_ioc else 1,
        x['bigram_score'], -x['ioc']
    ))

    if filtered_results:
        print(f"\n{YELLOW}--- TOP 10 RANKED SOLUTIONS ---{RESET}")
        for i, result in enumerate(filtered_results[:10]):
            range_marker = GREEN + " (In Range)" + RESET if min_ioc <= result['ioc'] <= max_ioc else ""
            phrase_marker = YELLOW + " (Phrase Match)" + RESET if result['is_phrase_match'] else ""
            highlighted = highlight_phrases(result['plaintext'], result['matched_phrases'])
            print(f"{GREY}-{'':-^50}{RESET}")
            print(f"Rank #{i+1}: Key: {YELLOW}{result['key']}{RESET}{phrase_marker}")
            print(f"Scores: IoC: {YELLOW}{result['ioc']:.4f}{RESET}{range_marker} | Bigram: {YELLOW}{result['bigram_score']:.2f}{RESET}")
            print(f"Plaintext: {highlighted}")
        
        if input(f"\nSave all {len(filtered_results)} results? ({YELLOW}Y/N{RESET}): ").upper() == 'Y':
            fname = input("Enter filename: ") or "vigenere_results.txt"
            utils.save_results_to_file(filtered_results, fname)
    else:
        print(f"\n{RED}NO SOLUTIONS FOUND{RESET}")

# --- SIMULATED ANNEALING ATTACK FUNCTIONS ---

def generate_neighbor_key(key: str, alphabet: str) -> str:
    """Creates a neighbor key by slightly modifying the current key."""
    key_list = list(key)
    pos = random.randint(0, len(key_list) - 1)
    key_list[pos] = random.choice(alphabet)
    return "".join(key_list)

def run_simulated_annealing_attack(ciphertext: str, alphabet_str: str, expected_freqs: Dict):
    """Attempts to find a non-dictionary key using simulated annealing."""
    print(f"\n{BLUE}--- Simulated Annealing Attack ---{RESET}")
    try:
        key_length = int(input("Enter the exact key length to search for: "))
        if key_length <= 0: raise ValueError
        iterations_str = input(f"Enter number of iterations (default: {YELLOW}200,000{RESET}): ")
        iterations = int(iterations_str) if iterations_str else 200_000
        if iterations <= 0: raise ValueError
    except ValueError:
        print(f"{RED}Invalid input. Please enter positive integers.{RESET}")
        return

    # Annealing Parameters
    initial_temp = 1000.0

    print(f"Running {YELLOW}{iterations:,}{RESET} iterations for a key of length {YELLOW}{key_length}{RESET}...")
    
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

            if (i + 1) % 1000 == 0:
                print(f"Progress: {((i + 1) / iterations):6.1%} | Best Score: {best_score:10.2f} | Key: {best_key}", end='\r')
        
        print("\n" + "="*80)
        print(f"\n{YELLOW}--- FINAL BEST RESULT ---{RESET}")
        best_plaintext = decrypt_vigenere(ciphertext, best_key, alphabet_str)
        print(f"Key: {YELLOW}{best_key}{RESET}")
        print(f"Bigram Score: {YELLOW}{best_score:.2f}{RESET} (Lower is better)")
        print(f"IoC: {YELLOW}{utils.calculate_ioc(best_plaintext):.4f}{RESET}")
        print(f"Plaintext: {best_plaintext.lower()}")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")

# --- MAIN APPLICATION ---

def run_direct_decrypt(ciphertext: str, alphabet_str: str):
    print(f"\n{BLUE}--- Direct Decryption ---{RESET}")
    key = input("Enter the decryption key: ").upper()
    if not key:
        print(f"{RED}Key cannot be empty.{RESET}")
        return
    
    plaintext = decrypt_vigenere(ciphertext, key, alphabet_str)
    print(f"\nKey: {YELLOW}{key}{RESET}")
    print(f"Plaintext: {plaintext}")

def run():
    print(f"{RED}V{RESET}igenere {RED}B{RESET}rute {RED}F{RESET}orcer")
    
    ciphertext = input(f"\nEnter ciphertext: ").upper()
    alphabet_input = input(f"Enter alphabet (default: {RED}A-Z{RESET}): ").upper()
    alphabet_str = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found. Scoring will be less accurate.{RESET}")
        return

    while True:
        print(f"\n{GREY}{'-'*50}{RESET}")
        print("Select an attack mode:")
        print(f"  ({YELLOW}1{RESET}) Direct Decryption (Known Key)")
        print(f"  ({YELLOW}2{RESET}) Dictionary Attack (Key is an English word)")
        print(f"  ({YELLOW}3{RESET}) Simulated Annealing (Unknown Key of a known length)")
        print(f"  ({YELLOW}4{RESET}) Exit")
        choice = input(">> ")

        if choice == '1':
            run_direct_decrypt(ciphertext, alphabet_str)
        elif choice == '2':
            run_dictionary_attack(ciphertext, alphabet_str, expected_freqs)
        elif choice == '3':
            run_simulated_annealing_attack(ciphertext, alphabet_str, expected_freqs)
        elif choice == '4':
            break
        else:
            print(f"{RED}Invalid choice.{RESET}")

    print(f"\n{GREY}Program complete.{RESET}")


if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file at '{bigram_freq_path}'...{RESET}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\n")
    run()

