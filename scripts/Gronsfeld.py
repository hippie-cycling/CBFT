import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Dict, Tuple
from functools import lru_cache
import random
import math
import time

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
                    if result.get('matched_phrases'):
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
RESET = '\033[0m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
BLUE = '\033[34m'
CYAN = '\033[36m'
WHITE = '\033[37m'
GREY = '\033[90m'

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
    highlighted_text = text.lower()
    for phrase in phrases:
        phrase_lower = phrase.lower()
        highlighted_text = highlighted_text.replace(phrase_lower, f"{RED}{phrase_lower}{RESET}")
    return highlighted_text

def gronsfeld_decrypt(ciphertext: str, key: str, alphabet: str) -> str:
    """Decrypt text using Gronsfeld cipher"""
    decrypted_text = []
    key_length = len(key)
    alphabet_length = len(alphabet)
    
    key_digits = [int(d) for d in key]

    for i, char in enumerate(ciphertext):
        char_upper = char.upper()
        if char_upper in alphabet:
            char_index = alphabet.index(char_upper)
            shift = key_digits[i % key_length]
            decrypted_index = (char_index - shift + alphabet_length) % alphabet_length
            decrypted_char = alphabet[decrypted_index]
            decrypted_text.append(decrypted_char.lower())
        else:
            decrypted_text.append(char.lower())
    return ''.join(decrypted_text)

# --- EXHAUSTIVE ATTACK FUNCTIONS ---

def process_exhaustive_batch(args):
    """Worker function for exhaustive attack: try a batch of keys and score them."""
    ciphertext, alphabet, keys_batch, expected_freqs = args
    results = []
    
    for key in keys_batch:
        try:
            decrypted = gronsfeld_decrypt(ciphertext, key, alphabet)
            ioc = utils.calculate_ioc(decrypted)
            bigram_score = calculate_bigram_score(decrypted, expected_freqs)
            results.append({
                'key': key, 'decrypted': decrypted,
                'ioc': ioc, 'bigram_score': bigram_score
            })
        except (ValueError, IndexError):
            continue
    return results

def run_exhaustive_attack(ciphertext: str, alphabet: str, expected_freqs: Dict):
    print(f"\n{BLUE}--- Exhaustive Brute-Force Attack ---{RESET}")
    try:
        key_length = int(input("Enter the exact key length to test (e.g., 5): "))
        if key_length <= 0 or key_length > 8:
            print(f"{RED}Key length must be between 1 and 8 for practical performance.{RESET}")
            return
    except ValueError:
        print(f"{RED}Invalid key length.{RESET}")
        return

    known_text = input(f"Enter known plaintext words for sorting/highlighting (optional, comma-separated): ").upper()
    known_plaintexts = [word.strip() for word in known_text.split(',')] if known_text.strip() else []

    total_combinations = 10 ** key_length
    print(f"\n{YELLOW}Trying all {total_combinations:,} possible {key_length}-digit keys...{RESET}")
    
    batch_size = max(1000, total_combinations // 100)
    num_processes = max(1, os.cpu_count() - 1)
    print(f"Processing with {num_processes} processes...")

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for start in range(0, total_combinations, batch_size):
            end = min(start + batch_size, total_combinations)
            batch = [str(i).zfill(key_length) for i in range(start, end)]
            args = (ciphertext, alphabet, batch, expected_freqs)
            futures.append(executor.submit(process_exhaustive_batch, args))
        
        for i, future in enumerate(as_completed(futures)):
            results.extend(future.result())
            print(f"Progress: {((i + 1) * batch_size) / total_combinations:.1%}", end='\r')
    
    print(f"\n\n{YELLOW}Processing complete!{RESET}")
    
    if results:
        for r in results:
            r['matched_phrases'] = [w for w in known_plaintexts if w.lower() in r['decrypted']]
        
        results.sort(key=lambda x: (-(len(x['matched_phrases'])), x['bigram_score'], -x['ioc']))
        
        print(f"\n{YELLOW}--- TOP 10 RANKED SOLUTIONS ---{RESET}")
        for i, result in enumerate(results[:10]):
            phrase_marker = f" {GREEN}({len(result['matched_phrases'])} words matched){RESET}" if result['matched_phrases'] else ""
            highlighted = highlight_phrases(result['decrypted'], result['matched_phrases'])
            print(f"{GREY}-{'':-^50}{RESET}")
            print(f"Rank #{i+1}: Key: {YELLOW}{result['key']}{RESET}{phrase_marker}")
            print(f"Scores: Bigram: {YELLOW}{result['bigram_score']:.2f}{RESET} | IoC: {YELLOW}{result['ioc']:.4f}{RESET}")
            print(f"Decrypted text: {highlighted}")
        
        if input(f"\nSave all {len(results)} results? ({YELLOW}Y/N{RESET}): ").upper() == 'Y':
            fname = input("Enter filename: ") or "gronsfeld_results.txt"
            utils.save_results_to_file(results, fname)
    else:
        print(f"\n{RED}NO SOLUTIONS FOUND{RESET}")

# --- SIMULATED ANNEALING ATTACK ---

def generate_neighbor_numeric_key(key: str) -> str:
    """Creates a neighbor key by changing one digit."""
    key_list = list(key)
    pos = random.randint(0, len(key_list) - 1)
    key_list[pos] = str(random.randint(0, 9))
    return "".join(key_list)

def run_simulated_annealing_attack(ciphertext: str, alphabet: str, expected_freqs: Dict):
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

    initial_temp = 1000.0
    print(f"Running {YELLOW}{iterations:,}{RESET} iterations for a key of length {YELLOW}{key_length}{RESET}...")
    
    current_key = "".join(str(random.randint(0, 9)) for _ in range(key_length))
    current_score = calculate_bigram_score(gronsfeld_decrypt(ciphertext, current_key, alphabet), expected_freqs)
    best_key, best_score = current_key, current_score
    
    try:
        for i in range(iterations):
            temp = initial_temp * (1.0 - (i + 1) / iterations)
            if temp <= 0: break
            
            neighbor_key = generate_neighbor_numeric_key(current_key)
            neighbor_score = calculate_bigram_score(gronsfeld_decrypt(ciphertext, neighbor_key, alphabet), expected_freqs)
            
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_key, current_score = neighbor_key, neighbor_score
                if current_score < best_score:
                    best_key, best_score = current_key, current_score
            
            if (i + 1) % 1000 == 0:
                print(f"Progress: {((i + 1) / iterations):6.1%} | Best Score: {best_score:10.2f} | Key: {best_key}", end='\r')

        print("\n" + "="*80)
        print(f"\n{YELLOW}--- FINAL BEST RESULT ---{RESET}")
        best_plaintext = gronsfeld_decrypt(ciphertext, best_key, alphabet)
        print(f"Key: {YELLOW}{best_key}{RESET}")
        print(f"Bigram Score: {YELLOW}{best_score:.2f}{RESET} (Lower is better)")
        print(f"IoC: {YELLOW}{utils.calculate_ioc(best_plaintext):.4f}{RESET}")
        print(f"Plaintext: {best_plaintext.lower()}")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")


# --- MAIN APPLICATION ---

def run_direct_decrypt(ciphertext: str, alphabet: str):
    print(f"\n{BLUE}--- Direct Decryption ---{RESET}")
    key = input("Enter the numeric key: ")
    if not key.isdigit():
        print(f"{RED}Invalid key. Must be a sequence of digits.{RESET}")
        return
    
    decrypted = gronsfeld_decrypt(ciphertext, key, alphabet)
    print(f"\nKey: {YELLOW}{key}{RESET}")
    print(f"Decrypted: {decrypted}")

def run():
    print(f"{RED}Gronsfeld Cipher Toolkit{RESET}")
    ciphertext = input("\nEnter the ciphertext: ").upper()
    alphabet = input(f"Enter custom alphabet (default: {RED}A-Z{RESET}): ").upper() or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found. Scoring will be less accurate.{RESET}")
        return

    while True:
        print(f"\n{GREY}{'-'*50}{RESET}")
        print("Select an attack mode:")
        print(f"  ({YELLOW}1{RESET}) Direct Decryption (Known Key)")
        print(f"  ({YELLOW}2{RESET}) Exhaustive Attack (Key Length <= 8)")
        print(f"  ({YELLOW}3{RESET}) Simulated Annealing (Unknown Key of any length)")
        print(f"  ({YELLOW}4{RESET}) Exit")
        choice = input(">> ")

        if choice == '1':
            run_direct_decrypt(ciphertext, alphabet)
        elif choice == '2':
            run_exhaustive_attack(ciphertext, alphabet, expected_freqs)
        elif choice == '3':
            run_simulated_annealing_attack(ciphertext, alphabet, expected_freqs)
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

