from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
from functools import lru_cache
import os
import math 
import random # NEW: For Memetic Algorithm
import string # NEW: For Memetic Algorithm

# This assumes you have a utils.py file with calculate_ioc and save_results_to_file
# If not, you'll need to implement or remove those calls.
# from utils import utils 

# Dummy utils for standalone execution
class DummyUtils:
    def calculate_ioc(self, text: str) -> float:
        text = text.upper()
        text = ''.join(filter(str.isalpha, text))
        n = len(text)
        if n < 2:
            return 0.0
        
        freqs = {}
        for char in text:
            freqs[char] = freqs.get(char, 0) + 1
            
        numerator = sum(count * (count - 1) for count in freqs.values())
        denominator = n * (n - 1)
        
        return numerator / denominator if denominator > 0 else 0.0

    def save_results_to_file(self, results: List[Dict], filename: str, include_phrases: bool = True):
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Key: {result['key']}\n")
                if 'ioc' in result:
                    f.write(f"IoC: {result['ioc']:.6f}\n")
                if 'score' in result:
                    f.write(f"Fitness Score: {result['score']:.2f}\n")
                if include_phrases and 'matched_phrases' in result:
                    f.write(f"Matched Phrases: {', '.join(result['matched_phrases'])}\n")
                f.write(f"Plaintext: {result['plaintext']}\n")
                f.write("-" * 20 + "\n")
        print(f"Results saved to {filename}")

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


# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'

# Default Playfair alphabet (I and J are treated as the same letter)
PLAYFAIR_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
dictionary_path = os.path.join(os.path.dirname(__file__), "data", "words_alpha.txt")
bigram_freq_path = os.path.join(os.path.dirname(__file__), "data", "english_bigrams.txt")


# --- FUNCTIONS FOR BIGRAM FREQUENCY ATTACK ---

@lru_cache(maxsize=None)
def load_bigram_frequencies(file_path: str) -> Dict[str, float]:
    """Loads English bigram frequencies from a file."""
    freqs = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    bigram, freq = parts
                    freqs[bigram.upper()] = float(freq)
        # Normalize frequencies to probabilities
        total = sum(freqs.values())
        for bigram in freqs:
            freqs[bigram] /= total
        return freqs
    except FileNotFoundError:
        print(f"{RED}Error: Bigram frequency file not found at {file_path}{RESET}")
        return {}

def calculate_fitness_score(text: str, expected_freqs: Dict[str, float]) -> float:
    """Calculates a fitness score using the Chi-Squared statistic for bigrams."""
    text = "".join(char for char in text.upper() if char.isalpha())
    if len(text) < 2:
        return float('inf')

    observed_counts = {}
    total_bigrams = 0
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        observed_counts[bigram] = observed_counts.get(bigram, 0) + 1
        total_bigrams += 1
    
    if total_bigrams == 0:
        return float('inf')

    chi_squared_score = 0
    # A small floor value to prevent division by zero for rare bigrams
    floor = 0.01 

    for bigram, expected_prob in expected_freqs.items():
        observed_count = observed_counts.get(bigram, 0)
        expected_count = expected_prob * total_bigrams
        
        # Chi-Squared formula component
        difference = observed_count - expected_count
        chi_squared_score += (difference * difference) / max(expected_count, floor)

    return chi_squared_score

def process_batch_frequency(args: Tuple) -> List[Dict]:
    """Processes a batch of keys and returns their fitness scores."""
    word_batch, ciphertext, expected_freqs = args
    results = []
    
    for word in word_batch:
        try:
            plaintext = decrypt_playfair(ciphertext, word)
            score = calculate_fitness_score(plaintext, expected_freqs)
            results.append({
                'key': word,
                'plaintext': plaintext,
                'score': score
            })
        except Exception:
            continue
    return results

def crack_playfair_by_frequency(ciphertext: str, dictionary_path: str, bigram_path: str) -> List[Dict]:
    """Attempts to crack Playfair using bigram frequency analysis on a dictionary."""
    print(f"\n{YELLOW}Loading English bigram frequencies...{RESET}")
    expected_freqs = load_bigram_frequencies(bigram_path)
    if not expected_freqs:
        return []

    print(f"\n{YELLOW}Loading dictionary...{RESET}")
    dictionary = load_dictionary(dictionary_path)
    if not dictionary:
        return []

    print(f"Loaded {YELLOW}{len(dictionary)}{RESET} valid words from dictionary")
    
    word_batches = batch_words(dictionary)
    num_processes = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    total_batches = len(word_batches)
    
    print(f"\n{YELLOW}Starting dictionary attack with bigram fitness scoring...{RESET}")
    print(f"Processing {YELLOW}{total_batches}{RESET} batches with {YELLOW}{num_processes}{RESET} processes")
    start_time = time.time()
    
    all_results = []
    processed_batches = 0
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_batch_frequency, (batch, ciphertext, expected_freqs))
            for batch in word_batches
        ]
        
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                processed_batches += 1
                
                progress_percent = (processed_batches / total_batches) * 100
                print(f"Progress: {processed_batches}/{total_batches} batches ({progress_percent:.1f}%)", end='\r')
            except Exception as e:
                print(f"{RED}Batch processing error: {e}{RESET}")
                processed_batches += 1
                continue
    
    print("\n") # Newline after progress bar
    end_time = time.time()
    print(f"{YELLOW}Cracking complete in {end_time - start_time:.2f} seconds{RESET}")
    
    # Sort results by score (lower is better)
    all_results.sort(key=lambda x: x['score'])
    
    print(f"Found {RED}{len(all_results)}{RESET} potential solutions, sorted by fitness score.")
    return all_results

# --- MEMETIC ALGORITHM FUNCTIONS (NEW) ---

def generate_random_key() -> str:
    """Generates a random key by shuffling the Playfair alphabet."""
    key_list = list(PLAYFAIR_ALPHABET)
    random.shuffle(key_list)
    return "".join(key_list)

def crossover(parent1: str, parent2: str) -> str:
    """Performs crossover between two parent keys to create a child."""
    # This is a simple order crossover (OX1)
    child = [''] * 25
    
    # Select a random slice from parent1
    start, end = sorted(random.sample(range(25), 2))
    
    # Copy the slice from parent1 to the child
    p1_slice = parent1[start:end+1]
    child[start:end+1] = p1_slice
    
    # Fill the remaining spots with characters from parent2
    p2_chars = [char for char in parent2 if char not in p1_slice]
    
    child_idx = 0
    for i in range(25):
        if child[i] == '':
            child[i] = p2_chars[child_idx]
            child_idx += 1
            
    return "".join(child)

def mutate(key: str, mutation_rate: float) -> str:
    """Mutates a key by swapping two characters."""
    key_list = list(key)
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(25), 2)
        key_list[idx1], key_list[idx2] = key_list[idx2], key_list[idx1]
    return "".join(key_list)

def crack_playfair_memetic(ciphertext: str, bigram_path: str, pop_size: int, generations: int, mutation_rate: float) -> List[Dict]:
    """Attempts to crack Playfair using a Memetic Algorithm."""
    print(f"\n{YELLOW}Loading English bigram frequencies...{RESET}")
    expected_freqs = load_bigram_frequencies(bigram_path)
    if not expected_freqs:
        return []
    
    print(f"\n{YELLOW}Starting Memetic Algorithm Attack...{RESET}")
    print(f"Population Size: {YELLOW}{pop_size}{RESET}, Generations: {YELLOW}{generations}{RESET}, Mutation Rate: {YELLOW}{mutation_rate}{RESET}")
    
    # 1. Initialization
    population = [generate_random_key() for _ in range(pop_size)]
    
    best_key_overall = ""
    best_score_overall = float('inf')
    
    start_time = time.time()

    # 2. Main evolutionary loop
    for gen in range(generations):
        # 3. Fitness Evaluation
        scores = []
        for key in population:
            plaintext = decrypt_playfair(ciphertext, key)
            score = calculate_fitness_score(plaintext, expected_freqs)
            scores.append((score, key))
        
        scores.sort(key=lambda x: x[0]) # Sort by score, lower is better
        
        # Update best solution found so far
        if scores[0][0] < best_score_overall:
            best_score_overall = scores[0][0]
            best_key_overall = scores[0][1]
            
            elapsed = time.time() - start_time
            print(f"Gen {gen+1}/{generations} | Best Score: {GREEN}{best_score_overall:.2f}{RESET} | Key: {best_key_overall[:15]}... | Time: {elapsed:.1f}s")

        # 4. Selection (Elitism)
        # Keep the top 20% of the population (the elites)
        elite_count = max(2, int(pop_size * 0.2))
        elites = [key for score, key in scores[:elite_count]]
        
        next_generation = elites
        
        # 5. Crossover and Mutation
        while len(next_generation) < pop_size:
            parent1, parent2 = random.choices(elites, k=2) # Select parents from the elite group
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)
            
        population = next_generation

    end_time = time.time()
    print(f"\n{YELLOW}Algorithm finished in {end_time - start_time:.2f} seconds.{RESET}")
    
    final_plaintext = decrypt_playfair(ciphertext, best_key_overall)
    
    return [{
        'key': best_key_overall,
        'plaintext': final_plaintext,
        'score': best_score_overall
    }]


# --- CORE PLAYFAIR AND UTILITY FUNCTIONS ---

@lru_cache(maxsize=None)
def create_playfair_matrix(key: str) -> List[List[str]]:
    """
    Creates a 5x5 Playfair matrix from a given key.
    The alphabet used is 'ABCDEFGHIKLMNOPQRSTUVWXYZ'.
    """
    matrix_chars = []
    
    # Normalize key by removing non-alphabetical characters and 'J'
    normalized_key = "".join(char for char in key.upper() if char.isalpha()).replace('J', 'I')
    
    # Add unique characters from the key
    for char in normalized_key:
        if char not in matrix_chars:
            matrix_chars.append(char)
            
    # Add remaining characters from the Playfair alphabet
    for char in PLAYFAIR_ALPHABET:
        if char not in matrix_chars:
            matrix_chars.append(char)
            
    # Construct the 5x5 matrix
    matrix = [matrix_chars[i:i + 5] for i in range(0, 25, 5)]
    return matrix

@lru_cache(maxsize=None)
def get_char_map(matrix: Tuple[Tuple[str, ...], ...]) -> Dict[str, Tuple[int, int]]:
    """
    Creates a mapping from a character to its (row, col) in the matrix.
    The matrix is passed as a tuple of tuples to be hashable for lru_cache.
    """
    return {char: (r, c) for r, row in enumerate(matrix) for c, char in enumerate(row)}

def decrypt_playfair(ciphertext: str, key: str) -> str:
    """
    Decrypts a Playfair cipher using the provided key.
    """
    matrix = create_playfair_matrix(key)
    # Convert matrix to a hashable tuple of tuples for caching
    matrix_tuple = tuple(tuple(row) for row in matrix)
    char_map = get_char_map(matrix_tuple)
    
    plaintext = []
    
    # Prepare ciphertext for decryption (remove non-letters, replace J with I)
    normalized_text = "".join(char for char in ciphertext.upper() if char.isalpha()).replace('J', 'I')
    
    if len(normalized_text) % 2 != 0:
        normalized_text = normalized_text[:-1]

    digraphs = [normalized_text[i:i+2] for i in range(0, len(normalized_text), 2)]

    for char1, char2 in digraphs:
        if char1 not in char_map or char2 not in char_map:
            plaintext.append(char1.lower())
            plaintext.append(char2.lower())
            continue

        r1, c1 = char_map[char1]
        r2, c2 = char_map[char2]
        
        p1, p2 = '', ''

        if r1 == r2: # Same row: move left
            p1 = matrix[r1][(c1 - 1 + 5) % 5]
            p2 = matrix[r2][(c2 - 1 + 5) % 5]
        elif c1 == c2: # Same column: move up
            p1 = matrix[(r1 - 1 + 5) % 5][c1]
            p2 = matrix[(r2 - 1 + 5) % 5][c2]
        else: # Rectangle: swap columns
            p1 = matrix[r1][c2]
            p2 = matrix[r2][c1]
            
        plaintext.append(p1)
        plaintext.append(p2)
        
    return ''.join(plaintext)

def load_dictionary(file_path: str, min_length: int = 3, max_length: int = 15) -> List[str]:
    """Load dictionary words that only contain characters from the Playfair alphabet."""
    alphabet_set = set(PLAYFAIR_ALPHABET)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [word.strip().upper().replace('J', 'I') for word in file 
                    if set(word.strip().upper()).issubset(alphabet_set.union({'J'})) 
                    and min_length <= len(word.strip()) <= max_length]
    except FileNotFoundError:
        print(f"{RED}Error: {file_path} not found. Please create it or change the path.{RESET}")
        print(f"{YELLOW}Proceeding without a dictionary for brute-force.{RESET}")
        return []

def contains_all_phrases(text: str, phrases: List[str]) -> bool:
    """Check if plaintext contains all target phrases."""
    if not phrases:
        return True
    
    if not any(char.isalpha() for phrase in phrases for char in phrase):
        return False
        
    return all(phrase.upper().replace(" ", "").replace("J", "I") in text.upper() for phrase in phrases)

def highlight_match(text: str, phrases: List[str]) -> str:
    """Highlight all matched phrases in the plaintext."""
    result = text
    display_text = result
    
    for phrase in phrases:
        phrase_upper = phrase.upper().replace(" ", "").replace("J", "I")
        start_index = 0
        temp_display_text = ""
        cursor = 0
        
        # This is a more robust way to handle highlighting multiple matches
        while cursor < len(display_text):
            text_upper = display_text.upper().replace(RED, "").replace(RESET, "")
            found_pos = text_upper.find(phrase_upper, start_index)
            
            if found_pos == -1:
                break
            
            # Find the actual position in the string with color codes
            plain_text_count = 0
            actual_start = -1
            i = 0
            in_escape = False
            while i < len(display_text):
                if display_text[i] == '\033':
                    in_escape = True
                elif in_escape and display_text[i] == 'm':
                    in_escape = False
                elif not in_escape:
                    if plain_text_count == found_pos:
                        actual_start = i
                        break
                    plain_text_count += 1
                i += 1
            
            if actual_start != -1:
                end = actual_start + len(phrase_upper)
                highlighted_part = f"{RED}{display_text[actual_start:end]}{RESET}"
                display_text = display_text[:actual_start] + highlighted_part + display_text[end:]
                start_index = found_pos + 1 # Continue search after this match
            else:
                break
                
    return display_text

def process_batch(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """Process a batch of dictionary words as potential keys."""
    word_batch, ciphertext, target_phrases, min_ioc, max_ioc = args
    phrase_matches = []
    ioc_matches = []
    
    for word in word_batch:
        try:
            plaintext = decrypt_playfair(ciphertext, word)
            ioc = utils.calculate_ioc(plaintext)
            
            is_ioc_match = min_ioc <= ioc <= max_ioc
            is_phrase_match = contains_all_phrases(plaintext, target_phrases)

            if is_ioc_match:
                ioc_matches.append({
                    'key': word,
                    'plaintext': plaintext,
                    'ioc': ioc
                })
            
            if is_phrase_match:
                phrase_matches.append({
                    'key': word,
                    'plaintext': plaintext,
                    'matched_phrases': target_phrases,
                    'ioc': ioc
                })
        except Exception:
            continue
            
    return phrase_matches, ioc_matches

def batch_words(words: List[str], batch_size: int = 500) -> List[List[str]]:
    """Split word list into batches for parallel processing."""
    return [words[i:i + batch_size] for i in range(0, len(words), batch_size)]

def crack_playfair(ciphertext: str, target_phrases: List[str], dictionary_path: str, 
                   specific_keys: List[str] = None, min_ioc: float = 0.065, max_ioc: float = 0.07) -> Tuple[List[Dict], List[Dict]]:
    """
    Attempt to crack Playfair cipher using two approaches:
    1. Target phrase matching
    2. English-like IoC values
    """
    phrase_results = []
    ioc_results = []
    
    if specific_keys:
        print(f"\n{YELLOW}Trying specific keys...{RESET}")
        for key in specific_keys:
            plaintext = decrypt_playfair(ciphertext, key)
            ioc = utils.calculate_ioc(plaintext)
            
            if min_ioc <= ioc <= max_ioc:
                ioc_results.append({
                    'key': key,
                    'plaintext': plaintext,
                    'ioc': ioc
                })
                print(f"IoC match found with key: {RED}{key}{RESET} (IoC: {ioc:.6f})")
            
            if contains_all_phrases(plaintext, target_phrases):
                phrase_results.append({
                    'key': key,
                    'plaintext': plaintext,
                    'matched_phrases': target_phrases,
                    'ioc': ioc
                })
                print(f"Phrase match found with key: {RED}{key}{RESET}")
    
    print(f"\n{YELLOW}Loading dictionary...{RESET}")
    dictionary = load_dictionary(dictionary_path)
    if not dictionary:
        return phrase_results, ioc_results

    print(f"Loaded {YELLOW}{len(dictionary)}{RESET} valid words from dictionary")
    
    word_batches = batch_words(dictionary)
    num_processes = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    total_batches = len(word_batches)
    
    print(f"\n{YELLOW}Trying potential keys from dictionary...{RESET}")
    print(f"Processing {YELLOW}{total_batches}{RESET} batches with {YELLOW}{num_processes}{RESET} processes")
    start_time = time.time()
    
    processed_batches = 0
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_batch, (batch, ciphertext, target_phrases, min_ioc, max_ioc))
            for batch in word_batches
        ]
        
        for future in as_completed(futures):
            try:
                batch_phrase_results, batch_ioc_results = future.result()
                phrase_results.extend(batch_phrase_results)
                ioc_results.extend(batch_ioc_results)
                processed_batches += 1
                
                progress_percent = (processed_batches / total_batches) * 100
                print(f"Progress: {processed_batches}/{total_batches} batches ({progress_percent:.1f}%)", end='\r')
            except Exception as e:
                print(f"{RED}Batch processing error: {e}{RESET}")
                processed_batches += 1
                continue
    
    print("\n") # Newline after progress bar
    end_time = time.time()
    print(f"{YELLOW}Cracking complete in {end_time - start_time:.2f} seconds{RESET}")
    
    phrase_keys = {res['key'] for res in phrase_results}
    ioc_results = [res for res in ioc_results if res['key'] not in phrase_keys]
    
    ioc_results.sort(key=lambda x: abs(0.0667 - x['ioc']))
    
    print(f"Found {RED}{len(phrase_results)}{RESET} phrase-matching solutions")
    print(f"Found {RED}{len(ioc_results)}{RESET} additional English-like IoC solutions")
    
    return phrase_results, ioc_results

def run_direct_decrypt():
    """Run the direct decryption mode for the Playfair cipher."""
    print(f"\n{YELLOW}--- Direct Playfair Decryption ---{RESET}")
    ciphertext = input(f"{GREY}Enter ciphertext: {RESET}").upper()
    key = input(f"{GREY}Enter the key: {RESET}").upper()

    if not ciphertext or not key:
        print(f"{RED}Error: Ciphertext and key cannot be empty.{RESET}")
        return

    try:
        plaintext = decrypt_playfair(ciphertext, key)
        print(f"\n{GREEN}Plaintext:{RESET}")
        print(f"{plaintext}")
        save_file = input(f"\n{GREY}Save plaintext to file? ({YELLOW}Y/N{RESET}): {RESET}").upper() == 'Y'
        if save_file:
            filename = input(f"{GREY}Enter filename: {RESET}")
            with open(filename, 'w') as f:
                f.write(plaintext)
            print(f"{GREEN}Plaintext saved to {filename}{RESET}")
    except Exception as e:
        print(f"{RED}Decryption Error: {e}{RESET}")

def run():
    print(f"""{GREY} 
    Playfair Cipher{RESET}""")
    print(f"{GREY}-{RESET}" * 50)
    
    # MODIFIED: Added option 4
    mode = input(f"Choose a mode: ({YELLOW}1{RESET} = Brute Force w/ Crib, {YELLOW}2{RESET} = Direct Decrypt, {YELLOW}3{RESET} = Dictionary Attack (Bigram Fitness), {YELLOW}4{RESET} = Memetic Algorithm Attack): ")
    
    if mode == '2':
        run_direct_decrypt()
        return
    
    if mode == '3':
        print(f"\n{YELLOW}--- Dictionary Attack (Bigram Fitness) ---{RESET}")
        ciphertext = input("Enter ciphertext: ").upper()
        if not ciphertext:
            print(f"{RED}Ciphertext cannot be empty.{RESET}")
            return
            
        results = crack_playfair_by_frequency(ciphertext, dictionary_path, bigram_freq_path)
        
        if results:
            print(f"\n{YELLOW}TOP 10 RESULTS (Sorted by Fitness Score - Lower is Better){RESET}")
            print(f"{GREY}-{RESET}" * 50)
            
            for i, result in enumerate(results[:10]):
                print(f"Solution #{i+1}:")
                print(f"Key: {YELLOW}{result['key']}{RESET}")
                print(f"Fitness Score: {YELLOW}{result['score']:.2f}{RESET}")
                print(f"Plaintext: {result['plaintext']}")
                print(f"{GREY}-{RESET}" * 50)

            save_results = input(f"\nSave all results to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if save_results:
                filename = input("Enter filename for results: ") or "playfair_freq_results"
                utils.save_results_to_file(results, f"{filename}.txt")
        else:
            print(f"\n{RED}NO SOLUTIONS FOUND.{RESET}")
        return

    # NEW: Logic for mode 4
    if mode == '4':
        print(f"\n{YELLOW}--- Memetic Algorithm Attack ---{RESET}")
        ciphertext = input("Enter ciphertext: ").upper()
        if not ciphertext:
            print(f"{RED}Ciphertext cannot be empty.{RESET}")
            return
        
        try:
            pop_size = int(input(f"Enter population size (e.g., 100): ") or "100")
            generations = int(input(f"Enter number of generations (e.g., 200): ") or "200")
            mutation_rate = float(input(f"Enter mutation rate (e.g., 0.2): ") or "0.2")
        except ValueError:
            print(f"{RED}Invalid input. Using default parameters.{RESET}")
            pop_size, generations, mutation_rate = 100, 200, 0.2

        results = crack_playfair_memetic(ciphertext, bigram_freq_path, pop_size, generations, mutation_rate)
        
        if results:
            print(f"\n{YELLOW}BEST SOLUTION FOUND{RESET}")
            print(f"{GREY}-{RESET}" * 50)
            result = results[0]
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"Final Fitness Score: {YELLOW}{result['score']:.2f}{RESET}")
            print(f"Plaintext: {result['plaintext']}")
            print(f"{GREY}-{RESET}" * 50)

            save_results = input(f"\nSave best result to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if save_results:
                filename = input("Enter filename for result: ") or "playfair_memetic_result"
                utils.save_results_to_file(results, f"{filename}.txt")
        else:
            print(f"\n{RED}ALGORITHM FAILED TO COMPLETE.{RESET}")
        return


    # --- Original Mode 1 Logic ---
    print(f"\n{YELLOW}--- Brute Force with Crib/IoC ---{RESET}")
    use_test = input(f"Use test case? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
    
    if use_test:
        ciphertext = "BMODZ BXDNA BEKUD MUIXM MOUVI F"
        target_phrases = ["HIDETHEGOLD"]
        expected_key = "PLAYFAIREXAMPLE"
        specific_keys = ["PLAYFAIREXAMPLE"]
        min_ioc = 0.065
        max_ioc = 0.07
        
        print(f"\n{GREY}----------------------")
        print(f"Running a test case...")
        print(f"Ciphertext: {ciphertext}")
        print(f"Target phrases: {', '.join(target_phrases)}")
        print(f"Expected key: {expected_key}")
        print(f"IoC range: {min_ioc}-{max_ioc}")
        print(f"----------------------{RESET}")
    else:
        ciphertext = input("Enter ciphertext: ").upper()
        
        target_phrases_input = input(f"Enter known plaintext words/phrases (comma-separated, or press Enter for none): ").upper()
        target_phrases = [phrase.strip() for phrase in target_phrases_input.split(",") if phrase.strip()] if target_phrases_input else []
        
        specific_keys_input = input(f"Enter specific keys to try first (comma-separated, or press Enter to skip): ").upper()
        specific_keys = [key.strip() for key in specific_keys_input.split(",") if key.strip()] if specific_keys_input else []
        
        use_default_ioc = input(f"Use default English IoC range ({YELLOW}0.065-0.07{RESET})? ({YELLOW}Y/N{RESET}): ").upper()
        if use_default_ioc != 'N':
            min_ioc = 0.065
            max_ioc = 0.07
        else:
            try:
                min_ioc = float(input(f"Enter minimum IoC value: "))
                max_ioc = float(input(f"Enter maximum IoC value: "))
            except ValueError:
                print(f"{RED}Invalid input, using default range.{RESET}")
                min_ioc = 0.065
                max_ioc = 0.07
    
    phrase_results, ioc_results = crack_playfair(
        ciphertext, 
        target_phrases, 
        dictionary_path, 
        specific_keys,
        min_ioc,
        max_ioc
    )
    
    if phrase_results:
        print(f"\n{YELLOW}PHRASE-MATCHING RESULTS{RESET}")
        print(f"{GREY}-{RESET}" * 50)
        
        for i, result in enumerate(phrase_results[:10]):
            print(f"Solution #{i+1}:")
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"IoC: {YELLOW}{result['ioc']:.6f}{RESET}")
            print(f"Matched phrases: {YELLOW}{', '.join(result['matched_phrases'])}{RESET}")
            
            highlighted = highlight_match(result['plaintext'], result['matched_phrases'])
            print(f"Plaintext: {highlighted}")
            print(f"{GREY}-{RESET}" * 50)
        
        if use_test:
            test_passed = any(r['key'] == expected_key for r in phrase_results)
            print(f"\n{YELLOW}PHRASE-MATCH TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")
        
        save_phrase = input(f"\nSave phrase-matching results to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
        if save_phrase:
            phrase_filename = input("Enter filename for phrase results: ") or "playfair_results"
            utils.save_results_to_file(phrase_results, f"{phrase_filename}-phrases.txt")
        
        analyze_phrase = input(f"Run frequency analysis on best phrase match? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
        if analyze_phrase:
            utils.analyze_frequency_vg(phrase_results[0]['plaintext'])

    if ioc_results:
        print(f"\n{YELLOW}ENGLISH-LIKE IoC RESULTS (IoC range: {min_ioc}-{max_ioc}){RESET}")
        print(f"{GREY}-{RESET}" * 50)
        
        for i, result in enumerate(ioc_results[:10]):
            print(f"IoC Solution #{i+1}:")
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"IoC: {YELLOW}{result['ioc']:.6f}{RESET}")
            print(f"Plaintext: {result['plaintext']}")
            print(f"{GREY}-{RESET}" * 50)
        
        if use_test:
            test_passed = any(r['key'] == expected_key for r in ioc_results)
            print(f"\n{YELLOW}IoC-MATCH TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")
        
        save_ioc = input(f"\nSave IoC-based results to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
        if save_ioc:
            ioc_filename = input("Enter filename for IoC results: ") or "playfair_results"
            utils.save_results_to_file(ioc_results, f"{ioc_filename}-ioc.txt", include_phrases=False)
        
        analyze_ioc = input(f"Run frequency analysis on best IoC match? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
        if analyze_ioc:
            utils.analyze_frequency_vg(ioc_results[0]['plaintext'])
    
    if not phrase_results and not ioc_results:
        print(f"\n{RED}NO SOLUTIONS FOUND WITH EITHER METHOD{RESET}")
        
    print(f"\n{GREY}Program complete.{RESET}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(dictionary_path):
        print(f"Creating dummy dictionary file at {dictionary_path}")
        with open(dictionary_path, 'w') as f:
            f.write("PLAYFAIR\nKEYWORD\nCIPHER\nSECRET\n")
            
    if not os.path.exists(bigram_freq_path):
        print(f"Creating dummy bigram frequency file at {bigram_freq_path}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\nRE 0.68\nES 0.59\n")
            f.write("ON 0.57\nST 0.55\nNT 0.51\nEN 0.50\nAT 0.46\nED 0.44\nND 0.42\n")
            f.write("TO 0.42\nOR 0.42\nEA 0.41\nIS 0.34\nIT 0.34\nOU 0.33\nAR 0.32\n")

    run()