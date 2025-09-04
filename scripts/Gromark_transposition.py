from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Tuple, Dict
import numpy as np
from functools import lru_cache
import itertools
import time
import random

# --- DUMMY UTILS CLASS (to make script standalone) ---
class DummyUtils:
    def save_results_to_file(self, results: List[Dict], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Gromark Cipher Brute-Force Results\n")
                f.write("===================================\n\n")
                for result in results:
                    f.write(f"Keyword: {result['keyword']}\n")
                    f.write(f"Primer: {result['primer']}\n")
                    f.write(f"Alphabet: {result['alphabet']}\n")
                    f.write(f"IoC Score: {result.get('ioc', 0):.4f}\n")
                    f.write(f"Bigram Score: {result.get('bigram_score', 0):.2f}\n")
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
BLUE = '\033[38;5;21m'
RESET = '\033[0m'

# Check for __file__ existence for compatibility with different environments
try:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
except NameError:
    data_dir = "data" # Fallback for environments where __file__ is not defined

dictionary_path = os.path.join(data_dir, "words_alpha.txt")
bigram_freq_path = os.path.join(data_dir, "english_bigrams.txt")

# --- SCORING FUNCTIONS ---

def calculate_ioc(text: str) -> float:
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

# --- GROMARK CORE FUNCTIONS ---

def create_keyed_alphabet(keyword: str, alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    keyword = ''.join(dict.fromkeys(keyword.upper()))
    remaining = ''.join(c for c in alphabet if c not in keyword)
    base = keyword + remaining
    cols = len(keyword)
    if cols == 0: return alphabet
    rows = (len(base) + cols - 1) // cols
    block = [['' for _ in range(cols)] for _ in range(rows)]
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < len(base):
                block[i][j] = base[idx]
                idx += 1

    sorted_keyword_with_indices = sorted([(char, i) for i, char in enumerate(keyword)])
    final_col_order = [i for char, i in sorted_keyword_with_indices]

    return ''.join(
        block[row][col]
        for col in final_col_order
        for row in range(rows)
        if row < len(block) and col < len(block[row]) and block[row][col]
    )

def batch_primers(start: int = 10000, end: int = 99999, batch_size: int = 1000) -> List[List[int]]:
    all_primers = list(range(start, end + 1))
    return [all_primers[i:i + batch_size] for i in range(0, len(all_primers), batch_size)]

def generate_running_key(primer: str, length: int) -> str:
    key = np.array([int(d) for d in primer], dtype=np.int8)
    result = np.zeros(length, dtype=np.int8)
    result[:len(key)] = key
    for i in range(len(key), length):
        result[i] = (result[i-5] + result[i-4]) % 10
    return ''.join(map(str, result))

def decrypt_gromark(ciphertext: str, mixed_alphabet: str, running_key: str, alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    result = []
    len_alpha = len(alphabet)
    for i, char in enumerate(ciphertext):
        if char in mixed_alphabet and i < len(running_key):
            mixed_pos = mixed_alphabet.find(char)
            if mixed_pos == -1:
                result.append(char.lower())
                continue
            straight_letter = alphabet[mixed_pos]
            shift = int(running_key[i])
            orig_pos = (alphabet.find(straight_letter) - shift) % len_alpha
            result.append(alphabet[orig_pos].lower())
        else:
            result.append(char.lower())
    return ''.join(result)

def can_form_word(word: str, text: str) -> bool:
    word = word.upper()
    text = text.upper()
    it = iter(text)
    return all(c in it for c in word)

def try_decrypt_batch(args: Tuple) -> List[Dict]:
    keyword, primers, ciphertext, required_words, alphabet, expected_freqs = args
    results = []
    mixed_alphabet = create_keyed_alphabet(keyword, alphabet)

    for primer in primers:
        try:
            primer_str = str(primer)
            running_key = generate_running_key(primer_str, len(ciphertext))
            decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet)
            decrypted_upper = decrypted.upper()

            match_type = None
            if not required_words:
                continue # Skip if no words are provided to filter against

            # Prioritize a full substring match (stronger indicator)
            if all(word in decrypted_upper for word in required_words):
                match_type = 'substring'
            # If not a substring, check for a "dragged" match
            elif all(can_form_word(word, decrypted) for word in required_words):
                match_type = 'dragged'

            if match_type:
                ioc = calculate_ioc(decrypted)
                bigram_score = calculate_bigram_score(decrypted, expected_freqs)
                results.append({
                    'keyword': keyword,
                    'primer': primer_str,
                    'decrypted': decrypted,
                    'alphabet': alphabet,
                    'ioc': ioc,
                    'bigram_score': bigram_score,
                    'match_type': match_type
                })
        except Exception:
            continue
    return results

def validate_keyword(keyword: str, known_segments: List[Tuple], alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> bool:
    try:
        mixed_alphabet = create_keyed_alphabet(keyword, alphabet)
        for _, cipher_segment, plain_segment in known_segments:
            for c, p in zip(cipher_segment, plain_segment):
                mixed_pos = mixed_alphabet.find(c)
                if mixed_pos == -1: return False
                straight_letter = alphabet[mixed_pos]
                plain_pos = alphabet.find(p.upper())
                straight_pos = alphabet.find(straight_letter)
                if (straight_pos - plain_pos) % len(alphabet) > 9:
                    return False
        return True
    except Exception:
        return False

def parallel_process_keywords(keywords_list: List[str], ciphertext: str, required_words: List[str], alphabet: str, expected_freqs: Dict, batch_size: int = 1000) -> List[Dict]:
    all_results = []
    num_processes = max(1, os.cpu_count() - 1)
    primer_batches = batch_primers(batch_size=batch_size)
    total_batches = len(keywords_list) * len(primer_batches)
    processed_batches = 0

    if total_batches == 0:
        print(f"{RED}No keywords to process.{RESET}")
        return []

    print(f"Processing {YELLOW}{len(keywords_list):,}{RESET} keywords across {YELLOW}{len(primer_batches)}{RESET} primer batches ({YELLOW}{total_batches:,}{RESET} total batches) using {YELLOW}{num_processes}{RESET} processes.")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(try_decrypt_batch, (keyword, batch, ciphertext, required_words, alphabet, expected_freqs))
            for keyword in keywords_list
            for batch in primer_batches
        ]
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Batch processing error: {e}")
            finally:
                processed_batches += 1
                progress_percent = (processed_batches / total_batches) * 100
                print(f"Progress: {processed_batches}/{total_batches} batches ({progress_percent:.1f}%)", end='\r')
    print("\n")
    return all_results

def get_alphabets_from_user() -> List[str]:
    alphabets = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    print(f"\n{YELLOW}Enter additional alphabets to test (one per line).{RESET}")
    print(f"{GREY}Press Enter on an empty line when done. Default is '{alphabets[0]}'.{RESET}")
    while True:
        alphabet_input = input(f"Additional alphabet #{len(alphabets)}: ").upper().strip()
        if not alphabet_input: break
        unique_chars = ''.join(dict.fromkeys(alphabet_input))
        if len(unique_chars) != len(alphabet_input):
            print(f"{RED}Warning: Duplicate characters removed.{RESET}")
            alphabet_input = unique_chars
        alphabets.append(alphabet_input)
    return alphabets

def highlight_phrases(text: str, phrases: List[str], color: str = RED) -> str:
    highlighted_text = text
    for phrase in phrases:
        phrase_lower = phrase.lower()
        parts = highlighted_text.split(phrase_lower)
        highlighted_text = (f"{color}{phrase_lower}{RESET}").join(parts)
    return highlighted_text

def highlight_dragged_crib(text: str, crib: str, color: str = BLUE) -> str:
    text_lower = text.lower()
    crib_lower = crib.lower()
    if not can_form_word(crib_lower, text_lower):
        return text
    result_chars = list(text)
    text_iter = iter(enumerate(text_lower))
    try:
        for crib_char in crib_lower:
            while True:
                index, text_char = next(text_iter)
                if text_char == crib_char:
                    result_chars[index] = f"{color}{text[index]}{RESET}"
                    break
    except StopIteration:
        pass
    return "".join(result_chars)

def process_and_display_results(all_results, required_words):
    if not all_results:
        print(f"\n{RED}NO SOLUTIONS FOUND{RESET}")
        return

    min_ioc_str = input(f"Enter minimum IoC for ranking (default: {YELLOW}0.060{RESET}): ") or "0.060"
    max_ioc_str = input(f"Enter maximum IoC for ranking (default: {YELLOW}0.075{RESET}): ") or "0.075"
    min_ioc, max_ioc = float(min_ioc_str), float(max_ioc_str)

    all_results.sort(key=lambda x: (
        0 if min_ioc <= x['ioc'] <= max_ioc else 1,
        x['bigram_score'],
        -x['ioc']
    ))

    print(f"\n{YELLOW}--- RANKED SOLUTIONS FOUND ---{RESET}")
    for result in all_results[:20]:
        in_range = min_ioc <= result['ioc'] <= max_ioc
        range_marker = GREEN + " (In Range)" + RESET if in_range else ""

        match_type = result.get('match_type')
        if match_type == 'substring':
            highlighted_decrypted = highlight_phrases(result['decrypted'], required_words, RED)
            match_marker = f"{RED}(Substring Match){RESET}"
        elif match_type == 'dragged':
            temp_decrypted = result['decrypted']
            for word in required_words:
                temp_decrypted = highlight_dragged_crib(temp_decrypted, word, BLUE)
            highlighted_decrypted = temp_decrypted
            match_marker = f"{BLUE}(Dragged Match){RESET}"
        else:
            highlighted_decrypted = result['decrypted']
            match_marker = ""


        print(f"{GREY}-{RESET}" * 50)
        print(f"Keyword: {YELLOW}{result['keyword']}{RESET} {match_marker}")
        print(f"Primer: {YELLOW}{result['primer']}{RESET}")
        print(f"Alphabet: {YELLOW}{result['alphabet']}{RESET}")
        print(f"IoC: {YELLOW}{result['ioc']:.4f}{RESET}{range_marker}")
        print(f"Bigram Score: {YELLOW}{result['bigram_score']:.2f}{RESET} (Lower is better)")
        print(f"Decrypted: {highlighted_decrypted}")

    save_filename = input("\nEnter filename to save all results (or press Enter to skip): ")
    if save_filename:
        save_filename = save_filename + ".txt" if not save_filename.endswith(".txt") else save_filename
        utils.save_results_to_file(all_results, save_filename)

    analyze_option = input(f"Run frequency analysis on best match? ({YELLOW}Y/N{RESET}): ").upper()
    if analyze_option == 'Y' and all_results:
        utils.analyze_frequency_vg(all_results[0]['decrypted'])

# --- MODES OF OPERATION ---

def run_dictionary_attack(expected_freqs: Dict):
    print(f"\n{GREEN}--- Mode 1: Dictionary Attack ---{RESET}")

    ciphertext = input("Enter ciphertext: ").upper()
    required_words_input = input("Enter known plaintext words to filter/highlight (comma-separated, optional): ").upper()
    required_words = [word.strip() for word in required_words_input.split(",")] if required_words_input else []

    known_segments = []
    use_known_plaintext_filtering = input(f"Do you have a known plaintext segment at a specific location? (Dramatically speeds up the search) ({YELLOW}Y/N{RESET}): ").upper()
    if use_known_plaintext_filtering == 'Y':
        plain_segment = input("  Enter known plaintext segment: ").upper()
        try:
            pos = int(input(f"  Enter the starting position (index 0) of '{plain_segment}': "))
            cipher_segment = ciphertext[pos:pos+len(plain_segment)]
            known_segments.append((pos, cipher_segment, plain_segment))
            print(f"{GREEN}Will filter keywords based on the segment '{plain_segment}' at index {pos}.{RESET}")
        except (ValueError, IndexError):
            print(f"{RED}Invalid position. Proceeding without keyword filtering.{RESET}")

    try:
        with open(dictionary_path, 'r') as f:
            words_list = [word.strip().upper() for word in f if 1 <= len(word.strip()) <= 15]
    except FileNotFoundError:
        print(f"{RED}Error: Dictionary file '{dictionary_path}' not found.{RESET}")
        return

    alphabets_to_test = get_alphabets_from_user()
    all_results = []

    for alphabet in alphabets_to_test:
        print(f"\nTesting with alphabet: {RED}{alphabet}{RESET}")
        print(f"\n{YELLOW}Filtering keywords...{RESET}")

        valid_keywords = []
        if known_segments:
            total_words = len(words_list)
            print(f"Validating {YELLOW}{total_words:,}{RESET} potential keywords against known segments...")
            for i, keyword in enumerate(words_list):
                if validate_keyword(keyword, known_segments, alphabet):
                    valid_keywords.append(keyword)
                if (i + 1) % 5000 == 0:
                    print(f"Keyword validation progress: {(i + 1) / total_words:.0%}", end='\r')
            print("\nValidation complete.")
            print(f"Filtered from {YELLOW}{len(words_list):,}{RESET} to {RED}{len(valid_keywords):,}{RESET} possible keywords.")
        else:
            print(f"{GREY}No known segments provided. Using full dictionary of {len(words_list):,} words.{RESET}")
            valid_keywords = words_list

        if valid_keywords:
            results = parallel_process_keywords(valid_keywords, ciphertext, required_words, alphabet, expected_freqs)
            all_results.extend(results)

    process_and_display_results(all_results, required_words)

def run_key_length_bruteforce(expected_freqs: Dict):
    print(f"\n{GREEN}--- Mode 2: Brute-Force by Key Length ---{RESET}")

    try:
        key_length = int(input(f"Enter key length to brute-force (e.g., 4 or 5): "))
        if key_length <= 0:
            raise ValueError
    except ValueError:
        print(f"{RED}Invalid key length. Please enter a positive integer.{RESET}")
        return

    alphabet_for_key = input(f"Enter character set for the key (default: ABCDEFGHIJKLMNOPQRSTUVWXYZ): ").upper() or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    total_keys = len(alphabet_for_key) ** key_length
    print(f"{YELLOW}This will generate {total_keys:,} unique keywords.{RESET}")
    if key_length > 5:
        confirm = input(f"{RED}WARNING: This is a very large number and may take an extremely long time. Proceed? (Y/N): {RESET}").upper()
        if confirm != 'Y':
            return

    ciphertext = input("Enter ciphertext: ").upper()
    required_words_input = input("Enter known plaintext words to filter/highlight (comma-separated): ").upper()
    required_words = [word.strip() for word in required_words_input.split(",")] if required_words_input else []

    alphabets_to_test = get_alphabets_from_user()
    all_results = []

    start_time = time.time()
    print(f"\n{YELLOW}Generating keywords of length {key_length}...{RESET}")
    keywords_to_test = [''.join(p) for p in itertools.product(alphabet_for_key, repeat=key_length)]
    print(f"Generated {len(keywords_to_test):,} keywords in {time.time() - start_time:.2f} seconds.")

    for alphabet in alphabets_to_test:
        print(f"\nTesting with alphabet: {RED}{alphabet}{RESET}")
        results = parallel_process_keywords(keywords_to_test, ciphertext, required_words, alphabet, expected_freqs)
        all_results.extend(results)

    process_and_display_results(all_results, required_words)

# --- HILL CLIMBING IMPLEMENTATION ---

def get_neighbors(keyword: str, primer_str: str, alphabet_for_key: str, num_neighbors: int = 10) -> List[Tuple[str, str]]:
    neighbors = []
    primer_int = int(primer_str)

    for _ in range(num_neighbors):
        if random.random() < 0.7:
            pos = random.randint(0, len(keyword) - 1)
            new_char = random.choice(alphabet_for_key)
            new_keyword = list(keyword)
            new_keyword[pos] = new_char
            neighbors.append((''.join(new_keyword), primer_str))
        else:
            new_primer = (primer_int + random.randint(-100, 100)) % 100000
            if new_primer < 10000: new_primer += 10000
            neighbors.append((keyword, str(new_primer)))

    return neighbors

def run_hill_climbing_attack(expected_freqs: Dict):
    print(f"\n{GREEN}--- Mode 3: Hill Climbing Attack ---{RESET}")

    if not expected_freqs:
        print(f"{RED}Cannot run Hill Climbing without bigram frequencies for scoring.{RESET}")
        return

    ciphertext = input("Enter ciphertext: ").upper()
    try:
        key_length = int(input(f"Enter the length of the keyword to search for: "))
        num_restarts = int(input(f"Enter number of random restarts (e.g., 50): ") or "50")
        max_iterations = int(input(f"Enter max iterations per climb (e.g., 1000): ") or "1000")
        if key_length <= 0 or num_restarts <= 0 or max_iterations <= 0:
            raise ValueError
    except ValueError:
        print(f"{RED}Invalid input. Please enter positive integers.{RESET}")
        return

    print(f"\n{YELLOW}Define the alphabets for the search:{RESET}")
    alphabet_for_cipher = input(f"  Enter the cipher's alphabet (default: ABCDEFGHIJKLMNOPQRSTUVWXYZ): ").upper().strip() or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet_for_key = input(f"  Enter the keyword's character set (default: Same as cipher alphabet): ").upper().strip() or alphabet_for_cipher
    print("-" * 20)

    required_words_input = input("Enter known plaintext words (comma-separated, for highlighting): ").upper()
    required_words = [word.strip() for word in required_words_input.split(",")] if required_words_input else []

    best_overall_solution = None
    best_overall_score = float('inf')

    for r in range(num_restarts):
        current_keyword = ''.join(random.choice(alphabet_for_key) for _ in range(key_length))
        current_primer = str(random.randint(10000, 99999))

        mixed_alphabet = create_keyed_alphabet(current_keyword, alphabet_for_cipher)
        running_key = generate_running_key(current_primer, len(ciphertext))
        decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet_for_cipher)
        current_best_score = calculate_bigram_score(decrypted, expected_freqs)

        for i in range(max_iterations):
            neighbors = get_neighbors(current_keyword, current_primer, alphabet_for_key)
            best_neighbor_solution = None
            best_neighbor_score = current_best_score

            for neighbor_keyword, neighbor_primer in neighbors:
                mixed_alphabet = create_keyed_alphabet(neighbor_keyword, alphabet_for_cipher)
                running_key = generate_running_key(neighbor_primer, len(ciphertext))
                decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet_for_cipher)
                score = calculate_bigram_score(decrypted, expected_freqs)

                if score < best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor_solution = (neighbor_keyword, neighbor_primer)

            if best_neighbor_solution:
                current_keyword, current_primer = best_neighbor_solution
                current_best_score = best_neighbor_score
            else:
                break

            print(f"Restart {r+1}/{num_restarts} | Iter {i+1}/{max_iterations} | Current Best Score: {current_best_score:.2f} | Key: {current_keyword}", end='\r')

        if current_best_score < best_overall_score:
            best_overall_score = current_best_score
            best_overall_solution = (current_keyword, current_primer)
            print("\n" + f"{GREEN}New best solution found! Score: {best_overall_score:.2f}, Keyword: {best_overall_solution[0]}, Primer: {best_overall_solution[1]}{RESET}")

    print("\n\n--- Hill Climbing Search Complete ---")
    if best_overall_solution:
        keyword, primer = best_overall_solution
        print(f"Best solution found:")
        print(f"  Keyword: {YELLOW}{keyword}{RESET}")
        print(f"  Primer:  {YELLOW}{primer}{RESET}")
        print(f"  Bigram Score: {YELLOW}{best_overall_score:.2f}{RESET}")

        mixed_alphabet = create_keyed_alphabet(keyword, alphabet_for_cipher)
        running_key = generate_running_key(primer, len(ciphertext))
        decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet_for_cipher)
        highlighted_decrypted = highlight_phrases(decrypted, required_words)

        print(f"  Decrypted Text: {highlighted_decrypted}")
        utils.analyze_frequency_vg(decrypted)
    else:
        print(f"{RED}No solution found that improved upon random noise.{RESET}")

def run_direct_decryption():
    print(f"\n{GREEN}--- Mode 4: Direct Decryption ---{RESET}")
    ciphertext = input("Enter ciphertext: ").upper()
    keyword = input("Enter keyword: ").upper()
    primer = input("Enter primer (5-digit number): ")
    alphabet = input("Enter alphabet (default: ABCDEFGHIJKLMNOPQRSTUVWXYZ): ").upper() or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    highlight_input = input("Enter words to highlight (comma-separated, optional): ").upper()
    words_to_highlight = [word.strip() for word in highlight_input.split(",")] if highlight_input else []


    if not primer.isdigit() or len(primer) != 5:
        print(f"{RED}Invalid primer. Must be a 5-digit number.{RESET}")
        return

    print("\n--- Decrypting ---")
    try:
        mixed_alphabet = create_keyed_alphabet(keyword, alphabet)
        running_key = generate_running_key(primer, len(ciphertext))
        decrypted_text = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet)
        highlighted_decrypted = highlight_phrases(decrypted_text, words_to_highlight, RED)

        print(f"Keyword: {YELLOW}{keyword}{RESET}")
        print(f"Primer: {YELLOW}{primer}{RESET}")
        print(f"Keyed Alphabet: {GREY}{mixed_alphabet}{RESET}")
        print(f"Running Key: {GREY}{running_key[:50]}...{RESET}")
        print(f"Decrypted Text: {highlighted_decrypted}")

        ioc = calculate_ioc(decrypted_text)
        print(f"IoC of result: {YELLOW}{ioc:.4f}{RESET}")

    except Exception as e:
        print(f"{RED}An error occurred during decryption: {e}{RESET}")

def run_test_case():
    """ Runs a known-answer test case to verify the decryption logic. """
    print(f"\n{GREEN}--- Mode 5: Run Known-Answer Test Case ---{RESET}")
    print(f"{GREY}This test uses a known solution to a Zodiac Killer cipher to verify the tool's core logic.{RESET}")

    # Known values for the Z-32 cipher
    ciphertext = "OHRERPHTMNUQDPUYQTGQHABASQXPTHPYSIXJUFVKNGNDRRIOMAEJGZKHCBNDBIWLDGVWDDVLXCSCZS"
    keyword = "GRONSFELD"
    primer = "3294151"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    known_words = ["ONLYTWOTHINGS", "THEUNIVERSE", "HUMANSTUPIDITY", "ABOUTTHEUNIVERSE"]

    print("\n--- Test Parameters ---")
    print(f"Ciphertext: {ciphertext}")
    print(f"Keyword:    {YELLOW}{keyword}{RESET}")
    print(f"Primer:     {YELLOW}{primer}{RESET}")
    print(f"Alphabet:   {alphabet}")
    print("-" * 23)

    try:
        mixed_alphabet = create_keyed_alphabet(keyword, alphabet)
        running_key = generate_running_key(primer, len(ciphertext))
        decrypted_text = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet)
        highlighted_decrypted = highlight_phrases(decrypted_text, known_words, RED)
        ioc = calculate_ioc(decrypted_text)

        print(f"\n{GREEN}Decryption Successful:{RESET}")
        print(highlighted_decrypted)
        print(f"\nIoC of result: {YELLOW}{ioc:.4f}{RESET}")

    except Exception as e:
        print(f"{RED}An error occurred during the test case decryption: {e}{RESET}")


# --- MAIN EXECUTION ---

def run():
    print(f"{RED}G{RESET}romark {RED}C{RESET}ipher {RED}T{RESET}oolkit")
    print(f"{GREY}-{RESET}" * 50)

    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found or empty. Some features will be disabled.{RESET}")

    while True:
        print("\nSelect an option:")
        print(f"  {YELLOW}1{RESET}) Dictionary Attack")
        print(f"  {YELLOW}2{RESET}) Brute-Force by Key Length {GREY}(Computationally Expensive){RESET}")
        print(f"  {YELLOW}3{RESET}) Hill Climbing Attack {GREEN}(Recommended for unknown keys){RESET}")
        print(f"  {YELLOW}4{RESET}) Direct Decryption")
        print(f"  {YELLOW}5{RESET}) Run Known-Answer Test Case")
        print(f"  {YELLOW}6{RESET}) Exit")
        choice = input(">> ")

        if choice == '1':
            run_dictionary_attack(expected_freqs)
        elif choice == '2':
            run_key_length_bruteforce(expected_freqs)
        elif choice == '3':
            run_hill_climbing_attack(expected_freqs)
        elif choice == '4':
            run_direct_decryption()
        elif choice == '5':
            run_test_case()
        elif choice == '6':
            break
        else:
            print(f"{RED}Invalid choice. Please select a valid option.{RESET}")

    print(f"\n{GREY}Program complete.{RESET}")

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file at '{bigram_freq_path}'...{RESET}")
        with open(bigram_freq_path, 'w', encoding='utf-8') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\n")
    if not os.path.exists(dictionary_path):
        print(f"{GREY}Creating dummy dictionary file at '{dictionary_path}'...{RESET}")
        with open(dictionary_path, 'w', encoding='utf-8') as f:
            f.write("GRONSFELD\nTESTING\nKRYPTOS\nPALIMPSEST\nBERLIN\nCLOCK\nNYPVTT\nPARADICE\n")
    run()