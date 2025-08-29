from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Tuple, Dict
import numpy as np
from functools import lru_cache

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
RESET = '\033[0m'

data_dir = os.path.join(os.path.dirname(__file__), "data")
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
    rows = (len(base) + cols - 1) // cols
    block = [['' for _ in range(cols)] for _ in range(rows)]
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < len(base):
                block[i][j] = base[idx]
                idx += 1
    sorted_chars = sorted(keyword)
    order = []
    for char in keyword:
        order.append(sorted_chars.index(char) + 1)
        sorted_chars[sorted_chars.index(char)] = None
    col_order = [p[0] for p in sorted(enumerate(order), key=lambda x: x[1])]
    return ''.join(
        block[row][col]
        for col in col_order
        for row in range(rows)
        if row < len(block) and block[row][col]
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
    for i, char in enumerate(ciphertext):
        if char in mixed_alphabet and i < len(running_key):
            mixed_pos = mixed_alphabet.index(char)
            straight_letter = alphabet[mixed_pos]
            shift = int(running_key[i])
            orig_pos = (alphabet.index(straight_letter) - shift) % len(alphabet)
            result.append(alphabet[orig_pos].lower())
        else:
            result.append(char.lower())
    return ''.join(result)

def try_decrypt_batch(args: Tuple) -> List[Dict]:
    keyword, primers, ciphertext, required_words, alphabet, expected_freqs = args
    results = []
    mixed_alphabet = create_keyed_alphabet(keyword, alphabet)

    for primer in primers:
        try:
            primer_str = str(primer)
            running_key = generate_running_key(primer_str, len(ciphertext))
            decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key, alphabet)

            if all(can_form_word(word, decrypted) for word in required_words):
                ioc = calculate_ioc(decrypted)
                bigram_score = calculate_bigram_score(decrypted, expected_freqs)
                results.append({
                    'keyword': keyword,
                    'primer': primer_str,
                    'decrypted': decrypted,
                    'alphabet': alphabet,
                    'ioc': ioc,
                    'bigram_score': bigram_score
                })
        except Exception:
            continue
    return results

def validate_keyword(keyword: str, known_segments: List[Tuple], alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> bool:
    try:
        mixed_alphabet = create_keyed_alphabet(keyword, alphabet)
        for _, cipher_segment, plain_segment in known_segments:
            for c, p in zip(cipher_segment, plain_segment):
                mixed_pos = mixed_alphabet.index(c)
                straight_letter = alphabet[mixed_pos]
                plain_pos = alphabet.index(p.upper())
                straight_pos = alphabet.index(straight_letter)
                if (straight_pos - plain_pos) % len(alphabet) > 9:
                    return False
        return True
    except Exception:
        return False

def parallel_process_keywords(valid_keywords: List[str], ciphertext: str, required_words: List[str], alphabet: str, expected_freqs: Dict, batch_size: int = 1000) -> List[Dict]:
    all_results = []
    num_processes = max(1, os.cpu_count() - 1)
    primer_batches = batch_primers(batch_size=batch_size)
    total_batches = len(valid_keywords) * len(primer_batches)
    processed_batches = 0

    print(f"Processing {YELLOW}{total_batches}{RESET} batches with {YELLOW}{num_processes}{RESET} processes")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(try_decrypt_batch, (keyword, batch, ciphertext, required_words, alphabet, expected_freqs))
            for keyword in valid_keywords
            for batch in primer_batches
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
                print(f"Batch processing error: {e}")
                processed_batches += 1
                continue
    print("\n")
    return all_results

def can_form_word(word: str, text: str) -> bool:
    word = word.upper()
    text = text.upper()
    word_ptr = 0
    for text_char in text:
        if word_ptr < len(word) and text_char == word[word_ptr]:
            word_ptr += 1
    return word_ptr == len(word)

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

def highlight_phrases(text: str, phrases: List[str]) -> str:
    """Highlights all occurrences of phrases in a given text."""
    highlighted_text = text
    for phrase in phrases:
        phrase_lower = phrase.lower()
        highlighted_text = highlighted_text.replace(
            phrase_lower,
            f"{RED}{phrase_lower}{RESET}"
        )
    return highlighted_text

def run():
    print(f"""{GREY} 
██████  ██████  ███████ 
██      ██  ██ ██      
██  ███ ██████  █████   
██   ██ ██  ██ ██      
 ██████  ██████  ██      
                       {RESET}""")
    print(f"{RED}G{RESET}romark {RED}B{RESET}rute {RED}F{RESET}orcer")
    print(f"{GREY}-{RESET}" * 50)

    use_test = input(f"Use test case? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'

    if use_test:
        ciphertext = "OHRERPHTMNUQDPUYQTGQHABASQXPTHPYSIXJUFVKNGNDRRIOMAEJGZKHCBNDBIWLDGVWDDVLXCSCZS"
        words_list = ["GRONSFELD", "TESTING", "GRONSFE"]
        required_words = ["ONLYTWOTHINGS"]
        print(f"\n{GREY}--- Running Test Case ---{RESET}")
        print(f"Ciphertext: {ciphertext}")
        print(f"Target phrases: {', '.join(required_words)}")
        print(f"{GREY}-------------------------{RESET}")
    else:
        ciphertext = input("Enter ciphertext: ").upper()
        required_words_input = input(f"Enter known plaintext words (comma-separated, default: {RED}BERLINCLOCK, EASTNORTHEAST{RESET}): ").upper()
        required_words = [word.strip() for word in required_words_input.split(",")] if required_words_input else ["BERLINCLOCK", "EASTNORTHEAST"]
        try:
            with open(dictionary_path, 'r') as f:
                words_list = [word.strip().upper() for word in f if 1 <= len(word.strip()) <= 15]
        except FileNotFoundError:
            print(f"{RED}Error: {dictionary_path} not found{RESET}")
            return

    # Get IoC range from user
    min_ioc_str = input(f"Enter minimum IoC (default: {YELLOW}0.060{RESET}): ") or "0.060"
    max_ioc_str = input(f"Enter maximum IoC (default: {YELLOW}0.075{RESET}): ") or "0.075"
    min_ioc, max_ioc = float(min_ioc_str), float(max_ioc_str)

    alphabets_to_test = get_alphabets_from_user()
    all_results = []
    
    # Load bigram frequencies once
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found. Scores will be less accurate.{RESET}")

    for alphabet in alphabets_to_test:
        print(f"\nTesting with alphabet: {RED}{alphabet}{RESET}")
        print(f"\n{YELLOW}Filtering keywords based on constraints...{RESET}")
        
        valid_keywords = []
        total_words = len(words_list)
        print(f"Processing {YELLOW}{total_words}{RESET} potential keywords...")
        
        known_segments = [(0, ciphertext[0:13], "ONLYTWOTHINGS")] if use_test else [(63, ciphertext[63:74], "BERLINCLOCK")]
        
        for i, keyword in enumerate(words_list):
            if validate_keyword(keyword, known_segments, alphabet):
                valid_keywords.append(keyword)
            if (i + 1) % (total_words // 10 or 1) == 0:
                print(f"Keyword validation progress: {(i + 1) / total_words:.0%}", end='\r')
        print("\n")

        print(f"Filtered from {YELLOW}{len(words_list)}{RESET} to {RED}{len(valid_keywords)}{RESET} possible keywords")

        if valid_keywords:
            results = parallel_process_keywords(valid_keywords, ciphertext, required_words, alphabet, expected_freqs)
            all_results.extend(results)

    if all_results:
        # Sort results based on the specified criteria
        all_results.sort(key=lambda x: (
            0 if min_ioc <= x['ioc'] <= max_ioc else 1, # Priority 1: In IoC range
            x['bigram_score'],                          # Priority 2: Bigram score (lower is better)
            -x['ioc']                                   # Priority 3: IoC (higher is better)
        ))
        
        print(f"\n{YELLOW}--- RANKED SOLUTIONS FOUND ---{RESET}")
        for result in all_results:
            in_range = min_ioc <= result['ioc'] <= max_ioc
            range_marker = GREEN + " (In Range)" + RESET if in_range else ""
            
            # Use the new highlighting function
            highlighted_decrypted = highlight_phrases(result['decrypted'], required_words)

            print(f"{GREY}-{RESET}" * 50)
            print(f"Keyword: {YELLOW}{result['keyword']}{RESET}")
            print(f"Primer: {YELLOW}{result['primer']}{RESET}")
            print(f"Alphabet: {YELLOW}{result['alphabet']}{RESET}")
            print(f"IoC: {YELLOW}{result['ioc']:.4f}{RESET}{range_marker}")
            print(f"Bigram Score: {YELLOW}{result['bigram_score']:.2f}{RESET} (Lower is better)")
            print(f"Decrypted: {highlighted_decrypted}") # Display highlighted text

        if use_test:
            test_passed = any(can_form_word("ONLYTWO", r['decrypted']) for r in all_results)
            print(f"\n{YELLOW}TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")

        save_filename = input("\nEnter filename to save results (or press Enter to skip): ")
        if save_filename:
            save_filename = save_filename + ".txt" if not save_filename.endswith(".txt") else save_filename
            utils.save_results_to_file(all_results, save_filename)

        analyze_option = input(f"Run frequency analysis on best match? ({YELLOW}Y/N{RESET}): ").upper()
        if analyze_option == 'Y':
            utils.analyze_frequency_vg(all_results[0]['decrypted'])
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