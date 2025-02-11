import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import os
from typing import List, Tuple, Optional, Dict
import multiprocessing
import numpy as np

def batch_primers(start: int = 10000, end: int = 99999, batch_size: int = 1000) -> List[List[int]]:
    """Create batches of primers for more efficient processing"""
    all_primers = list(range(start, end + 1))
    return [all_primers[i:i + batch_size] for i in range(0, len(all_primers), batch_size)]

def try_decrypt_batch(args: Tuple) -> List[Dict]:
    """Process a batch of primers for a given keyword"""
    keyword, primers, ciphertext, known_segments = args
    results = []
    
    # Create alphabet once for the batch
    mixed_alphabet = create_keyed_alphabet(keyword)
    
    # Pre-calculate segment indices and lengths for validation
    segment_details = [(start, len(cipher_segment)) for start, cipher_segment, _ in known_segments]
    
    for primer in primers:
        try:
            primer_str = str(primer)
            running_key = generate_running_key(primer_str, len(ciphertext))
            
            # Quick validation using numpy for known segments
            valid = True
            for start_idx, segment_len in segment_details:
                segment_key = running_key[start_idx:start_idx + segment_len]
                segment_cipher = ciphertext[start_idx:start_idx + segment_len]
                decrypted_segment = decrypt_gromark(segment_cipher, mixed_alphabet, segment_key)
                
                if decrypted_segment.upper() != known_segments[0][2].upper():
                    valid = False
                    break
            
            if valid:
                decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key)
                results.append({
                    'keyword': keyword,
                    'primer': primer_str,
                    'running_key': running_key,
                    'decrypted': decrypted
                })
                
        except Exception:
            continue
            
    return results

def parallel_process_keywords(valid_keywords: List[str], ciphertext: str, known_segments: List[Tuple], 
                            batch_size: int = 1000) -> List[Dict]:
    """Process keywords in parallel with batched primers"""
    all_results = []
    num_processes = max(1, os.cpu_count() - 1)  # Leave one core free
    primer_batches = batch_primers(batch_size=batch_size)
    
    total_batches = len(valid_keywords) * len(primer_batches)
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        
        # Submit batches for each keyword
        for keyword in valid_keywords:
            for primer_batch in primer_batches:
                args = (keyword, primer_batch, ciphertext, known_segments)
                futures.append(executor.submit(try_decrypt_batch, args))
        
        # Process results as they complete
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    pbar.update(1)
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    continue
    
    return all_results

def create_keyed_alphabet(keyword):
    """Create a keyed alphabet from a keyword following Gromark cipher rules"""
    # Remove duplicates while preserving order
    seen = set()
    keyword_unique = ''.join(c for c in keyword.upper() if not (c in seen or seen.add(c)))
    
    # Add remaining alphabet letters
    remaining = ''.join(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in keyword_unique)
    keyed_base = keyword_unique + remaining
    
    # Create transposition block
    cols = len(keyword)
    rows = (len(keyed_base) + cols - 1) // cols
    block = [['' for _ in range(cols)] for _ in range(rows)]
    
    # Fill the block row by row
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < len(keyed_base):
                block[i][j] = keyed_base[idx]
                idx += 1
    
    # Get column order based on keyword
    sorted_chars = sorted(keyword)
    order = []
    for char in keyword:
        order.append(sorted_chars.index(char) + 1)
        sorted_chars[sorted_chars.index(char)] = None
    
    # Create pairs of (order number, column index)
    pairs = list(enumerate(order))
    pairs.sort(key=lambda x: x[1])
    col_order = [p[0] for p in pairs]
    
    # Create mixed alphabet by reading down columns in the determined order
    mixed_alphabet = ''
    for col in col_order:
        for row in range(rows):
            if row < len(block) and block[row][col]:
                mixed_alphabet += block[row][col]
    
    return mixed_alphabet

def generate_running_key(primer, length):
    """Generate running key from a primer"""
    if len(primer) < 5:
        raise ValueError("Primer must be at least 5 digits long")
    
    key = list(map(int, primer))
    while len(key) < length:
        sum_digits = key[-5] + key[-4]
        if sum_digits > 9:
            sum_digits -= 10
        key.append(sum_digits)
    
    return ''.join(map(str, key[:length]))

def get_possible_shifts(cipher_char, plain_char, mixed_alphabet):
    """Get possible shifts (0-9) that could map plaintext to ciphertext"""
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mixed_pos = mixed_alphabet.index(cipher_char)
    straight_letter = straight_alphabet[mixed_pos]
    plain_pos = straight_alphabet.index(plain_char.upper())
    straight_pos = straight_alphabet.index(straight_letter)
    
    required_shift = (straight_pos - plain_pos) % 26
    return required_shift if required_shift <= 9 else None

def analyze_running_key_constraints(ciphertext_segment, plaintext_segment, mixed_alphabet, keyword):
    """Analyze running key constraints for a segment (Corrected)"""
    shifts = []
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for c, p in zip(ciphertext_segment, plaintext_segment):
        mixed_pos = mixed_alphabet.index(c)
        straight_letter = straight_alphabet[mixed_pos]
        plain_pos = straight_alphabet.index(p.upper())
        straight_pos = straight_alphabet.index(straight_letter)

        required_shift = (straight_pos - plain_pos) % 26
        if required_shift > 9:  # Gromark shifts are 0-9
            return None
        shifts.append(required_shift)
    return shifts

def decrypt_gromark(ciphertext, mixed_alphabet, running_key):
    """Decrypt text using Gromark cipher"""
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    decrypted = []
    
    for i, char in enumerate(ciphertext):
        if char in mixed_alphabet and i < len(running_key):
            mixed_pos = mixed_alphabet.index(char)
            straight_letter = straight_alphabet[mixed_pos]
            shift = int(running_key[i])
            orig_pos = (straight_alphabet.index(straight_letter) - shift) % 26
            decrypted.append(straight_alphabet[orig_pos].lower())
        else:
            decrypted.append(char.lower())
    
    return ''.join(decrypted)

def validate_keyword(keyword, known_segments):
    """Validate if a keyword could work with known plaintext segments (Corrected)"""
    try:
        mixed_alphabet = create_keyed_alphabet(keyword)

        for start_idx, cipher_segment, plain_segment in known_segments:
            shifts = analyze_running_key_constraints(cipher_segment, plain_segment, mixed_alphabet, keyword)
            if shifts is None:
                return False  # Early exit if shifts are invalid

        return True  # All segments passed initial shift check

    except Exception as e:
        return False

def try_decrypt_with_primer(args):
    """Try decryption with a given primer (Corrected)"""
    keyword, primer, ciphertext, known_segments = args

    try:
        mixed_alphabet = create_keyed_alphabet(keyword)
        running_key = generate_running_key(primer, len(ciphertext))
        decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key)

        # ***CRITICAL CHECK: Validate against known segments***
        valid_decryption = True
        for start_idx, cipher_segment, plain_segment in known_segments:
            segment_key = running_key[start_idx:start_idx + len(cipher_segment)]
            decrypted_segment = decrypt_gromark(cipher_segment, mixed_alphabet, segment_key)
            if decrypted_segment.upper() != plain_segment.upper():
                valid_decryption = False
                break  # Exit inner loop if segment doesn't match

        if valid_decryption:  # Only return if ALL segments match
            return {
                'keyword': keyword,
                'primer': primer,
                'running_key': running_key,
                'decrypted': decrypted
            }
        else:
            return None  # Return None if decryption doesn't match known segments

    except Exception as e:
        return None

def main():
    print("Optimized Gromark Cipher Decoder")
    print("-" * 50)

    # ***KEY CHANGE: Test Case Option***
    use_test_case = input("Use test case? (Y/N): ")

    if use_test_case == 'Y':
        ciphertext = "OHRERPHTMNUQDPUYQTGQHABASQXPTHPYSIXJUFVKNGNDRRIOMAEJGZKHCBNDBIWLDGVWDDVLXCSCZS" # Test ciphertext
        words_list = ["GRONSFELD", "TESTING", "GRONSFE"]  # Test word list
        known_segments = [(0, ciphertext[0:7], "ONLYTWO")] # Test known segment
        expected_keyword = "GRONSFELD"  # Expected keyword in test case
        expected_primer = "32941" # Expected primer in test case
        expected_plaintext = "onlytwothingsareinfinitetheuniverseandhumanstupidityandimnotsureabouttheformer" # Expected plaintext

    elif use_test_case == 'N':
        ciphertext = input("Enter the ciphertext: ").upper()
        known_segments = [(63, ciphertext[63:74], "BERLINCLOCK")] # Original known segment
        vowels = set('')
        try:
            with open('words_alpha.txt', 'r') as f:
                words_list = [
                    word.strip().upper()
                    for word in f
                    if 1 <= len(word.strip()) <= 15
                    and not any(
                        word[i].upper() in vowels and word[i + 1].upper() in vowels
                        for i in range(len(word.strip()) - 1)
                    )
                ]
        except FileNotFoundError:
            print("Warning: words_alpha.txt not found. Using default word list.")
    else:
        print("Invalid input. Exiting...")
        return

    # Filter keywords using constraints (Corrected)
    print("Filtering keywords based on constraints...")
    print(f"Known Plaintext: {known_segments}")
    valid_keywords = []
    for keyword in tqdm(words_list):
        if validate_keyword(keyword, known_segments):
            valid_keywords.append(keyword)
    
    print(f"\nFiltered from {len(words_list)} to {len(valid_keywords)} possible keywords")
    print(valid_keywords)
    
    # Process valid keywords with parallel primer testing
    # Replace the original parallel processing section with:
    results = parallel_process_keywords(valid_keywords, ciphertext, known_segments)
    
    # Display results (Corrected for Test Case)
    if results:
        print("\nPossible solutions found:")
        for result in results:
            print("\n" + "-" * 50)
            print(f"Keyword: {result['keyword']}")
            print(f"Primer: {result['primer']}")
            print(f"Running key: {result['running_key']}")
            print(f"Decrypted text: {result['decrypted']}")

            # ***KEY ADDITION: Test Case Verification***
            if use_test_case == 'Y':
                if result['keyword'] == expected_keyword and result['primer'] == expected_primer and result['decrypted'] == expected_plaintext:
                    print("\n***TEST CASE PASSED!***")
                else:
                    print("\n***TEST CASE FAILED!***")
                    if result['keyword'] != expected_keyword:
                        print(f"Expected Keyword: {expected_keyword}, Found: {result['keyword']}")
                    if result['primer'] != expected_primer:
                         print(f"Expected Primer: {expected_primer}, Found: {result['primer']}")
                    if result['decrypted'] != expected_plaintext:
                         print(f"Expected Plaintext: {expected_plaintext}, Found: {result['decrypted']}")

    else:
        print("\nNo solutions found with given parameters.")

if __name__ == "__main__":
    main()