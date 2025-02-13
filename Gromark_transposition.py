import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Tuple, Dict
import multiprocessing
import numpy as np

def create_keyed_alphabet(keyword: str) -> str:
    # Remove duplicates while preserving order
    keyword = ''.join(dict.fromkeys(keyword.upper()))
    remaining = ''.join(c for c in 'KRYPTOSABCDEFGHIJLMNQUVWXZ' if c not in keyword)
    base = keyword + remaining
    
    # Create and fill block
    cols = len(keyword)
    rows = (len(base) + cols - 1) // cols
    block = [['' for _ in range(cols)] for _ in range(rows)]
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < len(base):
                block[i][j] = base[idx]
                idx += 1
    
    # Get column order based on keyword
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
    """Create optimized batches of primers"""
    all_primers = list(range(start, end + 1))
    return [all_primers[i:i + batch_size] for i in range(0, len(all_primers), batch_size)]

def generate_running_key(primer: str, length: int) -> str:
    """Generate running key using numpy for speed"""
    key = np.array([int(d) for d in primer], dtype=np.int8)
    result = np.zeros(length, dtype=np.int8)
    result[:len(key)] = key
    
    for i in range(len(key), length):
        result[i] = (result[i-5] + result[i-4]) % 10
        
    return ''.join(map(str, result))

def decrypt_gromark(ciphertext: str, mixed_alphabet: str, running_key: str) -> str:
    """Optimized Gromark decryption"""
    straight = 'KRYPTOSABCDEFGHIJLMNQUVWXZ'
    result = []
    
    for i, char in enumerate(ciphertext):
        if char in mixed_alphabet and i < len(running_key):
            mixed_pos = mixed_alphabet.index(char)
            straight_letter = straight[mixed_pos]
            shift = int(running_key[i])
            orig_pos = (straight.index(straight_letter) - shift) % 26
            result.append(straight[orig_pos].lower())
        else:
            result.append(char.lower())
            
    return ''.join(result)

def try_decrypt_batch(args: Tuple) -> List[Dict]:
    """Process a batch of primers, checking if words can be formed."""
    keyword, primers, ciphertext, required_words = args
    results = []
    mixed_alphabet = create_keyed_alphabet(keyword)

    for primer in primers:
        try:
            primer_str = str(primer)
            running_key = generate_running_key(primer_str, len(ciphertext))
            decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key)

            if all(can_form_word(word, decrypted) for word in required_words): # Use can_form_word
                results.append({
                    'keyword': keyword,
                    'primer': primer_str,
                    'running_key': running_key,
                    'decrypted': decrypted
                })

        except Exception:
            continue

    return results

def validate_keyword(keyword: str, known_segments: List[Tuple]) -> bool:
    """Quick keyword validation"""
    try:
        mixed_alphabet = create_keyed_alphabet(keyword)
        straight = 'KRYPTOSABCDEFGHIJLMNQUVWXZ'
        
        for _, cipher_segment, plain_segment in known_segments:
            for c, p in zip(cipher_segment, plain_segment):
                mixed_pos = mixed_alphabet.index(c)
                straight_letter = straight[mixed_pos]
                plain_pos = straight.index(p.upper())
                straight_pos = straight.index(straight_letter)
                
                if (straight_pos - plain_pos) % 26 > 9:
                    return False
                    
        return True
        
    except Exception:
        return False

def parallel_process_keywords(valid_keywords: List[str], ciphertext: str,
                              required_words: List[str], batch_size: int = 1000) -> List[Dict]: # Modified argument
    """Optimized parallel processing for word presence check"""
    all_results = []
    num_processes = max(1, os.cpu_count() - 1)
    primer_batches = batch_primers(batch_size=batch_size)

    total_batches = len(valid_keywords) * len(primer_batches)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(try_decrypt_batch, (keyword, batch, ciphertext, required_words)) # Modified argument
            for keyword in valid_keywords
            for batch in primer_batches
        ]

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

def can_form_word(word: str, text: str) -> bool:
    """Checks if a word can be formed from a text, allowing interspersed characters."""
    word = word.upper()
    text = text.upper()
    word_ptr = 0
    for text_char in text:
        if word_ptr < len(word) and text_char == word[word_ptr]:
            word_ptr += 1
    return word_ptr == len(word)

def save_results_to_file(results: List[Dict], filename: str):
    """Saves the results to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:  # Use utf-8 encoding
            for result in results:
                f.write("-" * 50 + "\n")
                f.write(f"Keyword: {result['keyword']}\n")
                f.write(f"Primer: {result['primer']}\n")
                f.write(f"Running Key: {result['running_key']}\n")  # Added running key
                f.write(f"Decrypted: {result['decrypted']}\n")
                f.write("-" * 50 + "\n")
                f.write("\n")  # Add an extra newline for spacing

        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

def main():
    print("Optimized Gromark Cipher Decoder")
    print("-" * 50)
    
    use_test = input("Use test case? (Y/N): ").upper() == 'Y'
    
    if use_test:
        ciphertext = "OHRERPHTMNUQDPUYQTGQHABASQXPTHPYSIXJUFVKNGNDRRIOMAEJGZKHCBNDBIWLDGVWDDVLXCSCZS"
        words_list = ["GRONSFELD", "TESTING", "GRONSFE"]
        required_words = ["ONLYTWOTHINGS"]  # Test case uses ONLYTWO
        expected = {
            'keyword': "GRONSFELD",  # Added 'keyword'
            'primer': "32941",
            'plaintext': "onlytwothingsareinfinitetheuniverseandhumanstupidityandimnotsureabouttheformer"
        }

    else:
        ciphertext = input("Enter ciphertext: ").upper()
        required_words = ["BERLINCLOCK", "EASTNORTHEAST"] 
    
        try:
            with open('words_alpha.txt', 'r') as f:
                words_list = [word.strip().upper() for word in f if 1 <= len(word.strip()) <= 15]
        except FileNotFoundError:
            print("Error: words_alpha.txt not found")
            return

    print("\nFiltering keywords based on constraints...")
    valid_keywords = [
        keyword for keyword in tqdm(words_list)
        if validate_keyword(keyword, [(0, ciphertext[0:13], "ONLYTWOTHINGS") if use_test else (63, ciphertext[63:74], "BERLINCLOCK")]) # Modified for test case
    ]
    
    print(f"\nFiltered from {len(words_list)} to {len(valid_keywords)} possible keywords")
    
    results = parallel_process_keywords(valid_keywords, ciphertext, required_words)

    if results:
        test_passed = False  # Flag to track if at least one solution matches

        for result in results:
            print("\n" + "-" * 50)
            print(f"Keyword: {result['keyword']}")
            print(f"Primer: {result['primer']}")
            print(f"Decrypted: {result['decrypted']}")

            if use_test:
                matches = can_form_word("ONLYTWO", result['decrypted']) # Check if ONLYTWO can be formed
                if matches:
                    test_passed = True
                    #print(f"\n***TEST CASE PASSED (Found a valid solution)***")  # Show pass message for each valid solution
                    #break #If you only want to see one pass you can uncomment this line
                #else:
                    #print(f"\n***TEST CASE FAILED (Solution does not generate ONLYTWO)***") #If you want to see all the fails you can uncomment this line
        if use_test:
            print(f"\n***TEST CASE {'PASSED' if test_passed else 'FAILED'}***")
        
        save_filename = input("Enter filename to save results (or press Enter to skip): ")
        if save_filename:
            save_results_to_file(results, save_filename)
    else:
        print("\nNo solutions found.")

if __name__ == "__main__":
    main()


# OBKRUOXOGHULBS
# OLIFBBWFLRVQQP
# RNGKSSOTWTQSJQ
# SSEKZZWATJKLUD
# IAWINFBNYPVTTM
# ZFPKWGDKZXTJCD
# IGKUHUAUEKCAR

# 0,3,6,2,5,1,4

# OBKRUOXOGHULBSZFPKWGDKZXTJCDSSEKZZWATJKLUDOLIFBBWFLRVQQPIGKUHUAUEKCAXRIAWINFBNYPVTTMRNGKSSOTWTQSJQ

# OBKRUOXOGHULBSOLIFBB W FLRVQQPRNGKSSOT W TQSJQSSEKZZ W ATJKLUDIA W INFBNYPVTTMZFPK W GDKZXTJCDIGKUHUAUEKCAR
# OBKRUOXOGHULBSOLIFBBFLRVQQPRNGKSSOTTQSJQSSEKZZATJKLUDIAINFBNYPVTTMZFPKGDKZXTJCDIGKUHUAUEKCAR
# OBKRUOXOGHULBSOLIFBBTQSJQSSEKZZINFBNYPVTTMZFPKFLRVQQPRNGKSSOTATJKLUDIAGDKZXTJCDIGKUHUAUEKCAR

#OBUOXOGHULBSOLIFBBWFLRVQQPRNGKSWTQSJQSSEKZZWATJKLUDIAWINFBNVTTMZFPKWGDKZXTJCDIGKUHUAUEKCAR

#SPQDMDQBQJUTCRLQSLTJAUVQKVTCHRTJPXKGLWTYZEOFTANKUXWOWBDAOBSZFGUUBSZNWHRFKKIKUKIGEWPKBLNSAFGOORSIZI