from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
from functools import lru_cache
import os
from utils import utils

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'

@lru_cache(maxsize=None)
def get_alphabet(alphabet_str: str):
    """Get alphabet and corresponding dictionary mapping."""
    alphabet = alphabet_str.upper()
    alphabet_dict = {char: i for i, char in enumerate(alphabet)}
    return alphabet, alphabet_dict

def decrypt_vigenere(ciphertext: str, key: str, alphabet_str: str) -> str:
    """Decrypt Vigenere cipher using the provided key and alphabet."""
    alphabet, alphabet_dict = get_alphabet(alphabet_str)
    alphabet_length = len(alphabet)
    
    key_shifts = [alphabet_dict.get(k, 0) for k in key.upper() if k in alphabet_dict]
    if not key_shifts:
        return ciphertext

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
    """Load dictionary words that only contain characters from the given alphabet."""
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

def try_key_directly(key: str, ciphertext: str, alphabet_str: str) -> str:
    """Try a specific key directly."""
    return decrypt_vigenere(ciphertext, key, alphabet_str)

def contains_all_phrases(text: str, phrases: List[str]) -> bool:
    """Check if plaintext contains all target phrases."""
    if not phrases:  # If no phrases provided, return True
        return True
        
    return all(phrase.upper() in text.upper().replace(" ", "") for phrase in phrases)

def highlight_match(text: str, phrases: List[str]) -> str:
    """Highlight all matched phrases in the plaintext."""
    result = text
    for phrase in phrases:
        phrase_upper = phrase.upper().replace(" ", "")
        text_upper = result.upper().replace(" ", "")
        
        if phrase_upper in text_upper:
            start = text_upper.find(phrase_upper)
            end = start + len(phrase_upper)
            
            # Create highlighted version
            highlighted = f"{RED}{result[start:end]}{RESET}"
            result = result[:start] + highlighted + result[end:]
            
    return result

def process_batch(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """Process a batch of dictionary words as potential keys."""
    word_batch, ciphertext, target_phrases, alphabet_str, min_ioc, max_ioc = args
    phrase_matches = []
    ioc_matches = []
    
    for word in word_batch:
        plaintext = decrypt_vigenere(ciphertext, word, alphabet_str)
        
        # Calculate IoC for all solutions
        ioc = utils.calculate_ioc(plaintext)
        
        # Check if IoC is within the English-like range
        if min_ioc <= ioc <= max_ioc:
            ioc_matches.append({
                'key': word,
                'plaintext': plaintext,
                'ioc': ioc
            })
        
        # Check if plaintext contains all target phrases
        if contains_all_phrases(plaintext, target_phrases):
            phrase_matches.append({
                'key': word,
                'plaintext': plaintext,
                'matched_phrases': target_phrases,
                'ioc': ioc
            })
                
    return phrase_matches, ioc_matches

def batch_words(words: List[str], batch_size: int = 500) -> List[List[str]]:
    """Split word list into batches for parallel processing."""
    return [words[i:i + batch_size] for i in range(0, len(words), batch_size)]

def crack_vigenere(ciphertext: str, alphabet_str: str, target_phrases: List[str], dictionary_path: str, 
                   specific_keys: List[str] = None, min_ioc: float = 0.065, max_ioc: float = 0.07) -> Tuple[List[Dict], List[Dict]]:
    """
    Attempt to crack Vigenere cipher using two approaches:
    1. Target phrase matching
    2. English-like IoC values
    
    Returns two lists of possible solutions.
    """
    phrase_results = []
    ioc_results = []
    
    # First try specific keys if provided
    if specific_keys:
        print(f"\n{YELLOW}Trying specific keys...{RESET}")
        for key in specific_keys:
            plaintext = try_key_directly(key, ciphertext, alphabet_str)
            ioc = utils.calculate_ioc(plaintext)
            
            # Always add to ioc_results if within range
            if min_ioc <= ioc <= max_ioc:
                ioc_results.append({
                    'key': key,
                    'plaintext': plaintext,
                    'ioc': ioc
                })
                print(f"IoC match found with key: {RED}{key}{RESET} (IoC: {ioc:.6f})")
            
            # Add to phrase_results if phrases match
            if contains_all_phrases(plaintext, target_phrases):
                phrase_results.append({
                    'key': key,
                    'plaintext': plaintext,
                    'matched_phrases': target_phrases,
                    'ioc': ioc
                })
                print(f"Phrase match found with key: {RED}{key}{RESET}")
    
    # Load dictionary words for brute force
    print(f"\n{YELLOW}Loading dictionary...{RESET}")
    dictionary = load_dictionary(dictionary_path, alphabet_str)
    print(f"Loaded {YELLOW}{len(dictionary)}{RESET} valid words from dictionary")
    
    if dictionary:
        # Create batches for parallel processing
        word_batches = batch_words(dictionary)
        num_processes = max(1, os.cpu_count() - 1)
        
        total_batches = len(word_batches)
        
        print(f"\n{YELLOW}Trying potential keys from dictionary...{RESET}")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_batch, (batch, ciphertext, target_phrases, alphabet_str, min_ioc, max_ioc))
                for batch in word_batches
            ]
            
            with tqdm(total=total_batches, desc="Processing batches", colour="yellow") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_phrase_results, batch_ioc_results = future.result()
                        phrase_results.extend(batch_phrase_results)
                        ioc_results.extend(batch_ioc_results)
                        pbar.update(1)
                    except Exception as e:
                        print(f"{RED}Batch processing error: {e}{RESET}")
                        continue
        
        end_time = time.time()
        print(f"\n{YELLOW}Cracking complete in {end_time - start_time:.2f} seconds{RESET}")
    
    # Sort IoC results by closeness to ideal English IoC (0.0667)
    ioc_results.sort(key=lambda x: abs(0.0667 - x['ioc']))
    
    print(f"Found {RED}{len(phrase_results)}{RESET} phrase-matching solutions")
    print(f"Found {RED}{len(ioc_results)}{RESET} English-like IoC solutions")
    
    return phrase_results, ioc_results

def run():
    print(f"""{GREY} 
██    ██ ██  ██████  ███████ ███    ██ ███████ ██████  ███████ 
██    ██ ██ ██       ██      ████   ██ ██      ██   ██ ██      
██    ██ ██ ██   ███ █████   ██ ██  ██ █████   ██████  █████   
 ██  ██  ██ ██    ██ ██      ██  ██ ██ ██      ██   ██ ██      
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
        specific_keys = ["PALIMPSEST"]  # Try this key directly
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
        
        alphabet_input = input(f"Enter alphabet (press Enter for default {RED}ABCDEFGHIJKLMNOPQRSTUVWXYZ{RESET}): ").upper()
        alphabet_str = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        target_phrases_input = input(f"Enter known plaintext words/phrases (comma-separated, or press Enter for none): ").upper()
        target_phrases = [phrase.strip() for phrase in target_phrases_input.split(",")] if target_phrases_input else []
        
        specific_keys_input = input(f"Enter specific keys to try first (comma-separated, or press Enter to skip): ").upper()
        specific_keys = [key.strip() for key in specific_keys_input.split(",")] if specific_keys_input else []
        
        # Ask for IoC range
        use_default_ioc = input(f"Use default English IoC range ({YELLOW}0.065-0.07{RESET})? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
        if use_default_ioc:
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
    
    dictionary_path = "data\words_alpha.txt"
    
    # Run the cracking process with both methods
    phrase_results, ioc_results = crack_vigenere(
        ciphertext, 
        alphabet_str, 
        target_phrases, 
        dictionary_path, 
        specific_keys,
        min_ioc,
        max_ioc
    )
    
    # Display phrase-matching results if any
    if phrase_results and target_phrases:
        print(f"\n{YELLOW}PHRASE-MATCHING RESULTS{RESET}")
        print(f"{GREY}-{RESET}" * 50)
        
        for i, result in enumerate(phrase_results[:10]):  # Show top 10
            print(f"Solution #{i+1}:")
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"IoC: {YELLOW}{result['ioc']:.6f}{RESET}")
            print(f"Matched phrases: {YELLOW}{', '.join(result['matched_phrases'])}{RESET}")
            
            highlighted = highlight_match(result['plaintext'], result['matched_phrases'])
            print(f"Plaintext: {highlighted}")
            print(f"{GREY}-{RESET}" * 50)
        
        # Check if test case was successful
        if use_test:
            test_passed = any(r['key'] == expected_key for r in phrase_results)
            print(f"\n{YELLOW}PHRASE-MATCH TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")
        
        # Option to save phrase results
        if len(phrase_results) > 0:
            save_phrase = input(f"\nSave phrase-matching results to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if save_phrase:
                phrase_filename = input("Enter filename for phrase results: ")
                phrase_filename = phrase_filename + "-phrases.txt" if not phrase_filename.endswith(".txt") else phrase_filename.replace(".txt", "-phrases.txt")
                utils.save_results_to_file(phrase_results, phrase_filename)
            
            # Option to run frequency analysis
            analyze_phrase = input(f"Run frequency analysis on best phrase match? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if analyze_phrase:
                utils.analyze_frequency(phrase_results[0]['plaintext'])
    
    # Display IoC-based results
    if ioc_results:
        print(f"\n{YELLOW}ENGLISH-LIKE IoC RESULTS (IoC range: {min_ioc}-{max_ioc}){RESET}")
        print(f"{GREY}-{RESET}" * 50)
        
        for i, result in enumerate(ioc_results[:10]):  # Show top 10
            print(f"IoC Solution #{i+1}:")
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"IoC: {YELLOW}{result['ioc']:.6f}{RESET}")
            
            # Highlight any matched phrases if they exist
            if target_phrases and contains_all_phrases(result['plaintext'], target_phrases):
                print(f"Matched phrases: {YELLOW}{', '.join(target_phrases)}{RESET}")
                highlighted = highlight_match(result['plaintext'], target_phrases)
                print(f"Plaintext: {highlighted}")
            else:
                print(f"Plaintext: {result['plaintext']}")
            
            print(f"{GREY}-{RESET}" * 50)
        
        # Check if test case was successful
        if use_test:
            test_passed = any(r['key'] == expected_key for r in ioc_results)
            print(f"\n{YELLOW}IoC-MATCH TEST CASE {'PASSED' if test_passed else 'FAILED'}{RESET}")
        
        # Option to save IoC results
        if len(ioc_results) > 0:
            save_ioc = input(f"\nSave IoC-based results to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if save_ioc:
                ioc_filename = input("Enter filename for IoC results: ")
                ioc_filename = ioc_filename + "-ioc.txt" if not ioc_filename.endswith(".txt") else ioc_filename.replace(".txt", "-ioc.txt")
                utils.save_results_to_file(ioc_results, ioc_filename, include_phrases=False)
            
            # Option to run frequency analysis
            analyze_ioc = input(f"Run frequency analysis on best IoC match? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if analyze_ioc:
                utils.analyze_frequency(ioc_results[0]['plaintext'])
    
    # If no results found with either method
    if not phrase_results and not ioc_results:
        print(f"\n{RED}NO SOLUTIONS FOUND WITH EITHER METHOD{RESET}")
        
    print(f"\n{GREY}Program complete.{RESET}")

if __name__ == "__main__":
    run()