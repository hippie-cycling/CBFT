import itertools
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from collections import Counter
from utils import utils

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'

def highlight_match(text: str, phrases: list) -> str:
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

def gronsfeld_decrypt(ciphertext, key, alphabet):
    """Decrypt text using Gronsfeld cipher"""
    decrypted_text = []
    key_length = len(key)
    alphabet_length = len(alphabet)
    
    for i, char in enumerate(ciphertext):
        if char in alphabet:
            char_index = alphabet.index(char)
            shift = int(key[i % key_length])
            decrypted_index = (char_index - shift) % alphabet_length
            decrypted_char = alphabet[decrypted_index]
            decrypted_text.append(decrypted_char.lower())
        else:
            decrypted_text.append(char.lower())
    return ''.join(decrypted_text)

def try_decrypt(args):
    """Try decryption with a given key batch"""
    ciphertext, alphabet, keys, known_plaintexts, min_ioc, max_ioc = args
    results = []
    
    for key in keys:
        try:
            decrypted = gronsfeld_decrypt(ciphertext, key, alphabet)
            
            ioc = utils.calculate_ioc(decrypted)
            if min_ioc <= ioc <= max_ioc:
                print(f"\nIoC within range ({min_ioc}-{max_ioc}): {YELLOW}{ioc:.6f}{RESET}")
                print(f"Key: {YELLOW}{key}{RESET}")
                print(f"Decrypted text: {decrypted}")
            
            if any(word.lower() in decrypted.lower() for word in known_plaintexts):
                results.append({
                    'key': key,
                    'decrypted': decrypted,
                    'ioc': ioc,
                    'matched_phrases': [word for word in known_plaintexts if word.lower() in decrypted.lower()]
                })
        except (ValueError, IndexError) as e:
            continue
    
    return results

def run():
    print(f"""{GREY} 
 ██████  ██████   ██████  ███    ██ ███████ ███████ ███████ ██      ██████  
██       ██   ██ ██    ██ ████   ██ ██      ██      ██      ██      ██   ██ 
██   ███ ██████  ██    ██ ██ ██  ██ ███████ █████   █████   ██      ██   ██ 
██    ██ ██   ██ ██    ██ ██  ██ ██      ██ ██      ██      ██      ██   ██ 
 ██████  ██   ██  ██████  ██   ████ ███████ ██      ███████ ███████ ██████  
                                                                            {RESET}""")
    print(f"{RED}G{RESET}ronsfeld {RED}C{RESET}ipher {RED}D{RESET}ecoder")
    print(f"{GREY}-{RESET}" * 50)
    
    ciphertext = input("Enter the ciphertext: ").upper()
    
    alphabet_input = input(f"Enter custom alphabet (press Enter for default {RED}A-Z{RESET}): ").upper()
    if not alphabet_input:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        print(f"Using default alphabet: {RED}{alphabet}{RESET}")
    else:
        alphabet = alphabet_input
        print(f"Using custom alphabet: {RED}{alphabet}{RESET}")
    
    key_length = int(input("Enter the key length: "))
    
    known_text = input(f"Enter known plaintext words (comma-separated, press Enter for defaults {RED}THE, AND, THAT, FROM{RESET}): ").upper()
    if known_text.strip():
        known_plaintexts = [word.strip() for word in known_text.split(',') if word.strip()]
    else:
        known_plaintexts = ["THE", "AND", "THAT", "FROM"]
        print(f"Using default known plaintext words: {RED}{', '.join(known_plaintexts)}{RESET}")
    
    # Ask for IoC range
    use_default_ioc = input(f"Use default English IoC range ({YELLOW}0.062-0.071{RESET})? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
    if use_default_ioc:
        min_ioc = 0.062
        max_ioc = 0.071
    else:
        try:
            min_ioc = float(input(f"Enter minimum IoC value: "))
            max_ioc = float(input(f"Enter maximum IoC value: "))
        except ValueError:
            print(f"{RED}Invalid input, using default range.{RESET}")
            min_ioc = 0.062
            max_ioc = 0.071
    
    print(f"Using IoC range: {YELLOW}{min_ioc}-{max_ioc}{RESET}")
    
    # Generate keys in batches for better parallelization
    batch_size = 1000
    total_combinations = 10 ** key_length
    
    print(f"\n{YELLOW}Trying all {total_combinations} possible {key_length}-digit keys in parallel...{RESET}")
    
    # Calculate number of batches
    total_batches = (total_combinations + batch_size - 1) // batch_size
    processed_batches = 0
    processed_keys = 0
    
    print(f"Processing {YELLOW}{total_batches}{RESET} batches of up to {YELLOW}{batch_size}{RESET} keys each...")
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        
        # Process keys in batches
        for start in range(0, total_combinations, batch_size):
            end = min(start + batch_size, total_combinations)
            batch = [str(i).zfill(key_length) for i in range(start, end)]
            args = (ciphertext, alphabet, batch, known_plaintexts, min_ioc, max_ioc)
            futures.append(executor.submit(try_decrypt, args))
        
        # Collect results
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                processed_batches += 1
                processed_keys += batch_size
                
                # Ensure we don't exceed total combinations
                if processed_keys > total_combinations:
                    processed_keys = total_combinations
                
                # Print progress every 10% or on completion
                progress_percent = (processed_keys / total_combinations) * 100
                if processed_batches % max(1, total_batches // 10) == 0 or processed_batches == total_batches:
                    print(f"Progress: {processed_keys}/{total_combinations} keys ({progress_percent:.1f}%) - {processed_batches}/{total_batches} batches")
                    
            except Exception as e:
                print(f"\n{RED}Error processing batch: {str(e)}{RESET}")
                processed_batches += 1
                continue
    
    print(f"\n{YELLOW}Processing complete!{RESET}")
    
    if results:
        print(f"\n{YELLOW}POSSIBLE SOLUTIONS FOUND{RESET}")
        print(f"{GREY}-{RESET}" * 50)
        
        for i, result in enumerate(results):
            print(f"Solution #{i+1}:")
            print(f"Key: {YELLOW}{result['key']}{RESET}")
            print(f"IoC: {YELLOW}{result['ioc']:.6f}{RESET}")
            print(f"Matched phrases: {YELLOW}{', '.join(result['matched_phrases'])}{RESET}")
            
            # Highlight matched phrases in the decrypted text
            highlighted = highlight_match(result['decrypted'], result['matched_phrases'])
            print(f"Decrypted text: {highlighted}")
            print(f"{GREY}-{RESET}" * 50)
        
        # Option to save results
        save_results = input(f"\nSave results to file? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
        if save_results:
            filename = input("Enter filename for results: ")
            filename = filename + ".txt" if not filename.endswith(".txt") else filename
            utils.save_results_to_file(results, filename)
        
        # Option to run frequency analysis
        if results:
            analyze_option = input(f"Run frequency analysis on best match? ({YELLOW}Y/N{RESET}): ").upper() == 'Y'
            if analyze_option:
                utils.analyze_frequency_vg(results[0]['decrypted'])
                
    else:
        print(f"\n{RED}NO SOLUTIONS FOUND WITH GIVEN PARAMETERS{RESET}")
        
    print(f"\n{GREY}Program complete.{RESET}")

if __name__ == "__main__":
    run()