import sys
import os
import math
from functools import lru_cache
from typing import List, Dict, Tuple

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'

# Path for our bigram frequency file
data_dir = os.path.join(os.path.dirname(__file__), "data")
bigram_freq_path = os.path.join(data_dir, "english_bigrams.txt")

# --- SCORING FUNCTIONS ---

def calculate_ioc(text: str) -> float:
    """Calculates the Index of Coincidence for a given text."""
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

@lru_cache(maxsize=None)
def load_bigram_frequencies(file_path: str) -> Dict[str, float]:
    """Loads English bigram frequencies from a file and normalizes them."""
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
        if total > 0:
            for bigram in freqs:
                freqs[bigram] /= total
        return freqs
    except FileNotFoundError:
        return {}

def calculate_bigram_score(text: str, expected_freqs: Dict[str, float]) -> float:
    """Calculates a fitness score using the Chi-Squared statistic for bigrams."""
    if not expected_freqs:
        return float('inf') # Cannot score if frequency data is missing

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
    floor = 0.01 # A small floor value to prevent division by zero

    for bigram, expected_prob in expected_freqs.items():
        observed_count = observed_counts.get(bigram, 0)
        expected_count = expected_prob * total_bigrams
        difference = observed_count - expected_count
        chi_squared_score += (difference * difference) / max(expected_count, floor)

    return chi_squared_score

# --- CAESAR CIPHER FUNCTIONS ---

def caesar_decipher(ciphertext, shift, alphabet):
    """
    Decrypts text using a Caesar cipher with a given shift value.
    Note: A positive shift here means shifting the letters to the left (decryption).
    """
    decrypted_text = []
    alphabet_length = len(alphabet)
    
    for char in ciphertext:
        if char in alphabet:
            char_index = alphabet.index(char)
            # Apply the shift (subtract for decryption)
            decrypted_index = (char_index - shift + alphabet_length) % alphabet_length
            decrypted_char = alphabet[decrypted_index]
            decrypted_text.append(decrypted_char.lower())
        else:
            # Keep non-alphabet characters as they are
            decrypted_text.append(char.lower())
            
    return ''.join(decrypted_text)

def brute_force_caesar(ciphertext, alphabet):
    """
    Tries all possible shifts, scores them, and returns a ranked list of results.
    """
    print(f"\n{GREY}Loading bigram frequency data...{RESET}")
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram frequency file not found at '{bigram_freq_path}'. Cannot calculate bigram scores.{RESET}")

    results = []
    alphabet_length = len(alphabet)
    
    print(f"{GREY}Brute forcing {alphabet_length} possible shifts...{RESET}")

    for shift in range(alphabet_length):
        decrypted_text = caesar_decipher(ciphertext, shift, alphabet)
        ioc = calculate_ioc(decrypted_text)
        bigram_score = calculate_bigram_score(decrypted_text, expected_freqs)
        
        results.append({
            'shift': shift,
            'text': decrypted_text,
            'ioc': ioc,
            'bigram_score': bigram_score
        })
    
    # Sort results: primary key is bigram_score (lower is better), secondary key is IoC (higher is better)
    results.sort(key=lambda x: (x['bigram_score'], -x['ioc']))
    
    return results

def save_caesar_results(filename, results):
    """Saves the full list of brute-force results to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Caesar Cipher Brute-Force Results\n")
            f.write("===================================\n\n")
            for result in results:
                f.write(f"Shift: {result['shift']}\n")
                f.write(f"IoC Score: {result['ioc']:.4f}\n")
                f.write(f"Bigram Score: {result['bigram_score']:.2f}\n")
                f.write(f"Plaintext: {result['text']}\n")
                f.write("-" * 20 + "\n")
        print(f"{GREEN}Results successfully saved to {filename}{RESET}")
    except IOError as e:
        print(f"{RED}Error: Could not save file. {e}{RESET}")

# --- MAIN APPLICATION ---

def run():
    """Main function to run the Caesar decipher CLI."""
    print(f"{RED}============================={RESET}")
    print(f"{RED}= Caesar Shift Cipher Tool  ={RESET}")
    print(f"{RED}============================={RESET}")
    print(f"{GREY}This tool decrypts a Caesar cipher using a specific key or by brute force.{RESET}")
    print(f"{GREY}-{RESET}" * 60)
    
    # Get user inputs
    ciphertext = input(f"Enter the ciphertext: {GREEN}").upper()
    
    alphabet_input = input(f"{RESET}Enter custom alphabet (press Enter for default {YELLOW}A-Z{RESET}): {GREEN}").upper()
    alphabet = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"{RESET}Using alphabet: {YELLOW}{alphabet}{RESET}")
    
    # Display mode options
    print(f"\n{RESET}Choose a mode:")
    print(f"  ({YELLOW}1{RESET}) Decrypt with a specific shift value")
    print(f"  ({YELLOW}2{RESET}) Brute-force all possible shifts and rank results")
    
    mode_choice = input(f"Enter your choice (1 or 2): {GREEN}")
    print(f"{RESET}{GREY}-{RESET}" * 60)

    if mode_choice == '1':
        # Get the specific shift value from the user
        while True:
            try:
                shift_input = input(f"{RESET}Enter the shift value (e.g., 3 or -3): {GREEN}")
                shift = int(shift_input)
                # Perform single decryption
                decrypted_text = caesar_decipher(ciphertext, shift, alphabet)
                # Display the result
                print(f"\n{YELLOW}Ciphertext:{RESET} {ciphertext.lower()}")
                print(f"{YELLOW}Shift Value:{RESET} {shift}")
                print(f"{YELLOW}Decrypted Text:{RESET} {decrypted_text}")
                break
            except ValueError:
                print(f"{RED}Invalid input. Please enter an integer.{RESET}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{GREY}Program exited.{RESET}")
                sys.exit(0)

    elif mode_choice == '2':
        results = brute_force_caesar(ciphertext, alphabet)
        print(f"{YELLOW}--- Top 10 Brute-Force Results (Best First) ---{RESET}")
        for i, result in enumerate(results[:10]):
            color = GREEN if i == 0 else ""
            print(f"{color}Shift: {result['shift']:<2} | IoC: {result['ioc']:.4f} | Bigram Score: {result['bigram_score']:.2f} | Text: {result['text'].upper()}{RESET}")

        # Ask user to save results
        save_choice = input(f"\n{GREY}Save all {len(results)} results to a file? ({YELLOW}Y/N{RESET}): {GREEN}").upper()
        if save_choice == 'Y':
            filename = input(f"{RESET}Enter filename (default: caesar_results.txt): {GREEN}") or "caesar_results.txt"
            save_caesar_results(filename, results)
    
    else:
        print(f"{RED}Invalid choice. Please run the script again and select 1 or 2.{RESET}")

    print(f"\n{GREY}Program finished.{RESET}")

if __name__ == "__main__":
    # Create data directory and dummy bigram file if they don't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file at '{bigram_freq_path}'...{RESET}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\nRE 0.68\nES 0.59\n")
            f.write("ON 0.57\nST 0.55\nNT 0.51\nEN 0.50\nAT 0.46\nED 0.44\nND 0.42\n")

    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{GREY}Program exited.{RESET}")