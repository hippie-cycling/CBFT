import os
import math
from functools import lru_cache
from typing import List, Dict

# --- DUMMY UTILS CLASS (to make script standalone) ---
class DummyUtils:
    """Provides utility functions to make the script standalone."""
    def save_results_to_file(self, results: List[Dict], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Scytale Cipher Brute-Force Results\n")
                f.write("=====================================\n\n")
                for result in results:
                    f.write(f"Key (Sides/Diameter): {result['key']}\n")
                    f.write(f"IoC Score: {result.get('ioc', 0):.4f}\n")
                    f.write(f"Bigram Score: {result.get('bigram_score', 0):.2f}\n")
                    f.write(f"Decrypted: {result['plaintext']}\n")
                    f.write("-" * 20 + "\n")
            print(f"{GREEN}Results successfully saved to {filename}{RESET}")
        except IOError as e:
            print(f"{RED}Error saving file: {e}{RESET}")

utils = DummyUtils()

# --- COLOR CODES AND PATHS ---
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;21m'
RESET = '\033[0m'

data_dir = os.path.join(os.path.dirname(__file__), "data")
bigram_freq_path = os.path.join(data_dir, "english_bigrams.txt")

# --- SCORING FUNCTIONS ---
def calculate_ioc(text: str) -> float:
    """Calculates the Index of Coincidence for a given text."""
    text = ''.join(filter(str.isalpha, text.upper()))
    n = len(text)
    if n < 2: return 0.0
    freqs = {char: text.count(char) for char in set(text)}
    numerator = sum(count * (count - 1) for count in freqs.values())
    denominator = n * (n - 1)
    return numerator / denominator if denominator > 0 else 0.0

@lru_cache(maxsize=None)
def load_bigram_frequencies(file_path: str) -> Dict[str, float]:
    """Loads and normalizes English bigram frequencies."""
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
    """Calculates a fitness score based on bigram frequencies (lower is better)."""
    if not expected_freqs: return float('inf')
    text = "".join(char for char in text.upper() if char.isalpha())
    if len(text) < 2: return float('inf')
    
    observed_counts = {}
    total_bigrams = len(text) - 1
    for i in range(total_bigrams):
        bigram = text[i:i+2]
        observed_counts[bigram] = observed_counts.get(bigram, 0) + 1
        
    chi_squared_score = 0.0
    floor = 0.01
    for bigram, expected_prob in expected_freqs.items():
        observed_count = observed_counts.get(bigram, 0)
        expected_count = expected_prob * total_bigrams
        difference = observed_count - expected_count
        chi_squared_score += (difference ** 2) / max(expected_count, floor)
        
    return chi_squared_score

# --- SCYTALE CORE FUNCTIONS ---

def decrypt_scytale(ciphertext: str, sides: int) -> str:
    """
    Decrypts a Scytale cipher by simulating wrapping the text around a rod.
    The number of 'sides' is equivalent to the number of columns in the grid.
    """
    if not isinstance(sides, int) or sides <= 1:
        return ""
        
    n = len(ciphertext)
    if n == 0:
        return ""
        
    # The number of rows is determined by how many full 'wraps' are made.
    rows = math.ceil(n / sides)
    
    # Calculate how many columns are 'full' vs. 'short'.
    num_full_cols = n % sides
    if num_full_cols == 0 and n > 0:
        num_full_cols = sides
        
    grid = [['' for _ in range(sides)] for _ in range(rows)]
    
    text_idx = 0
    # Fill the grid column by column
    for col in range(sides):
        # The first `num_full_cols` have `rows` characters. The rest have `rows - 1`.
        rows_in_this_col = rows if col < num_full_cols else rows - 1
        
        for row in range(rows_in_this_col):
            if text_idx < n:
                grid[row][col] = ciphertext[text_idx]
                text_idx += 1
    
    # Read the plaintext row by row to decrypt.
    plaintext = "".join(grid[row][col] for row in range(rows) for col in range(sides))
    
    return plaintext


def brute_force_scytale(ciphertext: str):
    """
    Tries all possible numbers of sides for the Scytale and ranks the results.
    """
    print(f"\n{GREY}Loading bigram frequency data...{RESET}")
    expected_freqs = load_bigram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram frequency file not found. Scoring will be less accurate.{RESET}")

    results = []
    # We only need to check up to half the length, as sides > len/2 is trivial.
    max_sides = len(ciphertext) // 2
    
    print(f"{GREY}Brute forcing {max_sides - 1} possible key diameters...{RESET}")

    for sides in range(2, max_sides + 1):
        plaintext = decrypt_scytale(ciphertext, sides)
        if not plaintext:
            continue
            
        ioc = calculate_ioc(plaintext)
        bigram_score = calculate_bigram_score(plaintext, expected_freqs)
        
        results.append({
            'key': sides,
            'plaintext': plaintext,
            'ioc': ioc,
            'bigram_score': bigram_score
        })
    
    # Sort results: primary key is bigram_score (lower is better), secondary is IoC (higher is better)
    results.sort(key=lambda x: (x['bigram_score'], -x['ioc']))
    
    return results

# --- MAIN APPLICATION ---

def run():
    """Main function to run the Scytale cipher tool CLI."""
    print(f"{BLUE}=========================={RESET}")
    print(f"{BLUE}=   Scytale Cipher Tool  ={RESET}")
    print(f"{BLUE}=========================={RESET}")
    print(f"{GREY}A simple transposition cipher used in ancient Greece.{RESET}")
    print(f"{GREY}-{RESET}" * 60)
    
    ciphertext = input(f"Enter the ciphertext: {GREEN}").upper().replace(" ", "")
    
    print(f"\n{RESET}Choose a mode:")
    print(f"  ({YELLOW}1{RESET}) Decrypt with a specific key (number of sides/diameter)")
    print(f"  ({YELLOW}2{RESET}) Brute-force all possible key diameters and rank results")
    
    mode_choice = input(f"Enter your choice (1 or 2): {GREEN}")
    print(f"{RESET}{GREY}-{RESET}" * 60)

    if mode_choice == '1':
        while True:
            try:
                sides = int(input(f"{RESET}Enter the key (number of sides): {GREEN}"))
                if sides <= 1:
                    print(f"{RED}Number of sides must be greater than 1.{RESET}")
                    continue
                
                plaintext = decrypt_scytale(ciphertext, sides)
                
                print(f"\n{YELLOW}Ciphertext:{RESET} {ciphertext}")
                print(f"{YELLOW}Key (Sides):{RESET} {sides}")
                print(f"{YELLOW}Decrypted Text:{RESET} {plaintext}")
                break
            except ValueError:
                print(f"{RED}Invalid input. Please enter an integer.{RESET}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{GREY}Program exited.{RESET}")
                return

    elif mode_choice == '2':
        results = brute_force_scytale(ciphertext)
        
        if not results:
            print(f"{RED}Could not generate any brute-force results.{RESET}")
            return
            
        print(f"{YELLOW}--- Top 10 Brute-Force Results (Best First) ---{RESET}")
        for i, result in enumerate(results[:10]):
            color = GREEN if i == 0 else ""
            print(f"{color}Key: {result['key']:<3} | IoC: {result['ioc']:.4f} | Bigram Score: {result['bigram_score']:.2f} | Text: {result['plaintext']}{RESET}")

        save_choice = input(f"\n{GREY}Save all {len(results)} results to a file? ({YELLOW}Y/N{RESET}): {GREEN}").upper()
        if save_choice == 'Y':
            filename = input(f"{RESET}Enter filename (default: scytale_results.txt): {GREEN}") or "scytale_results.txt"
            utils.save_results_to_file(results, filename)
    
    else:
        print(f"{RED}Invalid choice. Please run the script again and select 1 or 2.{RESET}")

    print(f"\n{GREY}Program finished.{RESET}")

if __name__ == "__main__":
    # Create dummy data file if it doesn't exist for standalone execution
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        print(f"{GREY}Creating dummy bigram frequency file at '{bigram_freq_path}'...{RESET}")
        with open(bigram_freq_path, 'w') as f:
            f.write("TH 1.52\nHE 1.28\nIN 0.94\nER 0.94\nAN 0.82\n")

    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{GREY}Program exited.{RESET}")

