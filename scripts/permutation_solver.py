import os
import sys
import math
from itertools import permutations
from collections import Counter

# Standard ANSI colors (Matching your style)
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;4m'

# Path setup
data_dir = os.path.join(os.path.dirname(__file__), "data")
bigram_freq_path = os.path.join(data_dir, "english_bigrams.txt")

# --- UTILS (Re-implemented for standalone safety) ---

def load_bigram_frequencies():
    freqs = {}
    if not os.path.exists(bigram_freq_path): return {}
    with open(bigram_freq_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                freqs[parts[0].upper()] = float(parts[1])
    return freqs

def calculate_score(text, expected_freqs):
    """Calculates a simple fitness score based on Bigrams."""
    if not expected_freqs: return 0
    score = 0
    text = "".join(c for c in text.upper() if c.isalpha())
    for i in range(len(text) - 1):
        bg = text[i:i+2]
        score += expected_freqs.get(bg, 0)
    return score

def calculate_ioc(text):
    text = ''.join(c for c in text.upper() if c.isalpha())
    if len(text) <= 1: return 0.0
    counts = Counter(text)
    numerator = sum(n * (n - 1) for n in counts.values())
    denominator = len(text) * (len(text) - 1)
    return numerator / denominator

# --- CORE LOGIC ---

def get_factors(n):
    """Get all factors of n to determine possible matrix widths."""
    factors = []
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def decrypt_permutation(text, width, perm):
    """
    Reorders columns based on 'perm' tuple (e.g., (0, 2, 1)).
    Reads out rows.
    """
    num_rows = len(text) // width
    # 1. Create the grid (row by row)
    grid = [text[i * width : (i + 1) * width] for i in range(num_rows)]
    
    # 2. Reorder columns in every row
    new_grid = []
    for row in grid:
        new_row = "".join(row[p] for p in perm)
        new_grid.append(new_row)
    
    # 3. Read out text (Row by Row)
    return "".join(new_grid)

def solve_for_width(ciphertext, width, expected_freqs):
    """Generates ALL column permutations for a specific width."""
    # Column indices: 0, 1, 2 ... width-1
    cols = list(range(width))
    perms = list(permutations(cols))
    
    results = []
    total = len(perms)
    print(f"{GREY}  -> Testing width {width}: {total} permutations...{RESET}")

    for p in perms:
        plaintext = decrypt_permutation(ciphertext, width, p)
        score = calculate_score(plaintext, expected_freqs)
        ioc = calculate_ioc(plaintext)
        
        results.append({
            'width': width,
            'perm': p,
            'text': plaintext,
            'score': score,
            'ioc': ioc
        })
    return results

def run():
    print(f"{GREEN}=========================================={RESET}")
    print(f"{GREEN}= Full Matrix Permutation Solver         ={RESET}")
    print(f"{GREEN}=========================================={RESET}")
    print(f"{GREY}Generates all column transpositions for valid matrix sizes.{RESET}")

    # 1. Get Input (Using basic input here, but you can import the new util)
    ciphertext = input(f"\n{GREY}Enter ciphertext (clean, no spaces preferred): {RESET}").strip().upper().replace(" ", "")
    if not ciphertext: return

    length = len(ciphertext)
    factors = get_factors(length)
    
    if not factors:
        print(f"{RED}Prime length ({length}). Cannot form a rectangular matrix.{RESET}")
        return

    print(f"{BLUE}Text Length:{RESET} {length}")
    print(f"{BLUE}Possible widths:{RESET} {factors}")
    
    # 2. Load Data
    freqs = load_bigram_frequencies()
    if not freqs: print(f"{YELLOW}Warning: Bigrams not found. Scoring will be poor.{RESET}")

    all_results = []

    # 3. Process
    for width in factors:
        if width > 9:
            print(f"{YELLOW}Skipping width {width} (Too many permutations: {math.factorial(width):,}){RESET}")
            continue
            
        width_results = solve_for_width(ciphertext, width, freqs)
        all_results.extend(width_results)

    # 4. Sort (High Bigram Score + High IoC preferred)
    # Weighting: Score is roughly 0-1000 depending on length. IoC is 0.0-0.1. 
    # Let's simple sort by Bigram Score descending.
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # 5. Save/Display
    print(f"\n{YELLOW}Top 5 Results:{RESET}")
    for i, res in enumerate(all_results[:5]):
        print(f"[{i+1}] Width: {res['width']} | Perm: {res['perm']} | IoC: {res['ioc']:.4f}")
        print(f"    {GREEN}{res['text'][:60]}...{RESET}")

    save_choice = input(f"\n{GREY}Save all results to .txt? (Y/N): {RESET}").upper()
    if save_choice == 'Y':
        filename = "permutation_results.txt"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"Permutation Brute Force Results for length {length}\n")
            f.write("===================================================\n\n")
            for res in all_results:
                f.write(f"Width: {res['width']}, Permutation: {res['perm']}\n")
                f.write(f"Bigram Score: {res['score']:.2f}, IoC: {res['ioc']:.4f}\n")
                f.write(f"Text: {res['text']}\n")
                f.write("-" * 40 + "\n")
        print(f"{GREEN}Saved to {filename}{RESET}")

if __name__ == "__main__":
    run()