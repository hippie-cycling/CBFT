import os
import math
from collections import Counter
from typing import Dict, List

# --- DUMMY UTILS CLASS ---
class DummyUtils:
    def calculate_ioc(self, text: str) -> float:
        text = ''.join(filter(str.isalpha, text.upper()))
        n = len(text)
        if n < 2: return 0.0
        freqs = Counter(text)
        numerator = sum(count * (count - 1) for count in freqs.values())
        denominator = n * (n - 1)
        return numerator / denominator if denominator > 0 else 0.0

    def save_results_to_file(self, results: List[Dict], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Affine Cipher Brute-Force Results\n")
                f.write("===================================\n\n")
                for result in results:
                    f.write(f"Key: A={result['a']}, B={result['b']}\n")
                    f.write(f"Alphabet: {result['alphabet']}\n")
                    f.write(f"IoC Score: {result.get('ioc', 0):.4f}\n")
                    f.write(f"Bigram Score: {result.get('score', 0):.2f}\n")
                    f.write(f"Decrypted: {result['plaintext']}\n")
                    f.write("-" * 20 + "\n")
            print(f"{GREEN}Results successfully saved to {filename}{RESET}")
        except IOError as e:
            print(f"{RED}Error saving file: {e}{RESET}")

utils = DummyUtils()

# --- COLOR CODES AND PATHS ---
RED, YELLOW, GREY, GREEN, BLUE, RESET = '\033[38;5;88m', '\033[38;5;3m', '\033[38;5;238m', '\033[38;5;2m', '\033[38;5;21m', '\033[0m'
try: data_dir = os.path.join(os.path.dirname(__file__), "data")
except NameError: data_dir = "data"

bigram_freq_path = os.path.join(data_dir, "english_bigrams.txt")

# --- SCORING FUNCTIONS ---
def load_ngram_frequencies(file_path: str) -> Dict[str, float]:
    freqs = {}
    if not os.path.exists(file_path): return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2: freqs[parts[0].upper()] = float(parts[1])
    total = sum(freqs.values())
    if total > 0:
        for ngram in freqs: freqs[ngram] /= total
    return freqs

def calculate_ngram_score(text: str, expected_freqs: Dict[str, float], n: int = 2) -> float:
    if not expected_freqs: return float('inf')
    text = "".join(filter(str.isalpha, text.upper()))
    if len(text) < n: return float('inf')
    
    observed_counts = Counter(text[i:i+n] for i in range(len(text) - n + 1))
    total_ngrams = sum(observed_counts.values())
    if total_ngrams == 0: return float('inf')
    
    chi_squared_score = 0
    floor = 0.01
    for ngram, expected_prob in expected_freqs.items():
        observed_count = observed_counts.get(ngram, 0)
        expected_count = expected_prob * total_ngrams
        chi_squared_score += ((observed_count - expected_count) ** 2) / max(expected_count, floor)
    return chi_squared_score

# --- AFFINE CORE FUNCTIONS ---
def get_alphabet(alphabet_str: str):
    alphabet = ''.join(dict.fromkeys(alphabet_str.upper())) # Remove duplicates
    return alphabet, {char: i for i, char in enumerate(alphabet)}

def decrypt_affine(ciphertext: str, a: int, b: int, alphabet_str: str) -> str:
    alphabet, alphabet_dict = get_alphabet(alphabet_str)
    m = len(alphabet)
    
    try:
        mod_inv = pow(a, -1, m)
    except ValueError:
        return "" # 'a' and 'm' are not coprime
        
    plaintext = []
    for char in ciphertext:
        char_upper = char.upper()
        if char_upper in alphabet_dict:
            y = alphabet_dict[char_upper]
            x = (mod_inv * (y - b)) % m
            decrypted_char = alphabet[x]
            plaintext.append(decrypted_char.lower() if char.islower() else decrypted_char)
        else:
            plaintext.append(char)
    return ''.join(plaintext)

def brute_force_affine(ciphertext: str, alphabet_str: str, expected_freqs: Dict):
    alphabet, _ = get_alphabet(alphabet_str)
    m = len(alphabet)
    
    # Calculate all valid 'a' values (must be coprime to m)
    valid_a = [a for a in range(1, m) if math.gcd(a, m) == 1]
    total_keys = len(valid_a) * m
    
    print(f"\n{YELLOW}Alphabet Length (m): {m}{RESET}")
    print(f"{YELLOW}Valid multipliers (a): {len(valid_a)} (Coprimes of {m}){RESET}")
    print(f"{YELLOW}Total Keyspace: {total_keys} combinations{RESET}")
    print(f"{GREY}Executing brute force...{RESET}")
    
    results = []
    for a in valid_a:
        for b in range(m):
            pt = decrypt_affine(ciphertext, a, b, alphabet_str)
            if not pt: continue
            
            ioc = utils.calculate_ioc(pt)
            score = calculate_ngram_score(pt, expected_freqs)
            results.append({
                'a': a, 'b': b, 'alphabet': alphabet_str,
                'plaintext': pt, 'ioc': ioc, 'score': score
            })
            
    # Rank by Bigram Score (lowest is best)
    results.sort(key=lambda x: x['score'])
    return results

def run():
    print(f"{GREEN}===================================={RESET}")
    print(f"{GREEN}=====       Affine Cipher      ====={RESET}")
    print(f"{GREEN}===================================={RESET}")
    
    ciphertext = input(f"{GREY}Enter ciphertext: {RESET}")
    if not ciphertext: return
    
    alphabet_input = input(f"{GREY}Enter custom alphabet (default: A-Z): {RESET}").upper().strip()
    alphabet_str = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    expected_freqs = load_ngram_frequencies(bigram_freq_path)
    if not expected_freqs:
        print(f"{RED}Warning: Bigram file not found. Scoring will be disabled.{RESET}")
        
    results = brute_force_affine(ciphertext, alphabet_str, expected_freqs)
    
    if results:
        print(f"\n{YELLOW}--- TOP 10 MATCHES ---{RESET}")
        for i, res in enumerate(results[:10]):
            print(f"Rank #{i+1}: Key: {YELLOW}A={res['a']}, B={res['b']}{RESET}")
            print(f"Scores: Bigram: {YELLOW}{res['score']:.2f}{RESET} | IoC: {YELLOW}{res['ioc']:.4f}{RESET}")
            print(f"Text: {res['plaintext'][:80]}...\n{GREY}{'-'*40}{RESET}")
            
        save = input(f"\n{GREY}Save results to file? (Y/N): {RESET}").strip().upper()
        if save == 'Y':
            filename = input(f"{GREY}Enter filename (default: affine_results.txt): {RESET}").strip()
            utils.save_results_to_file(results, filename or "affine_results.txt")
    else:
        print(f"{RED}No solutions found.{RESET}")

if __name__ == "__main__":
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(bigram_freq_path):
        with open(bigram_freq_path, 'w') as f: f.write("TH 1.52\nHE 1.28\nIN 0.94\n")
    run()