import random
import os

RED, YELLOW, GREY, RESET, GREEN = '\033[38;5;88m', '\033[38;5;3m', '\033[38;5;238m', '\033[0m', '\033[38;5;2m'
bigram_freq_path = os.path.join(os.path.dirname(__file__), "data", "english_bigrams.txt")

def load_bigram_frequencies():
    freqs = {}
    try:
        with open(bigram_freq_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2: freqs[parts[0].upper()] = float(parts[1])
        total = sum(freqs.values())
        return {k: v / total for k, v in freqs.items()}
    except: return {}

def decrypt_transposition(ciphertext, key_order):
    cols = len(key_order)
    rows = len(ciphertext) // cols
    remainder = len(ciphertext) % cols
    
    # Calculate column lengths
    col_lengths = [rows + 1 if k < remainder else rows for k in key_order]
    
    # Read columns from ciphertext
    columns = {}
    idx = 0
    for i, length in enumerate(col_lengths):
        columns[key_order[i]] = ciphertext[idx:idx+length]
        idx += length
        
    # Read horizontally to get plaintext
    plaintext = ""
    for r in range(rows + 1):
        for c in range(cols):
            if c in columns and r < len(columns[c]):
                plaintext += columns[c][r]
    return plaintext

def calculate_fitness(text, expected_freqs):
    if len(text) < 2: return float('inf')
    observed = {}
    total = 0
    for i in range(len(text)-1):
        bg = text[i:i+2]
        observed[bg] = observed.get(bg, 0) + 1
        total += 1
        
    score = 0.0
    for bg, exp_prob in expected_freqs.items():
        obs = observed.get(bg, 0)
        exp = exp_prob * total
        diff = obs - exp
        score += (diff * diff) / max(exp, 0.01)
    return score

def run():
    print(f"{GREEN}===================================={RESET}")
    print(f"{GREEN}== Hill Climbing (Transposition)  =={RESET}")
    print(f"{GREEN}===================================={RESET}")
    
    ciphertext = input(f"{GREY}Enter ciphertext: {RESET}").upper()
    ciphertext = "".join(filter(str.isalpha, ciphertext))
    if not ciphertext: return
    
    key_len = int(input(f"{GREY}Enter suspected key length (number of columns): {RESET}"))
    freqs = load_bigram_frequencies()
    
    print(f"\n{YELLOW}Climbing the hill...{RESET}")
    
    # Start with a random column order
    current_key = list(range(key_len))
    random.shuffle(current_key)
    current_score = calculate_fitness(decrypt_transposition(ciphertext, current_key), freqs)
    
    iterations = 5000
    for i in range(iterations):
        # Mutate: swap two random columns
        idx1, idx2 = random.sample(range(key_len), 2)
        new_key = list(current_key)
        new_key[idx1], new_key[idx2] = new_key[idx2], new_key[idx1]
        
        new_text = decrypt_transposition(ciphertext, new_key)
        new_score = calculate_fitness(new_text, freqs)
        
        # Hill climb: Only accept strictly better (lower) scores
        if new_score < current_score:
            current_key = new_key
            current_score = new_score
            print(f"Improved Score: {GREEN}{current_score:.2f}{RESET} | Text: {new_text[:50]}...")
            
    final_text = decrypt_transposition(ciphertext, current_key)
    print(f"\n{YELLOW}FINAL RESULT:{RESET}")
    print(f"Key Order: {current_key}")
    print(f"Plaintext: {GREEN}{final_text}{RESET}")