# scripts/simulated_annealing.py
import random
import math
import time
import os
from collections import Counter

# ANSI Colors
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
bigram_freq_path = os.path.join(os.path.dirname(__file__), "data", "english_bigrams.txt")

def load_bigram_frequencies(file_path):
    freqs = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    freqs[parts[0].upper()] = float(parts[1])
        # Normalize
        total = sum(freqs.values())
        return {k: v / total for k, v in freqs.items()}
    except FileNotFoundError:
        print(f"{RED}Error: Bigram file not found at {file_path}{RESET}")
        return {}

def decrypt_substitution(ciphertext, key):
    """Decrypts text given a 26-letter substitution key."""
    translation_table = str.maketrans(key, ALPHABET)
    return ciphertext.translate(translation_table)

def calculate_fitness(text, expected_freqs):
    """Calculates Chi-Squared score for bigrams (Lower is better)."""
    text = "".join(filter(str.isalpha, text.upper()))
    if len(text) < 2: return float('inf')

    observed = Counter(text[i:i+2] for i in range(len(text)-1))
    total_bigrams = sum(observed.values())
    
    score = 0.0
    floor = 0.01 
    for bigram, expected_prob in expected_freqs.items():
        obs_count = observed.get(bigram, 0)
        exp_count = expected_prob * total_bigrams
        diff = obs_count - exp_count
        score += (diff * diff) / max(exp_count, floor)
        
    return score

def modify_key(key):
    """Swaps two random letters in the key to mutate it."""
    idx1, idx2 = random.sample(range(26), 2)
    key_list = list(key)
    key_list[idx1], key_list[idx2] = key_list[idx2], key_list[idx1]
    return "".join(key_list)

def run_simulated_annealing(ciphertext, expected_freqs, starting_temp=20.0, cooling_rate=0.995, steps_per_temp=1000):
    """
    The core Simulated Annealing loop.
    Starts hot (accepts bad mutations to explore), slowly cools down (only accepts good mutations).
    """
    # Initialize a completely random key
    current_key = "".join(random.sample(ALPHABET, 26))
    current_text = decrypt_substitution(ciphertext, current_key)
    current_score = calculate_fitness(current_text, expected_freqs)
    
    best_key = current_key
    best_score = current_score
    
    temp = starting_temp
    iteration = 0
    start_time = time.time()
    
    print(f"\n{YELLOW}Starting Simulated Annealing...{RESET}")
    print(f"{GREY}Starting Temp: {temp} | Cooling Rate: {cooling_rate}{RESET}\n")

    while temp > 0.1: # Stop when the "metal" has cooled
        for _ in range(steps_per_temp):
            # 1. Mutate the key
            new_key = modify_key(current_key)
            new_text = decrypt_substitution(ciphertext, new_key)
            new_score = calculate_fitness(new_text, expected_freqs)
            
            # 2. Check if the mutation is better (lower score is better for Chi-Squared)
            score_diff = current_score - new_score 
            
            # 3. If better, ACCEPT. If worse, maybe ACCEPT based on temperature.
            if score_diff > 0:
                # Better! Accept it.
                current_key = new_key
                current_score = new_score
                
                # Keep track of the all-time best
                if new_score < best_score:
                    best_score = new_score
                    best_key = new_key
                    
            elif temp > 0:
                # Worse. Calculate probability to accept it anyway.
                # math.exp takes a negative number here, returning a float between 0.0 and 1.0
                probability = math.exp(score_diff / temp) 
                if random.random() < probability:
                    current_key = new_key
                    current_score = new_score
        
        # Cool down the temperature for the next outer loop
        temp *= cooling_rate
        iteration += 1
        
        # Print progress every few iterations
        if iteration % 10 == 0:
            print(f"Temp: {temp:>5.2f} | Best Score: {GREEN}{best_score:>8.2f}{RESET} | Text: {decrypt_substitution(ciphertext, best_key)[:40]}...")

    time_taken = time.time() - start_time
    print(f"\n{GREEN}Annealing Complete in {time_taken:.2f}s!{RESET}")
    return best_key, best_score, decrypt_substitution(ciphertext, best_key)

def run():
    print(f"{GREEN}===================================={RESET}")
    print(f"{GREEN}=====    Simulated Annealing   ====={RESET}")
    print(f"{GREEN}=====  (Simple Substitution)   ====={RESET}")
    print(f"{GREEN}===================================={RESET}")
    
    ciphertext = input(f"{GREY}Enter substitution ciphertext: {RESET}").upper()
    if not ciphertext:
        return
        
    freqs = load_bigram_frequencies(bigram_freq_path)
    if not freqs: return
    
    # Because SA has a random starting point, it's standard practice to run it 
    # a few times from scratch to ensure it didn't get stuck in a "local maximum".
    runs = 3 
    overall_best_key = ""
    overall_best_score = float('inf')
    overall_best_text = ""
    
    for i in range(runs):
        print(f"\n{YELLOW}--- PASS {i+1} OF {runs} ---{RESET}")
        key, score, text = run_simulated_annealing(ciphertext, freqs)
        
        if score < overall_best_score:
            overall_best_score = score
            overall_best_key = key
            overall_best_text = text
            
    print(f"\n{YELLOW}================ FINAL BEST RESULT ================{RESET}")
    print(f"Key Mapping: {ALPHABET}  <-- Plaintext")
    print(f"             {overall_best_key}  <-- Ciphertext")
    print(f"Fitness Score: {GREEN}{overall_best_score:.2f}{RESET}")
    print(f"\nPlaintext:\n{overall_best_text}")

if __name__ == "__main__":
    run()