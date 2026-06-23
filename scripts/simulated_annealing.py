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

def calculate_ioc(text: str) -> float:
    text = ''.join(filter(str.isalpha, text.upper()))
    n = len(text)
    if n < 2: return 0.0
    freqs = Counter(text)
    numerator = sum(count * (count - 1) for count in freqs.values())
    denominator = n * (n - 1)
    return numerator / denominator if denominator > 0 else 0.0

def load_bigram_frequencies(file_path):
    freqs = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    freqs[parts[0].upper()] = float(parts[1])
        total = sum(freqs.values())
        return {k: v / total for k, v in freqs.items()}
    except FileNotFoundError:
        print(f"{RED}Error: Bigram file not found at {file_path}{RESET}")
        return {}

def decrypt_substitution(ciphertext, key):
    translation_table = str.maketrans(key, ALPHABET)
    return ciphertext.translate(translation_table)

def calculate_fitness(text, expected_freqs):
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
    idx1, idx2 = random.sample(range(26), 2)
    key_list = list(key)
    key_list[idx1], key_list[idx2] = key_list[idx2], key_list[idx1]
    return "".join(key_list)

def run_simulated_annealing(ciphertext, expected_freqs, starting_temp, cooling_rate, steps_per_temp):
    current_key = "".join(random.sample(ALPHABET, 26))
    current_text = decrypt_substitution(ciphertext, current_key)
    current_score = calculate_fitness(current_text, expected_freqs)
    
    best_key = current_key
    best_score = current_score
    
    temp = starting_temp
    iteration = 0
    start_time = time.time()
    
    print(f"\n{YELLOW}Starting Simulated Annealing...{RESET}")
    print(f"{GREY}Starting Temp: {temp} | Cooling Rate: {cooling_rate} | Steps/Temp: {steps_per_temp}{RESET}\n")

    try:
        while temp > 0.1:
            for _ in range(steps_per_temp):
                new_key = modify_key(current_key)
                new_text = decrypt_substitution(ciphertext, new_key)
                new_score = calculate_fitness(new_text, expected_freqs)
                
                score_diff = current_score - new_score 
                
                if score_diff > 0:
                    current_key = new_key
                    current_score = new_score
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_key = new_key
                        
                elif temp > 0:
                    probability = math.exp(score_diff / temp) 
                    if random.random() < probability:
                        current_key = new_key
                        current_score = new_score
            
            temp *= cooling_rate
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"Temp: {temp:>5.2f} | Best Score: {GREEN}{best_score:>8.2f}{RESET} | Text: {decrypt_substitution(ciphertext, best_key)[:40]}...", end='\r')
    except KeyboardInterrupt:
        print("\nManually interrupted.")

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

    # Prompt for SA Parameters
    print(f"\n{GREY}Configure Simulated Annealing Parameters:{RESET}")
    
    t_in = input(f"Starting Temperature (default {YELLOW}20.0{RESET}): ")
    starting_temp = float(t_in) if t_in else 20.0
    
    c_in = input(f"Cooling Rate (default {YELLOW}0.995{RESET}): ")
    cooling_rate = float(c_in) if c_in else 0.995
    
    s_in = input(f"Steps per temperature (default {YELLOW}1000{RESET}): ")
    steps_per_temp = int(s_in) if s_in else 1000
    
    r_in = input(f"Number of passes to run (default {YELLOW}3{RESET}): ")
    runs = int(r_in) if r_in else 3
    
    overall_best_key = ""
    overall_best_score = float('inf')
    overall_best_text = ""
    
    for i in range(runs):
        print(f"\n{YELLOW}--- PASS {i+1} OF {runs} ---{RESET}")
        key, score, text = run_simulated_annealing(ciphertext, freqs, starting_temp, cooling_rate, steps_per_temp)
        
        if score < overall_best_score:
            overall_best_score = score
            overall_best_key = key
            overall_best_text = text
            
    final_ioc = calculate_ioc(overall_best_text)
            
    print(f"\n{YELLOW}================ FINAL BEST RESULT ================{RESET}")
    print(f"Key Mapping: {ALPHABET}  <-- Plaintext")
    print(f"             {overall_best_key}  <-- Ciphertext")
    print(f"Fitness Score: {GREEN}{overall_best_score:.2f}{RESET}")
    print(f"Final IoC:     {GREEN}{final_ioc:.4f}{RESET}")
    print(f"\nPlaintext:\n{overall_best_text}")

if __name__ == "__main__":
    run()