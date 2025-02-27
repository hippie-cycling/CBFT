#!/usr/bin/env python3

import os
import datetime

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;4m'

# Hardcoded path for words_alpha.txt
WORDLIST_PATH = "words_alpha.txt"

def xor_strings(message, key):
    """XOR each character of the message with the corresponding character of the key."""
    # Extend the key if necessary to match the message length
    if len(key) < len(message):
        key = key * (len(message) // len(key) + 1)
    key = key[:len(message)]
    
    # Perform XOR operation
    result = []
    for m_char, k_char in zip(message, key):
        # XOR the ASCII values
        xor_value = ord(m_char) ^ ord(k_char)
        result.append(xor_value)
    
    return result

def map_to_alphabet(xor_result):
    """Map XOR result to A-Z (0-25)."""
    return [chr(65 + (value % 26)) for value in xor_result]

def calculate_ioc(text):
    """Calculate Index of Coincidence for the given text."""
    # Remove non-alphabetic characters and convert to uppercase
    text = ''.join(c for c in text if c.isalpha()).upper()
    
    if len(text) <= 1:
        return 0.0
    
    # Count frequency of each letter
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    
    # Calculate IoC
    n = len(text)
    numerator = sum(count * (count - 1) for count in freq.values())
    denominator = n * (n - 1)
    
    return numerator / denominator if denominator > 0 else 0.0

def analyze_frequency(text):
    """
    Analyze character frequency in the plaintext and display results.
    
    Args:
        text (str): The plaintext to analyze
    """
    result = []
    result.append(f"\n{YELLOW}Frequency Analysis{RESET}")
    result.append(f"{GREY}-{RESET}" * 50)
    
    # Ensure text is uppercase for consistency
    text = text.upper()
    
    # Count letter frequencies
    letter_count = {}
    total_letters = 0
    
    for char in text:
        if char.isalpha():
            letter_count[char] = letter_count.get(char, 0) + 1
            total_letters += 1
    
    # Calculate frequencies and sort by frequency (descending)
    frequencies = [(char, count, count/total_letters*100) for char, count in letter_count.items()]
    frequencies.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    result.append(f"{'Character':<10}{'Count':<10}{'Frequency %':<15}{'Bar Chart'}")
    result.append(f"{GREY}-{RESET}" * 50)
    
    for char, count, percentage in frequencies:
        bar_length = int(percentage) * 2  # Scale for better visualization
        bar = "█" * bar_length
        result.append(f"{char:<10}{count:<10}{percentage:.2f}%{'':<10}{RED}{bar}{RESET}")
    
    # Add some statistical analysis
    result.append(f"{GREY}-{RESET}" * 50)
    result.append(f"Total letters analyzed: {YELLOW}{total_letters}{RESET}")
    
    # Compare with English language frequency
    english_freq = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 7.31, 'N': 6.95,
        'S': 6.28, 'R': 6.02, 'H': 5.92, 'D': 4.32, 'L': 3.98, 'U': 2.88,
        'C': 2.71, 'M': 2.61, 'F': 2.30, 'Y': 2.11, 'W': 2.09, 'G': 2.03,
        'P': 1.82, 'B': 1.49, 'V': 1.11, 'K': 0.69, 'X': 0.17, 'Q': 0.11,
        'J': 0.10, 'Z': 0.07
    }
    
    # Calculate deviation from English frequency
    result.append(f"\n{YELLOW}Deviation from Standard English{RESET}")
    result.append(f"{'Character':<10}{'Text %':<15}{'English %':<15}{'Deviation'}")
    result.append(f"{GREY}-{RESET}" * 50)
    
    # Convert frequencies to a dict for easier lookup
    text_freq = {char: percentage for char, _, percentage in frequencies}
    
    # Calculate total deviation to determine English-like score
    total_deviation = 0
    for char in sorted(english_freq.keys()):
        text_percentage = text_freq.get(char, 0)
        eng_percentage = english_freq[char]
        deviation = text_percentage - eng_percentage
        total_deviation += abs(deviation)
        
        # Highlight significant deviations
        if abs(deviation) > 3:
            color = RED
        elif abs(deviation) > 1.5:
            color = YELLOW
        else:
            color = RESET
            
        result.append(f"{char:<10}{text_percentage:.2f}%{'':<10}{eng_percentage:.2f}%{'':<10}{color}{deviation:+.2f}%{RESET}")
    
    # Look for recurring patterns (potential key length indicators)
    result.append(f"\n{YELLOW}Common Bigrams and Trigrams{RESET}")
    
    # Analyze bigrams
    bigrams = {}
    for i in range(len(text) - 1):
        if text[i].isalpha() and text[i+1].isalpha():
            bigram = text[i:i+2]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
    
    # Analyze trigrams
    trigrams = {}
    for i in range(len(text) - 2):
        if text[i].isalpha() and text[i+1].isalpha() and text[i+2].isalpha():
            trigram = text[i:i+3]
            trigrams[trigram] = trigrams.get(trigram, 0) + 1
    
    # Show top bigrams
    top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:8]
    result.append(f"Top Bigrams: " + ", ".join([f"{RED}{b}{RESET}({c})" for b, c in top_bigrams]))
    
    # Show top trigrams
    top_trigrams = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)[:8]
    result.append(f"Top Trigrams: " + ", ".join([f"{RED}{t}{RESET}({c})" for t, c in top_trigrams]))
    
    # Add Index of Coincidence calculation
    ioc = calculate_ioc(text)
    result.append(f"\n{YELLOW}Index of Coincidence: {RED}{ioc:.6f}{RESET}")
    result.append(f"Typical English text IoC: {YELLOW}0.0667{RESET}")
    
    result.append(f"\n{GREY}Analysis complete.{RESET}")
    
    # Return English-likeness score (lower is better) and formatted analysis
    english_likeness_score = total_deviation
    return "\n".join(result), english_likeness_score

def save_to_file(results, filename):
    """Save the results to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"XOR Decryption Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"=" * 80 + "\n\n")
            
            for i, (key, plaintext, ioc, frequency_analysis, likeness_score) in enumerate(results):
                file.write(f"Match #{i+1}\n")
                file.write(f"Key: {key}\n")
                file.write(f"Plaintext: {plaintext}\n")
                file.write(f"IoC: {ioc:.6f}\n")
                file.write(f"English Likeness Score: {likeness_score:.2f} (lower is better)\n\n")
                
                # Strip color codes for file output
                clean_analysis = ""
                skip_mode = False
                for c in frequency_analysis:
                    if c == '\033':
                        skip_mode = True
                    elif skip_mode and c == 'm':
                        skip_mode = False
                    elif not skip_mode:
                        clean_analysis += c
                
                file.write(clean_analysis + "\n")
                file.write(f"{'=' * 80}\n\n")
                
            file.write(f"Total matches found: {len(results)}\n")
            
        print(f"\n{GREEN}Results saved to '{filename}'.{RESET}")
        return True
    except Exception as e:
        print(f"{RED}Error saving results: {e}{RESET}")
        return False

def print_colored_output(message, key, xor_result, alphabet_result=None, ioc=None):
    """Print the results with color formatting."""
    print(f"\n{GREY}Original message:{RESET} {message}")
    print(f"{GREY}Key used:{RESET} {key}")
    
    # Display raw XOR result
    print(f"\n{RED}XOR Result (decimal):{RESET}")
    print(" ".join(f"{x:3d}" for x in xor_result))
    
    # Display character representation
    print(f"\n{RED}XOR Result (ASCII):{RESET}")
    print("".join(chr(x) for x in xor_result))
    
    # Display hex representation
    print(f"\n{RED}XOR Result (hex):{RESET}")
    print(" ".join(f"{x:02X}" for x in xor_result))
    
    # If mapped to alphabet, display that too
    if alphabet_result:
        print(f"\n{YELLOW}XOR Result (mapped to A-Z):{RESET}")
        print("".join(alphabet_result))
        
        if ioc is not None:
            print(f"{BLUE}Index of Coincidence:{RESET} {ioc:.6f}")

def brute_force_with_ioc(ciphertext, min_ioc=0.065, max_ioc=0.07, perform_freq_analysis=False, freq_threshold=45.0):
    """Try each word in the wordlist as a key, filter by IoC, and optionally by frequency analysis."""
    try:
        if not os.path.exists(WORDLIST_PATH):
            print(f"{RED}Error: '{WORDLIST_PATH}' not found in the current directory.{RESET}")
            return []
            
        with open(WORDLIST_PATH, 'r', encoding='utf-8') as file:
            words = [word.strip() for word in file if word.strip()]
    except Exception as e:
        print(f"{RED}Error reading wordlist: {e}{RESET}")
        return []
    
    results = []
    total_words = len(words)
    
    print(f"{GREY}Starting brute force with {total_words} words...{RESET}")
    print(f"{GREY}IoC filter range: {min_ioc} to {max_ioc}{RESET}")
    if perform_freq_analysis:
        print(f"{GREY}Frequency analysis will be performed (threshold: {freq_threshold}){RESET}")
    
    words_checked = 0
    for word in words:
        words_checked += 1
        if words_checked % 1000 == 0:
            print(f"{GREY}Progress: {words_checked}/{total_words} words checked...{RESET}", end='\r')
        
        # Skip empty words or words with non-ASCII characters
        if not word or not all(ord(c) < 128 for c in word):
            continue
            
        xor_result = xor_strings(ciphertext, word)
        alphabet_result = ''.join(map_to_alphabet(xor_result))
        
        ioc = calculate_ioc(alphabet_result)
        
        if min_ioc <= ioc <= max_ioc:
            # If frequency analysis requested, perform it
            if perform_freq_analysis:
                freq_analysis, likeness_score = analyze_frequency(alphabet_result)
                if likeness_score <= freq_threshold:
                    results.append((word, alphabet_result, ioc, freq_analysis, likeness_score))
            else:
                # Add empty frequency analysis data if not performing it
                results.append((word, alphabet_result, ioc, "", 0))
    
    print(f"{GREY}Completed! {total_words} words checked.{RESET}                ")
    
    # Sort results by frequency likeness score if frequency analysis was performed
    if perform_freq_analysis:
        results.sort(key=lambda x: x[4])  # Sort by likeness score (lower is better)
    else:
        results.sort(key=lambda x: x[2], reverse=True)  # Sort by IoC
        
    return results

def main():
    print(f"{RED}======================================{RESET}")
    print(f"{RED}= String XOR Encryptor & IoC Cracker ={RESET}")
    print(f"{RED}======================================{RESET}")
    
    mode = input(f"\n{GREY}Choose mode (1 = Decrypt, 2 = Brute Force): {RESET}")
    
    if mode == '1':
        # Encryption mode
        message = input(f"\n{GREY}Enter the cipher: {RESET}")
        key = input(f"{GREY}Enter the key: {RESET}")
        
        # Perform XOR operation
        xor_result = xor_strings(message, key)
        
        # Ask if user wants to map to alphabet
        map_option = input(f"\n{GREY}Map result to A-Z? (Y/N): {RESET}").lower()
        
        if map_option == 'y':
            alphabet_result = map_to_alphabet(xor_result)
            ioc = calculate_ioc(''.join(alphabet_result))
            print_colored_output(message, key, xor_result, alphabet_result, ioc)
            
            # Option to perform frequency analysis
            freq_option = input(f"\n{GREY}Perform frequency analysis? (Y/N): {RESET}").lower()
            if freq_option == 'y':
                analysis, _ = analyze_frequency(''.join(alphabet_result))
                print(analysis)
        else:
            print_colored_output(message, key, xor_result)
    
    elif mode == '2':
        # Brute force mode
        ciphertext = input(f"\n{GREY}Enter the ciphertext to crack: {RESET}")
        
        # Check if wordlist exists
        if not os.path.exists(WORDLIST_PATH):
            print(f"{RED}Error: '{WORDLIST_PATH}' not found in the current directory.{RESET}")
            return
        
        min_ioc = 0.065
        max_ioc = 0.07
        
        custom_ioc = input(f"{GREY}Use default IoC range (0.065-0.07)? (Y/N): {RESET}").lower()
        if custom_ioc == 'n':
            try:
                min_ioc = float(input(f"{GREY}Enter minimum IoC value: {RESET}"))
                max_ioc = float(input(f"{GREY}Enter maximum IoC value: {RESET}"))
            except ValueError:
                print(f"{RED}Invalid IoC values, using defaults.{RESET}")
        
        # Ask if frequency analysis should be performed
        perform_freq = input(f"{GREY}Perform frequency analysis to find English-like text? (Y/N): {RESET}").lower()
        perform_freq_analysis = perform_freq == 'y'
        
        freq_threshold = 45.0
        if perform_freq_analysis:
            try:
                custom_threshold = input(f"{GREY}Use default frequency deviation threshold (45.0)? (Y/N): {RESET}").lower()
                if custom_threshold == 'n':
                    freq_threshold = float(input(f"{GREY}Enter frequency deviation threshold (lower values = more English-like): {RESET}"))
            except ValueError:
                print(f"{RED}Invalid threshold, using default.{RESET}")
        
        results = brute_force_with_ioc(ciphertext, min_ioc, max_ioc, perform_freq_analysis, freq_threshold)
        
        if results:
            print(f"\n{GREEN}Found {len(results)} potential matches:{RESET}")
            
            # Display top results
            display_count = min(10, len(results))
            for i, (key, plaintext, ioc, freq_analysis, likeness_score) in enumerate(results[:display_count]):
                print(f"\n{YELLOW}Match #{i+1}:{RESET}")
                print(f"{GREY}Key:{RESET} {key}")
                print(f"{GREY}Plaintext:{RESET} {plaintext}")
                print(f"{GREY}IoC:{RESET} {ioc:.6f}")
                if perform_freq_analysis:
                    print(f"{GREY}English Likeness Score:{RESET} {likeness_score:.2f} (lower is better)")
                
            print(f"\n{GREEN}Total matches found: {len(results)}{RESET}")
            
            # Ask to see more details of a specific match
            if perform_freq_analysis and len(results) > 0:
                see_details = input(f"\n{GREY}View frequency analysis for a specific match? (Enter match # or 'n'): {RESET}")
                if see_details.isdigit() and 1 <= int(see_details) <= len(results):
                    match_idx = int(see_details) - 1
                    print(results[match_idx][3])  # Print the frequency analysis
            
            # Ask to save results to a file
            save_option = input(f"\n{GREY}Save results to a file? (y/n): {RESET}").lower()
            if save_option == 'y':
                filename = input(f"{GREY}Enter filename (default: xor_results.txt): {RESET}")
                if not filename.strip():
                    filename = "xor_results.txt"
                save_to_file(results, filename)
        else:
            print(f"\n{RED}No matches found within the specified criteria.{RESET}")
    
    else:
        print(f"{RED}Invalid mode selected.{RESET}")
    
    print(f"\n{RED}======================================{RESET}")

if __name__ == "__main__":
    main()