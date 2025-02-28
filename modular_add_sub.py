import os
import datetime
import re

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;4m'

# Hardcoded path for words_alpha.txt
WORDLIST_PATH = "words_alpha.txt"

def parse_input(input_text, input_format='string', custom_alphabet=None):
    """Parse input based on the specified format and alphabet."""
    if input_format == 'string':
        if custom_alphabet:
            return [custom_alphabet.find(c.upper()) if c.upper() in custom_alphabet else -1 for c in input_text]
        else:
            return [ord(c.upper()) - ord('A') if c.isalpha() else -1 for c in input_text]
    elif input_format == 'decimal':
        return [int(x) for x in input_text.split() if x.strip()]
    elif input_format == 'hex':
        # Remove any non-hex characters and spaces
        clean_hex = re.sub(r'[^0-9A-Fa-f]', ' ', input_text)
        # Split by spaces and convert each hex value
        return [int(x, 16) for x in clean_hex.split() if x.strip()]
    return []

def modular_operation(message_values, key, operation='add', modulus=26, base_index=0):
    """
    Perform modular addition or subtraction of the message with the key.
    
    Args:
        message_values: List of numeric values representing the message
        key: String key to use for the operation
        operation: 'add' for encryption, 'subtract' for decryption
        modulus: The modulus to use (26 for standard A-Z)
        base_index: 0 if A=0, 1 if A=1
    
    Returns:
        List of numeric values after the operation
    """
    # Extend the key if necessary to match the message length
    if len(key) < len(message_values):
        key = key * (len(message_values) // len(key) + 1)
    key = key[:len(message_values)]
    
    # Perform modular operation
    result = []
    for i, m_val in enumerate(message_values):
        if m_val == -1:  # Skip non-alphabetic characters
            result.append(-1)
            continue
            
        # Get the key character value based on custom_alphabet or standard A-Z
        if isinstance(key[i], int):  # If key is already converted to numbers
            k_val = key[i]
        else:
            k_val = ord(key[i].upper()) - ord('A')
        
        # Adjust for base index (0 or 1)
        m_val_adj = m_val - base_index
        k_val_adj = k_val - base_index
        
        # Perform the operation
        if operation == 'add':
            op_result = (m_val_adj + k_val_adj) % modulus
        else:  # subtract
            op_result = (m_val_adj - k_val_adj) % modulus
            
        # Adjust back from base index
        result.append(op_result + base_index)
        
    return result

def map_to_alphabet(mod_result, custom_alphabet=None, base_index=0):
    """Map modular result to alphabet (either custom or A-Z)."""
    result = []
    for value in mod_result:
        if value == -1:  # Non-alphabetic character
            result.append(' ')
        else:
            value_adj = value - base_index  # Adjust for base index
            if custom_alphabet:
                if 0 <= value_adj < len(custom_alphabet):
                    result.append(custom_alphabet[value_adj])
                else:
                    result.append(' ')
            else:
                result.append(chr(65 + value_adj))
    return ''.join(result)

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
    
    if total_letters == 0:
        result.append(f"{RED}No alphabetic characters found for analysis.{RESET}")
        return "\n".join(result), float('inf')
    
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
    if top_bigrams:
        result.append(f"Top Bigrams: " + ", ".join([f"{RED}{b}{RESET}({c})" for b, c in top_bigrams]))
    else:
        result.append(f"No bigrams found.")
    
    # Show top trigrams
    top_trigrams = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)[:8]
    if top_trigrams:
        result.append(f"Top Trigrams: " + ", ".join([f"{RED}{t}{RESET}({c})" for t, c in top_trigrams]))
    else:
        result.append(f"No trigrams found.")
    
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
            file.write(f"Modular Cipher Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"=" * 80 + "\n\n")
            
            for i, result_data in enumerate(results):
                key, result_type, result_text, ioc, frequency_analysis, likeness_score = result_data
                
                file.write(f"Match #{i+1}\n")
                file.write(f"Key: {key}\n")
                file.write(f"Result Type: {result_type}\n")
                file.write(f"Decrypted Text: {result_text}\n")
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

def print_colored_output(input_data, input_format, key, mod_result, alphabet_result, operation, modulus, base_index, custom_alphabet=None, ioc=None):
    """Print the results with color formatting."""
    print(f"\n{GREY}Input data ({input_format}):{RESET} {input_data}")
    print(f"{GREY}Key used:{RESET} {key}")
    print(f"{GREY}Operation:{RESET} Modular {operation}")
    print(f"{GREY}Modulus:{RESET} {modulus} (Base index: {base_index})")
    
    if custom_alphabet:
        print(f"{GREY}Custom alphabet:{RESET} {custom_alphabet}")
    
    # Display raw modular result
    print(f"\n{RED}Modular Result (numeric):{RESET}")
    print(" ".join(f"{x:3d}" if x != -1 else "   " for x in mod_result))
    
    # Display character representation
    print(f"\n{RED}Result (text):{RESET} {alphabet_result}")
    #print(alphabet_result)
    
    if ioc is not None:
        print(f"{BLUE}IoC:{RESET} {ioc:.6f}")

def brute_force_with_ioc(ciphertext, operation, modulus, base_index, custom_alphabet=None, min_ioc=0.065, max_ioc=0.07, perform_freq_analysis=False, freq_threshold=45.0):
    """Try each word in the wordlist as a key, filter by IoC."""
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
    
    print(f"\n{YELLOW}Starting brute force with {total_words} words...{RESET}")
    print(f"\n-{GREY}IoC filter range: {min_ioc} to {max_ioc}{RESET}")
    print(f"-{GREY}Modular {operation} with modulus {modulus} (Base index: {base_index}){RESET}")
    if custom_alphabet:
        print(f"-{GREY}Using custom alphabet: {custom_alphabet}{RESET}")
    if perform_freq_analysis:
        print(f"-{GREY}Frequency analysis will be performed (threshold: {freq_threshold}){RESET}")
    
    words_checked = 0
    for word in words:
        words_checked += 1
        if words_checked % 1000 == 0:
            print(f"{GREY}Progress: {RED}{words_checked}{RESET}/{YELLOW}{total_words}{RESET} words checked...{RESET}", end='\r')
        
        # Skip empty words or words with non-ASCII characters
        if not word or not all(ord(c) < 128 for c in word):
            continue
            
        # Try with uppercase version of the word
        test_word = word.upper()
        
        # Convert key to numeric values if using custom alphabet
        if custom_alphabet:
            key_values = [custom_alphabet.find(c) if c in custom_alphabet else -1 for c in test_word]
            # Skip if any character in the key is not in the custom alphabet
            if -1 in key_values:
                continue
            mod_result = modular_operation(ciphertext, key_values, operation, modulus, base_index)
        else:
            mod_result = modular_operation(ciphertext, test_word, operation, modulus, base_index)
        
        # Map result to alphabet
        result_text = map_to_alphabet(mod_result, custom_alphabet, base_index)
        
        # Calculate IoC
        ioc = calculate_ioc(result_text)
        
        # Check if result matches IoC criteria
        if min_ioc <= ioc <= max_ioc and len(result_text.strip()) > 0:
            if perform_freq_analysis:
                freq_analysis, likeness_score = analyze_frequency(result_text)
                if likeness_score <= freq_threshold:
                    results.append((test_word, "Modular", result_text, ioc, freq_analysis, likeness_score))
            else:
                results.append((test_word, "Modular", result_text, ioc, "", float('inf')))
    
    print(f"{GREY}Completed! {total_words} words checked.{RESET}                ")
    
    # Sort results - first by frequency analysis score if available, then by IoC
    if perform_freq_analysis:
        results.sort(key=lambda x: (x[5], -x[3]))  # Sort by likeness score, then by IoC (descending)
    else:
        results.sort(key=lambda x: x[3], reverse=True)  # Sort by IoC
        
    return results

def run():
    print(f"{RED}=============================================={RESET}")
    print(f"{RED}= Modular Addition/Subtraction Cipher Tool  ={RESET}")
    print(f"{RED}=============================================={RESET}")
    
    # Ask if user wants to use a custom alphabet
    custom_alphabet_option = input(f"\n{GREY}Use custom alphabet? (y/n): {RESET}").lower()
    custom_alphabet = None
    if custom_alphabet_option == 'y':
        custom_alphabet_input = input(f"{GREY}Enter custom alphabet (e.g., KRYPTOSABCDEFGHIJLMNQUVWXZ): {RESET}").upper()
        if custom_alphabet_input:
            custom_alphabet = custom_alphabet_input
            print(f"{YELLOW}Using custom alphabet: {custom_alphabet}{RESET}")
            print(f"{GREY}Modulus will be set to alphabet length: {len(custom_alphabet)}{RESET}")
            modulus = len(custom_alphabet)
        else:
            print(f"{RED}No custom alphabet provided, using standard A-Z.{RESET}")
            modulus = 26
    else:
        modulus = 26  # Default modulus for A-Z
    
    # Ask for base index (0 or 1)
    base_index_option = input(f"{GREY}Use which base index? (0=A-Z is 0-25, 1=A-Z is 1-26): {RESET}")
    if base_index_option == '1':
        base_index = 1
    else:
        base_index = 0
    
    mode = input(f"\n{GREY}Choose mode ({RESET}{YELLOW}1 = Encrypt/Decrypt{RESET}{GREY},{RESET} {YELLOW}2 = IoC Brute Force{RESET}{GREY}): {RESET}")
    
    if mode == '1':
        # Encryption/Decryption mode
        operation = input(f"\n{GREY}Operation ({RESET}{YELLOW}1 = add (encrypt){RESET}{GREY},{RESET} {YELLOW}2 = subtract (decrypt){RESET}{GREY}): {RESET}")
        operation = 'add' if operation == '1' else 'subtract'
        
        input_format = input(f"\n{GREY}Input format ({RESET}{YELLOW}1 = string{RESET}{GREY},{RESET} {YELLOW}2 = decimal{RESET}{GREY},{RESET} {YELLOW}3 = hex{RESET}{GREY}): {RESET}")
        
        if input_format == '1':
            input_format = 'string'
            input_data = input(f"{GREY}Enter the text: {RESET}")
        elif input_format == '2':
            input_format = 'decimal'
            input_data = input(f"{GREY}Enter the decimal values (space-separated): {RESET}")
        elif input_format == '3':
            input_format = 'hex'
            input_data = input(f"{GREY}Enter the hex values (space-separated): {RESET}")
        else:
            print(f"{RED}Invalid format, defaulting to string.{RESET}")
            input_format = 'string'
            input_data = input(f"{GREY}Enter the text: {RESET}")
        
        key = input(f"{GREY}Enter the key: {RESET}")
        
        # Parse input based on format
        try:
            input_values = parse_input(input_data, input_format, custom_alphabet)
            
            if not input_values or all(v == -1 for v in input_values):
                print(f"{RED}Error: No valid input data found or characters not in alphabet.{RESET}")
                return
            
            # Convert key to values if using custom alphabet
            if custom_alphabet and input_format == 'string':
                key_values = [custom_alphabet.find(c.upper()) if c.upper() in custom_alphabet else -1 for c in key]
                if -1 in key_values:
                    print(f"{RED}Warning: Some key characters are not in the custom alphabet.{RESET}")
                mod_result = modular_operation(input_values, key_values, operation, modulus, base_index)
            else:
                mod_result = modular_operation(input_values, key, operation, modulus, base_index)
            
            # Map to alphabet
            alphabet_result = map_to_alphabet(mod_result, custom_alphabet, base_index)
            
            # Calculate IoC
            ioc = calculate_ioc(alphabet_result)
            
            # Print results
            print_colored_output(input_data, input_format, key, mod_result, alphabet_result, operation, modulus, base_index, custom_alphabet, ioc)
            
            # Option to perform frequency analysis
            freq_option = input(f"\n{GREY}Perform frequency analysis? (y/n): {RESET}").lower()
            if freq_option == 'y':
                analysis, _ = analyze_frequency(alphabet_result)
                print(analysis)
            
            # Ask if user wants to save result
            save_option = input(f"\n{GREY}Save result? (y/n): {RESET}").lower()
            if save_option == 'y':
                save_format = input(f"{GREY}Save as ({RESET}{YELLOW}1 = text{RESET}{GREY},{RESET} {YELLOW}2 = numeric{RESET}{GREY}): {RESET}")
                
                if save_format == '1':
                    result_str = alphabet_result
                else:
                    result_str = " ".join(str(x) for x in mod_result if x != -1)
                    
                filename = input(f"{GREY}Enter filename: {RESET}")
                if not filename:
                    filename = "modular_result.txt"
                    
                try:
                    with open(filename, 'w') as f:
                        f.write(result_str)
                    print(f"{GREEN}Result saved to '{filename}'.{RESET}")
                except Exception as e:
                    print(f"{RED}Error saving file: {e}{RESET}")
            
            # Ask if user wants to immediately verify with reverse operation
            verify_option = input(f"\n{GREY}Verify reversibility with opposite operation? (y/n): {RESET}").lower()
            if verify_option == 'y':
                # Apply opposite operation
                reverse_op = 'subtract' if operation == 'add' else 'add'
                
                # Convert key to values if using custom alphabet
                if custom_alphabet and input_format == 'string':
                    reverse_result = modular_operation(mod_result, key_values, reverse_op, modulus, base_index)
                else:
                    reverse_result = modular_operation(mod_result, key, reverse_op, modulus, base_index)
                
                # Map back to original alphabet
                reverse_text = map_to_alphabet(reverse_result, custom_alphabet, base_index)
                
                print(f"\n{YELLOW}Verification Result:{RESET}")
                print(f"{GREY}Original Input:{RESET} {input_data}")
                print(f"{GREY}After Reverse Operation:{RESET} {reverse_text}")
                
                if input_format == 'string':
                    # Compare handling non-alphabet characters
                    orig_text = ''.join(c if custom_alphabet is None or c.upper() in custom_alphabet else ' ' 
                                    for c in input_data.upper())
                    if reverse_text.replace(' ', '') == orig_text.replace(' ', ''):
                        print(f"{GREEN}VERIFIED: Reverse operation produced original text.{RESET}")
                    else:
                        print(f"{RED}WARNING: Reverse operation did not reproduce original text.{RESET}")
                else:
                    print(f"{YELLOW}NOTE: For non-string inputs, manual verification is needed.{RESET}")
                
        except Exception as e:
            print(f"{RED}Error during modular operation: {e}{RESET}")
    
    elif mode == '2':
        # Brute force mode
        operation = input(f"\n{GREY}Operation to test ({RESET}{YELLOW}1 = add{RESET}{GREY},{RESET} {YELLOW}2 = subtract{RESET}{GREY}): {RESET}")
        operation = 'add' if operation == '1' else 'subtract'
        
        input_format = input(f"\n{GREY}Input format ({RESET}{YELLOW}1 = string{RESET}{GREY},{RESET} {YELLOW}2 = decimal{RESET}{GREY},{RESET} {YELLOW}3 = hex{RESET}{GREY}): {RESET}")
        
        if input_format == '1':
            input_format = 'string'
            ciphertext_input = input(f"{GREY}Enter the ciphertext to crack: {RESET}")
        elif input_format == '2':
            input_format = 'decimal'
            ciphertext_input = input(f"{GREY}Enter the decimal values (space-separated): {RESET}")
        elif input_format == '3':
            input_format = 'hex'
            ciphertext_input = input(f"{GREY}Enter the hex values (space-separated): {RESET}")
        else:
            print(f"{RED}Invalid format, defaulting to string.{RESET}")
            input_format = 'string'
            ciphertext_input = input(f"{GREY}Enter the ciphertext to crack: {RESET}")
        
        # Parse input based on format
        try:
            ciphertext = parse_input(ciphertext_input, input_format, custom_alphabet)
            
            if not ciphertext or all(v == -1 for v in ciphertext):
                print(f"{RED}Error: No valid input data found or characters not in alphabet.{RESET}")
                return
            
            # Check if wordlist exists
            if not os.path.exists(WORDLIST_PATH):
                print(f"{RED}Error: '{WORDLIST_PATH}' not found in the current directory.{RESET}")
                return
            
            min_ioc = 0.065
            max_ioc = 0.07
            
            custom_ioc = input(f"{GREY}Use default IoC range (0.065-0.07)? (y/n): {RESET}").lower()
            if custom_ioc == 'n':
                try:
                    min_ioc = float(input(f"{GREY}Enter minimum IoC value: {RESET}"))
                    max_ioc = float(input(f"{GREY}Enter maximum IoC value: {RESET}"))
                except ValueError:
                    print(f"{RED}Invalid IoC values, using defaults.{RESET}")
            
            # Ask if frequency analysis should be performed
            perform_freq = input(f"{GREY}Perform frequency analysis to find English-like text? (y/n): {RESET}").lower()
            perform_freq_analysis = perform_freq == 'y'
            
            freq_threshold = 45.0
            if perform_freq_analysis:
                try:
                    custom_threshold = input(f"{GREY}Use default frequency deviation threshold (45.0)? (y/n): {RESET}").lower()
                    if custom_threshold == 'n':
                        freq_threshold = float(input(f"{GREY}Enter frequency deviation threshold (lower values = more English-like): {RESET}"))
                except ValueError:
                    print(f"{RED}Invalid threshold, using default.{RESET}")
            
            results = brute_force_with_ioc(
                ciphertext, operation, modulus, base_index, custom_alphabet,
                min_ioc, max_ioc, perform_freq_analysis, freq_threshold
            )
            
            if results:
                print(f"\n{GREEN}Found {len(results)} potential matches:{RESET}")
                
                # Display top results
                display_count = min(10, len(results))
                for i, (key, result_type, result_text, ioc, freq_analysis, likeness_score) in enumerate(results[:display_count]):
                    print(f"\n{YELLOW}Match #{i+1}:{RESET}")
                    print(f"{GREY}Key:{RESET} {key}")
                    print(f"{GREY}Type:{RESET} {result_type}")
                    print(f"{GREY}Decrypted text:{RESET} {result_text}")
                    print(f"{GREY}IoC:{RESET} {ioc:.6f}")
                    if perform_freq_analysis:
                        print(f"{GREY}English Likeness Score:{RESET} {likeness_score:.2f} (lower is better)")
                    
                print(f"\n{GREEN}Total matches found: {len(results)}{RESET}")
                
                # Ask to see more details of a specific match
                if perform_freq_analysis and len(results) > 0:
                    see_details = input(f"\n{GREY}View frequency analysis for a specific match? (Enter match # or 'n'): {RESET}")
                    if see_details.isdigit() and 1 <= int(see_details) <= len(results):
                        match_idx = int(see_details) - 1
                        print(results[match_idx][4])  # Print the frequency analysis
                
                # Ask to save results to a file
                save_option = input(f"\n{GREY}Save results to a file? (y/n): {RESET}").lower()
                if save_option == 'y':
                    filename = input(f"{GREY}Enter filename (default: xor_results.txt): {RESET}")
                    if not filename.strip():
                        filename = "xor_results.txt"
                    save_to_file(results, filename)
            else:
                print(f"\n{RED}No matches found within the specified criteria.{RESET}")
        except Exception as e:
            print(f"{RED}Error during brute force: {e}{RESET}")
    
    else:
        print(f"{RED}Invalid mode selected.{RESET}")
    
    print(f"\n{RED}======================================{RESET}")

if __name__ == "__main__":
    run()