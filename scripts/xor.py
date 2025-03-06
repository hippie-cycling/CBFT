import os
import datetime
import re
from utils import utils

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;4m'

# Hardcoded path for words_alpha.txt
WORDLIST_PATH = "data\words_alpha.txt"

def parse_input(input_text, input_format='string'):
    """Parse input based on the specified format."""
    if input_format == 'string':
        return [ord(c) for c in input_text]
    elif input_format == 'decimal':
        return [int(x) for x in input_text.split() if x.strip()]
    elif input_format == 'hex':
        # Remove any non-hex characters and spaces
        clean_hex = re.sub(r'[^0-9A-Fa-f]', ' ', input_text)
        # Split by spaces and convert each hex value
        return [int(x, 16) for x in clean_hex.split() if x.strip()]
    return []

def xor_values(message_values, key):
    """XOR each value in the message with the corresponding character of the key."""
    # Extend the key if necessary to match the message length
    if len(key) < len(message_values):
        key = key * (len(message_values) // len(key) + 1)
    key = key[:len(message_values)]
    
    # Perform XOR operation
    result = []
    for m_val, k_char in zip(message_values, key):
        # XOR the value with the ASCII of the key character
        xor_value = m_val ^ ord(k_char)
        result.append(xor_value)
    
    return result

def map_to_alphabet(xor_result):
    """Map XOR result to A-Z (0-25)."""
    return [chr(65 + (value % 26)) for value in xor_result]

def get_ascii_result(xor_result):
    """Convert XOR result to ASCII characters."""
    return ''.join(chr(x) for x in xor_result if 32 <= x <= 126)  # Printable ASCII only

def print_colored_output(input_data, input_format, key, xor_result, alphabet_result=None, ascii_result=None, ioc_az=None, ioc_ascii=None):
    """Print the results with color formatting."""
    print(f"\n{GREY}Input data ({input_format}):{RESET} {input_data}")
    print(f"{GREY}Key used:{RESET} {key}")
    
    # Display raw XOR result
    print(f"\n{RED}XOR Result (decimal):{RESET}")
    print(" ".join(f"{x:3d}" for x in xor_result))
    
    # Display character representation
    print(f"\n{RED}XOR Result (ASCII):{RESET}")
    print(ascii_result)
    if ioc_ascii is not None:
        print(f"{BLUE}ASCII IoC:{RESET} {ioc_ascii:.6f}")
    
    # Display hex representation
    print(f"\n{RED}XOR Result (hex):{RESET}")
    print(" ".join(f"{x:02X}" for x in xor_result))
    
    # If mapped to alphabet, display that too
    if alphabet_result:
        print(f"\n{YELLOW}XOR Result (mapped to A-Z):{RESET}")
        print(alphabet_result)
        
        if ioc_az is not None:
            print(f"{BLUE}A-Z IoC:{RESET} {ioc_az:.6f}")

def brute_force_with_ioc(ciphertext, min_ioc=0.065, max_ioc=0.07, perform_freq_analysis=False, freq_threshold=45.0):
    """Try each word in the wordlist as a key, filter by IoC for both ASCII and A-Z results."""
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
    if perform_freq_analysis:
        print(f"-{GREY}Frequency analysis will be performed (threshold: {freq_threshold}){RESET}")
    print(f"\n{YELLOW}Analyzing both ASCII and A-Z mapped results...{RESET}\n")
    
    # Add case variations testing
    print(f"{YELLOW}Testing both uppercase and lowercase versions of each word{RESET}")
    
    words_checked = 0
    for word in words:
        words_checked += 1
        if words_checked % 1000 == 0:
            print(f"{GREY}Progress: {RED}{words_checked}{RESET}/{YELLOW}{total_words}{RESET} words checked...{RESET}", end='\r')
        
        # Skip empty words or words with non-ASCII characters
        if not word or not all(ord(c) < 128 for c in word):
            continue
        
        # Try both lowercase and uppercase versions of the word
        for test_word in [word.upper()]:
            xor_result = xor_values(ciphertext, test_word)
            
            # Get both ASCII and A-Z mapped results
            ascii_result = get_ascii_result(xor_result)
            alphabet_result = ''.join(map_to_alphabet(xor_result))
            
            # Calculate IoC for both
            ioc_ascii = utils.calculate_ioc(ascii_result)
            ioc_az = utils.calculate_ioc(alphabet_result)
                
            # Check both results against IoC range
            ascii_match = min_ioc <= ioc_ascii <= max_ioc and len(ascii_result.strip()) > 0
            az_match = min_ioc <= ioc_az <= max_ioc
            
            if ascii_match or az_match:
                # Process ASCII match
                if ascii_match:
                    if perform_freq_analysis:
                        freq_analysis, likeness_score = utils.analyze_frequency(ascii_result)
                        if likeness_score <= freq_threshold:
                            results.append((test_word, "ASCII", ascii_result, ioc_ascii, freq_analysis, likeness_score))
                    else:
                        results.append((test_word, "ASCII", ascii_result, ioc_ascii, "", float('inf')))
                
                # Process A-Z match
                if az_match:
                    if perform_freq_analysis:
                        freq_analysis, likeness_score = utils.analyze_frequency(alphabet_result)
                        if likeness_score <= freq_threshold:
                            results.append((test_word, "A-Z", alphabet_result, ioc_az, freq_analysis, likeness_score))
                    else:
                        results.append((test_word, "A-Z", alphabet_result, ioc_az, "", float('inf')))
    
    print(f"{GREY}Completed! {total_words} words checked.{RESET}                ")
    
    # Sort results - first by frequency analysis score if available, then by IoC
    if perform_freq_analysis:
        results.sort(key=lambda x: (x[5], -x[3]))  # Sort by likeness score, then by IoC (descending)
    else:
        results.sort(key=lambda x: x[3], reverse=True)  # Sort by IoC
        
    return results

def run():
    print(f"{RED}==========================================={RESET}")
    print(f"{RED}= String XOR Decryptor & IoC Brute Forcer ={RESET}")
    print(f"{RED}==========================================={RESET}")
    
    mode = input(f"\n{GREY}Choose mode ({RESET}{YELLOW}1 = Encrypt/Decrypt{RESET}{GREY},{RESET} {YELLOW}2 = IoC Brute Force{RESET}{GREY}): {RESET}")
    
    if mode == '1':
        # Encryption/Decryption mode
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
            input_values = parse_input(input_data, input_format)
            
            if not input_values:
                print(f"{RED}Error: No valid input data found.{RESET}")
                return
                
            # Perform XOR operation
            xor_result = xor_values(input_values, key)
            
            # Get both ASCII and A-Z results
            ascii_result = get_ascii_result(xor_result)
            ioc_ascii = utils.calculate_ioc(ascii_result)
            
            # Ask if user wants to map to alphabet
            map_option = input(f"\n{GREY}Map result to A-Z? (y/n): {RESET}").lower()
            
            if map_option == 'y':
                alphabet_result = ''.join(map_to_alphabet(xor_result))
                ioc_az = utils.calculate_ioc(alphabet_result)
                print_colored_output(input_data, input_format, key, xor_result, alphabet_result, ascii_result, ioc_az, ioc_ascii)
                print(f"Please note that: {RED}A-Z Mapping is lossy thus non-reversible!{RESET}")
                # Option to perform frequency analysis
                freq_option = input(f"\n{GREY}Perform frequency analysis? (y/n): {RESET}").lower()
                if freq_option == 'y':
                    analyze_type = input(f"{GREY}Analyze which output? (1=ASCII, 2=A-Z, 3=Both): {RESET}")
                    
                    if analyze_type == '1' or analyze_type == '3':
                        print(f"\n{YELLOW}ASCII Result Analysis:{RESET}")
                        analysis, _ = utils.analyze_frequency(ascii_result)
                        print(analysis)
                    
                    if analyze_type == '2' or analyze_type == '3':
                        print(f"\n{YELLOW}A-Z Mapped Result Analysis :{RESET}")
                        analysis, _ = utils.analyze_frequency(alphabet_result)
                        print(analysis)
                        print(f"\nPlease note that: {RED}!!!Mapping is lossy thus non-reversible!!!{RESET}")
            else:
                print_colored_output(input_data, input_format, key, xor_result, None, ascii_result, None, ioc_ascii)
                
                # Option to perform frequency analysis on ASCII
                freq_option = input(f"\n{GREY}Perform frequency analysis on ASCII result? (y/n): {RESET}").lower()
                if freq_option == 'y':
                    analysis, _ = utils.analyze_frequency(ascii_result)
                    print(analysis)
                    
            # Ask if user wants to save XOR result for later use (for decrypt)
            save_option = input(f"\n{GREY}Save XOR result for later decryption? (y/n): {RESET}").lower()
            if save_option == 'y':
                save_format = input(f"{GREY}Save as ({RESET}{YELLOW}1 = decimal{RESET}{GREY},{RESET} {YELLOW}2 = hex{RESET}{GREY}): {RESET}")
                
                if save_format == '1':
                    result_str = " ".join(str(x) for x in xor_result)
                else:
                    result_str = " ".join(f"{x:02X}" for x in xor_result)
                    
                filename = input(f"{GREY}Enter filename: {RESET}")
                if not filename:
                    filename = "xor_result.txt"
                    
                try:
                    with open(filename, 'w') as f:
                        f.write(result_str)
                    print(f"{GREEN}Result saved to '{filename}'.{RESET}")
                except Exception as e:
                    print(f"{RED}Error saving file: {e}{RESET}")
                    
            # Ask if user wants to immediately decrypt with same key to verify
            verify_option = input(f"\n{GREY}Verify XOR reversibility with same key? (y/n): {RESET}").lower()
            if verify_option == 'y':
                # XOR the result again with the same key
                verify_result = xor_values(xor_result, key)
                
                # Convert to ASCII
                verify_ascii = get_ascii_result(verify_result)
                
                print(f"\n{YELLOW}Verification Result:{RESET}")
                print(f"{GREY}Original Input:{RESET} {input_data}")
                print(f"{GREY}After Double XOR:{RESET} {verify_ascii}")
                
                if input_format == 'string':
                    # For string input, we can directly compare the original text
                    if verify_ascii == input_data:
                        print(f"{GREEN}VERIFIED: Double XOR produced original text.{RESET}")
                    else:
                        print(f"{RED}WARNING: Double XOR did not reproduce original text.{RESET}")
                else:
                    print(f"{YELLOW}NOTE: For non-string inputs, manual verification is needed.{RESET}")
                
        except Exception as e:
            print(f"{RED}Error during XOR operation: {e}{RESET}")
    
    elif mode == '2':
        # Brute force mode
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
            ciphertext = parse_input(ciphertext_input, input_format)
            
            if not ciphertext:
                print(f"{RED}Error: No valid input data found.{RESET}")
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
            
            results = brute_force_with_ioc(ciphertext, min_ioc, max_ioc, perform_freq_analysis, freq_threshold)
            
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
                    utils.save_to_file_xor(results, filename)
            else:
                print(f"\n{RED}No matches found within the specified criteria.{RESET}")
        except Exception as e:
            print(f"{RED}Error during brute force: {e}{RESET}")
    
    else:
        print(f"{RED}Invalid mode selected.{RESET}")
    
    print(f"\n{RED}======================================{RESET}")

if __name__ == "__main__":
    run()