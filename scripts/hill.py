import os
import argparse
import numpy as np
from utils import utils
import itertools
import time

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;4m'

def matrix_to_key(matrix, alphabet):
    """Convert a matrix to a string key representation."""
    key_str = ""
    for row in matrix:
        for element in row:
            key_str += alphabet[element % len(alphabet)]
    return key_str

def key_to_matrix(key, matrix_size, alphabet):
    """Convert a string key to a matrix."""
    # Ensure key is long enough
    required_length = matrix_size * matrix_size
    if len(key) < required_length:
        key = (key * (required_length // len(key) + 1))[:required_length]
    
    # Convert key characters to indices in the alphabet
    key_indices = [alphabet.index(c) for c in key if c in alphabet]
    
    # Reshape into matrix
    matrix = np.array(key_indices).reshape(matrix_size, matrix_size)
    return matrix

def modular_inverse(matrix, modulus):
    """Calculate the modular inverse of a matrix"""
    # Get matrix size
    n = matrix.shape[0]
    
    # Calculate determinant
    det = int(round(np.linalg.det(matrix))) % modulus
    
    # Ensure the determinant is properly handled for negative values
    if det < 0:
        det = (det % modulus)
    
    # Check if the determinant has an inverse in the given modulus
    det_inv = None
    for i in range(1, modulus):
        if (det * i) % modulus == 1:
            det_inv = i
            break
    
    if det_inv is None:
        return None  # Matrix is not invertible in this modulus
    
    # For 2x2 matrix, the adjugate is easy to compute
    if n == 2:
        adj = np.array([
            [matrix[1, 1], -matrix[0, 1]],
            [-matrix[1, 0], matrix[0, 0]]
        ], dtype=int)
        
        # Handle negative values with modulus
        adj = (adj % modulus)
        
        # Multiply by determinant inverse and take modulo
        inv = (adj * det_inv) % modulus
        return inv
    
    # For larger matrices, compute the adjugate (cofactor matrix transposed)
    else:
        # Create cofactor matrix
        cofactors = np.zeros(matrix.shape, dtype=int)
        
        for i in range(n):
            for j in range(n):
                # Get minor by removing row i and column j
                minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                
                # Calculate determinant of minor
                minor_det = int(round(np.linalg.det(minor))) % modulus
                
                # Handle negative values
                if minor_det < 0:
                    minor_det = (minor_det % modulus)
                
                # Apply (-1)^(i+j) to get cofactor
                cofactors[i, j] = ((-1) ** (i + j)) * minor_det
                
                # Handle negative values with modulus
                if cofactors[i, j] < 0:
                    cofactors[i, j] = (cofactors[i, j] % modulus)
        
        # Transpose to get adjugate
        adj = cofactors.T
        
        # Multiply by determinant inverse and take modulo
        inv = (adj * det_inv) % modulus
        return inv

def decrypt_hill(ciphertext, key_matrix, alphabet):
    """Decrypt Hill cipher with the given key matrix."""
    matrix_size = key_matrix.shape[0]
    modulus = len(alphabet)
    
    # Check if matrix is invertible in the given modulus
    inverse_matrix = modular_inverse(key_matrix, modulus)
    if inverse_matrix is None:
        return None  # Non-invertible matrix for this modulus
    
    # Print the inverse matrix for debugging
    #print(f"Inverse matrix:\n{inverse_matrix}")
    
    # Split ciphertext into chunks of size n (matrix_size)
    plaintext = ""
    
    # Filter out non-alphabet characters
    filtered_ciphertext = [c for c in ciphertext if c in alphabet]
    
    # Pad the ciphertext if necessary
    while len(filtered_ciphertext) % matrix_size != 0:
        filtered_ciphertext.append(alphabet[0])  # Pad with first letter of alphabet
    
    # Process each chunk
    for i in range(0, len(filtered_ciphertext), matrix_size):
        chunk = filtered_ciphertext[i:i+matrix_size]
        
        # Convert the chunk to a vector of indices
        chunk_indices = [alphabet.index(c) for c in chunk]
        chunk_vector = np.array(chunk_indices)
        
        # Multiply by the inverse key matrix and take modulo
        decrypted_vector = np.dot(inverse_matrix, chunk_vector) % modulus
        
        # Convert back to characters
        for idx in decrypted_vector:
            plaintext += alphabet[int(idx)]
    
    return plaintext

def decrypt_hill_column_vectors(ciphertext, key_matrix, alphabet):
    """Decrypt Hill cipher using column vectors instead of row vectors."""
    matrix_size = key_matrix.shape[0]
    modulus = len(alphabet)
    
    # Check if matrix is invertible in the given modulus
    inverse_matrix = modular_inverse(key_matrix, modulus)
    if inverse_matrix is None:
        return None  # Non-invertible matrix for this modulus
    
    # Split ciphertext into chunks of size n (matrix_size)
    plaintext = ""
    
    # Filter out non-alphabet characters
    filtered_ciphertext = [c for c in ciphertext if c in alphabet]
    
    # Pad the ciphertext if necessary
    while len(filtered_ciphertext) % matrix_size != 0:
        filtered_ciphertext.append(alphabet[0])  # Pad with first letter of alphabet
    
    # Process each chunk
    for i in range(0, len(filtered_ciphertext), matrix_size):
        chunk = filtered_ciphertext[i:i+matrix_size]
        
        # Convert the chunk to a vector of indices
        chunk_indices = [alphabet.index(c) for c in chunk]
        chunk_vector = np.array(chunk_indices).reshape(matrix_size, 1)
        
        # Multiply by the inverse key matrix and take modulo
        decrypted_vector = np.dot(inverse_matrix, chunk_vector) % modulus
        
        # Convert back to characters
        for idx in decrypted_vector.flatten():
            plaintext += alphabet[int(idx)]
    
    return plaintext

def decrypt_with_specific_key():
    """Decrypt using a user-provided specific key."""
    print(f"\n{YELLOW}== Hill Cipher Direct Decryption =={RESET}")
    
    # Get the alphabet
    default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    custom_alphabet = input(f"\n{GREY}Use custom alphabet? (y/n, default is A-Z): {RESET}").lower()
    
    if custom_alphabet == 'y':
        alphabet = input(f"{GREY}Enter custom alphabet: {RESET}")
        # Validate: unique characters only
        if len(set(alphabet)) != len(alphabet):
            print(f"{RED}Warning: Duplicate characters in alphabet. Removing duplicates.{RESET}")
            alphabet = ''.join(dict.fromkeys(alphabet))
        print(f"{GREEN}Using alphabet: {alphabet}{RESET}")
    else:
        alphabet = default_alphabet
        print(f"{GREEN}Using standard alphabet: {alphabet}{RESET}")
    
    # Get the matrix size
    matrix_size = 2  # Default
    try:
        size_input = input(f"\n{GREY}Enter matrix size (2 or 3, default is 2): {RESET}")
        if size_input.strip():
            matrix_size = int(size_input)
            if matrix_size not in [2, 3]:
                print(f"{RED}Invalid matrix size. Using default size 2.{RESET}")
                matrix_size = 2
    except ValueError:
        print(f"{RED}Invalid input. Using default matrix size 2.{RESET}")
    
    print(f"{GREEN}Using {matrix_size}x{matrix_size} matrix for Hill cipher.{RESET}")
    
    # Get the key
    key_input_method = input(f"\n{GREY}Input key as (1) matrix elements or (2) text key? (1/2): {RESET}")
    
    if key_input_method == '1':
        # Input matrix elements directly
        matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        print(f"{YELLOW}Enter matrix elements (values from 0 to {len(alphabet)-1}):{RESET}")
        
        for i in range(matrix_size):
            row_input = input(f"Row {i+1} (space-separated values): ")
            values = row_input.split()
            
            if len(values) != matrix_size:
                print(f"{RED}Error: Expected {matrix_size} values for row {i+1}.{RESET}")
                return
            
            try:
                for j in range(matrix_size):
                    matrix[i, j] = int(values[j]) % len(alphabet)
            except ValueError:
                print(f"{RED}Error: Invalid matrix elements. Must be integers.{RESET}")
                return
    elif key_input_method == '2':
        # Input text key
        key_text = input(f"{GREY}Enter the key text: {RESET}")
        key_text = normalize_input(key_text, alphabet)
        
        if len(key_text) < matrix_size * matrix_size:
            print(f"{RED}Error: Key text too short. Need at least {matrix_size * matrix_size} characters.{RESET}")
            return
            
        matrix = key_to_matrix(key_text, matrix_size, alphabet)
    else:
        print(f"{RED}Invalid option. Exiting.{RESET}")
        return
    
    print(f"\n{GREEN}Using key matrix:{RESET}")
    for row in matrix:
        print(f"  {row}")
    
    # Check if matrix is valid (invertible)
    if not is_valid_key_matrix(matrix, len(alphabet)):
        print(f"{RED}Error: The provided matrix is not invertible in modulo {len(alphabet)}.{RESET}")
        print(f"{RED}Cannot use this key for decryption.{RESET}")
        return
    
    # Get ciphertext
    ciphertext = input(f"\n{GREY}Enter the ciphertext to decrypt: {RESET}")
    ciphertext = normalize_input(ciphertext, alphabet)
    
    if not ciphertext:
        print(f"{RED}Error: No valid characters found in ciphertext after normalization.{RESET}")
        return
    
    # Ask which convention to use
    convention = input(f"\n{GREY}Use (1) row vectors or (2) column vectors convention? (1/2, default 1): {RESET}")
    
    # Decrypt based on chosen convention
    if convention == '2':
        plaintext = decrypt_hill_column_vectors(ciphertext, matrix, alphabet)
        conv_name = "column vectors"
    else:
        plaintext = decrypt_hill(ciphertext, matrix, alphabet)
        conv_name = "row vectors"
    
    if plaintext:
        print(f"\n{GREEN}Decryption successful using {conv_name} convention:{RESET}")
        print(f"\n{YELLOW}Plaintext:{RESET}\n{plaintext}")
        
        # Calculate and show IoC
        ioc = utils.calculate_ioc(plaintext)
        print(f"\n{GREY}Index of Coincidence: {ioc:.6f}{RESET}")
        
        # Option to perform frequency analysis
        analyze = input(f"\n{GREY}Perform frequency analysis? (y/n): {RESET}").lower()
        if analyze == 'y':
            freq_analysis, likeness_score = utils.analyze_frequency(plaintext)
            print(f"\n{GREY}Frequency Analysis:{RESET}")
            print(freq_analysis)
            print(f"{GREY}English Likeness Score: {likeness_score:.2f} (lower is better){RESET}")
        
        # Option to save to file
        save_option = input(f"\n{GREY}Save results to a file? (y/n): {RESET}").lower()
        if save_option == 'y':
            filename = input(f"{GREY}Enter filename (default: hill_decryption.txt): {RESET}")
            if not filename.strip():
                filename = "hill_decryption.txt"
            
            try:
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(f"Hill Cipher Decryption Results\n")
                    file.write(f"============================\n\n")
                    file.write(f"Key Matrix:\n")
                    for row in matrix:
                        file.write(f"  {row}\n")
                    file.write(f"Convention: {conv_name}\n\n")
                    file.write(f"Ciphertext:\n{ciphertext}\n\n")
                    file.write(f"Plaintext:\n{plaintext}\n\n")
                    file.write(f"Index of Coincidence: {ioc:.6f}\n")
                    
                    if analyze == 'y':
                        file.write(f"\nFrequency Analysis:\n{freq_analysis}\n")
                        file.write(f"English Likeness Score: {likeness_score:.2f} (lower is better)\n")
                
                print(f"{GREEN}Results saved to {filename}{RESET}")
            except Exception as e:
                print(f"{RED}Error saving to file: {e}{RESET}")
    else:
        print(f"{RED}Decryption failed. Please check your key and ciphertext.{RESET}")

def contains_fragment(plaintext, fragment, case_sensitive=False):
    """Check if plaintext contains the specified fragment."""
    if not case_sensitive:
        if fragment.lower() in plaintext.lower():
            return True
        # Try without spaces
        if fragment.lower().replace(" ", "") in plaintext.lower().replace(" ", ""):
            return True
    else:
        if fragment in plaintext:
            return True
        # Try without spaces
        if fragment.replace(" ", "") in plaintext.replace(" ", ""):
            return True
    return False

def is_valid_key_matrix(matrix, modulus):
    """Check if a matrix is valid as a Hill cipher key (invertible in the given modulus)."""
    # Calculate determinant
    det = int(round(np.linalg.det(matrix))) % modulus
    
    # Valid if determinant is not 0 and is coprime with modulus
    return det != 0 and np.gcd(det, modulus) == 1

def generate_all_possible_matrices(matrix_size, alphabet_length):
    """Generate all possible matrices for the given size and alphabet length."""
    # For each position in the matrix, values can range from 0 to alphabet_length-1
    value_range = range(alphabet_length)
    
    # Generate all possible combinations
    for values in itertools.product(value_range, repeat=matrix_size*matrix_size):
        # Reshape into a matrix
        matrix = np.array(values).reshape(matrix_size, matrix_size)
        
        # Check if the matrix is valid (invertible)
        if is_valid_key_matrix(matrix, alphabet_length):
            yield matrix

def brute_force_hill_all_keys(ciphertext, alphabet, matrix_size=2, min_ioc=0, max_ioc=1.0, 
                              known_fragments=None, perform_freq_analysis=False, 
                              freq_threshold=45.0, max_results=100, progress_interval=1000,
                              consider_both_conventions=True):
    """Try all possible key matrices, filter by IoC and known fragments."""
    results = []
    modulus = len(alphabet)
    
    # Calculate total number of possible matrices (for progress reporting)
    total_matrices = modulus**(matrix_size*matrix_size)
    valid_matrices_count = 0
    matrices_checked = 0
    
    print(f"\n{YELLOW}Starting brute force with all possible {matrix_size}x{matrix_size} matrices...{RESET}")
    print(f"{YELLOW}Total possible matrices: {total_matrices} (before filtering for invertibility){RESET}")
    if min_ioc > 0 or max_ioc < 1.0:
        print(f"\n-{GREY}IoC filter range: {min_ioc} to {max_ioc}{RESET}")
    else:
        print(f"\n-{GREY}IoC filtering disabled{RESET}")
    
    if perform_freq_analysis:
        print(f"-{GREY}Frequency analysis will be performed (threshold: {freq_threshold}){RESET}")
    if known_fragments:
        print(f"-{GREY}Known fragments will be checked: {known_fragments}{RESET}")
    
    start_time = time.time()
    last_update_time = start_time

    # Generate and test all possible matrices
    for matrix in generate_all_possible_matrices(matrix_size, modulus):
        matrices_checked += 1
        valid_matrices_count += 1
        
        # Update progress periodically
        current_time = time.time()
        if matrices_checked % progress_interval == 0 or current_time - last_update_time >= 5:
            elapsed_time = current_time - start_time
            estimated_total_valid = (total_matrices * valid_matrices_count) / matrices_checked
            progress_percent = (matrices_checked / total_matrices) * 100
            
            print(f"{GREY}Progress: {RED}{matrices_checked:,}{RESET}/{YELLOW}{total_matrices:,}{RESET} matrices ({progress_percent:.2f}%)" + 
                  f" | Valid: {valid_matrices_count:,} | Time: {elapsed_time:.1f}s{RESET}", end='\r')
            last_update_time = current_time
        
        try:
            # # Skip the example matrix which we already tested
            # if matrix_size == 2 and np.array_equal(matrix, example_matrix):
            #     continue
                
            # Try both row and column vector conventions if enabled
            if consider_both_conventions:
                # Try row vector convention
                plaintext_row = decrypt_hill(ciphertext, matrix, alphabet)
                if plaintext_row:
                    ioc_row = utils.calculate_ioc(plaintext_row)
                    # Check against known fragments
                    fragment_match_row = False
                    if known_fragments:
                        fragment_match_row = any(contains_fragment(plaintext_row, fragment) for fragment in known_fragments)
                    
                    # Check IoC range
                    ioc_match_row = min_ioc <= ioc_row <= max_ioc
                    
                    if ioc_match_row or fragment_match_row:
                        key_rep = matrix_to_key(matrix, alphabet)
                        if perform_freq_analysis:
                            freq_analysis, likeness_score = utils.analyze_frequency(plaintext_row)
                            if likeness_score <= freq_threshold:
                                results.append((key_rep, matrix, plaintext_row, ioc_row, freq_analysis, likeness_score, "row"))
                        else:
                            results.append((key_rep, matrix, plaintext_row, ioc_row, "", float('inf'), "row"))
                
                # Try column vector convention
                plaintext_col = decrypt_hill_column_vectors(ciphertext, matrix, alphabet)
                if plaintext_col:
                    ioc_col = utils.calculate_ioc(plaintext_col)
                    # Check against known fragments
                    fragment_match_col = False
                    if known_fragments:
                        fragment_match_col = any(contains_fragment(plaintext_col, fragment) for fragment in known_fragments)
                    
                    # Check IoC range
                    ioc_match_col = min_ioc <= ioc_col <= max_ioc
                    
                    if ioc_match_col or fragment_match_col:
                        key_rep = matrix_to_key(matrix, alphabet)
                        if perform_freq_analysis:
                            freq_analysis, likeness_score = utils.analyze_frequency(plaintext_col)
                            if likeness_score <= freq_threshold:
                                results.append((key_rep, matrix, plaintext_col, ioc_col, freq_analysis, likeness_score, "column"))
                        else:
                            results.append((key_rep, matrix, plaintext_col, ioc_col, "", float('inf'), "column"))
            else:
                # Use only row vector convention (original behavior)
                plaintext = decrypt_hill(ciphertext, matrix, alphabet)
                if plaintext:
                    ioc = utils.calculate_ioc(plaintext)
                    
                    # Check against known fragments
                    fragment_match = False
                    if known_fragments:
                        fragment_match = any(contains_fragment(plaintext, fragment) for fragment in known_fragments)
                    
                    # Check IoC range
                    ioc_match = min_ioc <= ioc <= max_ioc
                    
                    if ioc_match or fragment_match:
                        key_rep = matrix_to_key(matrix, alphabet)
                        if perform_freq_analysis:
                            freq_analysis, likeness_score = utils.analyze_frequency(plaintext)
                            if likeness_score <= freq_threshold:
                                results.append((key_rep, matrix, plaintext, ioc, freq_analysis, likeness_score, "row"))
                        else:
                            results.append((key_rep, matrix, plaintext, ioc, "", float('inf'), "row"))
            
            # Limit the number of results to avoid memory issues
            if len(results) >= max_results:
                print(f"\n{YELLOW}Maximum number of results ({max_results}) reached. Stopping search.{RESET}")
                break
        except Exception as e:
            # Skip errors for individual matrices
            continue
    
    print(f"\n{GREY}Completed! {matrices_checked:,} matrices checked, {valid_matrices_count:,} valid matrices tested.{RESET}")
    
    # Sort results - first by fragment match, then by frequency analysis score if available, then by IoC
    if perform_freq_analysis:
        results.sort(key=lambda x: (not any(contains_fragment(x[2], f) for f in (known_fragments or [])), x[5], -x[3]))
    else:
        results.sort(key=lambda x: (not any(contains_fragment(x[2], f) for f in (known_fragments or [])), -x[3]))
        
    return results

def normalize_input(text, alphabet):
    """Normalize input text to match the alphabet."""
    result = ""
    for c in text:
        if c.upper() in alphabet.upper():
            # Match case with alphabet
            if alphabet[0].isupper():
                result += c.upper()
            else:
                result += c.lower()
    return result

def run():
    print(f"{GREEN}==============={RESET}")
    print(f"{GREEN}= Hill Cipher ={RESET}")
    print(f"{GREEN}==============={RESET}")
    
    print(f"\n{YELLOW}Select mode:{RESET}")
    print(f"1. Direct decryption with a known key")
    print(f"2. Brute force attack (all possible keys)")
    
    mode = input(f"\n{GREY}Enter your choice (1-2): {RESET}")
    
    if mode == '1':
        decrypt_with_specific_key()
    elif mode == '2':
        # Bruteforce
        default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        custom_alphabet = input(f"\n{GREY}Use custom alphabet? (y/n, default is A-Z): {RESET}").lower()
        
        if custom_alphabet == 'y':
            alphabet = input(f"{GREY}Enter custom alphabet: {RESET}")
            # Validate: unique characters only
            if len(set(alphabet)) != len(alphabet):
                print(f"{RED}Warning: Duplicate characters in alphabet. Removing duplicates.{RESET}")
                alphabet = ''.join(dict.fromkeys(alphabet))
            print(f"{GREEN}Using alphabet: {alphabet}{RESET}")
        else:
            alphabet = default_alphabet
            print(f"{GREEN}Using standard alphabet: {alphabet}{RESET}")
        
        # Get matrix size
        matrix_size = 2  # Default
        try:
            size_input = input(f"\n{GREY}Enter matrix size (2 or 3, default is 2): {RESET}")
            if size_input.strip():
                matrix_size = int(size_input)
                if matrix_size not in [2, 3]:
                    print(f"{RED}Invalid matrix size. Using default size 2.{RESET}")
                    matrix_size = 2
        except ValueError:
            print(f"{RED}Invalid input. Using default matrix size 2.{RESET}")
        
        print(f"{GREEN}Using {matrix_size}x{matrix_size} matrix for Hill cipher.{RESET}")
        
        # Calculate and display total possible matrices
        total_possible = len(alphabet) ** (matrix_size * matrix_size)
        print(f"{YELLOW}Total possible matrices: {total_possible:,}{RESET}")
        
        if total_possible > 10000000:  # 10 million
            confirm = input(f"{RED}Warning: This will generate a very large number of matrices to test. Continue? (y/n): {RESET}").lower()
            if confirm != 'y':
                print(f"{RED}Operation cancelled.{RESET}")
                return
        
        # Get ciphertext
        ciphertext = input(f"\n{GREY}Enter the ciphertext to crack: {RESET}")
        
        # Normalize ciphertext to match alphabet
        ciphertext = normalize_input(ciphertext, alphabet)
        if not ciphertext:
            print(f"{RED}Error: No valid characters found in ciphertext after normalization.{RESET}")
            return
        
        # Check if ciphertext length is valid for matrix size
        if len(ciphertext) % matrix_size != 0:
            print(f"{YELLOW}Warning: Ciphertext length ({len(ciphertext)}) is not a multiple of matrix size ({matrix_size}).{RESET}")
            print(f"{YELLOW}The ciphertext might be padded or some characters might be lost.{RESET}")
        
        # Get known plaintext fragments
        known_fragments = []
        use_fragments = input(f"\n{GREY}Do you have known plaintext fragments? (y/n): {RESET}").lower()
        if use_fragments == 'y':
            while True:
                fragment = input(f"{GREY}Enter a known fragment (or press Enter to finish): {RESET}")
                if not fragment:
                    break
                known_fragments.append(fragment)
        
        # Ask if they want to use IoC filtering
        use_ioc = input(f"\n{GREY}Use IoC filtering? (y/n, set to 'n' to disable filtering): {RESET}").lower()
        
        if use_ioc == 'y':
            min_ioc = 0.065
            max_ioc = 0.07
            
            custom_ioc = input(f"{GREY}Use default IoC range (0.065-0.07)? (y/n): {RESET}").lower()
            if custom_ioc == 'n':
                try:
                    min_ioc = float(input(f"{GREY}Enter minimum IoC value: {RESET}"))
                    max_ioc = float(input(f"{GREY}Enter maximum IoC value: {RESET}"))
                except ValueError:
                    print(f"{RED}Invalid IoC values, using defaults.{RESET}")
        else:
            # Disable IoC filtering
            min_ioc = 0
            max_ioc = 1.0
            print(f"{YELLOW}IoC filtering disabled. All matrices will be evaluated.{RESET}")
        
        # Ask if frequency analysis should be performed
        perform_freq = input(f"\n{GREY}Perform frequency analysis to find English-like text? (y/n): {RESET}").lower()
        perform_freq_analysis = perform_freq == 'y'
        
        freq_threshold = 45.0
        if perform_freq_analysis:
            try:
                custom_threshold = input(f"{GREY}Use default frequency deviation threshold (45.0)? (y/n): {RESET}").lower()
                if custom_threshold == 'n':
                    freq_threshold = float(input(f"{GREY}Enter frequency deviation threshold (lower values = more English-like): {RESET}"))
            except ValueError:
                print(f"{RED}Invalid threshold, using default.{RESET}")
        
        # Ask if both row and column vector conventions should be tested
        test_both = input(f"\n{GREY}Test both row and column vector conventions? (y/n, recommended 'y'): {RESET}").lower()
        consider_both_conventions = test_both == 'y'
        if consider_both_conventions:
            print(f"{YELLOW}Testing both row and column vector conventions (doubles processing time but more thorough){RESET}")
        
        # Get max results to avoid memory issues
        max_results = 100
        try:
            custom_max = input(f"\n{GREY}Maximum number of results to store (default 100): {RESET}")
            if custom_max.strip():
                max_results = int(custom_max)
        except ValueError:
            print(f"{RED}Invalid input. Using default max results: 100.{RESET}")
        
        # Get progress update interval
        progress_interval = 1000
        try:
            custom_interval = input(f"\n{GREY}Progress update interval (default 1000 matrices): {RESET}")
            if custom_interval.strip():
                progress_interval = int(custom_interval)
        except ValueError:
            print(f"{RED}Invalid input. Using default interval: 1000.{RESET}")
        
        # Run the brute force
        results = brute_force_hill_all_keys(
            ciphertext, 
            alphabet,
            matrix_size,
            min_ioc, 
            max_ioc, 
            known_fragments,
            perform_freq_analysis, 
            freq_threshold,
            max_results,
            progress_interval,
            consider_both_conventions
        )

        if results:
            print(f"\n{GREEN}Found {len(results)} potential matches:{RESET}")
            
            # Display top results
            display_count = min(10, len(results))
            for i, (key_rep, matrix, plaintext, ioc, freq_analysis, likeness_score, convention) in enumerate(results[:display_count]):
                print(f"\n{YELLOW}Match #{i+1}:{RESET}")
                print(f"{GREY}Key Representation:{RESET} {key_rep}")
                print(f"{GREY}Key Matrix:{RESET}")
                for row in matrix:
                    print(f"  {row}")
                print(f"{GREY}Vector Convention:{RESET} {convention}")
                print(f"{GREY}Decrypted text:{RESET} {plaintext}")
                print(f"{GREY}IoC:{RESET} {ioc:.6f}")
                if perform_freq_analysis:
                    print(f"{GREY}English Likeness Score:{RESET} {likeness_score:.2f} (lower is better)")
                
                # Check if this matches any known fragment
                if known_fragments:
                    matching_fragments = [f for f in known_fragments if contains_fragment(plaintext, f)]
                    if matching_fragments:
                        print(f"{GREEN}Matched fragments:{RESET} {', '.join(matching_fragments)}")
            
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
                filename = input(f"{GREY}Enter filename (default: hill_results.txt): {RESET}")
                if not filename.strip():
                    filename = "hill_results.txt"
                utils.save_to_file_hill(results, filename)
        else:
            print(f"\n{RED}No matches found within the specified criteria.{RESET}")
            
            # Offer to relax constraints
            relax = input(f"{GREY}Would you like to relax constraints and try again? (y/n): {RESET}").lower()
            if relax == 'y':
                print(f"\n{YELLOW}Suggestions:{RESET}")
                print(f"1. Disable IoC filtering")
                print(f"2. Widen the IoC range (try 0.05-0.08) if using IoC filtering")
                print(f"3. Increase frequency threshold (try 60.0) if using frequency analysis")
                print(f"4. Remove or simplify known plaintext fragments")
                print(f"5. Make sure to test both row and column vector conventions")
                try_again = input(f"\n{GREY}Try again with relaxed constraints? (y/n): {RESET}").lower()
                if try_again == 'y':
                    # Could implement recursive call to run() here
                    print(f"{YELLOW}Please restart the program and apply relaxed constraints.{RESET}")

def main():
    parser = argparse.ArgumentParser(description='Hill Cipher Brute Force Tool (All Possible Keys)')
    parser.add_argument('--file', help='Input ciphertext from file')
    parser.add_argument('--alphabet', default='ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='Custom alphabet')
    parser.add_argument('--matrix-size', type=int, default=2, choices=[2, 3], help='Matrix size (2 or 3)')
    parser.add_argument('--min-ioc', type=float, default=0.065, help='Minimum IoC threshold')
    parser.add_argument('--max-ioc', type=float, default=0.07, help='Maximum IoC threshold')
    parser.add_argument('--fragments', nargs='+', help='Known plaintext fragments')
    parser.add_argument('--freq-threshold', type=float, default=45.0, help='Frequency analysis threshold')
    parser.add_argument('--output', help='Output results to file')
    parser.add_argument('--max-results', type=int, default=100, help='Maximum number of results to store')
    parser.add_argument('--progress-interval', type=int, default=1000, help='Progress update interval')
    parser.add_argument('--test-both-conventions', action='store_true', help='Test both row and column vector conventions')
    parser.add_argument('--test-specific', action='store_true', help='Test with specific example matrix')
    parser.add_argument('--decrypt', action='store_true', help='Direct decryption mode')
    
    args = parser.parse_args()

    # If direct decrypt flag is specified
    if args.decrypt:
        decrypt_with_specific_key()
        return
    
    # If command line arguments are provided, use them
    if len(sys.argv) > 1:
        # Get ciphertext from file or prompt
        ciphertext = ""
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as file:
                    ciphertext = file.read()
            except Exception as e:
                print(f"{RED}Error reading input file: {e}{RESET}")
                return
        else:
            ciphertext = input(f"{GREY}Enter the ciphertext to crack: {RESET}")
            
        # Normalize ciphertext
        ciphertext = normalize_input(ciphertext, args.alphabet)
        
        # Run the brute force with command line parameters
        results = brute_force_hill_all_keys(
            ciphertext,
            args.alphabet,
            args.matrix_size,
            args.min_ioc,
            args.max_ioc,
            args.fragments,
            True if args.freq_threshold != 45.0 else False,
            args.freq_threshold,
            args.max_results,
            args.progress_interval
        )
        
        # Display results
        if results:
            print(f"\n{GREEN}Found {len(results)} potential matches:{RESET}")
            display_count = min(10, len(results))
            for i, (key_rep, matrix, plaintext, ioc, freq_analysis, likeness_score) in enumerate(results[:display_count]):
                print(f"\n{YELLOW}Match #{i+1}:{RESET}")
                print(f"{GREY}Key Representation:{RESET} {key_rep}")
                print(f"{GREY}Key Matrix:{RESET}")
                for row in matrix:
                    print(f"  {row}")
                print(f"{GREY}Decrypted text:{RESET} {plaintext}")
                print(f"{GREY}IoC:{RESET} {ioc:.6f}")
                if likeness_score != float('inf'):
                    print(f"{GREY}English Likeness Score:{RESET} {likeness_score:.2f} (lower is better)")
            
            # Save to file if requested
            if args.output:
                utils.save_to_file_hill(results, args.output)
        else:
            print(f"\n{RED}No matches found within the specified criteria.{RESET}")
    else:
        # Interactive mode
        run()

if __name__ == "__main__":
    import sys
    main()