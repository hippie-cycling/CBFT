import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

def create_keyed_alphabet(keyword):
    """Create a keyed alphabet from a keyword following Gromark cipher rules"""
    # Remove duplicates while preserving order
    seen = set()
    keyword_unique = ''.join(c for c in keyword.upper() if not (c in seen or seen.add(c)))
    
    # Add remaining alphabet letters
    remaining = ''.join(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in keyword_unique)
    keyed_base = keyword_unique + remaining
    
    # Create transposition block
    cols = len(keyword)
    rows = (len(keyed_base) + cols - 1) // cols
    block = [['' for _ in range(cols)] for _ in range(rows)]
    
    # Fill the block row by row
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < len(keyed_base):
                block[i][j] = keyed_base[idx]
                idx += 1
    
    # Get column order based on keyword
    sorted_chars = sorted(keyword)
    order = []
    for char in keyword:
        order.append(sorted_chars.index(char) + 1)
        sorted_chars[sorted_chars.index(char)] = None
    
    # Create pairs of (order number, column index)
    pairs = list(enumerate(order))
    pairs.sort(key=lambda x: x[1])
    col_order = [p[0] for p in pairs]
    
    # Create mixed alphabet by reading down columns in the determined order
    mixed_alphabet = ''
    for col in col_order:
        for row in range(rows):
            if row < len(block) and block[row][col]:
                mixed_alphabet += block[row][col]
    
    return mixed_alphabet

def generate_running_key(primer, length):
    """Generate running key from a primer"""
    if len(primer) < 5:
        raise ValueError("Primer must be at least 5 digits long")
    
    key = list(map(int, primer))
    while len(key) < length:
        sum_digits = key[-5] + key[-4]
        if sum_digits > 9:
            sum_digits -= 10
        key.append(sum_digits)
    
    return ''.join(map(str, key[:length]))

def get_possible_shifts(cipher_char, plain_char, mixed_alphabet):
    """Get possible shifts (0-9) that could map plaintext to ciphertext"""
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mixed_pos = mixed_alphabet.index(cipher_char)
    straight_letter = straight_alphabet[mixed_pos]
    plain_pos = straight_alphabet.index(plain_char.upper())
    straight_pos = straight_alphabet.index(straight_letter)
    
    required_shift = (straight_pos - plain_pos) % 26
    return required_shift if required_shift <= 9 else None

def analyze_running_key_constraints(ciphertext_segment, plaintext_segment, mixed_alphabet, keyword):
    """Analyze running key constraints for a segment (Corrected)"""
    shifts = []
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for c, p in zip(ciphertext_segment, plaintext_segment):
        mixed_pos = mixed_alphabet.index(c)
        straight_letter = straight_alphabet[mixed_pos]
        plain_pos = straight_alphabet.index(p.upper())
        straight_pos = straight_alphabet.index(straight_letter)

        required_shift = (straight_pos - plain_pos) % 26
        if required_shift > 9:  # Gromark shifts are 0-9
            return None
        shifts.append(required_shift)
    return shifts

def decrypt_gromark(ciphertext, mixed_alphabet, running_key):
    """Decrypt text using Gromark cipher"""
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    decrypted = []
    
    for i, char in enumerate(ciphertext):
        if char in mixed_alphabet and i < len(running_key):
            mixed_pos = mixed_alphabet.index(char)
            straight_letter = straight_alphabet[mixed_pos]
            shift = int(running_key[i])
            orig_pos = (straight_alphabet.index(straight_letter) - shift) % 26
            decrypted.append(straight_alphabet[orig_pos].lower())
        else:
            decrypted.append(char.lower())
    
    return ''.join(decrypted)

def validate_keyword(keyword, known_segments):
    """Validate if a keyword could work with known plaintext segments (Corrected)"""
    try:
        mixed_alphabet = create_keyed_alphabet(keyword)

        for start_idx, cipher_segment, plain_segment in known_segments:
            shifts = analyze_running_key_constraints(cipher_segment, plain_segment, mixed_alphabet, keyword)
            if shifts is None:
                return False  # Early exit if shifts are invalid

        return True  # All segments passed initial shift check

    except Exception as e:
        return False

def try_decrypt_with_primer(args):
    """Try decryption with a given primer (Corrected)"""
    keyword, primer, ciphertext, known_segments = args

    try:
        mixed_alphabet = create_keyed_alphabet(keyword)
        running_key = generate_running_key(primer, len(ciphertext))
        decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key)

        # ***CRITICAL CHECK: Validate against known segments***
        valid_decryption = True
        for start_idx, cipher_segment, plain_segment in known_segments:
            segment_key = running_key[start_idx:start_idx + len(cipher_segment)]
            decrypted_segment = decrypt_gromark(cipher_segment, mixed_alphabet, segment_key)
            if decrypted_segment.upper() != plain_segment.upper():
                valid_decryption = False
                break  # Exit inner loop if segment doesn't match

        if valid_decryption:  # Only return if ALL segments match
            return {
                'keyword': keyword,
                'primer': primer,
                'running_key': running_key,
                'decrypted': decrypted
            }
        else:
            return None  # Return None if decryption doesn't match known segments

    except Exception as e:
        return None

def main():
    print("Optimized Gromark Cipher Decoder")
    print("-" * 50)
    
    # User input for ciphertext
    ciphertext = input("Enter the ciphertext: ").upper()
    
    # Define known segments based on ciphertext length
    known_segments = [
        (0, ciphertext[0:7], "ONLYTWO")
    ]
    
    # Read words list
    vowels = set('AEIOU')
    try:
        with open('words_alpha_.txt', 'r') as f:
            words_list = [
                word.strip().upper()
                for word in f
                if 5 <= len(word.strip()) <= 12 
                and not any(
                    word[i].upper() in vowels and word[i+1].upper() in vowels
                    for i in range(len(word.strip()) - 1)
                )
            ]
    except FileNotFoundError:
        print("Warning: words_alpha.txt not found. Using default word list.")
        words_list = ["GRONSFELD", "KEYWORD", "CIPHER", "MATRIX"]
    
    # Filter keywords using constraints (Corrected)
    print("Filtering keywords based on constraints...")
    valid_keywords = []
    for keyword in tqdm(words_list):
        if validate_keyword(keyword, known_segments):
            valid_keywords.append(keyword)
    
    print(f"\nFiltered from {len(words_list)} to {len(valid_keywords)} possible keywords")
    print(valid_keywords)
    
    # Process valid keywords with parallel primer testing
    results = []
    total_combinations = len(valid_keywords) * 99999  # 5-digit primers from 10000-99999
    
    print("\nTesting possible combinations...")
    with ProcessPoolExecutor() as executor:
        futures = []
        
        for keyword in valid_keywords:
            for primer in range(10000, 100000):
                args = (keyword, str(primer), ciphertext, known_segments)
                futures.append(executor.submit(try_decrypt_with_primer, args))
        
        for future in tqdm(as_completed(futures), total=total_combinations):
            result = future.result()
            if result:
                results.append(result)
    
    # Display results
    if results:
        print("\nPossible solutions found:")
        for result in results:
            print("\n" + "-"*50)
            print(f"Keyword: {result['keyword']}")
            print(f"Primer: {result['primer']}")
            print(f"Running key: {result['running_key']}")
            print(f"Decrypted text: {result['decrypted']}")
    else:
        print("\nNo solutions found with given parameters.")

if __name__ == "__main__":
    main()