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
    # Convert keyword to numbers representing alphabetical order
    order = []
    sorted_chars = sorted(keyword)
    for char in keyword:
        order.append(sorted_chars.index(char) + 1)
        # Handle repeated letters
        sorted_chars[sorted_chars.index(char)] = None
    
    # Create pairs of (order number, column index)
    pairs = list(enumerate(order))
    # Sort by order number
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
        # Add successive pairs of digits
        sum_digits = key[-5] + key[-4]
        if sum_digits > 9:
            sum_digits -= 10
        key.append(sum_digits)
    
    return ''.join(map(str, key[:length]))

def decrypt_gromark(ciphertext, mixed_alphabet, running_key):
    """Decrypt text using Gromark cipher"""
    straight_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    decrypted = []
    
    for i, char in enumerate(ciphertext):
        if char in mixed_alphabet and i < len(running_key):
            # Find position in mixed alphabet
            mixed_pos = mixed_alphabet.index(char)
            # Find corresponding letter in straight alphabet
            straight_letter = straight_alphabet[mixed_pos]
            # Shift backwards by running key number
            shift = int(running_key[i])
            orig_pos = (straight_alphabet.index(straight_letter) - shift) % 26
            decrypted.append(straight_alphabet[orig_pos].lower())
        else:
            decrypted.append(char.lower())
    
    return ''.join(decrypted)

def try_decrypt(args):
    """Try decryption with a given primer batch"""
    ciphertext, mixed_alphabet, primers, known_plaintexts = args
    results = []
    
    for primer in primers:
        try:
            running_key = generate_running_key(primer, len(ciphertext))
            decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key)
            
            if any(word.lower() in decrypted.lower() for word in known_plaintexts):
                results.append({
                    'primer': primer,
                    'running_key': running_key,
                    'decrypted': decrypted
                })
        except (ValueError, IndexError) as e:
            continue
    
    return results

def main():
    print("Gromark Cipher Decoder")
    print("-" * 20)
    
    ciphertext = input("Enter the ciphertext: ").upper()
    keyword = input("Enter the keyword: ").upper()
    primer_length = int(input("Enter the primer length (e.g., 5 for 00000-99999): "))
    known_text = input("Enter known plaintext words (comma-separated): ").upper()
    known_plaintexts = [word.strip() for word in known_text.split(',') if word.strip()]
    
    if not known_plaintexts:
        known_plaintexts = ["ONLY", "THE", "UNIVERSE"]
    
    if primer_length < 5:
        print("Error: Primer length must be at least 5 digits")
        return
        
    # Create keyed alphabet
    mixed_alphabet = create_keyed_alphabet(keyword)
    print(f"\nMixed alphabet: {mixed_alphabet}")
    
    # Generate primers in batches for better parallelization
    batch_size = 1000
    total_combinations = 10 ** primer_length
    
    print(f"\nTrying all {total_combinations} possible {primer_length}-digit primers in parallel...")
    
    results = []
    with tqdm(total=total_combinations, desc="Testing primers", unit="primer") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = []
            
            # Process primers in batches
            for start in range(0, total_combinations, batch_size):
                end = min(start + batch_size, total_combinations)
                batch = [str(i).zfill(primer_length) for i in range(start, end)]
                args = (ciphertext, mixed_alphabet, batch, known_plaintexts)
                futures.append(executor.submit(try_decrypt, args))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    pbar.update(batch_size)
                except Exception as e:
                    print(f"\nError processing batch: {str(e)}")
                    continue
    
    if results:
        print("\nPossible solutions found:")
        for result in results:
            print(f"\nPrimer: {result['primer']}")
            print(f"Running key: {result['running_key']}")
            print(f"Decrypted text: {result['decrypted']}")
    else:
        print("\nNo solutions found with given parameters.")

if __name__ == "__main__":
    main()