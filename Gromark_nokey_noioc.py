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

def try_decrypt_with_keyword(args):
    """Try decryption with a given primer batch and keyword"""
    ciphertext, keyword, primers, known_plaintexts = args
    results = []
    
    # Create mixed alphabet for this keyword
    mixed_alphabet = create_keyed_alphabet(keyword)
    
    for primer in primers:
        try:
            running_key = generate_running_key(primer, len(ciphertext))
            decrypted = decrypt_gromark(ciphertext, mixed_alphabet, running_key)
            
            if any(word.lower() in decrypted.lower() for word in known_plaintexts):
                results.append({
                    'keyword': keyword,
                    'primer': primer,
                    'running_key': running_key,
                    'decrypted': decrypted
                })
        except (ValueError, IndexError):
            continue
    
    return results

def run():
    print("Gromark Cipher Decoder with Detailed Progress")
    print("-" * 40)
    
    # Read ciphertext
    ciphertext = input("Enter the ciphertext: ").upper()
    
    # Read words list and filter for 5-10 letter words
    try:
        with open('words_alpha.txt', 'r') as f:
            words_list = [word.strip().upper() for word in f 
                          if 5 <= len(word.strip()) <= 10]
    except FileNotFoundError:
        print("Error: words_alpha.txt not found. Using default words.")
        words_list = ["SPACE", "WORLD", "PLANET", "GALAXY"]
    
    # Get primer parameters
    primer_length = int(input("Enter the primer length (e.g., 5 for 00000-99999): "))
    if primer_length < 5:
        print("Error: Primer length must be at least 5 digits")
        return
    
    # Get known plaintext words
    known_text = input("Enter known plaintext words (comma-separated): ").upper()
    known_plaintexts = [word.strip() for word in known_text.split(',') if word.strip()]
    
    if not known_plaintexts:
        known_plaintexts = ["BERLINCLOCK", "EASTNORTH", "NORTHEAST"]
    
    # Generate primers in batches for better parallelization
    batch_size = 1000
    total_primers = 10 ** primer_length
    
    # Setup results tracking
    all_results = []
    
    # Progress bars for keywords and primers
    with tqdm(total=len(words_list), desc="Keywords", unit="word", position=0) as keywords_pbar, \
         tqdm(total=total_primers, desc="Primers", unit="primer", position=1) as primers_pbar:
        
        # Process keywords in batches
        keyword_batches = [words_list[i:i+5] for i in range(0, len(words_list), 5)]
        
        for keyword_batch in keyword_batches:
            batch_results = []
            
            with ProcessPoolExecutor() as executor:
                keyword_futures = []
                
                for keyword in keyword_batch:
                    # Process primers in batches
                    for start in range(0, total_primers, batch_size):
                        end = min(start + batch_size, total_primers)
                        batch = [str(i).zfill(primer_length) for i in range(start, end)]
                        args = (ciphertext, keyword, batch, known_plaintexts)
                        keyword_futures.append(executor.submit(try_decrypt_with_keyword, args))
                
                # Collect results for this batch of keywords
                for future in as_completed(keyword_futures):
                    try:
                        batch_results.extend(future.result())
                        # Update progress bars
                        primers_pbar.update(batch_size)
                    except Exception as e:
                        print(f"\nError processing batch: {str(e)}")
                
                # Update keyword progress
                keywords_pbar.update(len(keyword_batch))
            
            # Extend overall results
            all_results.extend(batch_results)
    
    # Display results
    if all_results:
        print("\nPossible solutions found:")
        for result in all_results:
            print("\n" + "-"*40)
            print(f"Keyword: {result['keyword']}")
            print(f"Primer: {result['primer']}")
            print(f"Running key: {result['running_key']}")
            print(f"Decrypted text: {result['decrypted']}")
    else:
        print("\nNo solutions found with given parameters.")

if __name__ == "__main__":
    run()