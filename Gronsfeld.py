import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from collections import Counter

def calculate_ioc(text):
    """Calculate the Index of Coincidence (IoC) of a given text"""
    text = text.upper()
    text = [char for char in text if char.isalpha()]
    length = len(text)
    if length < 2:
        return 0.0
    
    freq = Counter(text)
    ioc = sum(count * (count - 1) for count in freq.values()) / (length * (length - 1))
    return ioc

def gronsfeld_decrypt(ciphertext, key, alphabet):
    """Decrypt text using Gronsfeld cipher"""
    decrypted_text = []
    key_length = len(key)
    alphabet_length = len(alphabet)
    
    for i, char in enumerate(ciphertext):
        if char in alphabet:
            char_index = alphabet.index(char)
            shift = int(key[i % key_length])
            decrypted_index = (char_index - shift) % alphabet_length
            decrypted_char = alphabet[decrypted_index]
            decrypted_text.append(decrypted_char.lower())
        else:
            decrypted_text.append(char.lower())
    return ''.join(decrypted_text)

def try_decrypt(args):
    """Try decryption with a given key batch"""
    ciphertext, alphabet, keys, known_plaintexts = args
    results = []
    
    for key in keys:
        try:
            decrypted = gronsfeld_decrypt(ciphertext, key, alphabet)
            
            ioc = calculate_ioc(decrypted)
            if 0.062 <= ioc <= 0.071:
                print(f"\nIoC within range (0.062-0.071): {ioc}")
                print(f"Key: {key}")
                print(f"Decrypted text: {decrypted}")
            
            if any(word.lower() in decrypted.lower() for word in known_plaintexts):
                results.append({
                    'key': key,
                    'decrypted': decrypted,
                    'ioc': ioc
                })
        except (ValueError, IndexError) as e:
            continue
    
    return results

def run():
    print("Gronsfeld Cipher Decoder")
    print("-" * 20)
    
    ciphertext = input("Enter the ciphertext: ").upper()
    alphabet = input("Enter the custom alphabet (press Enter for default A-Z): ").upper()
    if not alphabet:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        print(f"Using default alphabet: {alphabet}")
    
    key_length = int(input("Enter the key length: "))
    known_text = input("Enter known plaintext words (comma-separated): ").upper()
    known_plaintexts = [word.strip() for word in known_text.split(',') if word.strip()]
    
    if not known_plaintexts:
        known_plaintexts = ["THE", "AND", "THAT", "FROM"]
        print(f"Using default known plaintext words: {', '.join(known_plaintexts)}")
    
    # Generate keys in batches for better parallelization
    batch_size = 1000
    total_combinations = 10 ** key_length
    
    print(f"\nTrying all {total_combinations} possible {key_length}-digit keys in parallel...")
    
    results = []
    with tqdm(total=total_combinations, desc="Testing keys", unit="key") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = []
            
            # Process keys in batches
            for start in range(0, total_combinations, batch_size):
                end = min(start + batch_size, total_combinations)
                batch = [str(i).zfill(key_length) for i in range(start, end)]
                args = (ciphertext, alphabet, batch, known_plaintexts)
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
            print(f"\nKey: {result['key']}")
            print(f"IoC: {result['ioc']:.4f}")
            print(f"Decrypted text: {result['decrypted']}")
    else:
        print("\nNo solutions found with given parameters.")

if __name__ == "__main__":
    run()