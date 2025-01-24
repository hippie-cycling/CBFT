import concurrent.futures
from itertools import product
from tqdm import tqdm  # For progress bar

def calculate_ioc(text):
    char_counts = {}
    for char in text:
        if char.isalpha():
            char_counts[char] = char_counts.get(char, 0) + 1
    
    n = sum(char_counts.values())
    numerator = sum(count * (count - 1) for count in char_counts.values())
    denominator = n * (n - 1)
    return numerator / denominator if denominator != 0 else 0

def gronsfeld_decrypt(ciphertext, key, alphabet):
    decrypted_text = []
    key_length = len(key)
    alphabet_length = len(alphabet)
    
    for i, char in enumerate(ciphertext):
        if char in alphabet:
            char_index = alphabet.index(char)
            shift = int(key[i % key_length])
            decrypted_index = (char_index - shift) % alphabet_length
            decrypted_char = alphabet[decrypted_index]
            decrypted_text.append(decrypted_char)
        else:
            decrypted_text.append(char)
    return ''.join(decrypted_text)

def test_key(key_str, ciphertext, alphabet, words_to_find):
    decrypted_text = gronsfeld_decrypt(ciphertext, key_str, alphabet)
    ioc = calculate_ioc(decrypted_text)
    
    if 0.06 <= ioc <= 0.07:
        return key_str, ioc, decrypted_text, "IOC"
    
    if any(word in decrypted_text for word in words_to_find):
        return key_str, ioc, decrypted_text, "WORD"
    
    return None

def brute_force_gronsfeld(ciphertext, key_length, words_to_find, alphabet):
    keys = (''.join(key) for key in product('0123456789', repeat=key_length))
    results = []

    # Calculate total number of keys
    total_keys = 10 ** key_length

    # Use tqdm for progress bar
    with tqdm(total=total_keys, desc="Brute-forcing", unit="key") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for key_str in keys:
                futures.append(executor.submit(test_key, key_str, ciphertext, alphabet, words_to_find))
                
                # Update progress bar as futures complete
                if len(futures) >= 1000:  # Process in chunks of 1000
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            results.append(result)
                        pbar.update(1)  # Update progress bar
                    futures = []  # Reset futures list

            # Process any remaining futures
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)  # Update progress bar

    # Print summary
    if results:
        print("\nBrute-force complete! Summary of results:")
        print("=" * 50)
        for key_str, ioc, decrypted_text, result_type in results:
            if result_type == "IOC":
                print(f"Key: {key_str}, IOC: {ioc:.4f}, Decrypted Text: {decrypted_text}")
            elif result_type == "WORD":
                print(f"Key: {key_str}, IOC: {ioc:.4f}, Decrypted Text: {decrypted_text}")
                print(f"Found one of the words: {[word for word in words_to_find if word in decrypted_text]}")
            print("-" * 50)
    else:
        print("\nNo results found.")

# Main program
if __name__ == "__main__":
    # Get ciphertext from user
    ciphertext = input("Enter the ciphertext: ").strip().upper()

    # Get alphabet from user
    alphabet = input("Enter the custom alphabet (e.g., KRYPTOSABCDEFGHIJLMNQUVWXZ): ").strip().upper()
    if not alphabet:
        print("Alphabet cannot be empty. Using default alphabet: ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Get words to find from user
    words_input = input("Enter the words to find (comma-separated, e.g., BERLIN,CLOCK,EAST,NORTH): ").strip().upper()
    words_to_find = [word.strip() for word in words_input.split(",")] if words_input else []
    if not words_to_find:
        print("No words to find provided. Using default words: THE, FROM, AND, THAT")
        words_to_find = ["THE", "FROM", "AND", "THAT"]

    # Get key length from user
    key_length = int(input("Enter the key length: "))

    # Brute-force the key
    brute_force_gronsfeld(ciphertext, key_length, words_to_find, alphabet)