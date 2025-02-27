# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'

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

def print_colored_output(message, key, xor_result, alphabet_result=None):
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

def run():
    print(f"{RED}==========================={RESET}")
    print(f"{RED}= String XOR Decryptor ={RESET}")
    print(f"{RED}==========================={RESET}")
    
    message = input(f"\n{GREY}Enter the cipher: {RESET}")
    key = input(f"{GREY}Enter the key: {RESET}")
    
    # Perform XOR operation
    xor_result = xor_strings(message, key)
    
    # Ask if user wants to map to alphabet
    map_option = input(f"\n{GREY}Map result to A-Z? (Y/N): {RESET}").lower()
    
    if map_option == 'y':
        alphabet_result = map_to_alphabet(xor_result)
        print_colored_output(message, key, xor_result, alphabet_result)
    else:
        print_colored_output(message, key, xor_result)
    
    print(f"\n{RED}==========================={RESET}")

if __name__ == "__main__":
    run()