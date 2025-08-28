import sys

# ANSI color codes for terminal output
RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
GREEN = '\033[38;5;2m'
BLUE = '\033[38;5;4m'
RESET = '\033[0m'

def caesar_decipher(ciphertext, shift, alphabet):
    """
    Decrypts text using a Caesar cipher with a given shift value.
    Handles both positive and negative shifts.
    """
    decrypted_text = []
    alphabet_length = len(alphabet)
    
    # Normalize the shift to be within the alphabet length
    shift = shift % alphabet_length
    
    for char in ciphertext:
        if char in alphabet:
            try:
                char_index = alphabet.index(char)
                # Apply the shift (subtract for decryption)
                decrypted_index = (char_index + shift) % alphabet_length
                decrypted_char = alphabet[decrypted_index]
                decrypted_text.append(decrypted_char.lower())
            except ValueError:
                # This case handles characters that might be in the input but not the alphabet
                decrypted_text.append(char.lower())
        else:
            # Keep non-alphabet characters as they are
            decrypted_text.append(char.lower())
            
    return ''.join(decrypted_text)

def run():
    """Main function to run the Caesar decipher CLI."""
    print(f"{RED}============================={RESET}")
    print(f"{RED}= Caesar Shift Cipher Tool  ={RESET}")
    print(f"{RED}============================={RESET}")
    print(f"{GREY}This tool applies a numeric shift to decrypt a Caesar cipher.{RESET}")
    print(f"{GREY}-{RESET}" * 60)
    
    # Get user inputs
    ciphertext = input(f"Enter the ciphertext: {GREEN}").upper()
    
    alphabet_input = input(f"{RESET}Enter custom alphabet (press Enter for default {YELLOW}A-Z{RESET}): {GREEN}").upper()
    alphabet = alphabet_input if alphabet_input else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"{RESET}Using alphabet: {YELLOW}{alphabet}{RESET}")
    
    # Get the shift value from the user
    while True:
        try:
            shift_input = input(f"{RESET}Enter the shift value (e.g., 3 or -3): {GREEN}")
            shift = int(shift_input)
            break
        except ValueError:
            print(f"{RED}Invalid input. Please enter an integer.{RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GREY}Program exited.{RESET}")
            sys.exit(0)

    print(f"{RESET}{GREY}-{RESET}" * 60)
    
    # Perform decryption
    decrypted_text = caesar_decipher(ciphertext, shift, alphabet)
    
    # Display the result
    print(f"{YELLOW}Ciphertext:{RESET} {ciphertext.lower()}")
    print(f"{YELLOW}Shift Value:{RESET} {shift}")
    print(f"{YELLOW}Decrypted Text:{RESET} {decrypted_text}")
    
    print(f"\n{GREY}Program finished.{RESET}")

if __name__ == "__main__":
    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{GREY}Program exited.{RESET}")
