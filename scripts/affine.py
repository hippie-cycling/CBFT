import math

RED, YELLOW, GREY, RESET, GREEN = '\033[38;5;88m', '\033[38;5;3m', '\033[38;5;238m', '\033[0m', '\033[38;5;2m'

def decrypt_affine(ciphertext, a, b):
    # Find modular inverse of 'a'
    mod_inv = pow(a, -1, 26)
    plaintext = ""
    for char in ciphertext.upper():
        if char.isalpha():
            p = (mod_inv * (ord(char) - 65 - b)) % 26
            plaintext += chr(p + 65)
        else:
            plaintext += char
    return plaintext

def run():
    print(f"{GREEN}===================================={RESET}")
    print(f"{GREEN}=====       Affine Cipher      ====={RESET}")
    print(f"{GREEN}===================================={RESET}")
    
    ciphertext = input(f"{GREY}Enter ciphertext: {RESET}")
    if not ciphertext: return
    
    print(f"\n{YELLOW}Brute-forcing all 312 combinations...{RESET}")
    
    # Valid 'a' values must be coprime with 26
    valid_a = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    
    for a in valid_a:
        for b in range(26):
            pt = decrypt_affine(ciphertext, a, b)
            # Basic visual filter: If it has spaces, check if short words look English
            if " " in pt:
                words = pt.split()
                if any(w in ["THE", "AND", "TO", "OF", "A", "IN", "IS", "IT"] for w in words):
                    print(f"Key A:{YELLOW}{a:<2}{RESET} B:{YELLOW}{b:<2}{RESET} | {GREEN}{pt[:60]}{RESET}...")
            else:
                # If no spaces, just print a few and let the user look
                if a == 3 and b < 5: # Just a sample
                    print(f"Key A:{YELLOW}{a:<2}{RESET} B:{YELLOW}{b:<2}{RESET} | {GREEN}{pt[:60]}{RESET}...")