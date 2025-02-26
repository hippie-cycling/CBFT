import Gromark_transposition
import Gronsfeld

RED = '\033[38;5;88m'
YELLOW = '\033[38;5;3m'
GREY = '\033[38;5;238m'
RESET = '\033[0m'

def display_menu():
    print("\n" + "="*40)
    print("Please select a cipher to run:")
    print(f"{YELLOW}0.{RESET} Help")
    print(f"{YELLOW}1.{RESET} Gromark Cipher")
    print(f"{YELLOW}2.{RESET} Gronsfeld Cipher")
    print(f"{YELLOW}3.{RESET} Exit")
    print("="*40)

def main():
    # Display the logo once at the start
    print(f"""{RED}
 ▄████▄   ▄▄▄▄     █████▒▄▄▄█████▓
▒██▀ ▀█  ▓█████▄ ▓██   ▒ ▓  ██▒ ▓▒
▒▓█    ▄ ▒██▒ ▄██▒████ ░ ▒ ▓██░ ▒░
▒▓▓▄ ▄██▒▒██░█▀  ░▓█▒  ░ ░ ▓██▓ ░ 
▒ ▓███▀ ░░▓█  ▀█▓░▒█░      ▒██▒ ░ 
░ ░▒ ▒  ░░▒▓███▀▒ ▒ ░      ▒ ░░   
  ░  ▒   ▒░▒   ░  ░          ░    
░         ░    ░  ░ ░      ░      
░ ░       ░                       
░              ░                  
{RESET}""")
    print("       Cipher Brute Force Tool")
    print("="*40)
    print(f"Developed by Daniel Navarro\n{YELLOW}(https://github.com/hippie-cycling){RESET}")
    print("="*40)
    
    while True:
        display_menu()
        
        choice = input(f"Enter your choice ({YELLOW}0-3{RESET}): ").strip()
        
        if choice == '0':
            print("""\n
- Gronsfield: The user can input a custom alphabet and plaintext words to be found. The brute force will check every key and output the keys that decript the plaintext words and the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
- Gromark: Input the ciphertext, the key for the keyed alphabet and the size of the primer key and it will brute force until one of the input plaintext is found and or the IoC of the plaintext closely matched english (in case the cipher is layered with transposition). Don't go crazy with the key size as it might literally tak forever to compute.
- Vigenere WIP: The user can input a custom alphabet and plaintext words to be found. The brute force will check every word in a large list of English words and output the keys that decript the plaintext words and/or the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
\nDon't go crazy with the key size as it might literally take forever to compute. Recommended key size is 1-9 digits, note the primer/s and extrapolate.""")
        elif choice == '1':
            print("\nYou selected: Gromark Cipher")
            print("Running brute force on Gromark Cipher...")
            Gromark_transposition.run()
        elif choice == '2':
            print("\nYou selected: Gronsfeld Cipher")
            print("Running brute force on Gronsfeld Cipher...")
            Gronsfeld.run()
        elif choice == '3':
            print(f"{RED}Exiting now...{RESET}")
            break
        else:
            print(f"\n{RED}Invalid choice!{RESET} Please enter a number between 0 and 3.")

if __name__ == "__main__":
    main()

