import Gromark_transposition
import Gronsfeld

def display_menu():
    print("\n" + "="*40)
    print("Please select a cipher to run:")
    print("0. Help")
    print("1. Gromark Cipher")
    print("2. Gronsfeld Cipher")
    print("3. Exit")
    print("="*40)

def main():
    # Display the logo once at the start
    print("""
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
""")
    print("       Cipher Brute Force Tool")
    print("="*40)
    print("Developed by Daniel Navarro (https://github.com/hippie-cycling)")
    print("="*40)
    
    while True:
        display_menu()
        
        choice = input("Enter your choice (0-3): ").strip()
        
        if choice == '0':
            print("""\n
- Gronsfield: The user can input a custom alphabet and plaintext words to be found. The brute force will check every key and output the keys that decript the plaintext words and the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
- Gromark: Input the ciphertext, the key for the keyed alphabet and the size of the primer key and it will brute force until one of the input plaintext is found and or the IoC of the plaintext closely matched english (in case the cipher is layered with transposition). Don't go crazy with the key size as it might literally tak forever to compute.
- Vigenere WIP: The user can input a custom alphabet and plaintext words to be found. The brute force will check every word in a large list of English words and output the keys that decript the plaintext words and/or the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
\nDon't go crazy with the key size as it might literally take forever to compute. Recommended key size is 1-9 digits, note the primer/s and extrapolate.""")
        elif choice == '1':
            print("\nYou selected: Gromark Cipher")
            print("Running brute force on Gromark Cipher...\n")
            Gromark_transposition.run()
        elif choice == '2':
            print("\nYou selected: Gronsfeld Cipher")
            print("Running brute force on Gronsfeld Cipher...\n")
            Gronsfeld.run()
        elif choice == '3':
            print("Exiting now...")
            break
        else:
            print("\nInvalid choice! Please enter a number between 0 and 3.")

if __name__ == "__main__":
    main()

