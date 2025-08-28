import os
import time
import random
import sys
from datetime import datetime
import importlib

# Add the 'scripts' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Define a mapping from user-friendly names to module file names
MODULE_MAP = {
    'vigenere': 'vigenere',
    'gromark': 'Gromark_transposition',
    'gronsfeld': 'Gronsfeld',
    'autoclave': 'autoclave',
    'hill': 'hill',
    'xor': 'xor',
    'mod_add_sub': 'modular_add_sub',
    'caesar': 'caesar',
    'playfair': 'playfair',
    'matrix_generator': 'Matrix Generator',
    'ioc': 'ioc',
    'freq_analysis': 'freq_analysis'
}

# Try to import the cipher and tool modules
CIPHER_MODULES = {name: None for name in MODULE_MAP.keys()}

for user_name, file_name in MODULE_MAP.items():
    try:
        # Construct the full module path from the 'scripts' directory
        module = importlib.import_module(file_name)
        CIPHER_MODULES[user_name] = module
    except ImportError as e:
        # Provide a specific error message to help with debugging
        print(f"Warning: Module 'scripts/{file_name}.py' not found. Error: {e}")
        pass

# Retro style colors and effects
class Style:
    # Colors
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    
    # Effects
    BOLD = '\033[1m'
    REVERSE = '\033[7m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def terminal_width():
    """Get terminal width"""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80  # Default width

def print_centered(text, color=Style.WHITE):
    """Print centered text with color"""
    print(f"{color}{text.center(terminal_width())}{Style.RESET}")

def print_divider(char='=', color=Style.GRAY):
    """Print a divider line"""
    print(f"{color}{char * terminal_width()}{Style.RESET}")

def fancy_box(text, color=Style.YELLOW, padding=1):
    """Create a retro box around text"""
    lines = text.split('\n')
    width = max(len(line) for line in lines) + padding * 2
    
    print(f"{color}┌{'─' * width}┐{Style.RESET}")
    
    for line in lines:
        plain_line = ''.join(c for c in line if c.isprintable() and ord(c) < 128)
        padding_needed = width - len(plain_line)
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        print(f"{color}│{' ' * left_padding}{Style.RESET}{line}{color}{' ' * right_padding}│{Style.RESET}")
    
    print(f"{color}└{'─' * width}┘{Style.RESET}")

def retro_effect(duration=1.5):
    """Create a retro terminal effect"""
    width = terminal_width()
    chars = "█▓▒░ ░▒▓█"
    
    end_time = time.time() + duration
    while time.time() < end_time:
        line = "".join(random.choice(chars) for _ in range(width))
        print(f"{Style.GREEN}{line}{Style.RESET}", end="\r")
        time.sleep(0.05)
    print(" " * width, end="\r")  # Clear the last line

def display_logo():
    """Display the CBFT logo in retro style"""
    logo = f"""
    {Style.GREEN}   
  ██████╗██████╗ ███████╗████████╗
 ██╔════╝██╔══██╗██╔════╝╚══██╔══╝
 ██║     ██████╔╝█████╗      ██║  
 ██║     ██╔══██╗██╔══╝      ██║  
 ╚██████╗██████╔╝██║         ██║  
  ╚═════╝╚═════╝ ╚═╝         ╚═╝
    {Style.RESET}
    """
    print(logo)
    print_centered(f"{Style.BOLD}{Style.REVERSE} CIPHER BRUTE FORCE TOOLKIT {Style.RESET}", Style.WHITE)
    print_divider()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_centered(f"github.com/hippie-cycling", Style.GRAY)
    print_centered(f"Session started: {now}", Style.BLUE)
    print_divider()

def display_menu():
    """Display the menu"""
    options = [
        (0, "Help & Documentation", Style.WHITE, 'help'),
        (-1, "CIPHER OPTIONS", Style.CYAN, None),
        (1, "Vigenere Cipher", Style.GREEN, 'vigenere'),
        (2, "Gromark Cipher", Style.GREEN, 'gromark'),
        (3, "Gronsfeld Cipher", Style.GREEN, 'gronsfeld'),
        (4, "Autoclave Cipher", Style.GREEN, 'autoclave'),
        (5, "Hill Cipher", Style.GREEN, 'hill'),
        (6, "XOR", Style.GREEN, 'xor'),
        (7, "Mod ADD-SUB", Style.GREEN, 'mod_add_sub'),
        (8, "Caesar Cipher", Style.GREEN, 'caesar'),
        (9, "Playfair Cipher", Style.GREEN, 'playfair'),
        (-2, "CRYPTANALYSIS TOOLS", Style.CYAN, None),
        (10, "Matrix Generator", Style.YELLOW, 'matrix_generator'),
        (11, "Calculate IoC", Style.YELLOW, 'ioc'),
        (12, "Frequency Analysis", Style.YELLOW, 'freq_analysis'),
        (-3, "OTHER OPTIONS", Style.CYAN, None),
        ('A', "About", Style.WHITE, 'about'),
        ('E', "Exit", Style.RED, 'exit')
    ]
    
    print(f"\n{Style.BOLD}{Style.WHITE}Select an option:{Style.RESET}")
    print_divider('-', Style.GRAY)
    
    for number, name, color, _ in options:
        if isinstance(number, int) and number < 0:
            print()
            print(f" {color}{Style.BOLD}{name}{Style.RESET}")
            print_divider('-', Style.GRAY)
        else:
            print(f" {color}[{number}]{Style.RESET} {Style.BOLD}{name}{Style.RESET}")
    
    print_divider('-', Style.GRAY)
    
    # Corrected dictionary comprehension to handle strings and integers separately
    return {
        str(number): name for number, name, _, _ in options 
        if isinstance(number, (int, str)) and (isinstance(number, int) and number >= 0 or isinstance(number, str))
    }

def display_help():
    """Display help information"""
    help_text = f"""
{Style.BOLD}{Style.YELLOW}CIPHER BRUTE FORCE TOOLKIT - HELP{Style.RESET}

{Style.UNDERLINE}AVAILABLE CIPHERS:{Style.RESET}

{Style.GREEN}CAESAR CIPHER:{Style.RESET}
A simple substitution cipher where each letter is shifted by a fixed number of
positions down the alphabet. This tool allows for both encryption and decryption
with a known key.

{Style.GREEN}PLAYFAIR CIPHER:{Style.RESET}
A digraph substitution cipher. This tool can brute force the key by checking
for English-like Index of Coincidence (IoC) values or known plaintext phrases.

{Style.GREEN}VIGENERE CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every word in English wordlist, 
outputs keys that decrypt plaintext or match a defined IoC range.

{Style.GREEN}GROMARK CIPHER:{Style.RESET}
Checks all words from a wordlist against ciphertext. Outputs the word and key if
target plaintext or words are found.

{Style.GREEN}GRONSFELD CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every key and outputs those that
decrypt plaintext or match a defined IoC range.

{Style.GREEN}AUTOCLAVE CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every key and outputs those that
decrypt plaintext or match a defined IoC range.

{Style.GREEN}HILL CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every possible key (2x2 or 3x3 
matrix) and outputs those that decrypt plaintext or match a defined IoC range.
Direct decryption is also available.

{Style.GREEN}XOR:{Style.RESET}
Performs XOR cipher with a key. Outputs in decimal, ASCII, hex, and can map to
A-Z (0-25). Includes an IoC brute forcer with frequency analysis.

{Style.GREEN}MOD ADD-SUB:{Style.RESET}
Performs modular addition or subtraction with a key. Includes an IoC brute forcer
with frequency analysis.

{Style.GREEN}MATRIX GENERATOR:{Style.RESET}
Given a ciphertext, it will calculate every possible n x m matrix and save the
results from reading the columns left to right and vice versa.

{Style.UNDERLINE}CRYPTANALYSIS TOOLS:{Style.RESET}

{Style.YELLOW}CALCULATE IoC:{Style.RESET}
Calculates the Index of Coincidence for a given text. Useful for determining if
a cipher is monoalphabetic or polyalphabetic.

{Style.YELLOW}FREQUENCY ANALYSIS:{Style.RESET}
Performs frequency analysis on a given text. Helps identify potential substitution
ciphers by comparing letter frequencies with standard English letter frequencies.

{Style.UNDERLINE}TIPS:{Style.RESET}
• Try common words like "FROM", "THE", "LIKE", "THAT".
• Use the IoC range to filter potential solutions.
• Use frequency analysis to filter potential solutions.
• Note that if many results are found, only a few will be printed in console. Use the
  "Save into file" functionality and filter the results accordingly.
"""
    clear_screen()
    print(help_text)
    input(f"\n{Style.YELLOW}Press Enter to return to the main menu...{Style.RESET}")

def display_about():
    """Display information about the toolkit"""
    about_text = f"""
{Style.BOLD}{Style.CYAN}CIPHER BRUTE FORCE TOOLKIT{Style.RESET}

A comprehensive toolkit designed for cryptanalysis and cipher breaking.
This toolkit provides methods for brute forcing various classical ciphers
including Vigenere, Gromark, Gronsfeld, Autokey, Hill, XOR, Caesar shift,
modulo-based Addition and Subtraction, and Playfair.

{Style.UNDERLINE}This is a WIP project and more features will be added in the future.{Style.RESET}

{Style.UNDERLINE}FEATURES:{Style.RESET}
• Custom alphabet support
• Word list attacks (350k English words)
• Matrix attacks for Hill cipher (2x2 and 3x3).
• Index of Coincidence (IoC) analysis
• Frequency analysis

{Style.UNDERLINE}DEVELOPER:{Style.RESET}
github.com/hippie-cycling

{Style.UNDERLINE}LICENSE:{Style.RESET}
MIT License
"""
    clear_screen()
    print(about_text)
    input(f"\n{Style.GREEN}Press Enter to return to the main menu...{Style.RESET}")

def run_module(module_name):
    """Run a specific cipher or tool module"""
    module = CIPHER_MODULES.get(module_name)
    name = module_name.replace('_', ' ').title()
    
    if module:
        clear_screen()
        print(f"{Style.YELLOW}Running {name} module...{Style.RESET}\n")
        module.run()
        print(f"\n{Style.GREEN}[{name} process completed]{Style.RESET}")
    else:
        fancy_box(f" ERROR: {name} MODULE NOT FOUND ", Style.RED)
        print(f"\n{Style.RED}The {name} module could not be imported.{Style.RESET}")
        print(f"{Style.YELLOW}Check that 'scripts/{MODULE_MAP.get(module_name, '???')}.py' exists.{Style.RESET}")
    
    input(f"\n{Style.YELLOW}Press Enter to return to the main menu...{Style.RESET}")

def main():
    """Main function"""
    if 'idlelib' not in sys.modules:
        retro_effect()  # Avoid effects in basic IDLE
    clear_screen()
    
    while True:
        clear_screen()
        display_logo()
        display_menu()
        
        try:
            choice = input(f"\n{Style.GREEN}Enter your choice: {Style.RESET}").strip().upper()
            
            if choice == '0':
                display_help()
            elif choice == '1':
                run_module('vigenere')
            elif choice == '2':
                run_module('gromark')
            elif choice == '3':
                run_module('gronsfeld')
            elif choice == '4':
                run_module('autoclave')
            elif choice == '5':
                run_module('hill')
            elif choice == '6':
                run_module('xor')
            elif choice == '7':
                run_module('mod_add_sub')
            elif choice == '8':
                run_module('caesar')
            elif choice == '9':
                run_module('playfair')
            elif choice == '10':
                run_module('matrix_generator')
            elif choice == '11':
                run_module('ioc')
            elif choice == '12':
                run_module('freq_analysis')
            elif choice == 'A':
                display_about()
            elif choice == 'E':
                clear_screen()
                if 'idlelib' not in sys.modules:
                    retro_effect()
                print(f"\n{Style.GREEN}Goodbye!{Style.RESET}")
                break
            else:
                print(f"\n{Style.RED}Invalid choice!{Style.RESET} Please enter a valid option.")
                time.sleep(1.5)
        except KeyboardInterrupt:
            print(f"\n\n{Style.YELLOW}Operation interrupted.{Style.RESET}")
            time.sleep(1)
        except Exception as e:
            print(f"\n{Style.RED}Error: {str(e)}{Style.RESET}")
            time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{Style.RED}Program terminated.{Style.RESET}")
        sys.exit(0)