import os
import time
import random
import sys
from datetime import datetime

# Try to import the cipher modules
CIPHER_MODULES = {
    'vigenere': None,
    'gromark': None,
    'Gronsfeld': None,
    'autoclave': None,
    'hill': None,
    'xor': None,
    'mod_add_sub': None,
    'caesar': None,  # Added Caesar Cipher
    'Matrix Generator': None,
    'ioc': None,
    'freq_analysis': None
}

for module_name in CIPHER_MODULES:
    try:
        # Special handling for module names that don't match filenames directly
        if module_name == 'gromark':
            module = __import__(f'scripts.Gromark_transposition', fromlist=['run'])
        elif module_name == 'mod_add_sub':
            module = __import__(f'scripts.modular_add_sub', fromlist=['run'])
        else:
            # Standard import for other modules
            module = __import__(f'scripts.{module_name}', fromlist=['run'])
        CIPHER_MODULES[module_name] = module
    except ImportError:
        pass # Module not found, will be handled in run_cipher/run_tool

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
        return 80 # Default width

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
        # Basic stripping of ANSI codes for length calculation
        plain_line = ''.join(c for c in line if c.isprintable())
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
 ██║     ██████╔╝█████╗     ██║   
 ██║     ██╔══██╗██╔══╝     ██║   
 ╚██████╗██████╔╝██║        ██║   
  ╚═════╝╚═════╝ ╚═╝        ╚═╝
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
        (0, "Help & Documentation", Style.WHITE),
        (-1, "CIPHER OPTIONS", Style.CYAN),
        (1, "Vigenere Cipher", Style.GREEN),
        (2, "Gromark Cipher", Style.GREEN),
        (3, "Gronsfeld Cipher", Style.GREEN),
        (4, "Autoclave Cipher", Style.GREEN),
        (5, "Hill Cipher", Style.GREEN),
        (6, "XOR", Style.GREEN),
        (7, "Mod ADD-SUB", Style.GREEN),
        (8, "Caesar Cipher", Style.GREEN), # Added
        (9, "Matrix Generator", Style.GREEN), # Renumbered
        (-2, "CRYPTANALYSIS TOOLS", Style.CYAN),
        (12, "Calculate IoC", Style.YELLOW), # Renumbered
        (13, "Frequency Analysis", Style.YELLOW), # Renumbered
        (-3, "OTHER OPTIONS", Style.CYAN),
        (10, "About", Style.WHITE), # Renumbered
        (11, "Exit", Style.RED) # Renumbered
    ]
    
    print(f"\n{Style.BOLD}{Style.WHITE}Select an option:{Style.RESET}")
    print_divider('-', Style.GRAY)
    
    for number, name, color in options:
        if number < 0:  # This is a section header
            print()
            print(f" {color}{Style.BOLD}{name}{Style.RESET}")
            print_divider('-', Style.GRAY)
        else:
            print(f" {color}[{number}]{Style.RESET} {Style.BOLD}{name}{Style.RESET}")
    
    print_divider('-', Style.GRAY)

def display_help():
    """Display help information"""
    help_text = f"""
{Style.BOLD}{Style.YELLOW}CIPHER BRUTE FORCE TOOLKIT - HELP{Style.RESET}

{Style.UNDERLINE}AVAILABLE CIPHERS:{Style.RESET}

{Style.GREEN}VIGENERE CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every word in 
English wordlist, outputs keys that decrypt plaintext or 
match defined IoC.

{Style.GREEN}GROMARK CIPHER:{Style.RESET}
Checks all words from wordlist against ciphertext.
Outputs word and key if target plaintext or words are found.

{Style.GREEN}GRONSFELD CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every key and
outputs those that decrypt plaintext or matches defined IoC.

{Style.GREEN}AUTOCLAVE CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every key and
outputs those that decrypt plaintext or matches defined IoC.

{Style.GREEN}HILL CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every posible key
(2x2 or 3x3 matrix) and outputs those that decrypt plaintext
or matches defined IoC. Direct decrytion is available.

{Style.GREEN}XOR:{Style.RESET}
XOR cipher and key.
Outputs in decimal, ASCII, hex and can map to A-Z (0-25).
Includes an IoC brute forcer with frequency analysis.

{Style.GREEN}MOD ADD-SUB:{Style.RESET}
Add or subtract cipher and key (modulo addition or subtraction).
Includes an IoC brute forcer with frequency analysis.

{Style.GREEN}Matrix Generator:{Style.RESET}
Given a ciphertext, it will calculate every possible n x m matrix
and save the results from reading the columns left to right and
viceversa.

{Style.UNDERLINE}CRYPTANALYSIS TOOLS:{Style.RESET}

{Style.YELLOW}CALCULATE IoC:{Style.RESET}
Calculate the Index of Coincidence for a given text.
Useful for determining if a cipher is monoalphabetic or
polyalphabetic.

{Style.YELLOW}FREQUENCY ANALYSIS:{Style.RESET}
Perform frequency analysis on a given text.
Helps identify potential substitution ciphers by comparing
letter frequencies with standard English letter frequencies.

{Style.UNDERLINE}TIPS:{Style.RESET}
• Try common words like "FROM", "THE", "LIKE", "THAT"
• Use the IoC range to filter potential solutions.
• Use frequency analysis to filter potential solutions.
• Note that if many results are found, only a few will
be printed in console. Use the Save into file functionality
and filter the results accordingly.
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
including Vigenere, Gromark, Gronsfeld, Autokey, Hill, XOR, Caesar shift, modulo based Addition
and Subtraction.

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

def display_help():
    """Display help information"""
    help_text = f"""
{Style.BOLD}{Style.YELLOW}CIPHER BRUTE FORCE TOOLKIT - HELP{Style.RESET}

{Style.UNDERLINE}AVAILABLE CIPHERS:{Style.RESET}
{Style.GREEN}CAESAR CIPHER:{Style.RESET}
A simple substitution cipher where each letter is shifted by a
fixed number of positions down the alphabet. This tool allows
for direct decryption with a known shift value.

{Style.GREEN}VIGENERE CIPHER:{Style.RESET}
Input custom alphabet and target words. Checks every word in 
English wordlist, outputs keys that decrypt plaintext or 
match defined IoC.

... (rest of the help text remains the same) ...
"""
    clear_screen()
    print(help_text)
    input(f"\n{Style.YELLOW}Press Enter to return to the main menu...{Style.RESET}")

def run_cipher(module_name):
    """Run a specific cipher module"""
    module = CIPHER_MODULES.get(module_name)
    name = module_name.replace('_', ' ').upper()
    
    if module:
        clear_screen()
        print(f"{Style.YELLOW}Running {name} cipher...{Style.RESET}\n")
        module.run()
        print(f"\n{Style.GREEN}[{name} process completed]{Style.RESET}")
    else:
        fancy_box(f" ERROR: {name} MODULE NOT FOUND ", Style.RED)
        print(f"\n{Style.RED}The {name} module could not be imported.{Style.RESET}")
        print(f"{Style.YELLOW}Check that 'scripts/{module_name}.py' exists.{Style.RESET}")
    
    input(f"\n{Style.YELLOW}Press Enter to return to the main menu...{Style.RESET}")

def run_tool(module_name):
    """Run a specific cryptanalysis tool module"""
    # This function remains largely unchanged
    run_cipher(module_name) # Can reuse the same logic

def main():
    """Main function"""
    if 'idlelib' not in sys.modules:
        retro_effect() # Avoid effects in basic IDLE
    clear_screen()
    
    while True:
        clear_screen()
        display_logo()
        display_menu()
        
        try:
            choice = input(f"\n{Style.GREEN}Enter your choice (0-13): {Style.RESET}").strip()
            
            if choice == '0': display_help()
            elif choice == '1': run_cipher('vigenere')
            elif choice == '2': run_cipher('gromark')
            elif choice == '3': run_cipher('Gronsfeld')
            elif choice == '4': run_cipher('autoclave')
            elif choice == '5': run_cipher('hill')
            elif choice == '6': run_cipher('xor')
            elif choice == '7': run_cipher('mod_add_sub')
            elif choice == '8': run_cipher('caesar') # Added
            elif choice == '9': run_cipher('Matrix Generator') # Renumbered
            elif choice == '10': display_about() # Renumbered
            elif choice == '11': # Renumbered
                clear_screen()
                if 'idlelib' not in sys.modules: retro_effect()
                print(f"\n{Style.GREEN}Goodbye!{Style.RESET}")
                break
            elif choice == '12': run_tool('ioc') # Renumbered
            elif choice == '13': run_tool('freq_analysis') # Renumbered
            else:
                print(f"\n{Style.RED}Invalid choice!{Style.RESET} Please enter a number between 0 and 13.")
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