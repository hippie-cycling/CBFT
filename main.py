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
    'affine': 'affine',
    'polybius': 'polybius',                                # NEW
    'bifid': 'bifid',                                      # NEW
    'rail_fence': 'rail_fence',
    'columnar_transposition': 'columnar_transposition',
    'scytale': 'Scytale',
    'permutation_solver': 'permutation_solver',
    'matrix_generator': 'Matrix Generator',
    'ioc': 'ioc',
    'freq_analysis': 'freq_analysis',
    'kasiski': 'kasiski',                             
    'friedman_test': 'friedman_test',
    'pattern_isomorphism': 'pattern_isomorphism',
    'crib_drag': 'crib_drag',
    'simulated_annealing': 'simulated_annealing',      
    'hill_climbing_transposition': 'hill_climbing_transposition',
    'vigenere_auto_solver': 'vigenere_auto_solver',        # NEW
    'text_formatter': 'text_formatter'                     # NEW
}

# Try to import the cipher and tool modules
CIPHER_MODULES = {name: None for name in MODULE_MAP.keys()}

for user_name, file_name in MODULE_MAP.items():
    try:
        module = importlib.import_module(file_name)
        CIPHER_MODULES[user_name] = module
    except ImportError as e:
        print(f"Warning: Module 'scripts/{file_name}.py' not found. Error: {e}")
        pass

# Retro style colors and effects
class Style:
    GREEN, YELLOW, RED, BLUE, CYAN, WHITE, GRAY = '\033[32m', '\033[33m', '\033[31m', '\033[34m', '\033[36m', '\033[37m', '\033[90m'
    BOLD, REVERSE, UNDERLINE, RESET = '\033[1m', '\033[7m', '\033[4m', '\033[0m'

def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')

def terminal_width():
    try: return os.get_terminal_size().columns
    except OSError: return 80

def print_centered(text, color=Style.WHITE): print(f"{color}{text.center(terminal_width())}{Style.RESET}")
def print_divider(char='=', color=Style.GRAY): print(f"{color}{char * terminal_width()}{Style.RESET}")

def fancy_box(text, color=Style.YELLOW, padding=1):
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
    width = terminal_width()
    chars = "█▓▒░ ░▒▓█"
    end_time = time.time() + duration
    while time.time() < end_time:
        line = "".join(random.choice(chars) for _ in range(width))
        print(f"{Style.GREEN}{line}{Style.RESET}", end="\r")
        time.sleep(0.05)
    print(" " * width, end="\r")

def display_logo():
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
    print_centered(f"github.com/hippie-cycling", Style.GRAY)
    print_centered(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Style.BLUE)
    print_divider()

def display_menu():
    options = [
        (0, "Help & Documentation", Style.WHITE, 'help'),
        (-1, "SUBSTITUTION CIPHER OPTIONS", Style.CYAN, None),
        (1, "Vigenere Cipher", Style.GREEN, 'vigenere'),
        (2, "Gromark Cipher", Style.GREEN, 'gromark'),
        (3, "Gronsfeld Cipher", Style.GREEN, 'gronsfeld'),
        (4, "Autoclave Cipher", Style.GREEN, 'autoclave'),
        (5, "Hill Cipher", Style.GREEN, 'hill'),
        (6, "XOR", Style.GREEN, 'xor'),
        (7, "Modular ADD-SUB", Style.GREEN, 'mod_add_sub'),
        (8, "Caesar Cipher", Style.GREEN, 'caesar'),
        (9, "Playfair Cipher", Style.GREEN, 'playfair'),
        (10, "Affine Cipher", Style.GREEN, 'affine'),
        (11, "Polybius Square", Style.GREEN, 'polybius'),
        (12, "Bifid Cipher", Style.GREEN, 'bifid'),
        (-2, "TRANSPOSITION CIPHER OPTIONS", Style.CYAN, None),
        (20, "Columnar Transposition Cipher", Style.GREEN, 'columnar_transposition'),
        (21, "Scytale Cipher", Style.GREEN, 'scytale'),
        (22, "Rail Fence Cipher", Style.GREEN, 'rail_fence'),
        (23, "Matrix Permutation Solver", Style.GREEN, 'permutation_solver'),
        (-3, "CRYPTANALYSIS TOOLS", Style.CYAN, None),
        (30, "Vigenere Auto-Solver", Style.YELLOW, 'vigenere_auto_solver'),
        (31, "Simulated Annealing (Substitution)", Style.YELLOW, 'simulated_annealing'),
        (32, "Hill Climbing (Transposition)", Style.YELLOW, 'hill_climbing_transposition'),
        (33, "Word Pattern Isomorphism", Style.YELLOW, 'pattern_isomorphism'),
        (34, "Automated Crib Dragging", Style.YELLOW, 'crib_drag'),
        (35, "Kasiski Examination", Style.YELLOW, 'kasiski'),
        (36, "Friedman Test (Key Length Estimator)", Style.YELLOW, 'friedman_test'),
        (-4, "UTILITIES", Style.CYAN, None),
        (40, "Calculate IoC", Style.GRAY, 'ioc'),
        (41, "Frequency Analysis", Style.GRAY, 'freq_analysis'),
        (42, "Matrix Generator", Style.GRAY, 'matrix_generator'),
        (43, "Text Formatter", Style.GRAY, 'text_formatter'),
        (-5, "SYSTEM", Style.CYAN, None),
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
    
    return {
        str(number): name for number, name, _, _ in options 
        if isinstance(number, (int, str)) and (isinstance(number, int) and number >= 0 or isinstance(number, str))
    }

def run_module(module_name):
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
    input(f"\n{Style.YELLOW}Press Enter to return to the main menu...{Style.RESET}")

def main():
    if 'idlelib' not in sys.modules: retro_effect()
    clear_screen()
    while True:
        clear_screen()
        display_logo()
        options_dict = display_menu()
        
        try:
            choice = input(f"\n{Style.GREEN}Enter your choice: {Style.RESET}").strip().upper()
            if choice == '0':
                clear_screen()
                print("Help Menu Updated! See README.md or individual tools for exact parameters.")
                input("\nPress Enter to return...")
            elif choice == 'A':
                clear_screen()
                print("Cipher Brute Force Toolkit. A comprehensive cryptanalysis and heuristic decryption suite.")
                input("\nPress Enter to return...")
            elif choice == 'E':
                clear_screen()
                print(f"\n{Style.GREEN}Goodbye!{Style.RESET}")
                break
            elif choice in options_dict:
                mapping = {
                    '1': 'vigenere', '2': 'gromark', '3': 'gronsfeld', '4': 'autoclave', '5': 'hill',
                    '6': 'xor', '7': 'mod_add_sub', '8': 'caesar', '9': 'playfair', '10': 'affine',
                    '11': 'polybius', '12': 'bifid',
                    '20': 'columnar_transposition', '21': 'scytale', '22': 'rail_fence', '23': 'permutation_solver',
                    '30': 'vigenere_auto_solver', '31': 'simulated_annealing', '32': 'hill_climbing_transposition',
                    '33': 'pattern_isomorphism', '34': 'crib_drag', '35': 'kasiski', '36': 'friedman_test',
                    '40': 'ioc', '41': 'freq_analysis', '42': 'matrix_generator', '43': 'text_formatter'
                }
                if choice in mapping:
                    run_module(mapping[choice])
            else:
                print(f"\n{Style.RED}Invalid choice!{Style.RESET}")
                time.sleep(1)
        except Exception as e:
            print(f"\n{Style.RED}Error: {str(e)}{Style.RESET}")
            time.sleep(2)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(0)