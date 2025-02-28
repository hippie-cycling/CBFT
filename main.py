import os
import time
import random
import sys
from datetime import datetime

#TO-DO Substitution ciphers
# Modular Addition
# Running key cipher
# Chaocipher
# Autokey cipher

# Try to import the cipher modules
try:
    import vigenere
    vigenere_imported = True
except ImportError:
    vigenere_imported = False

try:
    import Gromark_transposition
    gromark_imported = True
except ImportError:
    gromark_imported = False

try:
    import Gronsfeld
    gronsfeld_imported = True
except ImportError:
    gronsfeld_imported = False

try:
    import xor
    xor_imported = True
except ImportError:
    xor_imported = False

try:
    import modular_add_sub
    modular_add_sub_imported = True
except ImportError:
    modular_add_sub_imported = False

# Enhanced colors and styling
COLORS = {
    'red': '\033[38;5;88m',
    'bright_red': '\033[38;5;88m',
    'dark_red': '\033[38;5;88m',
    'yellow': '\033[38;5;3m',
    'gold': '\033[38;5;214m',
    'orange': '\033[38;5;208m',
    'green': '\033[38;5;46m',
    'lime': '\033[38;5;118m',
    'blue': '\033[38;5;39m',
    'cyan': '\033[38;5;51m',
    'magenta': '\033[38;5;201m',
    'purple': '\033[38;5;93m',
    'grey': '\033[38;5;240m',
    'dark_grey': '\033[38;5;236m',
    'white': '\033[38;5;255m',
    'black': '\033[38;5;232m',
    'reset': '\033[0m'
}

# Background colors
BG = {
    'red': '\033[48;5;196m',
    'dark_red': '\033[48;5;88m',
    'black': '\033[48;5;232m',
    'dark_grey': '\033[48;5;236m',
    'grey': '\033[48;5;240m',
    'blue': '\033[48;5;39m',
    'reset': '\033[0m'
}

# Text effects
EFFECTS = {
    'bold': '\033[1m',
    'dim': '\033[2m',
    'italic': '\033[3m',
    'underline': '\033[4m',
    'blink': '\033[5m',
    'reverse': '\033[7m',
    'reset': '\033[0m'
}

def matrix_effect(duration=2):
    """Create a matrix-like effect on the screen"""
    width = os.get_terminal_size().columns
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    end_time = time.time() + duration
    while time.time() < end_time:
        line = ""
        for i in range(width):
            if random.random() > 0.8:
                line += f"{COLORS['yellow']}{random.choice(chars)}{COLORS['reset']}"
            else:
                line += " "
        print(line, end="\r")
        time.sleep(0.05)
    print(" " * width, end="\r")  # Clear the last line

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def fancy_box(text, color='yellow', width=None, padding=1):
    """Create a fancy box around text"""
    if width is None:
        lines = text.split('\n')
        width = max(len(line) for line in lines) + padding * 2
    
    horizontal = f"{COLORS[color]}╭{'─' * width}╮{COLORS['reset']}"
    empty = f"{COLORS[color]}│{' ' * width}│{COLORS['reset']}"
    
    print(horizontal)
    if padding > 0:
        for _ in range(padding):
            print(empty)
    
    for line in text.split('\n'):
        padding_needed = width - len(line)
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        print(f"{COLORS[color]}│{' ' * left_padding}{COLORS['reset']}{line}{COLORS[color]}{' ' * right_padding}│{COLORS['reset']}")
    
    if padding > 0:
        for _ in range(padding):
            print(empty)
    
    bottom = f"{COLORS[color]}╰{'─' * width}╯{COLORS['reset']}"
    print(bottom)

def display_logo():
    """Display the enhanced CBFT logo"""
    logo = f"""{COLORS['dark_red']}
  ██████╗██████╗ ███████╗████████╗
 ██╔════╝██╔══██╗██╔════╝╚══██╔══╝
 ██║     ██████╔╝█████╗     ██║   
 ██║     ██╔══██╗██╔══╝     ██║   
 ╚██████╗██████╔╝██║        ██║   
  ╚═════╝╚═════╝ ╚═╝        ╚═╝   
{COLORS['reset']}"""

    sub_title = f"{BG['dark_grey']}{COLORS['white']}{EFFECTS['bold']} CIPHER BRUTE FORCE TOOLKIT {EFFECTS['reset']}{BG['reset']}"
    
    print(logo)
    print(sub_title.center(os.get_terminal_size().columns))
    print(f"{COLORS['grey']}{'═' * os.get_terminal_size().columns}{COLORS['reset']}")
    
    # Current timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = f"{COLORS['grey']}Developed by {COLORS['yellow']}(https://github.com/hippie-cycling){COLORS['reset']}"
    timestamp = f"{COLORS['blue']}Session started: {now}{COLORS['reset']}"
    
    print(footer.center(os.get_terminal_size().columns))
    print(timestamp.center(os.get_terminal_size().columns))
    print(f"{COLORS['grey']}{'═' * os.get_terminal_size().columns}{COLORS['reset']}")
                                                                  
def display_avatar():   
    avatar = f"""{COLORS['dark_red']}                                                                  
                                                                      
                          .=*: *%%%@@%#*:--..                         
                    .==+*+-                 :+**#                     
                .---                              =**                 
             +-            ....      .  :. :         =%-              
          +*.           .          ## -- --.:::. .     -#%            
        **                ==-:..-*@%@=*%      :..     .@  -%          
       %:             =.            .* .  @@@@@@@@@@@##@    *@:       
     @@                    %@@@@@@*  -%@  @@@@-  =:@@@=       @-      
    @@.        @@:       #@@@@@@@@= .@:        +%@@:     -     @.     
   %#:         .  -@-  *@@@#@@@%#:          =@#+@@@@#*@% .+@.   @     
   @@@               =*=.  @* #@@@@* -     @@%. @@@@@@%@@@@@+   :@    
  +@@*.         :        :.@@@@@@@@@@@    @@@+   =%@@+@@+    -   @    
   @*%     .      :=    +@@@@+@@@@@  %@.   @@#+%**#*@@@          @    
   @@@@           ::=@@@@@  +@@@@#  @@@@#   @@@*+=%@%=:          @    
   @@@@+            -:.  %@#. +*..+@@--+@#   %@. +#*            @%    
   -@@@+  -=        .=:=*@*@@@**@@-  .+#:%+  -- @+             %@     
    @-@#: :        -.   . *@ +         @- #     @# +.         +@      
     @%@@@@=       :@   *@           -@@@=.   :@@%+==@*     .@*       
      -@@@@=           -@+=         %%@@@@@@@@@@@@@@@@@  - -@*        
       .@#@@:        .: ..        * +@@@@@@@@@@@@@@@:*:   *=          
         +@#+. -#=:%%    +       **@@@@@@@@@@@@@@@@: +-.#*            
            @=    #@@    #*       .%=   .*@@@@@@*   -@@:              
           @-    .@@@@@@@@@%        *@=    :    :  *%                 
           @%  -@%+ *%@@@%#@=-.-     =#%%#-:.  #@@@:                  
            *@@@=           =++*--*@@%%*#++%@@@                       

    {COLORS['reset']}"""                                                                      
    print(avatar)                                                                      

def display_menu():
    """Display the enhanced menu"""
    options = [
        (0, "Help & Documentation", "white", "Get detailed information about each cipher"),
        (1, "Vigenere Cipher", "yellow", "Polyalphabetic substitution cipher using a keyword"),
        (2, "Gromark Cipher", "yellow", "Numerical key-based cipher with transposition"),
        (3, "Gronsfeld Cipher", "yellow", "Similar to Vigenere but using numbers as the key"),
        (4, "XOR", "yellow", "Perform XOR operation"),
        (5, "Mod. ADD / SUB", "yellow", "Perform modular addition or subtraction"),
        (6, "About", "white", "Information about this toolkit"),
        (7, "Exit", "red", "Exit the application")
    ]
    
    menu_width = os.get_terminal_size().columns - 10
    print(f"\n{COLORS['grey']}{'═' * menu_width}{COLORS['reset']}")
    print(f"{EFFECTS['bold']}{COLORS['white']}Select a cipher to run:{COLORS['reset']}{EFFECTS['reset']}")
    print(f"{COLORS['grey']}{'═' * menu_width}{COLORS['reset']}")
    
    # Check which modules are imported successfully
    module_status = {
        "Vigenere": vigenere_imported,
        "Gromark": gromark_imported,
        "Gronsfeld": gronsfeld_imported,
        "XOR": xor_imported,
        "Mod. ADD / SUB": modular_add_sub_imported
    }
    
    for opt in options:
        number, name, color, desc = opt
        status = ""
        
        # Add status indicator for cipher modules
        if name.split()[0] in module_status:
            if module_status[name.split()[0]]:
                status = f"{COLORS['green']}[READY]{COLORS['reset']}"
            else:
                status = f"{COLORS['red']}[NOT FOUND]{COLORS['reset']}"
        
        print(f"{COLORS[color]}[{number}]{COLORS['reset']} {EFFECTS['bold']}{name}{EFFECTS['reset']} {status}")
        print(f"   {COLORS['grey']}{desc}{COLORS['reset']}")
    
    print(f"{COLORS['grey']}{'═' * menu_width}{COLORS['reset']}")

def display_help():
    """Display help information with enhanced formatting"""
    help_text = f"""
{EFFECTS['bold']}{COLORS['yellow']}Cipher Brute Force Toolkit - Help Documentation{EFFECTS['reset']}{COLORS['reset']}

{EFFECTS['underline']}Available Ciphers:{EFFECTS['reset']}

{COLORS['yellow']}Vigenere Cipher:{COLORS['reset']}
The user can input a custom alphabet and plaintext words to be found. 
The brute force will check every word in a large list of English words and output 
the keys that decrypt the plaintext words and/or the keys that generate an IoC 
close to English (0.06 <= ioc <= 0.07) for further analysis.

{COLORS['yellow']}Gromark Cipher:{COLORS['reset']}
Input the ciphertext and it will brute force all words from words_alpha.text. 
The script will output the word and key if one of the input plaintext is found 
and/or the input words can be found in the output (even if transposed).

{COLORS['yellow']}Gronsfeld Cipher:{COLORS['reset']}
The user can input a custom alphabet and plaintext words to be found. 
The brute force will check every key and output the keys that decrypt the 
plaintext words and the keys that generate an IoC close to English 
(0.06 <= ioc <= 0.07) for further analysis.

{COLORS['yellow']}XOR:{COLORS['reset']}
The user can input a cipher and a key and the script will XOR both.
If the key length is shorter than the cipher, the key will be repeated.
The script will output the XOR result in decimal, ASCII, and hex format.
The user can also map the result to A-Z (0-25) for further analysis.
IoC brute force analysis is also available.
Frequency analysis is also available.

{COLORS['yellow']}Mod. ADD / SUB:{COLORS['reset']}
The user can input a cipher and a key and the script will add or subtract both (modulo).
If the key length is shorter than the cipher, the key will be repeated.
IoC brute force analysis is also available.
Frequency analysis is also available.

{EFFECTS['underline']}Tips & Warnings:{EFFECTS['reset']}
• You don´t know any plaintext word? try common words such as "FROM, "THE", "LIKE", "THAT", etc.
Note that a large quantity of outputs will be generated. So choose wisely and perform frequency
analysis to filter potential solutions.
"""
    clear_screen()
    print(help_text)
    input(f"\n{COLORS['yellow']}Press Enter to return to the main menu...{COLORS['reset']}")

def display_about():
    """Display information about the toolkit"""
    about_text = f"""
{EFFECTS['bold']}{COLORS['cyan']}Cipher Brute Force Toolkit{EFFECTS['reset']}{COLORS['reset']}

A comprehensive toolkit designed for cryptanalysis and cipher breaking.
This toolkit provides methods for brute forcing various classical ciphers
including Vigenere, Gromark, and Gronsfeld.

{EFFECTS['underline']}Features:{EFFECTS['reset']}
• Customizable alphabet support
• Word list-based attacks
• Index of Coincidence (IoC) analysis
• Frequency analysis
• XOR operation and analysis
• Modular addition and subtraction

{EFFECTS['underline']}Developer:{EFFECTS['reset']}
Daniel Navarro
{COLORS['yellow']}https://github.com/hippie-cycling{COLORS['reset']}

{EFFECTS['underline']}License:{EFFECTS['reset']}
This software is provided under the MIT License.
"""
    clear_screen()
    display_avatar()
    print(about_text)
    input(f"\n{COLORS['yellow']}Press Enter to return to the main menu...{COLORS['reset']}")

def run_cipher(module, name, color):
    """Run a specific cipher module with enhanced UI"""
    if module:
        clear_screen()
        module.run()
        print(f"\n{COLORS[color]}[{name} cipher process completed]{COLORS['reset']}")
    else:
        fancy_box(f" ERROR: {name.upper()} MODULE NOT FOUND ", "red", width=40)
        print(f"\n{COLORS['red']}The {name} cipher module could not be imported.{COLORS['reset']}")
        print(f"{COLORS['yellow']}Please ensure that the module file exists and is correctly formatted.{COLORS['reset']}")
    
    input(f"\n{COLORS['yellow']}Press Enter to return to the main menu...{COLORS['reset']}")

def main():
    """Main function with enhanced UI"""
    matrix_effect(1.0)
    clear_screen()
    while True:
        clear_screen()
        display_logo()
        display_menu()
        
        try:
            choice = input(f"\n{COLORS['yellow']}Enter your choice (0-6): {COLORS['reset']}").strip()
            
            if choice == '0':
                display_help()
            elif choice == '1':
                run_cipher(vigenere if vigenere_imported else None, "Vigenere", "yellow")
            elif choice == '2':
                run_cipher(Gromark_transposition if gromark_imported else None, "Gromark", "yellow")
            elif choice == '3':
                run_cipher(Gronsfeld if gronsfeld_imported else None, "Gronsfeld", "yellow")
            elif choice == '4':
                run_cipher(xor if xor_imported else None, "XOR", "yellow")
            elif choice == '5':
                run_cipher(modular_add_sub if modular_add_sub_imported else None, "Mod. ADD / SUB", "yellow")
            elif choice == '6':
                display_about()
            elif choice == '7':
                clear_screen()
                matrix_effect(1.0)
                break
            else:
                print(f"\n{COLORS['red']}Invalid choice!{COLORS['reset']} Please enter a number between 0 and 5.")
                time.sleep(1.5)
        except KeyboardInterrupt:
            print(f"\n\n{COLORS['yellow']}Operation interrupted by user.{COLORS['reset']}")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"\n{COLORS['red']}An error occurred: {str(e)}{COLORS['reset']}")
            time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{COLORS['red']}Program terminated by user.{COLORS['reset']}")
        sys.exit(0)