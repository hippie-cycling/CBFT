RED, YELLOW, GREY, RESET, GREEN = '\033[38;5;88m', '\033[38;5;3m', '\033[38;5;238m', '\033[0m', '\033[38;5;2m'

def run():
    print(f"{GREEN}===================================={RESET}")
    print(f"{GREEN}====  Crypto Text Pre-Processor ===={RESET}")
    print(f"{GREEN}===================================={RESET}")
    
    text = input(f"{GREY}Enter text to format: {RESET}")
    if not text: return
    
    print(f"\n{YELLOW}Available Operations:{RESET}")
    print(f"({YELLOW}1{RESET}) Strip everything except letters (Uppercase)")
    print(f"({YELLOW}2{RESET}) Remove spaces only")
    print(f"({YELLOW}3{RESET}) Format into blocks of 5 (Standard Crypto Format)")
    print(f"({YELLOW}4{RESET}) Reverse the text")
    
    choice = input(f"{GREY}Selection: {RESET}")
    
    print(f"\n{GREEN}Result:{RESET}")
    if choice == '1':
        print("".join(filter(str.isalpha, text)).upper())
    elif choice == '2':
        print(text.replace(" ", ""))
    elif choice == '3':
        clean = "".join(filter(str.isalpha, text)).upper()
        print(" ".join(clean[i:i+5] for i in range(0, len(clean), 5)))
    elif choice == '4':
        print(text[::-1])