RED, YELLOW, GREY, RESET, GREEN = '\033[38;5;88m', '\033[38;5;3m', '\033[38;5;238m', '\033[0m', '\033[38;5;2m'

def decrypt_rail_fence(ciphertext, rails):
    if rails <= 1: return ciphertext
    
    # Build a template to see where characters go
    template = [['\n' for i in range(len(ciphertext))] for j in range(rails)]
    dir_down = None
    row, col = 0, 0
    
    for i in range(len(ciphertext)):
        if row == 0: dir_down = True
        if row == rails - 1: dir_down = False
        template[row][col] = '*'
        col += 1
        row += 1 if dir_down else -1
        
    # Fill the template with actual ciphertext characters
    idx = 0
    for i in range(rails):
        for j in range(len(ciphertext)):
            if template[i][j] == '*' and idx < len(ciphertext):
                template[i][j] = ciphertext[idx]
                idx += 1
                
    # Read the zigzag to get plaintext
    result = []
    row, col = 0, 0
    for i in range(len(ciphertext)):
        if row == 0: dir_down = True
        if row == rails-1: dir_down = False
        if template[row][col] != '*':
            result.append(template[row][col])
            col += 1
        row += 1 if dir_down else -1
        
    return "".join(result)

def run():
    print(f"{GREEN}===================================={RESET}")
    print(f"{GREEN}=====     Rail Fence Cipher    ====={RESET}")
    print(f"{GREEN}===================================={RESET}")
    
    ciphertext = input(f"{GREY}Enter ciphertext: {RESET}").upper()
    if not ciphertext: return
    
    max_rails = min(20, len(ciphertext))
    print(f"\n{YELLOW}Brute-forcing rails 2 through {max_rails}...{RESET}")
    print(f"{GREY}-{RESET}" * 50)
    
    for r in range(2, max_rails + 1):
        pt = decrypt_rail_fence(ciphertext, r)
        print(f"Rails: {YELLOW}{r:<2}{RESET} | {GREEN}{pt[:60]}{RESET}")