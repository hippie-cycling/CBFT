# CBFT [Cipher Brute Force Toolkit]

A comprehensive console-based (TUI) toolkit designed for cryptanalysis and cipher breaking.

![image](https://github.com/user-attachments/assets/27209098-d9f5-44da-b79d-671123d2d418)

This toolkit provides methods for brute forcing various classical ciphers including:
- Vigenere
- Gromark
- Gronsfeld
- Autoclave/Autokey
- Hill
- Playfair
- XOR
- Modulo addition/subtraction

It also provides some basic tools such as:
- Ioc
- Frequency Analysis
- String Matrix Generator (outputs all possible n x m matrices)

More ciphers and tools to be implemented, **WIP**

## Features

- Customizable alphabet support.
- Known Plaintext list-based attack.
- Bruteforce using dictionary attack (400k English-latin words).
- Bigram frequency fitness scoring
- Playfair cracking via a Memetic Algorithm (evolutionary search) to find the best key.
- Bruteforce (for Hill) using all possible 2x2 or 3x3 matrix keys.
- IoC based analysis. Range defined by the user.
- Saving results to files for further analysis.

To run the program, download the repository and launch main.py.

## License

[MIT](https://choosealicense.com/licenses/mit/)
