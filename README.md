# Cipher-Brute-Forcers

This is a repository of Python scripts to attack different known ciphers.
- **Gronsfield**: The user can input a custom alphabet and plaintext words to be found. The brute force will check every key and output the keys that decript the plaintext words and the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
- **Vigenere** WIP: The user can input a custom alphabet and plaintext words to be found. The brute force will check every word in a large list of English words and output the keys that decript the plaintext words and/or the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
