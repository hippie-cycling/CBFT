# Cipher-Brute-Forcers

This is a repository of Python scripts to attack different known ciphers. The different attacks are launched using bruteforcer.py

- **Gronsfield**: The user can input a custom alphabet and plaintext words to be found. The brute force will check every key and output the keys that decript the plaintext words and the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
- **Gromark**: Input the ciphertext, the key for the keyed alphabet and the size of the primer key and it will brute force until one of the input plaintext is found and or the IoC of the plaintext closely matched english (in case the cipher is layered with transposition). Don't go crazy with the key size as it might literally tak forever to compute.
- **Vigenere** WIP: The user can input a custom alphabet and plaintext words to be found. The brute force will check every word in a large list of English words and output the keys that decript the plaintext words and/or the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.
