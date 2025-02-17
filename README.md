# CBFT [Cipher Brute Force Tool]

**WIP**

A simple console based script launcher to attack different known ciphers.<br/><br/> The Gronsfeld and Gromark scripts generate the primers (keys) in batches for better parallelization. I can achieve up to 120k primers/s with a 12th Gen Intel(R) Core(TM) i7-1280P - 16GB of RAM.

- **Gronsfeld**: The user can input a custom alphabet and plaintext words to be found. The brute force will check every key and output the keys that decript the plaintext words and or the keys that generate an IoC close to English (0.063 <= ioc <= 0.070) (in case the cipher is layered with transposition).
- **Gromark**: Input the ciphertext, the key for the keyed alphabet and the size of the primer key and it will brute force until one of the input plaintext is found and or the IoC of the plaintext closely matched English (0.063 <= ioc <= 0.070) (in case the cipher is layered with transposition). **Note that the AC Gromark cihper uses a primer size of 5 digits.**
- **Vigenere** WIP: The user can input a custom alphabet and plaintext words to be found. The brute force will check every word in a large list of English words and output the keys that decript the plaintext words and/or the keys that generate an IoC close to English (0.06 <= ioc <= 0.07) for further analysis.

**Don't go crazy with the key size as it might literally take forever to compute.**
**I recommend using key sizes between 1 and 9 digits [9 - 999999999]. Note your primers/s speed and then extrapolate to larger key sizes.**
**Note that the AC Gromark cihper uses a primer size of 5 digits.**
