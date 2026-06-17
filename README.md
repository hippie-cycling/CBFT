# Cipher Brute Force Toolkit (CBFT)

A comprehensive, modular command-line toolkit for classical cryptanalysis, cipher breaking, and automated brute-forcing. Designed for CTF (Capture The Flag) challenges, cryptography enthusiasts, and logic puzzle solvers. 

![CBFT Interface Concept](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

CBFT is divided into four main categories, accessible via an interactive retro-style terminal launcher.

### 1. Substitution Ciphers
Tools to encrypt, decrypt, and brute-force classical substitution ciphers:
* **Caesar Cipher**
* **Vigenère Cipher**
* **Playfair Cipher** (Includes Memetic Algorithm optimization)
* **Affine Cipher**
* **Hill Cipher**
* **Autoclave Cipher**
* **Gronsfeld Cipher**
* **Gromark Cipher**
* **Polybius Square**
* **Bifid Cipher**
* **XOR Cipher**
* **Modular Add/Sub**

### 2. Transposition Ciphers
Tools for ciphers that rearrange the plaintext without changing the alphabet:
* **Columnar Transposition**
* **Scytale Cipher**
* **Rail Fence Cipher**

### 3. Advanced Cryptanalysis & Heuristics
The core strength of the CBFT lies in its algorithmic approaches to cracking ciphers without knowing the key:
* **Simulated Annealing (Substitution):** Cracks completely random Simple Substitution ciphers from scratch using thermal cooling and N-gram scoring.
* **Hill Climbing (Transposition):** Unscrambles complex columnar transpositions purely by optimizing bigram fitness.
* **Vigenère Auto-Solver:** Fully automated frequency analysis and key deduction for Vigenère ciphers.
* **Word Pattern Isomorphism:** Instantly solves space-preserving substitution ciphers by mapping word shapes (e.g., `1-2-2-1-3-4`) against a dictionary (e.g., `ATTACK`).
* **Automated Crib Dragging:** "Drags" a suspected plaintext word across a ciphertext to reveal underlying key snippets.
* **Kasiski Examination:** Analyzes polyalphabetic ciphers to find repeating substrings and deduce the key length.
* **Friedman Test:** Mathematically estimates polyalphabetic key lengths using the Index of Coincidence (IoC).
* **Matrix Permutation Solver:** Generates and solves matrix-based permutations.

### 4. Utilities
* **Calculate IoC (Index of Coincidence):** Determines whether a cipher is monoalphabetic or polyalphabetic.
* **Frequency Analysis:** Plots character deviations against standard English text.
* **Text Formatter:** Cleans and formats ciphertexts.
* **Matrix Generator:** Dumps all possible `n x m` permutations of a ciphertext string.

---

## 🛠️ Installation

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/hippie-cycling/CBFT.git](https://github.com/hippie-cycling/CBFT.git)