import re
from collections import Counter

def analyze_frequency(text):
    """Performs frequency analysis and compares to English."""

    # English letter frequencies (approximate)
    english_freq = {
        'e': 12.70, 't': 9.05, 'a': 8.16, 'o': 7.51, 'i': 6.97, 'n': 6.75,
        's': 6.30, 'h': 6.09, 'r': 5.98, 'd': 4.25, 'l': 4.02, 'u': 2.76,
        'c': 2.78, 'm': 2.40, 'f': 2.23, 'y': 2.02, 'g': 1.97, 'p': 1.93,
        'w': 1.75, 'b': 1.49, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
        'q': 0.10, 'z': 0.07
    }

    text = text.lower()  # Case-insensitive
    letter_counts = Counter(c for c in text if 'a' <= c <= 'z')
    total_letters = sum(letter_counts.values())

    if total_letters == 0:  # Handle empty strings
        return 0

    text_freq = {
        letter: (count / total_letters) * 100 for letter, count in letter_counts.items()
    }

    # Calculate frequency difference (a simple metric)
    difference = 0
    for letter, freq in english_freq.items():
        difference += abs(text_freq.get(letter, 0) - freq) #handles if a letter doesn't appear in the text

    return difference



def process_file(filename):
    """Opens a file, extracts decrypted text, and analyzes frequency."""

    try:
        with open(filename, 'r') as f:
            file_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    entries = file_content.split("--------------------------------------------------")
    for entry in entries:
        if entry.strip():  # Skip empty entries
            match = re.search(r"Decrypted: (.+)", entry)
            if match:
                decrypted_text = match.group(1)
                difference = analyze_frequency(decrypted_text)
                print(f"Decrypted: {decrypted_text}")
                print(f"Frequency Difference: {difference:.2f}")

                # You can adjust this threshold for "closeness" to English
                if difference < 50:  # Example threshold
                    print("Possibly English-like text.")
                print("-" * 50)


# Example usage:
filename = "test.txt"  # Replace with your file name
process_file(filename)