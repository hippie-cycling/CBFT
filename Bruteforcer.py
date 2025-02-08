import Gromark
import Gronsfeld

def main():
    while True:
        print("\nSelect a cipher to run the brute force:")
        print("1. Gromark")
        print("2. Gronsfeld")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            Gromark.run()
        elif choice == '2':
            Gronsfeld.run()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
