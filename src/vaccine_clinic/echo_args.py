# echo_args.py
import sys

def main():
    print(f"Command line args: {sys.argv}\n")

    if len(sys.argv) != 6:
        print(f"Five positional args required, only {len(sys.argv) - 1} specified.")
        exit(1)

    for i, arg in enumerate(sys.argv):
            print(f"sys.argv[{i}]: {arg}")

if __name__ == '__main__':
   main()