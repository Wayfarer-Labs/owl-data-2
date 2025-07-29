import os
import argparse

def count_jpegs_in_dir(root_dir):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg'):
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Count the number of JPEG files in a directory recursively.")
    parser.add_argument("directory", type=str, help="Directory to search for JPEG files")
    args = parser.parse_args()

    num_jpegs = count_jpegs_in_dir(args.directory)
    print(f"Total number of JPEG files in '{args.directory}': {num_jpegs}")

if __name__ == "__main__":
    main()
