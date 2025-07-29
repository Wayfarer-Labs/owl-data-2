import os
import argparse
import time

def count_jpegs_in_dir(root_dir):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg'):
                count += 1
    return count

def watch_jpegs(directory, interval=30):
    print(f"Watching '{directory}' for new JPEGs every {interval} seconds. Press Ctrl+C to stop.")
    prev_count = count_jpegs_in_dir(directory)
    prev_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initial count: {prev_count} JPEGs")
    try:
        while True:
            time.sleep(interval)
            curr_count = count_jpegs_in_dir(directory)
            curr_time = time.time()
            delta_count = curr_count - prev_count
            delta_time = curr_time - prev_time
            rate = delta_count / delta_time if delta_time > 0 else 0
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total: {curr_count} JPEGs | "
                  f"New: {delta_count} | Rate: {rate:.2f} JPEGs/sec over last {delta_time:.1f}s")
            prev_count = curr_count
            prev_time = curr_time
    except KeyboardInterrupt:
        print("\nStopped watching.")

def main():
    parser = argparse.ArgumentParser(description="Periodically count JPEG files in a directory and report generation rate.")
    parser.add_argument("directory", type=str, help="Directory to watch for JPEG files")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds between checks (default: 30)")
    args = parser.parse_args()

    watch_jpegs(args.directory, args.interval)

if __name__ == "__main__":
    main()
