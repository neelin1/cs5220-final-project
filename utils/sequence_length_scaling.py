import argparse
import subprocess
import tempfile
import os
import random
import string
import matplotlib.pyplot as plt

def generate_random_sequence(length):
    # Generate a random sequence of uppercase letters
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def write_test_case(filepath, X, Y, expected=0):
    # Writes a single test case (X Y expected_lcs_length) to the file
    with open(filepath, 'w') as f:
        f.write(f"{X} {Y} {expected}\n")

def parse_execution_time(output):
    # The output line of interest is something like:
    # "Execution time: 0.000123 seconds"
    # We'll parse that line.
    for line in output.split('\n'):
        if "Execution time:" in line:
            # Line format: "Execution time: X.XXXXXX seconds"
            parts = line.strip().split()
            # parts might look like ["Execution", "time:", "0.000123", "seconds"]
            # The time should be the third element
            try:
                exec_time = float(parts[2])
                return exec_time
            except (IndexError, ValueError):
                pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Sequence length scaling experiment")
    parser.add_argument("--model", required=True, help="Path to the LCS executable (e.g., ./build/grid_gpu)")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum sequence length")
    parser.add_argument("--step", type=int, default=100, help="Step size for sequence lengths")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per length for averaging")
    parser.add_argument("--output_file", default="sequence_length_scaling.png", help="Output image file for the plot")
    args = parser.parse_args()

    lengths = range(args.min_length, args.max_length + 1, args.step)
    avg_times = []

    temp_dir = tempfile.mkdtemp(prefix="lcs_tests_")

    try:
        for length in lengths:
            times = []
            for _ in range(args.trials):
                # Generate random sequences
                X = generate_random_sequence(length)
                Y = generate_random_sequence(length)

                test_file = os.path.join(temp_dir, f"test_{length}.txt")
                # Using expected LCS length as 0 (not essential for timing)
                write_test_case(test_file, X, Y, expected=0)

                # Run the model
                cmd = [args.model, "--input", test_file]
                try:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
                    output = result.stdout
                    exec_time = parse_execution_time(output)
                    if exec_time is not None:
                        times.append(exec_time)
                    else:
                        print(f"Warning: Could not parse execution time for length={length}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running model for length={length}: {e}")
                    times.append(float('nan'))

            # Compute average time over trials
            if times:
                avg_time = sum(times) / len(times)
            else:
                avg_time = float('nan')
            avg_times.append(avg_time)
            print(f"Length={length}, Average execution time={avg_time} seconds")

    finally:
        # Cleanup temporary files
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(list(lengths), avg_times, marker='o')
    plt.title("LCS Execution Time vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.savefig(args.output_file)
    plt.close()
    print(f"Plot saved as {args.output_file}")

if __name__ == "__main__":
    main()
