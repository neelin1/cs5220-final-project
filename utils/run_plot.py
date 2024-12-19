import os
import subprocess
import matplotlib.pyplot as plt

# Define the test files and their corresponding sequence lengths
test_files = [
    "test1000.txt",
    "test3000.txt",
    "test10000.txt",
    "test30000.txt",
    "test100000.txt",
    "test300000.txt",
    "test500000.txt",
    "test1000000.txt",
    "test2000000.txt",
    "test3000000.txt",
]

sequence_lengths = [1000, 3000, 10000, 30000, 100000, 300000, 500000, 1000000, 2000000, 3000000]
execution_times = []

# Path to the executable and test files
executable = "./build/grid_gpu"
test_dir = "tests"
output_file = "execution_time_data.txt"

# Open the output file for writing
with open(output_file, "w") as f:
    # Write the header
    f.write("Sequence Length\tExecution Time (seconds)\n")

    # Run the executable with each input file and measure execution time
    for test_file, seq_len in zip(test_files, sequence_lengths):
        input_path = os.path.join(test_dir, test_file)
        try:
            result = subprocess.run(
                [executable, "--input", input_path],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output to find execution time
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "Execution time:" in line:
                    execution_time = float(line.split()[2])  # Extract the numerical value
                    execution_times.append(execution_time)
                    f.write(f"{seq_len}\t{execution_time}\n")  # Write to the file
                    break
            else:
                raise ValueError(f"Execution time not found in output for {test_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {test_file}: {e.stderr}")
            execution_times.append(None)
            f.write(f"{seq_len}\tError\n")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sequence_lengths, execution_times, marker="o", linestyle="-", label="Execution Time")
plt.title("Execution Time vs. Sequence Length")
plt.xlabel("Sequence Length")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig("execution_time_vs_sequence_length.png")
plt.show()

print(f"Execution time data saved to {output_file}")