#!/usr/bin/env bash

# List of executables (all shown in the image)
EXECUTABLES=(
  "./build/grid_mpi"
  "./build/grid_openmp"
  "./build/prefixsum_openmp"
  "./build/wavefront_mpi"
  "./build/wavefront_openmp"
  "./build/wavefrontblocking_openmp"
)

# Directory containing test files
TEST_DIR="./tests"

# Output file for results
RESULTS_FILE="results_1.txt"

# Max allowed execution time in seconds
TIMEOUT_SECONDS=50

# Clear or create the results file
echo "Executable | Input | Execution Time" > "$RESULTS_FILE"

# Loop through each executable
for EXEC in "${EXECUTABLES[@]}"; do
    # Loop from 10000 to 100000 in increments of 10000
    for (( i=10000; i<=100000; i+=10000 )); do
        INPUT_FILE="${TEST_DIR}/test${i}.txt"

        # Check if input file exists to avoid errors
        if [[ ! -f "$INPUT_FILE" ]]; then
            echo "Input file $INPUT_FILE does not exist. Skipping..."
            continue
        fi

        # Run the executable with a timeout
        OUTPUT=$(timeout "${TIMEOUT_SECONDS}s" $EXEC --input "$INPUT_FILE" 2>&1)
        EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 124 ]]; then
            # Command timed out
            EXEC_TIME_LINE="skip"
        else
            # Extract the execution time line
            EXEC_TIME_LINE=$(echo "$OUTPUT" | grep "Execution time:")

            # If no execution time line found, note that as well
            if [[ -z "$EXEC_TIME_LINE" ]]; then
                EXEC_TIME_LINE="No execution time line found."
            fi
        fi

        # Append to results file
        echo "$(basename $EXEC) | test${i}.txt | $EXEC_TIME_LINE" >> "$RESULTS_FILE"
    done
done
