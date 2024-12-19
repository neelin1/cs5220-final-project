import matplotlib.pyplot as plt

# Function to plot processes' execution times vs sequence lengths
def plot_execution_times(processes):
    plt.figure(figsize=(10, 6))
    
    # Assign unique colors to each process
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Expand this if needed
    
    # Iterate through processes and plot them
    for i, (process_name, data) in enumerate(processes.items()):
        x, y = zip(*data)
        plt.plot(x, y, 'o-', label=process_name, color=colors[i % len(colors)])
    
    # Graph labels and legend
    plt.title("Execution Time Comparison vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Manually fill in the data points for each process
processes = {
    "Serial DP": [
        (10000, 0.214874),
        (20000, 0.853845),
        (30000, 1.93247),
        (40000, 4.38933),
        (50000, 6.78337),
        (60000, 9.96836),
        (70000, 14.2119),
        (80000, 18.9719),
        (90000, 24.2393),
        (100000, 31.3114)
    ],
    "Serial Wavefront": [
        (10000, 0.44055),
        (20000, 1.79287),
        (30000, 5.24032),
        (40000, 9.77086),
        (50000, 25.1028),
        (60000, 45.0863),
        (70000, 70.5597),
        (80000, 96.6774),
        (90000, 126.56),
        (100000, 186.641)
    ]
}

# Call the function to plot the data
plot_execution_times(processes)
