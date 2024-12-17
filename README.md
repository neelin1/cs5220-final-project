# Final Project - DNA Sequence Similarity with Longest Common Subsequence

Make a method with:
```
make naive_serial
```
Then run with, where you put as input a series of test cases:
```
./build/naive_serial --input ./tests/testcase1.txt
```

Test cases must be of the following form, where you have at least one test case. string1 string2 true_longest_subseqence_length. Test case cases can only consist of the characters CGATXYZ
```
AGGGCTA AGGCTAA 6
ATA AT 2
GATTACA ATTACA 6
AAAAA CCCCC 0
A AAAAA 1
```


## Project Writeup

While 3 of our team members were working at a longevity-focused health startup, we came across research on aging biomarkers that led us down a rabbit hole of DNA sequencing and its applications. DNA sequence analysis allows geneticists to compare DNA sequences between individuals, looking for patterns that have been preserved across generations. The Longest Common Subsequence (LCS) problem finds the longest sequence of characters that appear in order (though not necessarily consecutively) across multiple strings. LCS algorithms play a crucial role in this, efficiently identifying the longest subsequence of genetic markers that appear in the same order across multiple DNA samples (while still allowing for the discontinuities and variations caused by evolution). This similarity matching helps scientists piece together evolutionary relatedness. The computational challenge of solving the LCS problem, which becomes NP-hard with multiple sequences, caught our attention.

Cornell Professor Guidi directed us toward Vahidi et al.'s implementation of parallel LCS using a grid-based approach in Chapel (https://ieeexplore.ieee.org/document/10363472). Their work demonstrates efficient computation of LCS between two sequences using a two-dimensional grid method. Building on their foundation and using concepts from parallel computing, we want to implement their methods with both MPI and CUDA, as efficiently as possible. We also want to potentially extend their algorithm to handle three sequences simultaneously through a three-dimensional grid implementation, allowing testing for genetic similarity across 3 individuals at once. 

For our data, we will use actual DNA sequences from sources like NCBI datasets to evaluate the practical applicability of each implementation. We also have the actual sequenced genome of one of our group members, as provided by 23andMe, which we will be comparing to other human genomes and ape genomes. Though for experimentation, we will just use simulated strings to start out with (so we can test accuracy reliably). We will start with 1 string, and mix distinct character sets into 2 different versions of it (all Xs in one, and Zs in the other), so we know the true longest subsequence. This helps assess whether the parallel speedup achieved matches theoretical expectations when working with biological datasets, providing insight into the real-world utility of each approach.

When assessing system features, we consider both input characteristics and performance metrics. For input characteristics, we systematically vary sequence lengths ($10^2$, $10^4$, and $10^6$), while measuring runtime and memory usage. We'll extend testing from 2-sequence to 3-sequence LCS, requiring higher-dimensional DP tables. We'll also evaluate how varying sequence similarity levels (90%, 50%, 10%) affect performance and potential optimizations.

Key performance metrics include execution time, speedup, and memory usage. We'll measure how runtime scales with sequence length and similarity, while comparing parallel implementations (OpenMP, MPI, CUDA) against sequential baselines. Both strong scaling (fixed problem size, increasing processors) and weak scaling (proportionally increasing both) will be evaluated. Memory usage tracking is especially critical for larger sequences and multi-sequence comparisons. We'll use profiling tools like Nsight (CUDA) and Intel VTune (CPUs) to assess resource utilization and communication overhead.

Our implementation strategy starts with a sequential baseline, then explores three parallel approaches: OpenMP (shared memory, focusing on inner loop parallelization), MPI (distributed memory, testing row/diagonal distribution across nodes), and CUDA (GPU acceleration, optimizing memory access patterns). We'll extend to three-sequence LCS where feasible, evaluating how runtime and memory usage scale with additional dimensions. Final evaluation will focus on sequence length scaling, implementation-specific bottlenecks, and detailed scalability testing across architectures – varying thread count (OpenMP), nodes (MPI), and block/thread configurations (CUDA).