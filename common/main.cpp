#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>
#include "solver.hpp"

#ifdef MPI
#include <mpi.h>
#endif

struct TestCase {
    std::string X;
    std::string Y;
    int expected_lcs_length;
};

void read_test_cases(const std::string &file_path, std::vector<TestCase> &test_cases) {
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open input file: " + file_path);
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string X, Y;
        int expected_lcs_length;

        // Expecting: X Y expected_length
        if (!(iss >> X >> Y >> expected_lcs_length)) {
            throw std::runtime_error("Invalid test case format in file: " + file_path);
        }

        test_cases.push_back({X, Y, expected_lcs_length});
    }

    infile.close();
}

int main(int argc, char **argv) {
    std::string input_file;

    if (argc < 2 || strcmp(argv[1], "--input") != 0 || argc < 3) {
        std::cerr << "Usage: " << argv[0] << " --input <testcase_file>\n";
        return 1;
    }
    input_file = argv[2];

    std::vector<TestCase> test_cases;

    try {
        read_test_cases(input_file, test_cases);
    } catch (const std::runtime_error &e) {
        std::cerr << "Error reading test cases: " << e.what() << std::endl;
        return 1;
    }

#ifdef MPI
    MPI_Init(&argc, &argv);
    std::cout << "MPI is enabled." << std::endl;
#else
    std::cout << "MPI is NOT enabled." << std::endl;
#endif

    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto &test = test_cases[i];
        std::cout << "Running test case " << i + 1 << ":\n"
                  << "  X = " << test.X << " (Len " << test.X.size() << ")\n"
                  << "  Y = " << test.Y << " (Len " << test.Y.size() << ")\n"
                  << "  Z (same as Y) = " << test.Y << " (Len " << test.Y.size() << ")\n"
                  << "  Expected LCS length = " << test.expected_lcs_length << "\n";

        // Since we only have two sequences in the file, let's use Y as Z
        std::string Z = test.Y;

        // Initialize the sequences in the solver
        init(test.X, test.Y, Z);

        auto start_time = std::chrono::high_resolution_clock::now();
        int computed_lcs_length = compute_lcs();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Computed LCS length: " << computed_lcs_length 
                  << " (Expected: " << test.expected_lcs_length << ")\n";
        std::cout << "Execution time: " << elapsed.count() << " seconds\n";

        if (computed_lcs_length == test.expected_lcs_length) {
            std::cout << "Test case " << i + 1 << " passed.\n";
        } else {
            std::cout << "Test case " << i + 1 << " failed.\n";
        }

        std::cout << "-----------------------------------\n";
    }

    free_memory();

#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}
