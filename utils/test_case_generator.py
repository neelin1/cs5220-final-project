#!/usr/bin/env python3
import argparse
import random
import os

def generate_dna_test_case(length, lcs_length):
    if (lcs_length > length):
      lcs_length = length
    
    dna_bases = ['A', 'T', 'G', 'C']
    sequence = [random.choice(dna_bases) for _ in range(lcs_length)]
    
    num_diff = length - lcs_length

    positions_str1 = set(random.sample(range(length), lcs_length))
    positions_str2 = set(random.sample(range(length), lcs_length))
    
    str1 = ['X'] * length
    str1_pos = 0

    str2 = ['Y'] * length
    str2_pos = 0
    
    for i in range(length):
        if (i in positions_str1):
          str1[i] = sequence[str1_pos]
          str1_pos+=1
        if (i in positions_str2):
          str2[i] = sequence[str2_pos]
          str2_pos+=1
    
    return ''.join(str1), ''.join(str2), lcs_length

def main():
    parser = argparse.ArgumentParser(description='Generate DNA test cases')
    parser.add_argument('--output', type=str, required=True, help='Output file name')
    parser.add_argument('--length', type=int, required=True, help='Length of strings')
    parser.add_argument('--lcs', type=int, required=True, help='Lcs length')
    
    args = parser.parse_args()
    
    # Generate test case
    str1, str2, lcs_length = generate_dna_test_case(args.length, args.lcs)
    
    # Ensure tests directory exists
    os.makedirs('tests', exist_ok=True)
    
    # Write to file
    with open(f'tests/{args.output}.txt', 'w') as f:
        f.write(f"{str1} {str2} {lcs_length}\n")
    
    print(f"Generated test case:")
    print(f"String 1 (length {len(str1)}): {str1}")
    print(f"String 2 (length {len(str2)}): {str2}")
    print(f"Expected LCS length: {lcs_length}")
    print(f"Written to: tests/{args.output}.txt")

if __name__ == "__main__":
    main()