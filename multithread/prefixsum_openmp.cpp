#include "../common/solver.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <unordered_map>

struct MatchPoint {
    int i, j;
    int len;
    MatchPoint(int i = 0, int j = 0, int len = 0) : i(i), j(j), len(len) {}
};

static std::string X_global;
static std::string Y_global;
static std::vector<MatchPoint> match_points;
static std::vector<std::vector<int>> max_lengths; 

void init(const std::string &X_input, const std::string &Y_input) {
    X_global = X_input;
    Y_global = Y_input;
    match_points.clear();
    
    int m = X_input.size();
    int n = Y_input.size();
    
    max_lengths.resize(m + 1, std::vector<int>(n + 1, 0));
}

void find_match_points() {
    int m = X_global.size();
    int n = Y_global.size();
    
    std::vector<std::vector<MatchPoint>> thread_match_points(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (X_global[i] == Y_global[j]) {
                    thread_match_points[thread_id].emplace_back(i, j, 0);
                }
            }
        }
    }
    
    // combing match points from all threads
    for (const auto& thread_points : thread_match_points) {
        match_points.insert(match_points.end(), thread_points.begin(), thread_points.end());
    }
    
    // sort match points by X position, then Y position
    std::sort(match_points.begin(), match_points.end(),
              [](const MatchPoint& a, const MatchPoint& b) {
                  return a.i < b.i || (a.i == b.i && a.j < b.j);
              });
}

void compute_max_lengths() {
    for (auto& mp : match_points) {
        int i = mp.i;
        int j = mp.j;
        
        int max_prev = 0;
        
        #pragma omp parallel for reduction(max:max_prev)
        for (int pi = 0; pi < i; pi++) {
            for (int pj = 0; pj < j; pj++) {
                max_prev = std::max(max_prev, max_lengths[pi][pj]);
            }
        }
        
        mp.len = max_prev + 1;
        
        max_lengths[i][j] = mp.len;
    }
}

int compute_lcs() {
    if (X_global.empty() || Y_global.empty()) {
        return 0;
    }
    
    find_match_points();
    
    if (match_points.empty()) {
        return 0;
    }
    compute_max_lengths();

    int max_length = 0;
    #pragma omp parallel for reduction(max:max_length)
    for (size_t i = 0; i < match_points.size(); i++) {
        max_length = std::max(max_length, match_points[i].len);
    }
    
    return max_length;
}

void free_memory() {
    match_points.clear();
    match_points.shrink_to_fit();
    max_lengths.clear();
    max_lengths.shrink_to_fit();
}