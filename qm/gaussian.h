#pragma once 

#include <cmath>
#include <numbers>
#include <random>
#include "matrix.h"

const long double pi = std::numbers::pi_v<long double>;

namespace qm {

// Helper function to generate random long doubles
inline long double random_double(long double min, long double max) {
    // Thread_local ensures thread safety for multithreading
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<long double> dis(min, max);
    return dis(gen);
}

struct gaus {
    matrix A; 
    matrix s;
    
    // Default constructor
    gaus() = default;
    ~gaus() = default;
    
    // Constructor allocates memory and immediately randomizes
    gaus(size_t dim, long double min_A = 0.001, long double max_A = 0.1) {
        A = matrix(dim, dim);
        s = matrix(dim, 3); 
        randomize(min_A, max_A);
    }

    gaus(const matrix& A_in, const matrix& s_in) : A(A_in), s(s_in) {}

    // Method to randomize an existing Gaussian without reallocating memory
    void randomize(long double min_A, long double max_A) {
        size_t dim = A.size1();
        matrix L(dim, dim);
        long double log_min = std::log(std::sqrt(min_A));
        long double log_max = std::log(std::sqrt(max_A));

        // 1. Build the Lower Triangular matrix L
        for(size_t i = 0; i < dim; i++) {
            // The diagonal MUST be strictly positive. 
            L(i, i) = std::exp(random_double(log_min, log_max));
            
            // The off-diagonals can be negative or positive.
            for(size_t j = 0; j < i; j++) {
                    if (j < i) {
                    // Lower triangle: Below the diagonal
                    long double bound = std::min(L(i, i), L(j, j));
                    L(i, j) = random_double(-bound, bound);
                } 
                else if (j > i) {
                    // Upper triangle: Must be zero for a Lower Triangular matrix
                    L(i, j) = 0.0;
                }
            }
        }

        // 2. Ensure positive-definiteness via A = L * L^T
        FOR_MAT(A) {
            long double sum = 0;
            for(size_t k = 0; k <= std::min(i, j); k++) {
                sum += L(i, k) * L(j, k);
            }
            A(i, j) = sum;
        }
        //std::cout << A << "\n" <<std::endl;

        // 3. Generate random shift vectors s
        for(size_t i = 0; i < dim; i++) {
            for(size_t j = 0; j < 3; j++) {
                s(i, j) = 0.0; // Keeping 0.0 for now
            }
        }
    }

    long double operator()(const matrix& r) const {
        long double rAr = 0; 
        long double sr = 0; 
        for(size_t i = 0; i < A.size1(); i++) {
            sr += dot(s[i], r[i]);
            for(size_t j = 0; j < A.size2(); j++) {
                rAr += A(i, j) * dot(r[i], r[j]);
            }
        } 
        return std::exp(-rAr + sr);
    }
    
    void update(const matrix& A_new, const matrix& s_new) {
        A = A_new; 
        s = s_new;
    }

    size_t dim() const { return A.size1(); }
};

long double overlap(const gaus& a, const gaus& b) {
    matrix v = a.s + b.s;
    matrix B = a.A + b.A;
    size_t n = B.size1();

    long double vBv = 0; 
    matrix B_inv = B.inverse();
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < n; j++) {
            vBv += B_inv(i, j) * dot(v[i], v[j]);
        }
    }
    long double front = std::pow((std::pow(pi, (long double)n) / B.determinant()), 1.5);
    
    return front * std::exp(0.25 * vBv);
}

}