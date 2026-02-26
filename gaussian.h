#pragma once 

#include <cmath>
#include <numbers>
#include <initializer_list>
#include <random>
#include "matrix.h"

const long double pi = std::numbers::pi_v<long double>;

namespace qm {

// Helper function to generate random long doubles
inline long double random_double(long double min, long double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<long double> dis(min, max);
    return dis(gen);
}

struct gaus {
    matrix A; 
    matrix s;
    
    // Default constructor
    gaus() = default;
    ~gaus() = default;
    
    // Constructor to generate a random Gaussian of dimension 'dim'
    gaus(size_t dim, long double min_A = 0.001, long double max_A = 0.1) {
        A = matrix(dim, dim);
        s = matrix(dim, 3); // Assuming spatial shift vectors are 3D

        // 1. Generate random positive-definite A using A = B * B^T trick
        matrix B(dim, dim);
        FOR_MAT(B) {
            min_A = std::log(min_A);
            max_A = std::log(max_A);
            long double rd = random_double(min_A, max_A);
            B(i, j) = std::exp(rd); 
        }
        
        FOR_MAT(B) {
            long double sum = 0;
            for(size_t k = 0; k < dim; k++) {
                sum += B(i, k) * B(j, k);
            }
            // Add shift to diagonal for strict positive-definiteness
            if (i == j) sum += 1e-15; 
            A(i, j) = sum;
        }

        // 2. Generate random shift vectors s
        // For ground state L=0, you often keep these small or zero.
        for(size_t i = 0; i < dim; i++) {
            for(size_t j = 0; j < 3; j++) {
                s(i, j) = random_double(-0.0, 0.0); 
            }
        }
    }

    gaus(const matrix& A_in, const matrix& s_in) : A(A_in), s(s_in) {}

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

inline long double overlap(const gaus& a, const gaus& b) {
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
    
    return std::pow((std::pow(pi, (long double)n) / B.determinant()), 1.5) * std::exp(0.25 * vBv);
}

}