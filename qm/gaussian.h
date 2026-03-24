#pragma once

#include <cmath>
#include <vector>
#include <random>
#include "matrix.h"
#include "jacobi.h"

using namespace qm; 

inline long double random_ld(long double lo, long double hi) {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<long double> dist(lo, hi);
    return dist(rng);
}

struct Gaussian {
    rmat A; // Correlation matrix (positive definite)
    rvec s; // Shift vector

    Gaussian() = default;
    Gaussian(const rmat& A_in, const rvec& s_in) : A(A_in), s(s_in) {}

    ~Gaussian() = default;

    ld evaluate(const rvec& r) const {
        // r^T * A * r
        ld rAr = dot_no_conj(r, A * r); 
        // s^T * r
        ld sr = dot_no_conj(s, r);      
        
        return std::exp(-rAr + sr);
    }

    void set(const rmat& A_in, const rvec& s_in) {SELF.A = A_in; SELF.s = s_in; }
};


struct SpatialWavefunction {
    rmat A;
    rvec s;
    int parity_sign; // +1 for symmetric (pn), -1 for antisymmetric (pn pi)

    SpatialWavefunction(const rmat& A_, const rvec& s_, int parity) 
        : A(A_), s(s_), parity_sign(parity) {}

    SpatialWavefunction() = default;
    ~SpatialWavefunction() = default;

    void randomize(const Jacobian& jac, ld b_range) {
        size_t N = jac.N;           // Number of physical particles
        size_t dim = N - 1;         // Internal Jacobi dimensions (ignoring CM)
        
        rmat A_new = zeros<ld>(dim, dim); 

        // 1. Loop over pairs (i < j), not matrix dimensions!
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                
                // Get full N-dimensional physical selection vectors
                rvec w_i = jac.transform_w(i);
                rvec w_j = jac.transform_w(j);
                rvec w_ij_full = w_i - w_j;
                
                // 2. Truncate the Center of Mass coordinate (drop the last element)
                rvec w_ij(dim);
                for (size_t k = 0; k < dim; ++k) {
                    w_ij[k] = w_ij_full[k];
                }
                
                // The outer product is now safely (N-1) x (N-1)
                rmat outer = outer_no_conj(w_ij, w_ij);
                
                // 3. Stochastically pick b_ij
                ld u = random_ld(1e-10, 1.0); 
                ld b_ij = -std::log(u) * b_range;
                
                A_new += outer / (b_ij * b_ij);
            }
        }
        
        SELF.A = A_new;

        // 4. Randomize the shift vector 
        ld range = 0.5 * b_range; 
        for(size_t i = 0; i < SELF.s.size(); ++i) { 
            SELF.s[i] = random_ld(-range, range);
        }
    }

    ld evaluate(const rvec& r) const {
        ld rAr = dot_no_conj(r, A * r);
        ld sr  = dot_no_conj(s, r);
        
        ld g_plus  = std::exp(-rAr + sr);
        ld g_minus = std::exp(-rAr - sr);
        
        return g_plus + parity_sign * g_minus;
    }
};

inline ld overlap(Spacial g1, )