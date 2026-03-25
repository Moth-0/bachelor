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
    rmat s; // Shift matrix (N-1 x 3)

    Gaussian() = default;
    Gaussian(const rmat& A_in, const rmat& s_in) : A(A_in), s(s_in) {}

    ~Gaussian() = default;

    // r is now an (N-1) x 3 matrix representing the 3D Jacobi vectors
    ld evaluate(const rmat& r) const {
        ld rAr = 0.0;
        ld sr = 0.0;
        
        // Loop over x (0), y (1), z (2) columns
        for (size_t c = 0; c < 3; ++c) {
            rAr += dot_no_conj(r[c], A * r[c]);
            sr  += dot_no_conj(s[c], r[c]);
        }
        
        return std::exp(-rAr + sr);
    }

    void set(const rmat& A_in, const rmat& s_in) { SELF.A = A_in; SELF.s = s_in; }
};


struct SpatialWavefunction {
    rmat A;
    rmat s;
    int parity_sign; // +1 for symmetric (pn), -1 for antisymmetric (pn pi)

    // FIXED: s_ is now an rmat
    SpatialWavefunction(const rmat& A_, const rmat& s_, int parity) 
        : A(A_), s(s_), parity_sign(parity) {}

    SpatialWavefunction() = default;
    ~SpatialWavefunction() = default;

    // ... [Your randomize function stays exactly as you wrote it!] ...

    ld evaluate(const rmat& r) const {
        ld rAr = 0.0;
        ld sr = 0.0;
        
        // Loop over x (0), y (1), z (2) columns
        for (size_t c = 0; c < 3; ++c) {
            rAr += dot_no_conj(r[c], A * r[c]);
            sr  += dot_no_conj(s[c], r[c]);
        }
        
        ld g_plus  = std::exp(-rAr + sr);
        ld g_minus = std::exp(-rAr - sr);
        
        return g_plus + parity_sign * g_minus;
    }
};

// 1. Primitive Overlap: <A1, s1 | A2, s2>
inline ld gaussian_overlap(const rmat& A1, const rmat& s1, const rmat& A2, const rmat& s2) {
    rmat B = A1 + A2;
    rmat v = s1 + s2; // matrix.h natively adds the (N-1)x3 matrices!
    size_t dim = B.size1();

    ld detB = B.determinant();
    if (std::abs(detB) < 1e-25) return 0.0; // Singular, reject

    ld pi_to_d = std::pow(M_PI, dim);
    ld prefactor = std::pow(pi_to_d / detB, 1.5);

    rmat B_inv = B.inverse();
    ld vBv = 0.0;

    // Evaluate v^T B^-1 v for x, y, and z columns
    for (size_t c = 0; c < 3; ++c) {
        vBv += dot_no_conj(v[c], B_inv * v[c]);
    }

    return prefactor * std::exp(0.25 * vBv);
}

// 2. Full Spatial Overlap incorporating Parity
inline ld spactial_overlap(const SpatialWavefunction& g1, const SpatialWavefunction& g2) {
    // A SpatialWavefunction is |+s> + p |-s>. 
    // Expanding <g1 | g2> gives 4 terms:
    
    // <+s1 | +s2>
    ld term1 = gaussian_overlap(g1.A, g1.s, g2.A, g2.s);
    
    // p2 * <+s1 | -s2>
    ld term2 = g2.parity_sign * gaussian_overlap(g1.A, g1.s, g2.A, -1.0L * g2.s);
    
    // p1 * <-s1 | +s2>
    ld term3 = g1.parity_sign * gaussian_overlap(g1.A, -1.0L * g1.s, g2.A, g2.s);
    
    // p1 * p2 * <-s1 | -s2>
    ld term4 = (g1.parity_sign * g2.parity_sign) * gaussian_overlap(g1.A, -1.0L * g1.s, g2.A, -1.0L * g2.s);

    return term1 + term2 + term3 + term4;
}