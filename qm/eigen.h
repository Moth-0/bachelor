#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include "matrix.h"

namespace qm {

// Struct to hold both eigenvalues and the corresponding eigenvector matrix
struct EigenResult {
    vector evals;
    matrix evecs; // Columns are the eigenvectors
};

// Decomposes a symmetric positive-definite matrix A into L * L^T
// Returns the lower triangular matrix L
inline matrix cholesky(const matrix& A) {
    size_t n = A.size1();
    matrix L(n, n); 

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            long double sum = 0;
            for (size_t k = 0; k < j; k++) {
                sum += L(i, k) * L(j, k);
            }

            if (i == j) {
                long double val_under_sqrt = A(i, i) - sum;
                if (val_under_sqrt <= ZERO_LIMIT) { 
                    return matrix(0, 0); // Ill-conditioned
                }
                L(i, j) = std::sqrt(val_under_sqrt);
            } else {
                L(i, j) = (1.0 / L(j, j)) * (A(i, j) - sum);
            }
        }
    }
    return L;
}

// Solves the standard eigenvalue problem for a symmetric matrix A
// Returns both eigenvalues and eigenvectors
inline EigenResult jacobi_eigensystem(const matrix& M) {
    matrix A = M;
    size_t n = A.size1();
    long double tolerance = ZERO_LIMIT; 
    int max_iterations = 20 * n * n;
    
    // Initialize V as the Identity Matrix
    matrix V(n, n);
    for (size_t i = 0; i < n; ++i) V(i, i) = 1.0;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        long double max_val = 0.0;
        size_t p = 0, q = 1;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                if (std::abs(A(i, j)) > max_val) {
                    max_val = std::abs(A(i, j));
                    p = i;
                    q = j;
                }
            }
        }
        
        if (max_val < tolerance) break;
        
        long double theta = 0.5 * std::atan2(2.0 * A(p, q), A(q, q) - A(p, p));
        long double c = std::cos(theta);
        long double s = std::sin(theta);
        
        // Apply rotation to A
        long double App = c * c * A(p, p) - 2.0 * s * c * A(p, q) + s * s * A(q, q);
        long double Aqq = s * s * A(p, p) + 2.0 * s * c * A(p, q) + c * c * A(q, q);
        
        A(p, q) = 0.0;
        A(q, p) = 0.0;
        
        for (size_t i = 0; i < n; i++) {
            if (i != p && i != q) {
                long double Aip = A(i, p);
                long double Aiq = A(i, q);
                A(i, p) = c * Aip - s * Aiq;
                A(p, i) = A(i, p);
                A(i, q) = s * Aip + c * Aiq;
                A(q, i) = A(i, q);
            }
        }
        A(p, p) = App;
        A(q, q) = Aqq;

        // Apply EXACT SAME rotation to V (to track eigenvectors)
        for (size_t i = 0; i < n; i++) {
            long double Vip = V(i, p);
            long double Viq = V(i, q);
            V(i, p) = c * Vip - s * Viq;
            V(i, q) = s * Vip + c * Viq;
        }
    }
    
    EigenResult result;
    result.evals = vector(n);
    result.evecs = V;
    for (size_t i = 0; i < n; i++) {
        result.evals[i] = A(i, i);
    }
    return result;
}

// ---------------------------------------------------------
// Solves Generalized Eigenvalue Problem: H*c = E*N*c
// Returns SORTED eigenvalues and eigenvectors (Ground State is Index 0)
// ---------------------------------------------------------
inline EigenResult solve_generalized_eigensystem(const matrix& H, const matrix& N) {
    matrix L = cholesky(N);
    if (L.size1() == 0) return EigenResult{vector(), matrix()}; 

    matrix L_inv = L.inverse_lower();
    if (L_inv.size1() == 0) return EigenResult{vector(), matrix()};

    // Transform H: H' = L^{-1} * H * (L^{-1})^T
    matrix H_prime = L_inv * H * L_inv.transpose(); // Note: Changed to .transpose() assuming it's standard

    // Solve standard problem for H'
    EigenResult sys = jacobi_eigensystem(H_prime);

    // Transform eigenvectors back to original basis: c = (L^T)^{-1} * v
    sys.evecs = L_inv.transpose() * sys.evecs;

    // Sort eigenvalues and corresponding eigenvectors from lowest to highest
    size_t n = sys.evals.size();
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = 0; j < n - i - 1; j++) {
            if (sys.evals[j] > sys.evals[j + 1]) {
                // Swap energies
                std::swap(sys.evals[j], sys.evals[j + 1]);
                // Swap corresponding eigenvector columns
                for (size_t k = 0; k < n; k++) {
                    long double temp = sys.evecs(k, j);
                    sys.evecs(k, j) = sys.evecs(k, j + 1);
                    sys.evecs(k, j + 1) = temp;
                }
            }
        }
    }

    return sys;
}

} // namespace qm