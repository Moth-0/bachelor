#pragma once

#include<cmath>
#include<vector>
#include"qm/matrix.h"

namespace qm {
// Decomposes a symmetric positive-definite matrix A into L * L^T
// Returns the lower triangular matrix L
matrix cholesky(const matrix& A) {
    size_t n = A.size1();
    matrix L(n, n); // Assumes constructor initializes to 0.0

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            long double sum = 0;
            for (size_t k = 0; k < j; k++) {
                sum += L(i, k) * L(j, k);
            }

            if (i == j) {
                // DIAGONAL ELEMENTS: Calculate value before square root
                long double val_under_sqrt = A(i, i) - sum;
                
                // Only diagonals must be strictly positive
                if (val_under_sqrt <= ZERO_LIMIT) { 
                    std::cerr << "Rejected step: Matrix ill-conditioned." << std::endl;
                    return matrix(0, 0);
                }
                
                L(i, j) = std::sqrt(val_under_sqrt);
            } else {
                // OFF-DIAGONAL ELEMENTS: No square root, use A(i,j), can be negative/zero
                L(i, j) = (1.0 / L(j, j)) * (A(i, j) - sum);
            }
        }
    }
    return L;
}

// Solves the standard eigenvalue problem for a symmetric matrix A
// Returns a std::vector containing the eigenvalues
vector jacobi_eigenvalues(const matrix& M) {
    matrix A = M;
    size_t n = A.size1();
    long double tolerance = ZERO_LIMIT; // Stop when off-diagonals are tiny
    int max_iterations = 1000;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 1. Find the largest off-diagonal element A(p, q)
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
        
        // Check for convergence
        if (max_val < tolerance) break;
        
        // 2. Calculate the rotation angle (theta)
        long double theta = 0.5 * std::atan2(2.0 * A(p, q), A(q, q) - A(p, p));
        long double c = std::cos(theta);
        long double s = std::sin(theta);
        
        // 3. Apply the rotation to the matrix
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
    }
    
    // Extract the diagonal elements (which are now the eigenvalues)
    vector eigenvalues(n);
    for (size_t i = 0; i < n; i++) {
        eigenvalues[i] = A(i, i);
    }
    return eigenvalues;
}

}