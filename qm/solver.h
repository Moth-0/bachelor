#pragma once

#include <tuple>
#include <algorithm>

#include "matrix.h"

using namespace qm;

// A helper function to find the lowest eigenvalue of a standard Hermitian matrix
// using the Jacobi rotation method.
ld jacobi_lowest_eigenvalue(cmat A, int max_sweeps = 50) {
    size_t n = A.size1();
    ld tolerance = ZERO_LIMIT;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        ld max_off_diag = 0.0;
        
        // Standard Jacobi sweep over the upper triangle
        for (size_t p = 0; p < n - 1; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                ld off_diag_mag = std::abs(A(p, q));
                if (off_diag_mag > max_off_diag) max_off_diag = off_diag_mag;

                if (off_diag_mag > tolerance) {
                    // Calculate the rotation angles
                    ld app = std::real(A(p, p));
                    ld aqq = std::real(A(q, q));
                    cld apq = A(p, q);
                    
                    ld theta = 0.5 * std::atan2(2.0 * off_diag_mag, aqq - app);
                    ld cos_t = std::cos(theta);
                    ld sin_t = std::sin(theta);
                    cld phase = std::conj(apq) / off_diag_mag; // Phase to handle complex elements

                    // Apply the Givens rotation to the matrix
                    for (size_t i = 0; i < n; ++i) {
                        cld ip = A(i, p);
                        cld iq = A(i, q);
                        A(i, p) = cos_t * ip - sin_t * phase * iq;
                        A(i, q) = sin_t * std::conj(phase) * ip + cos_t * iq;
                    }
                    for (size_t i = 0; i < n; ++i) {
                        cld pi = A(p, i);
                        cld qi = A(q, i);
                        A(p, i) = cos_t * pi - sin_t * std::conj(phase) * qi;
                        A(q, i) = sin_t * phase * pi + cos_t * qi;
                    }
                }
            }
        }
        if (max_off_diag < tolerance) break; // Converged!
    }

    // Find and return the lowest eigenvalue on the diagonal
    ld lowest_E = std::real(A(0, 0));
    for (size_t i = 1; i < n; ++i) {
        if (std::real(A(i, i)) < lowest_E) {
            lowest_E = std::real(A(i, i));
        }
    }
    return lowest_E;
}


// The Main GEVP Solver
ld solve_ground_state_energy(const cmat& H, const cmat& N) {
    // 1. Cholesky Decomposition: N = L * L^dag
    cmat L = N.cholesky();
    
    // Check if the basis is mathematically unstable (linearly dependent)
    if (L.size1() == 0) {
        return 999999.0; // Return a huge energy penalty so the SVM rejects this state!
    }

    // 2. Invert L
    cmat L_inv = L.inverse_lower();
    if (L_inv.size1() == 0) {
        return 999999.0; 
    }

    // 3. Create the standard Hermitian matrix: H' = L^{-1} * H * L^{-dag}
    cmat L_inv_dag = L_inv.adjoint();
    cmat H_prime = L_inv * H * L_inv_dag;

    // 4. Diagonalize to find the ground state!
    return jacobi_lowest_eigenvalue(H_prime);
}