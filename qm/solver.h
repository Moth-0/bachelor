#pragma once

#include <tuple>
#include <algorithm>
#include <numeric>
#include <vector>

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



template <typename ObjectiveFunc>
rvec nelder_mead(rvec p0, ObjectiveFunc objective) {
    size_t n = p0.size();
    ld alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5; // Standard NM coefficients
    ld tolerance = 1e-5; // Stop when the simplex is tiny enough
    int max_iter = 300; 

    // 1. Initialize the Simplex (n+1 vertices)
    // We use std::vector to hold a list of rvecs
    std::vector<rvec> simplex(n + 1, p0);
    std::vector<ld> f_vals(n + 1);
    
    // Create initial spread (proportional step size)
    for (size_t i = 1; i <= n; ++i) {
        qm::ld current_val = p0[i - 1];
        // Step by 10% of the value, or 0.05 if the value is very close to 0
        qm::ld step_size = std::abs(current_val) * 0.1 + 0.05; 
        simplex[i][i - 1] += step_size;
    }
    for (size_t i = 0; i <= n; ++i) {
        f_vals[i] = objective(simplex[i]);
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        // 2. Order the vertices by their objective function values
        std::vector<size_t> indices(n + 1);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { 
            return f_vals[i] < f_vals[j]; 
        });

        size_t best = indices[0];
        size_t worst = indices[n];
        size_t second_worst = indices[n - 1];

        // Check for convergence
        if (std::abs(f_vals[worst] - f_vals[best]) < tolerance) break;

        // 3. Calculate Centroid of all vertices except the worst
        rvec centroid(n); // Initializes to zeros natively
        for (size_t i = 0; i < n; ++i) {
            centroid += simplex[indices[i]];
        }
        centroid /= static_cast<ld>(n);

        // 4. Reflection (Look how clean your overloaded operators make this!)
        rvec reflected = centroid + alpha * (centroid - simplex[worst]);
        ld f_ref = objective(reflected);

        if (f_ref >= f_vals[best] && f_ref < f_vals[second_worst]) {
            simplex[worst] = reflected; 
            f_vals[worst] = f_ref;
            continue;
        }

        // 5. Expansion
        if (f_ref < f_vals[best]) {
            rvec expanded = centroid + gamma * (reflected - centroid);
            ld f_exp = objective(expanded);
            if (f_exp < f_ref) { 
                simplex[worst] = expanded; 
                f_vals[worst] = f_exp; 
            } else { 
                simplex[worst] = reflected; 
                f_vals[worst] = f_ref; 
            }
            continue;
        }

        // 6. Contraction
        rvec contracted = centroid + rho * (simplex[worst] - centroid);
        ld f_con = objective(contracted);
        if (f_con < f_vals[worst]) {
            simplex[worst] = contracted; 
            f_vals[worst] = f_con;
            continue;
        }

        // 7. Shrink (If all else fails, pull all vertices toward the best)
        for (size_t i = 1; i <= n; ++i) {
            size_t idx = indices[i];
            simplex[idx] = simplex[best] + sigma * (simplex[idx] - simplex[best]);
            f_vals[idx] = objective(simplex[idx]);
        }
    }
    
    // Find absolute best to return
    size_t best_idx = 0;
    for(size_t i = 1; i <= n; ++i) {
        if(f_vals[i] < f_vals[best_idx]) best_idx = i;
    }
    return simplex[best_idx];
}