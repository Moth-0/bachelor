/*
╔════════════════════════════════════════════════════════════════════════════════╗
║          solver.h - GENERALIZED EIGENVALUE SOLVER & OPTIMIZATION               ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   1. Solve the Generalized Eigenvalue Problem (GEVP): H·c = E·N·c              ║
║   2. Optimize basis parameters via Nelder-Mead simplex method                  ║
║                                                                                ║
║   The GEVP is the heart of the Rayleigh-Ritz variational method:               ║
║     E_0 = min_c [<c|H|c> / <c|N|c>]                                            ║
║                                                                                ║
║ GEVP SOLVER ALGORITHM:                                                         ║
║                                                                                ║
║   Input: Hamiltonian matrix H (dimensional: basis_size × basis_size)           ║
║          Overlap matrix N (metric; must be positive definite)                  ║
║                                                                                ║
║   Step 1: Cholesky decomposition N = L L†                                      ║
║           Ensures N is invertible; fails if N singular                         ║
║                                                                                ║
║   Step 2: Invert L (lower-triangular specialization)                           ║
║           L⁻¹ is also lower-triangular (faster than full inverse)              ║
║                                                                                ║
║   Step 3: Transform to standard Hermitian problem                              ║
║           H' = L⁻¹ H (L†)⁻¹                                                    ║
║           Solves: H' c' = E c'  (standard eigenvalue equation)                 ║
║                                                                                ║
║   Step 4: Jacobi rotation diagonalization                                      ║
║           Rotates matrix to extract eigenvalues from diagonal                  ║
║                                                                                ║
║   Output: E_0 = lowest eigenvalue on diagonal of H' (ground state)             ║
║                                                                                ║
║                                                                                ║
║ JACOBI ROTATION METHOD:                                                        ║
║                                                                                ║
║   Iterative diagonalization via sweeps of Givens rotations:                    ║
║                                                                                ║
║   for each sweep:                                                              ║
║     for each off-diagonal pair (p,q):                                          ║
║       if |H[p,q]| > tolerance:                                                 ║
║         compute rotation angle θ from H[p,p], H[q,q], H[p,q]                   ║
║         apply rotation: H ← R_θ† H R_θ                                         ║
║                                                                                ║
║   Converges when all off-diagonal elements < ZERO_LIMIT                        ║
║                                                                                ║
║                                                                                ║
║ NELDER-MEAD OPTIMIZATION:                                                      ║
║                                                                                ║
║   Derivative-free simplex method for basis parameter tuning:                   ║
║                                                                                ║
║   Simplex: (N+1)-vertex polytope in N-dimensional parameter space              ║
║            each vertex = one candidate basis parameter set                     ║
║                                                                                ║
║   per iteration:                                                               ║
║     1. Sort vertices by objective function value (energy)                      ║
║     2. Try reflection: reflect worst vertex through centroid                   ║
║     3. If better than best → try expansion to go further                       ║
║     4. If worse than second-worst → try contraction                            ║
║     5. If all else fails → shrink all toward best vertex                       ║
║                                                                                ║
║   Why Nelder-Mead?                                                             ║
║     • No gradient computation required (neural net-like robustness)            ║
║     • Handles non-smooth objective (energy function)                           ║
║     • Empirically fast for low-dimensional problems (10-20 variables)          ║
║     • Tolerant to scattered/noisy evaluations                                  ║
║                                                                                ║
║   Scaling notes:                                                               ║
║     • Gaussian basis (N-1)×(N-1) correlation matrix → ~(N-1)² parameters       ║
║     • For 3-body (PN+pion): 2×2 matrix A + 2×3 matrix s → 4+6 = 10 variables   ║
║     • Each evaluation: build matrices + solve GEVP → O(basis_size³)            ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include <tuple>
#include <algorithm>
#include <numeric>
#include <vector>

#include "matrix.h"

using namespace qm;

// Helper function to find lowest eigenvalue and eigenvector using Jacobi diagonalization
// Returns: pair of (eigenvalue, eigenvector) where eigenvector is normalized
std::pair<ld, cvec> jacobi_with_eigenvector(cmat A, int max_sweeps = 50) {
    size_t n = A.size1();
    ld tolerance = ZERO_LIMIT;

    // Initialize eigenvector matrix as identity (tracks rotations)
    cmat V = eye<cld>(n);

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

                    // Apply Givens rotation to A
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

                    // Apply same rotation to eigenvector matrix V
                    for (size_t i = 0; i < n; ++i) {
                        cld vip = V(i, p);
                        cld viq = V(i, q);
                        V(i, p) = cos_t * vip - sin_t * std::conj(phase) * viq;
                        V(i, q) = sin_t * phase * vip + cos_t * viq;
                    }
                }
            }
        }
        if (max_off_diag < tolerance) break; // Converged!
    }

    // Find the lowest eigenvalue and its index
    ld lowest_E = std::real(A(0, 0));
    size_t min_idx = 0;
    for (size_t i = 1; i < n; ++i) {
        if (std::real(A(i, i)) < lowest_E) {
            lowest_E = std::real(A(i, i));
            min_idx = i;
        }
    }

    // Extract the corresponding eigenvector from V
    cvec eigvec = V[min_idx];

    // Normalize the eigenvector
    cld norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        norm += std::conj(eigvec[i]) * eigvec[i];
    }
    norm = std::sqrt(norm);

    if (std::abs(norm) > ZERO_LIMIT) {
        for (size_t i = 0; i < n; ++i) {
            eigvec[i] /= norm;
        }
    }

    return {lowest_E, eigvec};
}

// A helper function to find the lowest eigenvalue of a standard Hermitian matrix
// using the Jacobi rotation method.
ld jacobi_lowest_eigenvalue(cmat A, int max_sweeps = 50) {
    return jacobi_with_eigenvector(A, max_sweeps).first;
}


// The Main GEVP Solver - with eigenvector
std::pair<ld, cvec> solve_ground_state_with_eigenvector(const cmat& H, const cmat& N) {
    // 1. Cholesky Decomposition: N = L * L^dag
    cmat L = N.cholesky();

    // Check if the basis is mathematically unstable (linearly dependent)
    if (L.size1() == 0) {
        return {999999.0, cvec()};
    }

    // 2. Invert L
    cmat L_inv = L.inverse_lower();
    if (L_inv.size1() == 0) {
        return {999999.0, cvec()};
    }

    // 3. Create the standard Hermitian matrix: H' = L^{-1} * H * L^{-dag}
    cmat L_inv_dag = L_inv.adjoint();
    cmat H_prime = L_inv * H * L_inv_dag;

    // 4. Diagonalize to find the ground state with eigenvector!
    auto [E0, c_prime] = jacobi_with_eigenvector(H_prime);

    // 5. Transform back to original basis: c = L^{-dag} * c'
    cvec c = L_inv_dag * c_prime;

    return {E0, c};
}

// The Main GEVP Solver - energy only (backward compatible)
ld solve_ground_state_energy(const cmat& H, const cmat& N) {
    return solve_ground_state_with_eigenvector(H, N).first;
}



template <typename ObjectiveFunc>
rvec nelder_mead(rvec p0, ObjectiveFunc objective, int max_iter = 100) {
    size_t n = p0.size();
    const ld alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5; // Standard NM coefficients
    const ld tolerance = 1e-4; // Relaxed convergence: was 1e-5
    const int max_no_improve = 10; // Stop if no improvement for this many iterations

    // 1. Initialize the Simplex (n+1 vertices) as vector of vectors
    std::vector<rvec> simplex(n + 1, rvec(n));
    rvec f_vals(n + 1);

    // First vertex is the initial point
    simplex[0] = p0;

    // Create remaining vertices with proportional perturbations (step by 10% or 0.05)
    for (size_t i = 1; i <= n; ++i) {
        simplex[i] = p0;
        ld step_size = std::abs(p0[i - 1]) * 0.1 + 0.05;
        simplex[i][i - 1] += step_size;
    }

    // Evaluate all vertices
    for (size_t i = 0; i <= n; ++i) {
        f_vals[i] = objective(simplex[i]);
    }

    int no_improve_count = 0;
    ld prev_best = 1e10;

    for (int iter = 0; iter < max_iter; ++iter) {
        // 2. Order the vertices by their objective function values
        std::vector<size_t> indices(n + 1);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            return f_vals[i] < f_vals[j];
        });

        const size_t best = indices[0];
        const size_t worst = indices[n];
        const size_t second_worst = indices[n - 1];

        // Check for convergence: range is small OR no improvement
        ld range = std::abs(f_vals[worst] - f_vals[best]);
        if (range < tolerance) break;

        if (std::abs(f_vals[best] - prev_best) < tolerance * 0.1) {
            no_improve_count++;
            if (no_improve_count > max_no_improve) break;
        } else {
            no_improve_count = 0;
        }
        prev_best = f_vals[best];

        // 3. Calculate Centroid of all vertices except the worst
        rvec centroid(n);
        for (size_t i = 0; i < n; ++i) {
            centroid += simplex[indices[i]];
        }
        ld n_inv = 1.0L / static_cast<ld>(n);
        centroid *= n_inv;

        // Precompute direction from worst to centroid (optimize reflections/contractions)
        rvec direction = centroid - simplex[worst];

        // 4. Reflection: reflected = centroid + alpha * (centroid - worst)
        rvec reflected = centroid + direction * alpha;
        ld f_ref = objective(reflected);

        if (f_ref >= f_vals[best] && f_ref < f_vals[second_worst]) {
            simplex[worst] = reflected;
            f_vals[worst] = f_ref;
            continue;
        }

        // 5. Expansion: expanded = centroid + gamma * (reflected - centroid)
        if (f_ref < f_vals[best]) {
            rvec expanded = centroid + (reflected - centroid) * gamma;
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

        // 6. Contraction: contracted = centroid + rho * (worst - centroid)
        rvec contracted = centroid - direction * rho;
        ld f_con = objective(contracted);
        if (f_con < f_vals[worst]) {
            simplex[worst] = contracted;
            f_vals[worst] = f_con;
            continue;
        }

        // 7. Shrink (If all else fails, pull all vertices toward the best)
        for (size_t i = 1; i <= n; ++i) {
            size_t idx = indices[i];
            simplex[idx] = simplex[best] + (simplex[idx] - simplex[best]) * sigma;
            f_vals[idx] = objective(simplex[idx]);
        }
    }

    // Find and return absolute best vertex
    ld best_f = f_vals[0];
    size_t best_idx = 0;
    for (size_t i = 1; i <= n; ++i) {
        if (f_vals[i] < best_f) {
            best_f = f_vals[i];
            best_idx = i;
        }
    }

    //size_t absolute_best_idx = indices[0]; 
    return simplex[best_idx];
}