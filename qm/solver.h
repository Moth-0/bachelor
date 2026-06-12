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
║       if |H(p,q)| > tolerance:                                                 ║
║         compute rotation angle θ from H(p,p), H(q,q), H(p,q)                   ║
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

namespace qm {

// Jacobi diagonalization of a complex Hermitian matrix M.
// Iteratively zeroes off-diagonal elements via complex Givens rotations.
// Returns real eigenvalues (diagonal of rotated M) and complex eigenvector matrix V.
std::pair<ld, cvec> jacobi_with_eigenvector(cmat& A, size_t nvals = 0) {
    size_t n = A.size1();
    if (nvals == 0) nvals = n;

    rvec w(n);
    cmat V(n, n);
    V.setid(); // Assuming cmat has a setid() method like rmat

    bool changed;
    do {
        changed = false;
        for (size_t p = 0; p < nvals; p++) {
            for (size_t q = p + 1; q < n; q++) {
                // Diagonals of a Hermitian matrix are strictly real
                ld app_real = std::real(A(p, p));
                ld aqq_real = std::real(A(q, q));
                
                std::complex<ld> apq = A(p, q);
                ld apq_mag = std::abs(apq);

                // Skip if the off-diagonal is effectively zero to prevent division by zero
                if (apq_mag == 0.0) continue; 

                // The standard rotation angle uses the magnitude of the complex element
                ld phi = 0.5 * std::atan2(2.0 * apq_mag, aqq_real - app_real);
                ld c = std::cos(phi);
                ld s_real = std::sin(phi);

                // Derive the complex rotation scalar and its conjugate
                std::complex<ld> sc = s_real * (apq / apq_mag);
                std::complex<ld> sc_conj = std::conj(sc);

                // Update the diagonals (which remain purely real)
                ld app1 = c * c * app_real - 2.0 * s_real * c * apq_mag + s_real * s_real * aqq_real;
                ld aqq1 = s_real * s_real * app_real + 2.0 * s_real * c * apq_mag + c * c * aqq_real;

                if (app1 != app_real || aqq1 != aqq_real) {
                    changed = true;
                    
                    // Assign updated reals back as complex types
                    A(p, p) = std::complex<ld>(app1, 0.0);
                    A(q, q) = std::complex<ld>(aqq1, 0.0);
                    A(p, q) = std::complex<ld>(0.0, 0.0);

                    // Upper-left block bounds
                    for (size_t i = 0; i < p; i++) {
                        std::complex<ld> aip = A(i, p);
                        std::complex<ld> aiq = A(i, q);
                        A(i, p) = c * aip - sc_conj * aiq;
                        A(i, q) = c * aiq + sc * aip;
                    }

                    // Middle block bounds (conjugates required for Hermitian symmetry)
                    for (size_t i = p + 1; i < q; i++) {
                        std::complex<ld> api = A(p, i);
                        std::complex<ld> aiq = A(i, q);
                        A(p, i) = c * api - sc * std::conj(aiq);
                        A(i, q) = c * aiq + sc * std::conj(api);
                    }

                    // Lower-right block bounds
                    for (size_t i = q + 1; i < n; i++) {
                        std::complex<ld> api = A(p, i);
                        std::complex<ld> aqi = A(q, i);
                        A(p, i) = c * api - sc * aqi;
                        A(q, i) = c * aqi + sc_conj * api;
                    }

                    // Update the unitary eigenvector matrix
                    for (size_t i = 0; i < n; i++) {
                        std::complex<ld> vip = V(i, p);
                        std::complex<ld> viq = V(i, q);
                        V(i, p) = c * vip - sc_conj * viq;
                        V(i, q) = c * viq + sc * vip;
                    }
                }
            }
        }
    } while (changed);

    // Extract real eigenvalues
    for(size_t i = 0; i < n; i++) {
        w[i] = std::real(A(i, i));
    }
    
    return {w[0], V[0]};
}

// The Main GEVP Solver - with eigenvector
std::pair<ld, cvec> solve_ground_state_with_eigenvector(const cmat& H, const cmat& N, size_t nvals=0) {
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
    auto [E0, c_prime] = jacobi_with_eigenvector(H_prime, nvals);

    // 5. Transform back to original basis: c = L^{-dag} * c'
    cvec c = L_inv_dag * c_prime;

    return {E0, c};
}

// The Main GEVP Solver - energy only (backward compatible)
ld solve_ground_state_energy(const cmat& H, const cmat& N, size_t nvals = 0) {
    return solve_ground_state_with_eigenvector(H, N, nvals).first;
}



template <typename ObjectiveFunc>
rvec nelder_mead(const rvec& p0, const ObjectiveFunc& objective, int max_iter = 500) {
    size_t n = p0.size();
    const ld alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5; // Standard NM coefficients

    // Adaptive tolerance: single-state optimization (n<=20) converges quickly
    ld tolerance = 1e-4;
    const int max_no_improve = 20;

    // 1. Initialize the Simplex (n+1 vertices)
    std::vector<rvec> simplex(n + 1, rvec(n));
    rvec f_vals(n + 1);

    // First vertex is the initial point
    simplex[0] = p0;

    // FAST SIMPLEX INITIALIZATION: No rand() calls, use deterministic pattern
    rvec scales(n);
    for (size_t i = 0; i < n; ++i) {
        ld scale = std::abs(p0[i]) * 0.2;
        if (scale < 0.01) scales[i] = 0.01;
        else scales[i] = scale;
    }

    // Create vertices with alternating +/- perturbations (faster, deterministic)
    for (size_t i = 1; i <= n; ++i) {
        simplex[i] = p0;
        ld perturbation = (i % 2 == 1) ? scales[i - 1] : -0.7 * scales[i - 1];
        simplex[i][i - 1] += perturbation;
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
        ld n_inv = 1.0 / (n);
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

        // 5. Expansion
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

        // 6. Contraction 
        bool contracted_successfully = false;
        if (f_ref < f_vals[worst]) {
            // Outside Contraction (reflection was better than worst)
            rvec contracted = centroid + (reflected - centroid) * rho;
            ld f_con = objective(contracted);
            if (f_con <= f_ref) {
                simplex[worst] = contracted;
                f_vals[worst] = f_con;
            } else {
                simplex[worst] = reflected;
                f_vals[worst] = f_ref;
            }
            contracted_successfully = true;

        } else {
            // Inside Contraction (reflection was worse than worst)
            rvec contracted = centroid - direction * rho; 
            ld f_con = objective(contracted);
            if (f_con < f_vals[worst]) {
                simplex[worst] = contracted;
                f_vals[worst] = f_con;
                contracted_successfully = true;
            }
        }

        if (contracted_successfully) continue;

        // 7. Shrink (Only if both reflection and contraction totally failed)
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

    return simplex[best_idx];
}
}