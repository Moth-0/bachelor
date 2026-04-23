/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                oscillator.h - HARMONIC OSCILLATOR SANDBOX                      ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/
#pragma once

#include <vector>
#include <algorithm>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h" // USING YOUR EXISTING CODE!
#include "qm/solver.h"
#include "qm/jacobi.h"

namespace qm {

constexpr ld HO_K = 10.0; // Spring constant (MeV / fm^2)

// Custom Full-Diagonalization Jacobi Routine
inline std::vector<ld> get_all_eigenvalues(cmat A) {
    size_t n = A.size1();
    ld tolerance = 1e-12;
    for (int sweep = 0; sweep < 100; ++sweep) {
        ld max_off_diag = 0.0;
        for (size_t p = 0; p < n - 1; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                ld off_diag_mag = std::abs(A(p, q));
                if (off_diag_mag > max_off_diag) max_off_diag = off_diag_mag;
                if (off_diag_mag > tolerance) {
                    ld app = std::real(A(p, p));
                    ld aqq = std::real(A(q, q));
                    cld apq = A(p, q);
                    ld theta = 0.5 * std::atan2(2.0 * off_diag_mag, aqq - app);
                    ld cos_t = std::cos(theta);
                    ld sin_t = std::sin(theta);
                    cld phase = std::conj(apq) / off_diag_mag;
                    for (size_t i = 0; i < n; ++i) {
                        cld ip = A(i, p); cld iq = A(i, q);
                        A(i, p) = cos_t * ip - sin_t * phase * iq;
                        A(i, q) = sin_t * std::conj(phase) * ip + cos_t * iq;
                    }
                    for (size_t i = 0; i < n; ++i) {
                        cld pi = A(p, i); cld qi = A(q, i);
                        A(p, i) = cos_t * pi - sin_t * std::conj(phase) * qi;
                        A(q, i) = sin_t * phase * pi + cos_t * qi;
                    }
                }
            }
        }
        if (max_off_diag < tolerance) break;
    }
    std::vector<ld> evals;
    for(size_t i = 0; i < n; ++i) evals.push_back(std::real(A(i, i)));
    std::sort(evals.begin(), evals.end());
    return evals;
}

// Build Matrix Elements using YOUR operators!
inline std::tuple<cmat, cmat> build_ho_matrices(const std::vector<SpatialWavefunction>& basis, const Jacobian& jac) {
    size_t size = basis.size();
    cmat H = zeros<cld>(size, size);
    cmat N = zeros<cld>(size, size);

    // Dummy charges: We want r^2 of the first coordinate (the distance between the two particles)
    rvec dummy_charges = {1.0, 0.0}; 

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            
            // 1. USE YOUR KINETIC ENERGY OPERATOR
            ld T_val = total_kinetic_energy(basis[i], basis[j], jac, {false}, Integrator::GAUSS_LEGENDRE);
            
            // 2. USE YOUR R^2 OPERATOR (V = 0.5 * k * r^2)
            ld r2_val = charge_radius_operator(basis[i], basis[j], jac, dummy_charges);
            ld V_val = 0.5 * HO_K * (4.0 * r2_val);

            // 3. USE YOUR SPATIAL OVERLAP
            ld N_val = spactial_overlap(basis[i], basis[j]);
            
            H(i, j) = cld(T_val + V_val, 0.0);
            N(i, j) = cld(N_val, 0.0);
            if (i != j) {
                H(j, i) = std::conj(H(i, j));
                N(j, i) = std::conj(N(i, j));
            }
        }
    }
    return {H, N};
}

// Safe GEVP Evaluator
inline ld evaluate_ho_sum(const std::vector<SpatialWavefunction>& basis, const Jacobian& jac, std::vector<ld>& out_energies, bool debug = false) {
    auto [H, N] = build_ho_matrices(basis, jac);

    // CRITICAL: Multi-level overlap check prevents basis collapse
    // Level 1: Standard overlap (catches identical channels)
    // Level 2: Spatial overlap (catches hidden clones with different spin/isospin)
    ld tol = 0.95;
    for (size_t i = 0; i < N.size1(); ++i) {
        for (size_t j = i + 1; j < N.size2(); ++j) {
            ld overlap = std::abs(N(i, j)) / std::sqrt(std::abs(N(i, i)) * std::abs(N(j, j)));

            if (overlap > tol) {
                if (debug) {
                    //std::cerr << "  [REJECT GEVP] Near-duplicate detected (Overlap: " << overlap << " > " << tol << ").\n";
                }
                return 999999.0;
            }
        }
    }

    // // 2. Tikhonov Regularization (Relative Scaling)
    // for (size_t i = 0; i < N.size1(); ++i) {
    //     N(i, i) *= cld(1.0 + 1e-6, 0.0);
    // }

    cmat L = N.cholesky();

    // 3. Total Failure Check: The matrix is explicitly not Positive-Definite
    if (L.size1() == 0) {
        if (debug) {
            std::cerr << "  [REJECT GEVP] Overlap matrix N is not positive-definite.\n";
        }
        return 999999.0;
    }

    // 4. Near-Singularity Check (Dynamic Safety net)
    for (size_t i = 0; i < L.size1(); ++i) {
        ld num = std::abs(L(i, i));
        ld thresh = (1-tol) * std::sqrt(std::abs(N(i, i)));
        if (num < thresh) {
            if (debug) {
                std::cerr << "  [REJECT GEVP] Near-linear dependence at index " << num << " < " << thresh << ".\n";
            }
            return 999999.0;
        }
    }

    cmat L_inv = L.inverse_lower();
    cmat H_prime = L_inv * H * L_inv.adjoint();
    out_energies = get_all_eigenvalues(H_prime);

    if (out_energies.size() < 2) return 999999.0;
    return out_energies[0] + out_energies[1]; 
}

} // namespace qm