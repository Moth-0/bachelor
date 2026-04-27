#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"

namespace qm {

struct SvmResult {
    ld energy;
    ld energy_excited;
    rvec coefficients;
    ld charge_radius;
    ld avg_kinetic_energy;
    ld prob_bare;
    ld prob_dressed;
    rvec convergence_history;
};

// Template: Evaluate energy: build H,N and solve GEVP for E0 only
// Uses overlap checking logic from the original evaluate_energy_sum
template <typename BasisStateType>
ld evaluate_basis_energy(const std::vector<BasisStateType>& basis, ld b, ld S,
                        const std::vector<bool>& relativistic, ld ho_k = 0.0, bool debug = false) {
    auto [H, N] = build_matrices(basis, b, S, relativistic, ho_k);

    // CRITICAL: Multi-level overlap check prevents basis collapse
    // Uses the stricter tolerance (0.99) from evaluate_energy_sum for stability
    ld tol = 0.99;
    for (size_t i = 0; i < N.size1(); ++i) {
        for (size_t j = i + 1; j < N.size2(); ++j) {
            ld overlap = std::abs(N(i, j)) / std::sqrt(std::abs(N(i, i)) * std::abs(N(j, j)));

            // Level 2: Pure spatial cross-channel check (expensive but necessary!)
            if (overlap <= tol && basis[i].psi.A.size1() == basis[j].psi.A.size1()) {
                BasisStateType temp = basis[j];
                temp.type = basis[i].type; // Force channel match to strip spin/isospin orthogonality
                ld spatial_overlap = std::abs(calc_N_elem(basis[i], temp));
                overlap = spatial_overlap / std::sqrt(std::abs(N(i, i)) * std::abs(N(j, j)));
            }

            if (overlap > tol) {
                if (debug) {
                    std::cerr << "  [REJECT GEVP] Near-duplicate detected (Overlap: " << overlap << " > " << tol << ").\n";
                }
                return 999999.0;
            }
        }
    }

    cmat L = N.cholesky();

    // Total Failure Check: The matrix is explicitly not Positive-Definite
    if (L.size1() == 0) {
        if (debug) {
            std::cerr << "  [REJECT GEVP] Overlap matrix N is not positive-definite.\n";
        }
        return 999999.0;
    }

    // Near-Singularity Check (Dynamic Safety net)
    for (size_t i = 0; i < L.size1(); ++i) {
        ld num = std::abs(L(i, i));
        ld thresh = 1e-3 * std::sqrt(std::abs(N(i, i)));
        if (num < thresh) {
            if (debug) {
                std::cerr << "  [REJECT GEVP] Near-linear dependence at index " << num << " < " << thresh << ".\n";
            }
            return 999999.0;
        }
    }

    // Return E0 only (NO E1)
    return solve_ground_state_energy(H, N);
}

template <typename BasisStateType>
SvmResult evaluate_observables(const std::vector<BasisStateType>& basis, ld b, ld S,
                               const std::vector<bool>& relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    cld detN = N.determinant();
    if (std::abs(detN) < ZERO_LIMIT) {
        return {999999.0, 999999.0, {}, 99999.0, 0.0, 0.0, 0.0, {}};
    }

    auto [E0, eigvec] = solve_ground_state_with_eigenvector(H, N);
    ld E1 = solve_kth_state_energy(H, N, 1);

    rvec coeff(eigvec.size());
    for (size_t i = 0; i < coeff.size(); ++i) {
        coeff[i] = abs(eigvec[i]);
    }

    // Build observable matrices
    cmat R2 = build_r2_matrix(basis);
    cmat T_mat = build_T_matrix(basis, relativistic);

    cld r2_expectation  = 0.0;
    cld t_expectation   = 0.0;

    for (size_t i = 0; i < basis.size(); ++i) {
        for (size_t j = 0; j < basis.size(); ++j) {
            r2_expectation += std::conj(eigvec[i]) * R2(i, j) * eigvec[j];
            t_expectation += std::conj(eigvec[i]) * T_mat(i, j) * eigvec[j];
        }
    }

    ld r2_point = std::real(r2_expectation);
    ld r_p_sq = 0.8414 * 0.8414;
    ld r_n_sq = -0.1161;

    ld r2_total_charge = r2_point + r_p_sq + r_n_sq;
    ld charge_radius = (r2_total_charge > 0.0) ? std::sqrt(r2_total_charge) : 0.0;

    return {E0, E1, coeff, charge_radius, std::real(t_expectation), 0.0, 0.0, {}};
}

// Check if a candidate state has positive kinetic energy (rejects unphysical states)
template <typename BasisStateType>
bool has_positive_kinetic_energy(const BasisStateType& state, const std::vector<bool>& relativistic) {
    // Build appropriate relativistic flags for this state's Jacobian dimension
    size_t expected_size = state.jac.masses.size() - 1;
    std::vector<bool> rel_flags(expected_size, false);

    for (size_t i = 0; i < expected_size && i < relativistic.size(); ++i) {
        rel_flags[i] = relativistic[i];
    }

    // Skip kinetic energy check for bare nucleon (0 internal coordinates)
    if (state.jac.masses.size() == 1) {
        return true;  // Bare nucleon is always physical
    }

    ld kinetic = total_kinetic_energy(state.psi, state.psi, state.jac, rel_flags);
    ld overlap = spactial_overlap(state.psi, state.psi);
    if (overlap < ZERO_LIMIT) return false;

    ld kinetic_expectation = kinetic / overlap;
    return kinetic_expectation > -1e-6;
}

// Physics constraint checker - validates Gaussian state is physical
bool is_physical_gaussian(const SpatialWavefunction& psi, bool debug = false) {
    const ld min_width = 1.0 / (500.0 * 500.0); 
    const ld max_width = 1.0 / (0.01 * 0.01); 

    // Check diagonal widths and shifts
    for (size_t i = 0; i < psi.A.size1(); ++i) {
        ld width = psi.A(i, i);

        if (width < min_width) {
            if (debug) std::cerr << "  [REJECT] Width[" << i << "]=" << width << " < min=" << min_width << "\n";
            return false;
        }
        if (width > max_width) {
            if (debug) std::cerr << "  [REJECT] Width[" << i << "]=" << width << " > max=" << max_width << "\n";
            return false;
        }

        // Calculate physical shift distance in fm: r = s / (2A)
        ld shift_sq = 0.0;
        for (size_t col = 0; col < 3; ++col) {
            shift_sq += psi.s(i, col) * psi.s(i, col);
        }
        ld absolute_shift_magnitude = std::sqrt(shift_sq);
        ld physical_shift_fm = absolute_shift_magnitude / (2.0 * width);

        // THE PHYSICAL LIMIT in fm
        ld limit = 10.0;
        if (physical_shift_fm > limit) {
            if (debug) std::cerr << "  [REJECT] Shift too large: " << physical_shift_fm << " fm > " << limit <<" fm\n";
            return false; 
        }

        // if (psi.parity_sign == -1 && i > 0 && total_shift < ZERO_LIMIT) {
        //     if (debug) std::cerr << "  [REJECT] Odd-parity shift too small (collapsed state).\n";
        //     return false;
        // }
    }

    // Positive definiteness check
    ld det = psi.A.determinant();
    if (det <= ZERO_LIMIT) {
        if (debug) std::cerr << "  [REJECT] det(A)=" << det << " <= " << ZERO_LIMIT << "\n";
        return false;
    }

    return true;
}

// Debug helper: Print final basis state parameters with constraint validation
void print_basis_details(const std::vector<BasisState>& basis, const rvec& coeff) {
    std::cerr << "\n === FINAL BASIS STATE (ALL CYCLES COMPLETE) === \n";
    std::cerr << "Basis Size: " << basis.size() << "\n";
    std::cerr << std::string(120, '=') << "\n";

    for (size_t k = 0; k < basis.size(); ++k) {
        std::cerr << "\nState " << k << " (Type " << (int)basis[k].type  
                  << ") c_i: " << coeff[k] << ":\n";

        for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
            ld width = basis[k].psi.A(i, i);
            std::cerr << "  Gaussian[" << i << "] width r≈ "
                      << (1.0/std::sqrt(width)) << " fm | ";

            ld shift_sq = 0.0;
            for (size_t col = 0; col < 3; ++col) {
                shift_sq += basis[k].psi.s(i, col) * basis[k].psi.s(i, col);
            }
            ld total_shift = std::sqrt(shift_sq);

            std::cerr << "Total Shift: " << total_shift << " fm | \n";
        }
    }
    std::cerr << std::string(120, '=') << "\n";
}

// Helper to safely apply random noise to a Gaussian state
// If lock_A00=true, skip perturbing the first width (for pion states to keep PN block locked)
SpatialWavefunction perturb_wavefunction(const SpatialWavefunction& original, ld noise_scale, bool lock_A00 = false) {
    SpatialWavefunction perturbed = original;

    // 1. Perturb the Widths (Multiplicative noise to stay positive)
    for (size_t i = 0; i < perturbed.A.size1(); ++i) {
        // LOCK: Skip A[0,0] if this is a pion state (lock_A00=true)
        if (lock_A00 && i == 0) continue;

        // Generate random factor between (1 - noise) and (1 + noise)
        ld factor = 1.0 + ((static_cast<ld>(rand()) / RAND_MAX) * 2.0 - 1.0) * noise_scale;
        perturbed.A(i, i) *= std::abs(factor); // Ensure strictly positive
    }

    // 2. Perturb the Shifts (Additive noise in fm)
    for (size_t i = 1; i < perturbed.s.size1(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ld shift_bump = ((static_cast<ld>(rand()) / RAND_MAX) * 2.0 - 1.0) * noise_scale;
            perturbed.s(i, j) += shift_bump;
        }
    }
    return perturbed;
}

// Refine a single basis state using fast Matrix Caching and Micro-Darts
// TEMPLATE VERSION: Works with any BasisState type
template <typename BasisStateType>
bool refine_basis_state(std::vector<BasisStateType>& basis, size_t k, ld noise_scale,
                        ld b_form, ld S, const std::vector<bool>& relativistic)
{
    // CACHING: Build the full matrix once
    auto [H_full, N_full] = build_matrices(basis, b_form, S, relativistic);
    ld original_E = solve_ground_state_energy(H_full, N_full);

    int cands = 200; // Micro-dart cloud size
    ld best_E = original_E;
    BasisStateType best_state = basis[k];

    #pragma omp parallel
    {
        ld local_best_E = original_E;
        BasisStateType local_best_state = basis[k];

        // Thread-local matrix copies
        cmat H_test = H_full;
        cmat N_test = N_full;

        #pragma omp for
        for (int c = 0; c < cands; ++c) {
            BasisStateType test_candidate = basis[k];

            // Skip perturbation for bare nucleon (0-coordinate system)
            if (basis[k].jac.masses.size() > 1) {
                test_candidate.psi = perturb_wavefunction(basis[k].psi, noise_scale, false);

                // Check if the random kick made it unphysical
                if (!is_physical_gaussian(test_candidate.psi)) continue;

                // Reject candidates with negative kinetic energy
                if (!has_positive_kinetic_energy(test_candidate, relativistic)) continue;

                // FAST EVALUATION: Only update the k-th row and column
                for (size_t i = 0; i < basis.size(); ++i) {
                    if (i == k) continue; // Skip self

                    cld h_ik = calc_H_elem(basis[i], test_candidate, b_form, S, relativistic);
                    cld n_ik = calc_N_elem(basis[i], test_candidate);

                    H_test(i, k) = h_ik;
                    N_test(i, k) = n_ik;
                    H_test(k, i) = std::conj(h_ik);
                    N_test(k, i) = std::conj(n_ik);
                }

                H_test(k, k) = calc_H_elem(test_candidate, test_candidate, b_form, S, relativistic);
                N_test(k, k) = calc_N_elem(test_candidate, test_candidate);

                // Solve updated matrix
                ld E_estimate = solve_ground_state_energy(H_test, N_test);

                if (E_estimate < local_best_E - 1e-6) {
                    local_best_E = E_estimate;
                    local_best_state = test_candidate;
                }
            }
        }

        #pragma omp critical
        {
            if (local_best_E < best_E) {
                best_E = local_best_E;
                best_state = local_best_state;
            }
        }
    }

    // Apply the best micro-dart
    if (best_E < original_E) {
        basis[k] = best_state;
        return true;
    }
    return false;
}

// Performs one full cycle of competitive basis growth using ROBUST Matrix Caching
// TEMPLATE VERSION: Works with any BasisState type
// OPTIMIZES E0 ONLY (no E0+E1)
template <typename BasisStateType>
inline void competitive_search(std::vector<BasisStateType>& basis,
                               const std::vector<BasisStateType>& channel_templates,
                               int num_candidates, ld b_range, ld b_form, ld S,
                               const std::vector<bool>& relativistic, ld ho_k = 0.0)
{
    // 1. Get the current baseline energy (E0 only)
    ld E_core = evaluate_basis_energy(basis, b_form, S, relativistic, ho_k);
    size_t K = basis.size();
    auto [H_core, N_core] = build_matrices(basis, b_form, S, relativistic, ho_k);

    for (size_t t = 1; t < channel_templates.size(); ++t) {
        BasisStateType best_candidate = channel_templates[t];

        ld best_E = E_core;
        bool found_valid_dart = false;

        #pragma omp parallel
        {
            BasisStateType local_best_candidate = channel_templates[t];
            ld local_best_E = E_core;

            cmat H_test = zeros<cld>(K + 1, K + 1);
            cmat N_test = zeros<cld>(K + 1, K + 1);

            for (size_t i = 0; i < K; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    H_test(i, j) = H_core(i, j);
                    N_test(i, j) = N_core(i, j);
                }
            }

            int reject_unphysical = 0, reject_energy = 0, reject_kinetic = 0, reject_singular = 0, reject_improve = 0;

            #pragma omp for
            for (int c = 0; c < num_candidates; ++c) {
                BasisStateType test_candidate = channel_templates[t];
                Gaussian g;
                g.randomize(test_candidate.jac, b_range, b_form);
                test_candidate.psi.set_from_gaussian(g);
                if (!is_physical_gaussian(test_candidate.psi)) {
                    reject_unphysical++;
                    continue;
                }

                // Calculate the dart's isolated diagonal elements
                cld H_xx = calc_H_elem(test_candidate, test_candidate, b_form, S, relativistic);
                cld N_xx = calc_N_elem(test_candidate, test_candidate);

                // FAST REJECTION GATE: unphysical Gaussians
                ld isolated_energy = (std::real(N_xx) > 1e-15) ? std::real(H_xx) / std::real(N_xx) : 999999.0;
                if (isolated_energy > 5000.0 || std::real(N_xx) < 1e-10) {
                    reject_energy++;
                    continue;
                }

                // Reject candidates with negative kinetic energy
                if (!has_positive_kinetic_energy(test_candidate, relativistic)) {
                    reject_kinetic++;
                    continue;
                }

                for (size_t i = 0; i < K; ++i) {
                    cld h_ik = calc_H_elem(basis[i], test_candidate, b_form, S, relativistic);
                    cld n_ik = calc_N_elem(basis[i], test_candidate);

                    H_test(i, K) = h_ik;
                    N_test(i, K) = n_ik;
                    H_test(K, i) = std::conj(h_ik);
                    N_test(K, i) = std::conj(n_ik);
                }

                H_test(K, K) = calc_H_elem(test_candidate, test_candidate, b_form, S, relativistic);
                N_test(K, K) = calc_N_elem(test_candidate, test_candidate);

                // SAFEGUARD: Reject linearly dependent darts
                if (std::abs(N_test.determinant()) < ZERO_LIMIT) {
                    reject_singular++;
                    continue;
                }

                // Compute E0 ONLY (no E1)
                ld E0 = solve_ground_state_energy(H_test, N_test);
                ld E_estimate = (E0 < 999999.0) ? E0 : 999999.0;

                // Strict improvement gate
                if (E_estimate < local_best_E - 1e-6) {
                    local_best_E = E_estimate;
                    local_best_candidate = test_candidate;
                } else {
                    reject_improve++;
                }
            }

            #pragma omp critical
            {
                if (local_best_E < best_E) {
                    best_E = local_best_E;
                    best_candidate = local_best_candidate;
                    found_valid_dart = true;
                }
                // std::cerr << "  Channel " << t << ": Unphysical=" << reject_unphysical
                //           << " BadEnergy=" << reject_energy << " NegKE=" << reject_kinetic
                //           << " Singular=" << reject_singular << " NoImprove=" << reject_improve << "\n";
            }
        }

        // 3. LOCK AND VERIFY
        if (found_valid_dart) {
            basis.push_back(best_candidate);

            // CRITICAL: Update the core matrices
            K = basis.size();
            std::tie(H_core, N_core) = build_matrices(basis, b_form, S, relativistic, ho_k);
            E_core = best_E;

            std::cout << "\rAdded State " << basis.size() << " (Ch " << t << ") -> E0 = "
                      << std::fixed << std::setprecision(5) << best_E << " MeV    " << std::flush;
        }
    }
    std::cout << "\n";
}

// Pack ALL basis states into a single vector for global Nelder-Mead optimization
rvec pack_all_basis_states(const std::vector<BasisState>& basis, bool optimize_shift) {
    rvec p;
    for (const auto& state : basis) {
        // All states (PN and pion) can optimize shifts in Phase 2
        rvec state_params = pack_wavefunction(state.psi, optimize_shift);
        // Manually concatenate (rvec is a custom type, no insert method)
        for (size_t i = 0; i < state_params.size(); ++i) {
            p.push_back(state_params[i]);
        }
    }
    return p;
}

// Unpack a single vector back into all basis states
void unpack_all_basis_states(std::vector<BasisState>& basis, const rvec& p, bool optimize_shift) {
    size_t idx = 0;
    for (size_t k = 0; k < basis.size(); ++k) {
        auto& state = basis[k];

        // Calculate size of parameters for this state
        size_t state_size;
        size_t dim = state.psi.A.size1();
        if (optimize_shift) {
            // L lower triangle (not upper!)
            state_size = (dim * (dim + 1)) / 2;  // Lower triangle = dim*(dim+1)/2
            // All states (PN and pion) can optimize shifts: dim rows × 3 coords
            state_size += dim * 3;
        } else {
            state_size = (dim * (dim + 1)) / 2;  // Just L
        }

        // Bounds check
        if (idx + state_size > p.size()) {
            std::cerr << "[ERROR] unpack_all_basis_states: tried to unpack state " << k
                      << " size " << state_size << " at idx " << idx << " but p.size()=" << p.size() << "\n";
            return;
        }

        // Extract the portion of p for this state
        rvec state_params;
        for (size_t i = 0; i < state_size; ++i) {
            state_params.push_back(p[idx + i]);
        }

        unpack_wavefunction(state.psi, state_params, optimize_shift);
        idx += state_size;
    }
}


// Optimize basis parameters using Single-State Sweeping
// TEMPLATE VERSION: Works with any BasisState type
// OPTIMIZES E0 ONLY (no E0+E1)
template <typename BasisStateType>
void sweep_optimize_basis(std::vector<BasisStateType>& basis, ld b, ld S, const std::vector<bool>& relativistic,
                          rvec& convergence_energies, int max_sweeps = 100, ld threshold = 1e-4, ld ho_k = 0.0) {
    int nm_max_iter = 150;
    int patience = 2;

    ld previous_E = evaluate_basis_energy(basis, b, S, relativistic, ho_k);
    int no_improve_count = 0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        ld sweep_start_E = previous_E;

        // SWEEP: Optimize each state ONE AT A TIME
        for (size_t k = 0; k < basis.size(); ++k) {
            // Optimize shifts for dressed states, not for bare nucleon
            bool opt_shift = (basis[k].jac.masses.size() > 1);

            rvec p0 = pack_wavefunction(basis[k].psi, opt_shift);

            // Local objective: Minimizes E0 only
            auto local_objective = [&](const qm::rvec& p_test) -> qm::ld {
                SpatialWavefunction test_psi = basis[k].psi;
                unpack_wavefunction(test_psi, p_test, opt_shift);

                if (!is_physical_gaussian(test_psi)) {
                    return 999999.0;
                }

                BasisStateType temp_state = basis[k];
                temp_state.psi = test_psi;
                if (!has_positive_kinetic_energy(temp_state, relativistic)) {
                    return 999999.0;
                }

                // Temporarily replace state k's psi
                SpatialWavefunction orig_psi = basis[k].psi;
                basis[k].psi = test_psi;
                ld E = evaluate_basis_energy(basis, b, S, relativistic, ho_k);
                basis[k].psi = orig_psi;  // Restore

                return E;
            };

            // Run Nelder-Mead on just this state's parameters
            rvec p_best = nelder_mead(p0, local_objective, nm_max_iter);

            // Apply the optimized parameters
            unpack_wavefunction(basis[k].psi, p_best, opt_shift);
        }

        // Single energy evaluation after the full sweep
        ld current_E = evaluate_basis_energy(basis, b, S, relativistic, ho_k, true);
        convergence_energies.push_back(current_E);

        ld improvement = sweep_start_E - current_E;
        if (improvement < threshold) {
            no_improve_count++;
        } else {
            no_improve_count = 0;
        }

        std::cout << "\rSweep " << sweep << " (Basis=" << basis.size() << "): E0 = " << current_E
                  << " MeV  (ΔE = " << improvement << ")     " << std::flush;

        if (no_improve_count >= patience) {
            std::cout << "\n  - Converged: improvement < " << threshold << " MeV for " << patience << " sweeps\n";
            break;
        }

        previous_E = current_E;
    }
    std::cout << "\n";
}

} // Namespace qm