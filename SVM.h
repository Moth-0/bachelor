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
#include "deuterium.h" 

namespace qm {

struct SvmResult {
    ld energy;
    rvec coefficients;
    ld charge_radius;
    ld avg_kinetic_energy;
    ld prob_bare;
    ld prob_dressed;
    rvec convergence_history; 
};

// Evaluate energy: build H,N and solve GEVP
ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, const std::vector<bool>& relativistic, bool debug = false) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    
    cmat L = N.cholesky();
    
    // 1. Total Failure Check: The matrix is explicitly not Positive-Definite
    if (L.size1() == 0) { 
        if (debug) {
            std::cerr << "  [REJECT GEVP] Overlap matrix N is not positive-definite (Cholesky failed). "
                      << "Basis size: " << basis.size() << ".\n"
                      << "  -> Cause: Basis functions are linearly dependent (numerical collapse).\n";
        }
        return 999999.0; 
    }
    
    // 2. Near-Singularity Check: The diagonal of L approaches 0
    for (size_t i = 0; i < L.size1(); ++i) {
        if (std::abs(L(i, i)) < 0.05) { 
            if (debug) {
                std::cerr << "  [REJECT GEVP] Near-linear dependence detected at basis index " << i << ".\n"
                          << "  -> L(" << i << "," << i << ") = " << std::abs(L(i, i)) << " < threshold.\n";
            }
            return 999999.0;
        }
    }

    return solve_ground_state_energy(H, N);
}

SvmResult evaluate_observables(const std::vector<BasisState>& basis, ld b, ld S,
                               const std::vector<bool>& relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    cld detN = N.determinant();
    if (std::abs(detN) < ZERO_LIMIT) { 
        return {999999.0, {}, 99999.0, 0.0, {}}; 
    }

    auto [E0, eigvec] = solve_ground_state_with_eigenvector(H, N);
    
    rvec coeff(eigvec.size());
    for (size_t i = 0; i < coeff.size(); ++i) {
        coeff[i] = abs(eigvec[i]);
    }

    // Build observable matrices
    cmat R2 = build_r2_matrix(basis);
    cmat T_mat = build_T_matrix(basis, relativistic);

    cld r2_expectation  = 0.0;
    cld t_expectation   = 0.0;
    ld prob_bare        = 0.0;
    ld prob_dressed     = 0.0;
    
    for (size_t i = 0; i < basis.size(); ++i) {
        for (size_t j = 0; j < basis.size(); ++j) {
            r2_expectation += std::conj(eigvec[i]) * R2(i, j) * eigvec[j];
            
            t_expectation += std::conj(eigvec[i]) * T_mat(i, j) * eigvec[j];
            
            cld overlap_term = std::conj(eigvec[i]) * N(i, j) * eigvec[j];
            if (basis[i].type == Channel::PN && basis[j].type == Channel::PN) {
                prob_bare += std::real(overlap_term);
            } 
            else if (basis[i].type != Channel::PN && basis[j].type != Channel::PN) {
                prob_dressed += std::real(overlap_term);
            }
        }
    }

    ld r2_point = std::real(r2_expectation);
    ld r_p_sq = 0.8414 * 0.8414;  
    ld r_n_sq = -0.1161;          

    ld r2_total_charge = r2_point + r_p_sq + r_n_sq;
    ld charge_radius = (r2_total_charge > 0.0) ? std::sqrt(r2_total_charge) : 0.0;

    return {E0, coeff, charge_radius, std::real(t_expectation), prob_bare, prob_dressed, {}};
}

// Physics constraint checker - validates Gaussian state is physical
bool is_physical_gaussian(const SpatialWavefunction& psi, bool debug = false) {
    const ld min_width = 1.0 / (200.0 * 200.0); 
    const ld max_width = 1.0 / (0.05 * 0.05); 

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

        // Shift constraint: |s_i| ≤ 2.0 * width * r_max (keep Gaussian localized)
        ld total_shift = 0;
        for (size_t col = 0; col < 3; ++col) {
            ld shift = std::abs(psi.s(i, col));
            ld limit = 2.0 * width * 5.0;
            if (shift > limit) {
                if (debug) std::cerr << "  [REJECT] |s[" << i << "," << col << "]|=" << shift
                                     << " > limit=" << limit << " (width=" << width << ")\n";
                return false;
            }

            total_shift += shift * shift; 
        }
        if (psi.parity_sign == -1 && total_shift < ZERO_LIMIT) {
            if (debug) std::cerr << "  [REJECT] Odd-parity shift too small (collapsed state).\n";
            return false;
        }
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
            ld total_position = total_shift / (2.0 * width); 

            std::cerr << "Total Shift: " << total_position << " fm | \n";
        }
    }
    std::cerr << std::string(120, '=') << "\n";
}

// Helper to safely apply random noise to a Gaussian state
SpatialWavefunction perturb_wavefunction(const SpatialWavefunction& original, ld noise_scale) {
    SpatialWavefunction perturbed = original;
    
    // 1. Perturb the Widths (Multiplicative noise to stay positive)
    for (size_t i = 0; i < perturbed.A.size1(); ++i) {
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
bool refine_basis_state(std::vector<BasisState>& basis, size_t k, ld noise_scale, 
                        ld b_form, ld S, const std::vector<bool>& relativistic) 
{
    // CACHING: Build the full matrix once
    auto [H_full, N_full] = build_matrices(basis, b_form, S, relativistic);
    ld original_E = solve_ground_state_energy(H_full, N_full);
    
    int cands = 200; // Micro-dart cloud size
    ld best_E = original_E;
    BasisState best_state = basis[k];

    #pragma omp parallel
    {
        ld local_best_E = original_E;
        BasisState local_best_state = basis[k];
        
        // Thread-local matrix copies
        cmat H_test = H_full;
        cmat N_test = N_full;

        #pragma omp for
        for (int c = 0; c < cands; ++c) {
            BasisState test_candidate = basis[k];
            
            // Generate a micro-dart perturbation
            test_candidate.psi = perturb_wavefunction(basis[k].psi, noise_scale);
            
            // Check if the random kick made it unphysical
            if (!is_physical_gaussian(test_candidate.psi, false)) continue;

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
inline void competitive_search(std::vector<BasisState>& basis, 
                               const std::vector<BasisState>& channel_templates,
                               int num_candidates, ld b_range, ld b_form, ld S, 
                               const std::vector<bool>& relativistic) 
{
    // 1. Get the current baseline energy. Darts MUST beat this!
    ld E_core = evaluate_basis_energy(basis, b_form, S, relativistic);
    size_t K = basis.size();
    auto [H_core, N_core] = build_matrices(basis, b_form, S, relativistic);

    for (size_t t = 1; t < channel_templates.size(); ++t) {
        BasisState best_candidate = channel_templates[t];
        
        // 2. Set the baseline to E_core
        ld best_E = E_core; 
        bool found_valid_dart = false;

        #pragma omp parallel
        {
            BasisState local_best_candidate = channel_templates[t];
            ld local_best_E = E_core; 

            cmat H_test = zeros<cld>(K + 1, K + 1);
            cmat N_test = zeros<cld>(K + 1, K + 1);
            
            for (size_t i = 0; i < K; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    H_test(i, j) = H_core(i, j);
                    N_test(i, j) = N_core(i, j);
                }
            }

            #pragma omp for
            for (int c = 0; c < num_candidates; ++c) {
                BasisState test_candidate = channel_templates[t];
                Gaussian g;
                g.randomize(test_candidate.jac, b_range, b_form);
                test_candidate.psi.set_from_gaussian(g);

                // Calculate the dart's isolated diagonal elements first
                cld H_xx = calc_H_elem(test_candidate, test_candidate, b_form, S, relativistic);
                cld N_xx = calc_N_elem(test_candidate, test_candidate);

                // 1. FAST REJECTION GATE: Is this dart completely unphysical?
                // Calculate the isolated energy of this single Gaussian
                ld isolated_energy = std::real(H_xx) / std::real(N_xx);

                // If the Gaussian itself costs more than +500 MeV, it's garbage. 
                // Skip the matrix build and GEVP solver completely!
                if (isolated_energy > 500.0 || std::real(N_xx) < 1e-10) {
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

                // SAFEGUARD 1: Reject linearly dependent darts immediately
                if (std::abs(N_test.determinant()) < 1e-10) continue;

                ld E_estimate = solve_ground_state_energy(H_test, N_test);

                // SAFEGUARD 2: The Strict Improvement Gate
                // It MUST be noticeably better than the current basis to be considered!
                if (E_estimate < local_best_E - 1e-6) {
                    local_best_E = E_estimate;
                    local_best_candidate = test_candidate;
                }
            }

            #pragma omp critical
            {
                if (local_best_E < best_E) {
                    best_E = local_best_E;
                    best_candidate = local_best_candidate;
                    found_valid_dart = true;
                }
            }
        }

        // 3. LOCK AND VERIFY
        if (found_valid_dart) {
            basis.push_back(best_candidate);
            
            // CRITICAL: Update the core matrices so the next channel sees the new state!
            K = basis.size();
            std::tie(H_core, N_core) = build_matrices(basis, b_form, S, relativistic);
            E_core = best_E;

            std::cout << "\rAdded State " << basis.size() << " (Ch " << t << ") -> E = "
                      << std::fixed << std::setprecision(5) << best_E << " MeV    " << std::flush;
        } else {
            // If no dart was good enough, skip the channel entirely instead of breaking the matrix!
            std::cout << "\r[Skipped] Ch " << t << " yielded no improving darts.    " << std::flush;
        }
    }
    std::cout << "\n";
}

// Optimize basis parameters via Nelder-Mead sweeping with early exit
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, const std::vector<bool>& relativistic, rvec& convergence_energies) {
    int max_sweeps = 200;
    int nm_max_iter = 5000; 
    ld improvement_threshold = 1e-5; 
    int patience = 3; 

    ld previous_E = 999999.0;
    int no_improve_count = 0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (size_t k = 0; k < basis.size(); ++k) {
            // Optimize shifts for pions, but NOT for the bare PN state.
            bool opt_shift = true; //(basis[k].type != Channel::PN);
            if (basis[k].type == Channel::PN) continue;

            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi, opt_shift);

            ld E_before = evaluate_basis_energy(basis, b, S, relativistic);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                std::vector<BasisState> test_basis = basis;
                unpack_wavefunction(test_basis[k].psi, p_test, opt_shift);
                if (!is_physical_gaussian(test_basis[k].psi, false)) return 999999.0;
                return evaluate_basis_energy(test_basis, b, S, relativistic, false);
            };

            // THE RESTART ENGINE: Force Nelder-Mead to blow up its simplex 3 times
            rvec p_current = p0;
            int num_restarts = 3; 
            
            for (int restart = 0; restart < num_restarts; ++restart) {
                // nelder_mead now builds a huge simplex around p_current
                p_current = nelder_mead(p_current, objective_func, nm_max_iter);
            }
            
            rvec p_best = p_current;
            unpack_wavefunction(basis[k].psi, p_best, opt_shift);

            ld E_after = evaluate_basis_energy(basis, b, S, relativistic);
            if (E_after > E_before) {
                basis[k].psi = backup_psi;
            }
        }

        ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
        convergence_energies.push_back(current_E);

        ld improvement = previous_E - current_E; 
        if (improvement < improvement_threshold) {
            no_improve_count++;
        } else {
            no_improve_count = 0; 
        }

        std::cout << "\r" << "Sweep " << sweep << " (Basis=" << basis.size() << "): E=" << current_E
                  << " MeV  (ΔE=" << improvement << ")     " << std::flush;

        if (no_improve_count >= patience) {
            std::cout << "\n  - Converged: improvement < " << improvement_threshold << " MeV for " << patience << " sweeps\n";
            break;
        }

        previous_E = current_E;
    }
    std::cout << "\n";
}

} // Namespace qm