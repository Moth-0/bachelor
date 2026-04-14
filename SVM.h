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
        if (std::abs(L(i, i)) < ZERO_LIMIT) { 
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
        return {999999.0, 99999.0, 0.0, {}}; 
    }

    auto [E0, eigvec] = solve_ground_state_with_eigenvector(H, N);

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

    return {E0, charge_radius, std::real(t_expectation), prob_bare, prob_dressed, {}};
}

// Physics constraint checker - validates Gaussian state is physical
bool is_physical_gaussian(const SpatialWavefunction& psi, bool debug = false) {
    const ld min_width = 1.0 / (50.0 * 50.0); 
    const ld max_width = 1.0 / (0.1 * 0.1); 

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
            ld limit = 2.0 * width * 3.0;
            if (shift > limit) {
                if (debug) std::cerr << "  [REJECT] |s[" << i << "," << col << "]|=" << shift
                                     << " > limit=" << limit << " (width=" << width << ")\n";
                return false;
            }

            total_shift += shift * shift; 
        }
        if (psi.parity_sign == -1 && total_shift < 1e-6) {
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
void print_basis_details(const std::vector<BasisState>& basis, const std::string& label) {
    std::cerr << "\n" << label << "\n";
    std::cerr << "Basis Size: " << basis.size() << "\n";
    std::cerr << std::string(120, '=') << "\n";

    for (size_t k = 0; k < basis.size(); ++k) {
        std::cerr << "State " << k << " (Type " << (int)basis[k].type << "):\n";

        for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
            ld width = basis[k].psi.A(i, i);
            std::cerr << "  Gaussian[" << i << "] width=" << width << " fm⁻² (r≈"
                      << (1.0/std::sqrt(width)) << " fm) | ";

            ld shift_sq = 0.0;
            for (size_t col = 0; col < 3; ++col) {
                shift_sq += basis[k].psi.s(i, col) * basis[k].psi.s(i, col);
            }
            ld total_shift = std::sqrt(shift_sq);
            ld total_position = total_shift / (2.0 * width); 

            std::cerr << "Total Shift: " << total_shift << " fm⁻¹ (" << total_position << " fm) | ";
            std::cerr << "| det(A)=" << basis[k].psi.A.determinant() << "\n";
        }
    }
    std::cerr << std::string(120, '=') << "\n";
}


// Refine a single basis state by trying random parameter replacements
bool refine_basis_state(std::vector<BasisState>& basis, size_t k, ld b_range, ld b_form, ld S, const std::vector<bool>& relativistic) {
    SpatialWavefunction original_psi = basis[k].psi;
    ld original_E = evaluate_basis_energy(basis, b_form, S, relativistic);
    int cands = 500;

    ld best_E = original_E;
    SpatialWavefunction best_psi = original_psi;

    #pragma omp parallel
    {
        ld local_best_E = original_E;
        SpatialWavefunction local_best_psi = original_psi;
        std::vector<BasisState> local_basis = basis; 

        #pragma omp for
        for (int c = 0; c < cands; ++c) {
            Gaussian g;
            g.randomize(local_basis[k].jac, b_range);
            local_basis[k].psi.set_from_gaussian(g);

            if (!is_physical_gaussian(local_basis[k].psi)) continue;

            ld E = evaluate_basis_energy(local_basis, b_form, S, relativistic);
            if (E < local_best_E) {
                local_best_E = E;
                local_best_psi = local_basis[k].psi;
            }
        }

        #pragma omp critical
        {
            if (local_best_E < best_E) {
                best_E = local_best_E;
                best_psi = local_best_psi;
            }
        }
    }

    basis[k].psi = best_psi;
    return (best_E < original_E);
}

// Performs one full cycle of competitive basis growth across all channels
inline void competitive_search(std::vector<BasisState>& basis, 
                               const std::vector<BasisState>& channel_templates,
                               int num_candidates, ld b_range, ld b_form, ld S, 
                               const std::vector<bool>& relativistic, int cycle) 
{
    for (size_t t = 0; t < channel_templates.size(); ++t) {
        BasisState best_candidate = channel_templates[t];
        ld best_E = 999999.0;

        #pragma omp parallel
        {
            BasisState local_best_candidate = channel_templates[t];
            ld local_best_E = 999999.0;
            std::vector<BasisState> local_basis = basis; // Thread-local copy

            #pragma omp for
            for (int c = 0; c < num_candidates; ++c) {
                BasisState test_candidate = channel_templates[t];
                Gaussian g;
                g.randomize(test_candidate.jac, b_range);
                test_candidate.psi.set_from_gaussian(g);

                local_basis.push_back(test_candidate);
                
                ld E = evaluate_basis_energy(local_basis, b_form, S, relativistic, false);
                local_basis.pop_back();

                if (E < local_best_E) {
                    local_best_E = E;
                    local_best_candidate = test_candidate;
                }
            }

            #pragma omp critical
            {
                if (local_best_E < best_E) {
                    best_E = local_best_E;
                    best_candidate = local_best_candidate;
                }
            }
        }

        // Lock the global winner into the actual basis
        basis.push_back(best_candidate);
        ld current_E = evaluate_basis_energy(basis, b_form, S, relativistic);

        std::cout << "\rAdded State " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> E = "
                  << std::fixed << std::setprecision(5) << current_E << " MeV    " << std::flush;
    }
}

// Performs iterative stochastic refinement on the existing basis until convergence
inline void refinement(std::vector<BasisState>& basis, int max_passes, ld tolerance, 
                       ld b_range, ld b_form, ld S, const std::vector<bool>& relativistic, 
                       rvec& convergence_energies) 
{
    ld previous_pass_E = evaluate_basis_energy(basis, b_form, S, relativistic);

    for (int pass = 0; pass < max_passes; ++pass) {
        int num_improved_this_pass = 0;
        
        for (size_t k = 0; k < basis.size(); ++k) {
            bool improved = refine_basis_state(basis, k, b_range, b_form, S, relativistic);
            if (improved) {
                num_improved_this_pass++;
            }
            std::cout << "\rPass " << pass+1 << "/" << max_passes 
                      << " | Refined state " << k+1 << "/" << basis.size()
                      << " (" << num_improved_this_pass << " improved)    " << std::flush;
        }

        ld current_pass_E = evaluate_basis_energy(basis, b_form, S, relativistic);
        ld delta_E = previous_pass_E - current_pass_E;

        std::cout << "\n -> End of Pass " << pass+1 << ": E = " 
                  << std::fixed << std::setprecision(6) << current_pass_E 
                  << " MeV (ΔE = " << delta_E << " MeV)\n";

        if (delta_E < tolerance) {
            std::cout << " -> Refinement Converged: Total improvement below " << tolerance << " MeV.\n";
            break; 
        }

        previous_pass_E = current_pass_E;
        convergence_energies.push_back(previous_pass_E);
    }
}

// Optimize basis parameters via Nelder-Mead sweeping with early exit
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, const std::vector<bool>& relativistic, rvec& convergence_energies) {
    int max_sweeps = 100;
    int nm_max_iter = 200; 
    ld improvement_threshold = 1e-3; 
    int patience = 3; 

    ld previous_E = 999999.0;
    int no_improve_count = 0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (size_t k = 0; k < basis.size(); ++k) {
            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            ld E_before = evaluate_basis_energy(basis, b, S, relativistic);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                std::vector<BasisState> test_basis = basis;
                unpack_wavefunction(test_basis[k].psi, p_test);

                if (!is_physical_gaussian(test_basis[k].psi, false)) return 999999.0;

                return evaluate_basis_energy(test_basis, b, S, relativistic, false);
            };

            rvec p_best = nelder_mead(p0, objective_func, nm_max_iter);
            unpack_wavefunction(basis[k].psi, p_best);

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