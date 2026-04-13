/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                  deu.cc - DEUTERON SVM GROUND STATE FINDER                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver orchestrating the complete Stochastic Variational Method (SVM)   ║
║   to find the ground state energy of a deuteron with pion exchange coupling.   ║
║                                                                                ║
║ WORKFLOW OVERVIEW:                                                             ║
║   main()                                                                       ║
║     └─→ Iterates over 4 kinematic configurations (Classic/Relativistic)        ║
║     └─→ run_deuteron_svm(flags)                                                ║
║          └─→ Phase 1: Skeleton basis (14 states) → sweep_optimize()            ║
║          └─→ Phase 2: Competitive growth (2 cycles) →                          ║
║               For each channel: test candidates →                              ║
║               add best → sweep_optimize()                                      ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

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

using namespace qm;

struct SvmResult {
    ld energy;
    ld charge_radius;
    rvec convergence_history;
};

// Evaluate energy: build H,N and solve GEVP
ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, const std::vector<bool>& relativistic, bool debug = true) {
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

struct GroundStateResult{
    ld energy;
    ld charge_radius;
};

// Calculate charge radius from ground state
GroundStateResult evaluate_with_radius(const std::vector<BasisState>& basis, ld b, ld S,
                                       const std::vector<bool>& relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    cld detN = N.determinant();
    if (std::abs(detN) < ZERO_LIMIT) { 
        return {999999.0, 99999.0}; 
    }

    auto [E0, eigvec] = solve_ground_state_with_eigenvector(H, N);

    // Build r² matrix using the internal state jacobians
    cmat R2 = build_r2_matrix(basis);

    // Calculate <ψ₀|r²_point|ψ₀>
    cld r2_expectation = 0.0;
    for (size_t i = 0; i < basis.size(); ++i) {
        for (size_t j = 0; j < basis.size(); ++j) {
            r2_expectation += std::conj(eigvec[i]) * R2(i, j) * eigvec[j];
        }
    }

    ld r2_point = std::real(r2_expectation);
    
    // Finite Nucleon Size Correction
    ld r_p_sq = 0.8414 * 0.8414;  
    ld r_n_sq = -0.1161;          

    ld r2_total_charge = r2_point + r_p_sq + r_n_sq;
    ld charge_radius = (r2_total_charge > 0.0) ? std::sqrt(r2_total_charge) : 0.0;

    return {E0, charge_radius};
}

// Physics constraint checker - validates Gaussian state is physical
bool is_physical_gaussian(const SpatialWavefunction& psi, bool debug = true) {
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
bool refine_basis_state(std::vector<BasisState>& basis, size_t k, ld b_range, ld b_form, ld S, const std::vector<bool>& relativistic, rvec& convergence_energies) {
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


// Optimize basis parameters via Nelder-Mead sweeping with early exit
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, const std::vector<bool>& relativistic, rvec& convergence_energies) {
    int max_sweeps = 50;
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

// Run two-phase SVM: skeleton (Phase 1) + competitive growth (Phase 2)
// Returns: SvmResult struct containing E, R, and convergence history
SvmResult run_deuteron_svm(const std::vector<bool>& relativistic, ld b_range, ld b_form, ld S) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;
    ld m_pi0 = 134.97, m_pic = 139.57;

    ld iso_c = std::sqrt(2);

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    rvec convergence_energies;

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        
        // pi^0 channel (Phase: +1)
        {SpatialWavefunction(-1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        
        // pi^+ channel (Phase: +1)
        {SpatialWavefunction(-1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        
        // pi^- channel (Phase: -1)
        {SpatialWavefunction(-1), Channel::PI_mc_0f, NO_FLIP, -iso_c, jac_dressed_c, m_pic}, 
        {SpatialWavefunction(-1), Channel::PI_mc_1f, FLIP_PARTICLE_1, -iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_2f, FLIP_PARTICLE_2, -iso_c, jac_dressed_c, m_pic}
    };

    // ------- PHASE 1: SKELETON BASIS WITH GEOMETRIC GRID --------
    std::cout << "--- 1. Planting Geometric PN Grid & Pion Seeds ---\n";
    
    std::vector<ld> deterministic_widths = {0.05, 0.3, 1.2, 2.0};
    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * width;
        rmat s_fixed = zeros<ld>(1, 3);

        SpatialWavefunction psi_fixed(A_fixed, s_fixed, 1);
        basis.push_back({psi_fixed, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    for (size_t t = 1; t < channel_templates.size(); ++t) {
        BasisState seed = channel_templates[t];
        Gaussian g;
        g.randomize(seed.jac, b_range);
        seed.psi.set_from_gaussian(g);
        basis.push_back(seed);
    }
    
    std::cout << "Skeleton Size: " << basis.size() << " states.\n";
    sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);
    ld skeleton_E = evaluate_basis_energy(basis, b_form, S, relativistic);
    convergence_energies.push_back(skeleton_E);
    std::cout << "\nSkeleton Energy: " << skeleton_E << " MeV\n\n";

    // ------- PHASE 2: COMPETITIVE SVM GROWTH --------
    int num_cycles = 3;

    std::cout << "--- 2. Competitive SVM Growth ---\n";
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        int num_candidates_per_step = 500;

        for (size_t t = 0; t < channel_templates.size(); ++t) {
            BasisState best_candidate = channel_templates[t];
            ld best_E = 999999.0;

            #pragma omp parallel
            {
                BasisState local_best_candidate = channel_templates[t];
                ld local_best_E = 999999.0;
                std::vector<BasisState> local_basis = basis;

                #pragma omp for
                for (int c = 0; c < num_candidates_per_step; ++c) {
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

            basis.push_back(best_candidate);

            ld current_E = evaluate_basis_energy(basis, b_form, S, relativistic);

            std::cout << "\rAdded State " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> E = "
                      << std::fixed << std::setprecision(5) << current_E << " MeV    " << std::flush;
        }

        std::cout << "\n - Sweeping Cycle " << cycle+1 << " basis -\n";
        sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);

        std::cout << "\n=== Refinement Cycle " << cycle+1 << " ===\n";
        
        int max_refine_passes = 2;       
        ld refine_tolerance = 1e-3;      
        ld previous_pass_E = evaluate_basis_energy(basis, b_form, S, relativistic);

        for (int pass = 0; pass < max_refine_passes; ++pass) {
            int num_improved_this_pass = 0;
            
            for (size_t k = 0; k < basis.size(); ++k) {
                bool improved = refine_basis_state(basis, k, b_range, b_form, S, relativistic, convergence_energies);
                if (improved) {
                    num_improved_this_pass++;
                }
                std::cout << "\rPass " << pass+1 << "/" << max_refine_passes 
                          << " | Refined state " << k+1 << "/" << basis.size()
                          << " (" << num_improved_this_pass << " improved)    " << std::flush;
            }

            ld current_pass_E = evaluate_basis_energy(basis, b_form, S, relativistic);
            ld delta_E = previous_pass_E - current_pass_E;

            std::cout << "\n -> End of Pass " << pass+1 << ": E = " 
                      << std::fixed << std::setprecision(6) << current_pass_E 
                      << " MeV (ΔE = " << delta_E << " MeV)\n";

            if (delta_E < refine_tolerance) {
                std::cout << " -> Refinement Converged: Total improvement below " << refine_tolerance << " MeV.\n";
                break; 
            }

            previous_pass_E = current_pass_E;
            convergence_energies.push_back(previous_pass_E);
        }

        std::cout << " - Final sweep after refinement cycle " << cycle+1 << " -\n";
        ld E_before_sweep = evaluate_basis_energy(basis, b_form, S, relativistic);
        sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);
        ld E_after_sweep = evaluate_basis_energy(basis, b_form, S, relativistic);

        if (E_after_sweep > E_before_sweep) {
            std::cout << "  WARNING: Sweep made energy WORSE! (" << E_before_sweep << " -> " << E_after_sweep << " MeV, ΔE=" << (E_after_sweep - E_before_sweep) << ")\n";
        } else {
            std::cout << "  Sweep improved: " << E_before_sweep << " -> " << E_after_sweep << " MeV (ΔE=" << (E_after_sweep - E_before_sweep) << ")\n";
        }

        std::cout << "\n";
    }

    print_basis_details(basis, " === FINAL BASIS STATE (ALL CYCLES COMPLETE) ===");
    std::cout << "\n";
    
    // Return struct instead of saving to file here
    auto result = evaluate_with_radius(basis, b_form, S, relativistic);
    return {result.energy, result.charge_radius, convergence_energies};
}

int main(int argc, char* argv[]) {
    // Default values
    ld b_range = 3.6;
    ld b_form = 1.4;
    ld S = 30.0;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-b_range" || arg == "--b_range") && i + 1 < argc) {
            b_range = std::stold(argv[++i]);
        } else if ((arg == "-b_form" || arg == "--b_form") && i + 1 < argc) {
            b_form = std::stold(argv[++i]);
        } else if ((arg == "-S" || arg == "--S") && i + 1 < argc) {
            S = std::stold(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./deu [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -b_range <value>  Search space for Gaussian width (default: 1.4 fm)\n";
            std::cout << "  -b_form <value>   Pion interaction range (default: 1.4 fm)\n";
            std::cout << "  -S <value>        Pion coupling strength (default: 100.0 MeV)\n";
            std::cout << "  -h, --help        Show this help message\n";
            return 0;
        }
    }

    // Enable nested parallelism
    omp_set_nested(1);
    omp_set_max_active_levels(2);

    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n";
    std::cout << "Parameters:\n";
    std::cout << "  b_range = " << b_range << " fm\n";
    std::cout << "  b_form  = " << b_form << " fm\n";
    std::cout << "  S       = " << S << " MeV\n";
    std::cout << "========================================\n\n";

    // Define the 4 kinematics configurations: {Nucleons_Relativistic, Pion_Relativistic}
    std::vector<std::pair<std::string, std::vector<bool>>> configurations = {
        {"Classic_PN_Classic_Pi", {false, false}},
        {"Relativ_PN_Classic_Pi", {true,  false}},
        {"Classic_PN_Relativ_Pi", {false, true}},
        {"Relativ_PN_Relativ_Pi", {true,  true}}
    };

    std::ofstream outfile("convergence_master.data");
    std::vector<SvmResult> all_results;

    // Run the configurations loop
    for (const auto& config : configurations) {
        std::string label = config.first;
        std::vector<bool> flags = config.second;

        std::cout << "\n>>>>>>>> RUNNING CONFIGURATION: " << label << " <<<<<<<<\n";
        
        SvmResult res = run_deuteron_svm(flags, b_range, b_form, S);
        all_results.push_back(res);

        std::cout << "--> FINAL " << label << " | E: " << res.energy << " MeV, R: " << res.charge_radius << " fm\n";

        // Write block to the master file
        // Put the label directly into the data column header
        outfile << "\"Iteration\"\t\"" << label << "\"\n";
        for (size_t iter = 0; iter < res.convergence_history.size(); ++iter) {
            outfile << iter << "\t" << std::fixed << std::setprecision(8) << res.convergence_history[iter] << "\n";
        }
        
        // Two blank lines separate index blocks for Gnuplot
        outfile << "\n\n"; 
    }

    outfile.close();

    // Print final summary comparison table
    std::cout << "\n=======================================================\n";
    std::cout << "                 FINAL RESULTS SUMMARY                 \n";
    std::cout << "=======================================================\n";
    std::cout << std::fixed << std::setprecision(5);
    
    for (size_t i = 0; i < configurations.size(); ++i) {
        std::cout << std::setw(24) << std::left << configurations[i].first 
                  << " | Energy: " << std::setw(9) << std::right << all_results[i].energy << " MeV"
                  << " | Radius: " << std::setw(8) << all_results[i].charge_radius << " fm\n";
    }
    
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Experimental Target      | Energy:  -2.22400 MeV | Radius:  2.12800 fm\n";
    std::cout << "=======================================================\n";
    std::cout << "\nPlot ready: 'convergence_master.data' saved successfully.\n";

    return 0;
}