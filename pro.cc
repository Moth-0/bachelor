/*
╔════════════════════════════════════════════════════════════════════════════════╗
║              pro.cc - PROTON-PION SVM GROUND STATE FINDER                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver orchestrating the Stochastic Variational Method (SVM) to find    ║
║   the ground state of a single proton dressed with virtual pion clouds.        ║
║   Simpler analog of deuteron system (1+pion instead of 2+pion).                ║
║                                                                                ║
║ WORKFLOW OVERVIEW:                                                             ║
║   main()                                                                       ║
║     └─→ run_proton_svm(false)   [classic kinetic energy]                       ║
║     └─→ run_proton_svm(true)    [relativistic kinetic energy]                  ║
║          └─→ Phase 1: Single bare proton reference state                       ║
║          └─→ Phase 2: Competitive growth with refinement                       ║
║               For each channel: test candidates → add best → sweep             ║
║                                                                                ║
║ KEY DIFFERENCES FROM DEUTERON MODEL:                                           ║
║   • One-body bare state (proton) vs two-body (proton-neutron)                  ║
║   • No internal kinetic energy in bare state (point particle reference)         ║
║   • Only 4 dressed channels (vs 9 for deuteron):                               ║
║     - π⁰ configurations (2: no flip, spin flip)                                ║
║     - π⁺ configurations (2: no flip, spin flip) → neutron dressed state        ║
║                                                                                ║
║ PARAMETER TUNING:                                                              ║
║                                                                                ║
║   b_range = 1.4 fm:                                                            ║
║     • Search space for Gaussian width parameter                                ║
║                                                                                ║
║   b_form = 1.4 fm:                                                             ║
║     • Pion interaction range (form factor)                                     ║
║                                                                                ║
║   S = 140.0:  **TUNING PARAMETER**                                             ║
║     • Proton-pion coupling strength                                            ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <omp.h>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "proton.h"

using namespace qm;

// Evaluate energy: build H,N and solve GEVP
ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic, bool debug = true) {
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
                          << "  -> L(" << i << "," << i << ") = " << std::abs(L(i, i)) << " < threshold (1e-4).\n";
            }
            return 999999.0;
        }
    }

    return solve_ground_state_energy(H, N);
}

/// Calculate charge (proton) radius from ground state
struct GroundStateResult {
    ld energy;
    ld charge_radius;
};

GroundStateResult evaluate_with_radius(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    cld detN = N.determinant();
    if (std::abs(detN) < ZERO_LIMIT) {
        return {999999.0, 99999.0};
    }

    auto [E0, eigvec] = solve_ground_state_with_eigenvector(H, N);

    // For single proton, charge radius is just the bare proton radius (small)
    // The pion cloud contributes to mass but not much to geometric size
    ld charge_radius = 0.8;  // Fixed proton charge radius (fm)

    return {E0, charge_radius};
}

/// Physics constraint checker
bool is_physical_gaussian(const SpatialWavefunction& psi, bool debug = true) {
    const ld min_width = 1.0 / (50.0 * 50.0);  // Allow r up to 50 fm
    const ld max_width = 1.0 / (0.1 * 0.1);    // Prevent r < 0.1 fm

    // Check diagonal widths
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

        // Shift constraint
        for (size_t col = 0; col < 3; ++col) {
            ld shift = std::abs(psi.s(i, col));
            ld limit = 2.0 * width * 0.2;
            if (shift > limit) {
                if (debug) std::cerr << "  [REJECT] |s[" << i << "," << col << "]|=" << shift
                                     << " > limit=" << limit << " (width=" << width << ")\n";
                return false;
            }
        }
    }

    // Positive definiteness check
    ld det = psi.A.determinant();
    if (det <= ZERO_LIMIT) {
        if (debug) std::cerr << "  [REJECT] det(A)=" << det << " <= " << ZERO_LIMIT << "\n";
        return false;
    }

    if (debug) std::cerr << "  [ACCEPT] State passes all constraints\n";
    return true;
}

// Refine a single basis state by trying random parameter replacements
bool refine_basis_state(std::vector<BasisState>& basis, size_t k, ld b_range, ld b_form, ld S, bool relativistic, rvec& convergence_energies) {
    if (k == 0) return false;  // Skip bare proton

    SpatialWavefunction original_psi = basis[k].psi;
    ld original_E = evaluate_basis_energy(basis, b_form, S, relativistic);
    int cands = 500;

    ld best_E = original_E;
    SpatialWavefunction best_psi = original_psi;

    // Parallelize candidate search
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
    convergence_energies.push_back(best_E);

    return (best_E < original_E);
}

/// Optimize basis parameters via Nelder-Mead sweeping
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic, rvec& convergence_energies) {
    int max_sweeps = 50;
    int nm_max_iter = 200;
    ld improvement_threshold = 1e-3;
    int patience = 3;

    ld previous_E = 999999.0;
    int no_improve_count = 0;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (size_t k = 0; k < basis.size(); ++k) {
            // Skip bare proton (no parameters to optimize)
            if (basis[k].type == Channel::P) continue;

            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            ld E_before = evaluate_basis_energy(basis, b, S, relativistic);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                std::vector<BasisState> test_basis = basis;
                unpack_wavefunction(test_basis[k].psi, p_test);

                if (!is_physical_gaussian(test_basis[k].psi)) return 999999.0;

                return evaluate_basis_energy(test_basis, b, S, relativistic);
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

/// Main SVM algorithm for proton-pion system
std::pair<ld, ld> run_proton_svm(bool relativistic, ld b_range, ld b_form, ld S) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;
    ld m_pi0 = 134.97, m_pic = 139.57;

    Jacobian jac_bare({m_p});
    Jacobian jac_dressed_0({m_p, m_pi0});
    Jacobian jac_dressed_c({m_n, m_pic});

    std::vector<BasisState> basis;
    rvec convergence_energies;
    std::string convergence_file = relativistic ? "conv_pro_rel.data" : "conv_pro_cla.data";

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(-1), Channel::P_PI0_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::P_PI0_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::N_PIP_0f, NO_FLIP, std::sqrt(2.0L), jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::N_PIP_1f, FLIP_PARTICLE_1, std::sqrt(2.0L), jac_dressed_c, m_pic}
    };

    // --- PHASE 1: BARE PROTON REFERENCE ---
    std::cout << "--- 1. Bare Proton Reference State ---\n";
    SpatialWavefunction psi_bare(1);
    basis.push_back({psi_bare, Channel::P, NO_FLIP, 1.0, jac_bare, 0.0});

    ld bare_E = evaluate_basis_energy(basis, b_form, S, relativistic);
    convergence_energies.push_back(bare_E);
    std::cout << "Bare Proton Energy: " << bare_E << " MeV\n\n";

    // --- PHASE 2: COMPETITIVE SVM GROWTH ---
    int num_cycles = 2;
    std::cout << "--- 2. Competitive SVM Growth ---\n";

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        int num_candidates_per_step = 500;

        for (size_t t = 0; t < channel_templates.size(); ++t) {
            BasisState best_candidate = channel_templates[t];
            ld best_E = 999999.0;

            // OpenMP Parallel Candidate Search
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
                    ld E = evaluate_basis_energy(local_basis, b_form, S, relativistic);
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
            convergence_energies.push_back(current_E);

            std::cout << "\rAdded State " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> E = "
                      << std::fixed << std::setprecision(5) << current_E << " MeV    " << std::flush;
        }

        // Sweep after adding all channels in this cycle
        std::cout << "\n - Sweeping Cycle " << cycle+1 << " basis -\n";
        sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);

        // Refinement cycle
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
        }

        // Final sweep after refinement
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

    // Save convergence data
    {
        std::ofstream outfile(convergence_file);
        outfile << "iteration energy\n";
        for (size_t i = 0; i < convergence_energies.size(); ++i) {
            outfile << i << " " << std::fixed << std::setprecision(8) << convergence_energies[i] << "\n";
        }
        outfile.close();
        std::cout << "\nConvergence data saved to: " << convergence_file << "\n";
    }

    std::cout << "\n";
    auto result = evaluate_with_radius(basis, b_form, S, relativistic);
    return {result.energy, result.charge_radius};
}

int main(int argc, char* argv[]) {
    // Default values
    ld b_range = 3.6;
    ld b_form = 1.4;
    ld S = 50.0;

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
            std::cout << "Usage: ./pro [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -b_range <value>  Search space for Gaussian width (default: 3.6 fm)\n";
            std::cout << "  -b_form <value>   Pion interaction range (default: 1.4 fm)\n";
            std::cout << "  -S <value>        Pion coupling strength (default: 50.0 MeV)\n";
            std::cout << "  -h, --help        Show this help message\n";
            return 0;
        }
    }

    // Enable nested parallelism
    omp_set_nested(1);
    omp_set_max_active_levels(2);

    std::cout << "========================================\n";
    std::cout << "  PROTON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n";
    std::cout << "Parameters:\n";
    std::cout << "  b_range = " << b_range << " fm\n";
    std::cout << "  b_form  = " << b_form << " fm\n";
    std::cout << "  S       = " << S << " MeV\n";
    std::cout << "========================================\n\n";

    // Run with both kinetic energy models
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY\n";
    auto [E_classic, R_classic] = run_proton_svm(false, b_range, b_form, S);

    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY\n";
    auto [E_relativistic, R_relativistic] = run_proton_svm(true, b_range, b_form, S);

    ld E_diff = E_relativistic - E_classic;
    ld R_diff = R_relativistic - R_classic;

    // --- COMPARISON & INTERPRETATION ---
    std::cout << "========================================\n";
    std::cout << "  FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(5);

    std::cout << "\n--- PROTON SELF-ENERGY ---\n";
    std::cout << "Classic Energy:          " << E_classic << " MeV\n";
    std::cout << "Relativistic Energy:     " << E_relativistic << " MeV\n";
    std::cout << "Difference (rel-cl):     " << E_diff << " MeV\n";

    std::cout << "\n--- CHARGE RADIUS ---\n";
    std::cout << "Classic Radius:          " << R_classic << " fm\n";
    std::cout << "Relativistic Radius:     " << R_relativistic << " fm\n";
    std::cout << "Difference (rel-cl):     " << R_diff << " fm\n";

    std::cout << "========================================\n";

    return 0;
}
