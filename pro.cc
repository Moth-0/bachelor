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
║          └─→ Phase 2: Competitive growth (6 states × 4 channels = 24 dressed)  ║
║               Test 20 candidates per state → pick best → sweep optimize        ║
║                                                                                ║
║ KEY DIFFERENCES FROM DEUTERON MODEL:                                           ║
║   • One-body bare state (proton) vs two-body (proton-neutron)                  ║
║   • No internal kinetic energy in bare state (point particle reference)         ║
║   • Only 4 dressed channels (vs 9 for deuteron):                               ║
║     - π⁰ configurations (2: no flip, spin flip)                                ║
║     - π⁺ configurations (2: no flip, spin flip) → neutron dressed state        ║
║   • Total basis size ~25 states (similar to sigma model)                       ║
║   • Simpler physics but fully coupled via W-operator                           ║
║                                                                                ║
║ PARAMETER TUNING:                                                              ║
║                                                                                ║
║   b_range = 1.4 fm:                                                            ║
║     • Search space for Gaussian width parameter (same as deuteron)             ║
║                                                                                ║
║   b_form = 1.4 fm:                                                             ║
║     • Pion interaction range (same as deuteron)                                ║
║                                                                                ║
║   S = 140.0:  **TUNING PARAMETER**                                             ║
║     • Proton-pion coupling strength (same as deuteron pion S)                  ║
║     • Both models use same fundamental coupling                                ║
║     • Tuned to match experimental pion-nucleon dynamics                        ║
║                                                                                ║
║ PHYSICAL PARAMETERS:                                                           ║
║   m_p = 938.27 MeV, m_n = 939.56 MeV:  nucleon masses                          ║
║   m_pi0 = 134.97 MeV, m_pic = 139.57 MeV:  pion masses                         ║
║   iso_c = √2:  isospin weighting for π⁺                                        ║
║                                                                                ║
║ PHYSICAL INSIGHT:                                                              ║
║   Proton mass = bare mass + self-energy from pion cloud                        ║
║   This explains the constituent quark model concept: due to QCD running,       ║
║   dressed nucleons acquire "constituent" masses different from bare quarks.    ║
║   The pion cloud is the low-energy shadow of gluon/quark dynamics.             ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "proton.h"

using namespace qm;

/// Evaluates the ground state energy for a given basis configuration.
/// Constructs H and N matrices, solves GEVP, returns lowest eigenvalue.
ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    return solve_ground_state_energy(H, N);
}

/// Optimizes all basis state parameters via Nelder-Mead sweeping.
/// Runs until convergence (energy change < tolerance) or max_sweeps reached.
/// Skips bare proton (no parameters to optimize). See deu.cc for detailed explanation.
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
    
    ld previous_E = 999999.0;
    ld sweep_tolerance = 1e-5; 
    int max_sweeps = 500;
    int sweep = 0;

    while (sweep < max_sweeps && std::abs(previous_E - current_E) > sweep_tolerance) {
        previous_E = current_E;
        
        for (size_t k = 0; k < basis.size(); ++k) {
            
            // CRITICAL FIX: Skip the bare proton! It has no parameters.
            if (basis[k].type == Channel::P) continue;

            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                unpack_wavefunction(basis[k].psi, p_test);
                
                // --- THE PHYSICS BOUNCER ---
                bool is_physical = true;

                for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
                    // Mirrored from Deuteron: Prevent the pion from evaporating!
                    if (basis[k].psi.A(i, i) <= 0.02) is_physical = false;
                }
                if (basis[k].psi.A.determinant() <= ZERO_LIMIT) is_physical = false;

                for (size_t i = 0; i < basis[k].psi.s.size1(); ++i) {
                    for (size_t col = 0; col < 3; ++col) {
                        // Mirrored from Deuteron: Keep the pion close to the proton!
                        if (std::abs(basis[k].psi.s(i, col)) > 5.0) is_physical = false;
                    }
                }

                if (!is_physical) return 999999.0;

                return evaluate_basis_energy(basis, b, S, relativistic);
            };

            rvec p_best = nelder_mead(p0, objective_func);
            
            unpack_wavefunction(basis[k].psi, p_best);
            current_E = evaluate_basis_energy(basis, b, S, relativistic);
        }
        sweep++;
    }
}

/// --- NEW WRAPPER FUNCTION ---
/// Main SVM algorithm for proton-pion system.
/// Two-phase approach: bare proton reference → competitive pion dressing.
/// Simpler than deuteron (1 nucleon), but fully coupled via W-operator.
/// See deu.cc for detailed explanation of SVM algorithm.
///
/// Proton-specific notes:
///   • Bare proton: N(P,P)=1.0, H(P,P)=0.0 (reference point)
///   • Only 4 dressed channels (π⁰ no flip, π⁰ flip, π⁺ no flip, π⁺ flip)
///   • Total: 1 bare + 6×4 = 25 basis states
///   • No internal proton structure (unlike deuteron's p-n dynamics)
// --- NEW WRAPPER FUNCTION ---
// Runs the entire competitive construction and returns the final energy
ld run_proton_svm(bool relativistic) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;  
    ld m_pi0 = 134.97, m_pic = 139.57; 
    ld b_range = 1.4, b_form = 1.4, S = 80.0;      

    Jacobian jac_bare({m_p});
    Jacobian jac_dressed_0({m_p, m_pi0});
    Jacobian jac_dressed_c({m_n, m_pic});

    std::vector<BasisState> basis;

    // 1. INITIALIZE THE BARE STATE
    SpatialWavefunction psi_bare(1);

    basis.push_back({psi_bare, Channel::P, NO_FLIP, 1.0, jac_bare, 0.0});

    // 2. COMPETITIVE BASIS CONSTRUCTION
    int num_dressed_per_channel = 6;
    int num_candidates_per_step = 20;
    ld iso_c = std::sqrt(2.0L);

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(-1), Channel::P_PI0_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::P_PI0_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::N_PIP_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::N_PIP_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic}
    };

    for (int i = 0; i < num_dressed_per_channel; ++i) {
        for (size_t t = 0; t < channel_templates.size(); ++t) {
            
            BasisState best_candidate = channel_templates[t];
            ld best_E = 999999.0;

            for (int c = 0; c < num_candidates_per_step; ++c) {
                BasisState test_candidate = channel_templates[t];
                Gaussian g;
                g.randomize(test_candidate.jac, b_range);
                test_candidate.psi.set_from_gaussian(g);

                basis.push_back(test_candidate);
                ld E = evaluate_basis_energy(basis, b_form, S, relativistic);
                basis.pop_back();

                if (E < best_E) {
                    best_E = E;
                    best_candidate = test_candidate;
                }
            }

            basis.push_back(best_candidate);
            sweep_optimize_basis(basis, b_form, S, relativistic);
            
            std::cout << "\r" << "Added state " << basis.size() << " (Channel " << t << ") -> Energy: " 
                      << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV    " << std::flush;
        }
    }
    
    std::cout << "\n";
    return evaluate_basis_energy(basis, b_form, S, relativistic);
}


int main() {
    std::cout << "========================================\n";
    std::cout << "  PROTON-PION SYSTEM (COMPETITIVE SVM)\n";
    std::cout << "========================================\n\n";

    // Run SVM twice: classic and relativistic kinetic energy
    // Compare to understand relativistic corrections in proton self-energy
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY (T = p^2 / 2m)\n";
    ld E_classic = run_proton_svm(false);

    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY (T = sqrt(p^2 + m^2) - m)\n";
    ld E_relativistic = run_proton_svm(true);

    // Display final results
    std::cout << "\n========================================\n";
    std::cout << "  FINAL RESULTS COMPARISON\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Classic Energy:      " << E_classic << " MeV\n";
    std::cout << "Relativistic Energy: " << E_relativistic << " MeV\n";

    ld diff = std::abs(E_classic - E_relativistic);
    std::cout << "Difference:          " << diff << " MeV\n";
    std::cout << "========================================\n";

    return 0;
}