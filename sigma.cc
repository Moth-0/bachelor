/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                sigma.cc - SIGMA-MESON SVM GROUND STATE FINDER                  ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver orchestrating the Stochastic Variational Method (SVM) to find    ║
║   the ground state of a deuteron with sigma-meson exchange coupling           ║
║   (Fedorov 2020 model). Simplified single-meson model compared to pion.        ║
║                                                                                ║
║ WORKFLOW OVERVIEW:                                                             ║
║   main()                                                                       ║
║     └─→ run_sigma_svm(false)   [classic kinetic energy]                        ║
║     └─→ run_sigma_svm(true)    [relativistic kinetic energy]                   ║
║          └─→ Phase 1: Plant 10 PN states with geometric widths                 ║
║          └─→ Phase 2: Competitive growth (25 sigma-meson states)               ║
║               Test 20 candidates per state → pick best → sweep optimize        ║
║                                                                                ║
║ KEY DIFFERENCES FROM DEUTERON MODEL:                                           ║
║   • Single meson type (σ ≈ 500 MeV) instead of 3 pion types                    ║
║   • Scalar coupling (no spin flip) → simpler W-operator                        ║
║   • No isospin factors (neutral meson for all configurations)                  ║
║   • Larger PN basis (10 states) to span wider kinetic energy range             ║
║   • Fewer growth cycles (25 total states vs. 34 in deuteron)                   ║
║   • Different S value (~20 vs ~140): reflects weaker binding of σ vs πs        ║
║                                                                                ║
║ PARAMETER TUNING:                                                              ║
║                                                                                ║
║   b_range = 3.0 fm:                                                            ║
║     • Search space for Gaussian width parameter                                ║
║     • Larger than pion model (pion uses 1.4) because σ is more diffuse         ║
║                                                                                ║
║   S = 20.35:  **TUNING PARAMETER FOR SIGMA MODEL**                             ║
║     • Sigma-nucleon coupling strength (MeV)                                    ║
║     • Much smaller than pion S (140) due to weaker σ dynamics                  ║
║     • Fedorov fixed S=20.35 from fit to experimental data                      ║
║     • Compare results to understand pion vs sigma binding contributions         ║
║                                                                                ║
║   m_sigma ≈ 500 MeV:                                                           ║
║     • Rest mass of sigma meson (poorly measured resonance)                     ║
║     • Fedorov uses 500 MeV as canonical value                                  ║
║                                                                                ║
║ PHYSICAL PARAMETERS:                                                           ║
║   m_p ≈ m_n ≈ 939 MeV:       nucleon masses (isospin-averaged)                 ║
║   m_sigma ≈ 500 MeV:         scalar meson rest mass                            ║
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
#include "sigma.h"
#include "SVM.h"


using namespace qm;

/// Main SVM algorithm for sigma-meson deuteron model.
/// Two-phase approach: geometric PN basis → competitive sigma growth.
/// See deu.cc for detailed explanation of SVM algorithm.
///
/// Sigma-specific notes:
///   • Plant 10 PN states (wider spacing than pion's 5) for kinetic energy range
///   • Add 25 sigma-dressed states competitively (simpler than 9 pion channels)
///   • No spin flips or isospin factors (scalar W-operator)
///   • Different S parameter (~20.35 from Fedorov fit)
ld run_sigma_svm(bool rel) {
    // Physical Constants from Fedorov (2020)
    ld m_n = 939.0, m_p = 939.0;    // 
    ld m_sigma = 500.0;             // [cite: 156, 157]
    ld b_range = 3.0, S = 20.35;    // 

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed({m_p, m_n, m_sigma});

    std::vector<BasisState> basis;
    std::vector<bool> relativistic = {rel};

    // 1. PLANT THE GEOMETRIC BARE CORE (10 states as recommended) [cite: 159]
    std::cout << "--- 1. Planting PN Bare Core ---\n";
    std::vector<ld> deterministic_widths = {0.02, 0.05, 0.1, 0.3, 0.8, 1.5, 3.0, 5.0, 8.0, 12.0};

    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * width;
        rmat s_fixed = zeros<ld>(1, 3);
        SpatialWavefunction psi_bare(A_fixed, s_fixed, 1);
        basis.push_back({psi_bare, Channel::PN, jac_bare, 0.0});
    }

    // 2. COMPETITIVE BASIS CONSTRUCTION (Add 25 dressed states) [cite: 152, 160]
    int num_dressed = 25;
    int num_candidates_per_step = 20;

    BasisState dressed_template = {SpatialWavefunction(1), Channel::PN_SIGMA, jac_dressed, m_sigma};

    std::cout << "--- 2. Competitive SVM Growth (Sigma Meson Cloud) ---\n";
    for (int i = 0; i < num_dressed; ++i) {
        
        BasisState best_candidate = dressed_template;
        ld best_E = 999999.0;

        #pragma omp parallel 
        {
            BasisState local_best_candidate = dressed_template;
            ld local_best_E = 999999.0;
            std::vector<BasisState> local_basis = basis; 
            
            #pragma omp for
            for (int c = 0; c < num_candidates_per_step; ++c) {
                BasisState test_candidate = dressed_template;
                Gaussian g;
                g.randomize(test_candidate.jac, b_range, b_range);
                test_candidate.psi.set_from_gaussian(g);

                local_basis.push_back(test_candidate);
                ld E = evaluate_basis_energy(local_basis, b_range, S, relativistic);
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
    
        
        std::cout << "\r" << "Added Dressed State " << i+1 << "/" << num_dressed << " -> Energy: " 
                  << std::fixed << std::setprecision(5) << evaluate_basis_energy(basis, b_range, S, relativistic) << " MeV    " << std::flush;
    }
    
    std::cout << "\n";
    sweep_optimize_basis(basis, b_range, S, relativistic);

    return evaluate_basis_energy(basis, b_range, S, relativistic);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  FEDOROV SIGMA-MESON DEUTERON MODEL\n";
    std::cout << "========================================\n\n";

    // Run SVM twice: classic and relativistic kinetic energy
    // Compare to understand relativistic corrections in sigma model
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY (T = p^2 / 2m)\n";
    ld E_classic = run_sigma_svm(false);

    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY (T = sqrt(p^2 + m^2) - m)\n";
    ld E_relativistic = run_sigma_svm(true);

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