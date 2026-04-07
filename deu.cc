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
║     └─→ run_deuteron_svm(false)   [classic kinetic energy]                     ║
║     └─→ run_deuteron_svm(true)    [relativistic kinetic energy]                ║
║          └─→ Phase 1: Skeleton basis (14 states) → sweep_optimize()            ║
║          └─→ Phase 2: Competitive growth (2 cycles) →                          ║
║               For each channel: test 100 candidates →                          ║
║               add best → sweep_optimize()                                      ║
║                                                                                ║
║ KEY CONCEPTS:                                                                  ║
║                                                                                ║
║   1. SKELETON (Phase 1):                                                       ║
║      • 5 deterministic PN states: geometric widths {0.02, 0.08, 0.3, 1.2, 4.0} ║
║        These ensure spatial scales from short-range to long-range are covered. ║
║      • 9 random pion states: one per channel × spin configuration              ║
║        Seed the optimization for each 3-body coupling mode.                    ║
║      • Polish with Nelder-Mead sweeps: optimize all 14 states' A,s params      ║
║                                                                                ║
║   2. GROWTH (Phase 2):                                                         ║
║      • Competitive: each channel independently tests 100 random candidates     ║
║      • Selects the one lowering total system energy the most                   ║
║      • Locked into basis permanently                                           ║
║      • After all channels: full basis sweep to polish all parameters           ║
║                                                                                ║
║      Why competitive?                                                          ║
║        - Prevents basis bloat (only keeps impactful states)                    ║
║        - Each state must justify its computational cost                        ║
║        - Parallelizable via OpenMP (each candidate tested independently)       ║
║                                                                                ║
║ PARAMETER TUNING:                                                              ║
║                                                                                ║
║   b_range = 1.4 fm:                                                            ║
║     • Search space for Gaussian width parameter                                ║
║     • log-uniform sampling: b_ij = -log(u) × b_range                           ║
║     • Larger → explores wider spatial scales                                   ║
║                                                                                ║
║   b_form = 1.4 fm:                                                             ║
║     • Pion interaction range (form factor)                                     ║
║     • f(r) = exp(-r²/b_form²) coupling strength                                ║
║     • Controls how "soft" the pion exchange is                                 ║
║     • 1.4 fm choosen from compton wavelength of pion                           ║
║                                                                                ║
║   S = 140.0:  **CRITICAL TUNING KNOB**                                         ║
║     • Pion coupling strength parameter                                         ║
║     • Higher S → stronger binding → more negative energy                       ║
║     • Tune to match experimental target E = -2.224 MeV                         ║
║     • Example: S=100 might give -1.8 MeV, S=180 might give -2.5 MeV            ║
║                                                                                ║
║   num_cycles = 2:                                                              ║
║     • How many SVM growth phases to run                                        ║
║     • Each cycle: 10 channels × 100 candidates = 1000 evaluations (parallel)   ║
║     • More cycles → more time but better convergence                           ║
║                                                                                ║
║   num_candidates_per_step = 100:                                               ║
║     • Test this many random states per channel per cycle                       ║
║     • Larger → better chance of finding good state, but slower                 ║
║                                                                                ║
║ PHYSICAL PARAMETERS:                                                           ║
║                                                                                ║
║   m_p = 938.27 MeV, m_n = 939.56 MeV:  nucleon masses                          ║
║   m_pi0 = 134.97 MeV, m_pic = 139.57 MeV:  pion masses                         ║
║   iso_c = sqrt(2):  isospin weighting for charged pions                        ║
║                                                                                ║
║ CHANNEL DESCRIPTION:                                                           ║
║                                                                                ║
║   Channel enum defines 10 distinct physics states:                             ║
║     Channel::PN          - bare proton-neutron (no pion)                       ║
║     Channel::PI_0c_0f    - π⁰ without spin flip                                ║
║     Channel::PI_0c_1f    - π⁰ with particle 1 flipped                          ║
║     Channel::PI_0c_2f    - π⁰ with particle 2 flipped                          ║
║     Channel::PI_pc_0f    - π⁺ without spin flip                                ║
║     Channel::PI_pc_1f    - π⁺ with particle 1 flipped                          ║
║     Channel::PI_pc_2f    - π⁺ with particle 2 flipped                          ║
║     Channel::PI_mc_0f    - π⁻ without spin flip                                ║
║     Channel::PI_mc_1f    - π⁻ with particle 1 flipped                          ║
║     Channel::PI_mc_2f    - π⁻ with particle 2 flipped                          ║
║                                                                                ║
║   Note: spin flips on charged pions give different final states due to         ║
║   Pauli matrix structure. All encoded in SpinChannel enum.                     ║
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
#include "deuterium.h" 

using namespace qm;

// Evaluate energy: build H,N and solve GEVP
ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    return solve_ground_state_energy(H, N);
}

// Optimize basis parameters via Nelder-Mead sweeping
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
    
    ld previous_E = 999999.0;
    ld sweep_tolerance = 1e-5; 
    int max_sweeps = 20;       // Hard cap to prevent infinite stalling
    int sweep = 0;

    while (sweep < max_sweeps && std::abs(previous_E - current_E) > sweep_tolerance) {
        previous_E = current_E;
        
        for (size_t k = 0; k < basis.size(); ++k) {
            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                unpack_wavefunction(basis[k].psi, p_test);
                
                // --- THE PHYSICS BOUNCER ---
                bool is_physical = true;

                for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
                    if (basis[k].psi.A(i, i) <= 0.02) is_physical = false; // Grid floor
                }
                if (basis[k].psi.A.determinant() <= ZERO_LIMIT) is_physical = false;

                for (size_t i = 0; i < basis[k].psi.s.size1(); ++i) {
                    for (size_t col = 0; col < 3; ++col) {
                        if (std::abs(basis[k].psi.s(i, col)) > 5.0) is_physical = false; // Range limit
                    }
                }

                if (!is_physical) return 999999.0;
                
                return evaluate_basis_energy(basis, b, S, relativistic);
            };

            rvec p_best = nelder_mead(p0, objective_func);
            unpack_wavefunction(basis[k].psi, p_best);
            current_E = evaluate_basis_energy(basis, b, S, relativistic);
            std::cout << "\r" << "Optimized Energy (Sweep " << sweep << ") = " << current_E << " MeV" << std::flush;
        }
        sweep++;
    }
}

// Run two-phase SVM: skeleton (Phase 1) + competitive growth (Phase 2)
ld run_deuteron_svm(bool relativistic) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;  
    ld m_pi0 = 134.97, m_pic = 139.57; 
    
    ld b_range = 1.4, b_form = 1.4;
    //
    // *** CRITICAL TUNING PARAMETER ***
    // S = coupling strength for pion exchange (in MeV)
    //     Directly controls how strongly pions bind the nucleons
    
    ld S = 140.0;      

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    ld iso_c = std::sqrt(2.0L);

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        {SpatialWavefunction(-1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic}
    };

    // ------- PHASE 1: SKELETON BASIS WITH GEOMETRIC GRID --------
    // Purpose: Build initial 14-state basis with wide spatial coverage
    //
    // Strategy: 5 explicit geometric PN states + 9 random pion channels
    
    std::cout << "--- 1. Planting Geometric PN Grid & Pion Seeds ---\n";
    
    // The Anchor: 5 deterministic PN states
    std::vector<ld> deterministic_widths = {0.02, 0.08, 0.3, 1.2, 4.0};
    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * width;
        rmat s_fixed = zeros<ld>(1, 3);
        SpatialWavefunction psi_fixed(A_fixed, s_fixed, 1);
        basis.push_back({psi_fixed, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    // The Cloud: 9 random Pion seeds
    for (size_t t = 1; t < channel_templates.size(); ++t) {
        BasisState seed = channel_templates[t];
        Gaussian g;
        g.randomize(seed.jac, b_range);
        seed.psi.set_from_gaussian(g);
        basis.push_back(seed);
    }
    
    std::cout << "Skeleton Size: " << basis.size() << " states.\n";
    sweep_optimize_basis(basis, b_form, S, relativistic);
    std::cout << "\nSkeleton Energy: " << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV\n\n";

    // ------- PHASE 2: COMPETITIVE SVM GROWTH (INCREMENTAL BASIS EXPANSION) -------
    // Purpose: Grow basis incrementally with best candidates, avoiding bloat
    //
    // Algorithm per cycle:
    //   1. For each of 10 pion channels (π⁰, π⁺, π⁻ × 3 spin flips):
    //      - Generate 100 random candidate states independently (parallel OpenMP)
    //      - Evaluate each: add to basis temporarily → solve GEVP → remove
    //      - Select candidate with LOWEST energy
    //      - Lock it permanently into the basis
    //   2. After all channels: sweep-optimize the expanded basis for polish
    
    int num_cycles = 2; 
    int num_candidates_per_step = 100;

    std::cout << "--- 2. Competitive SVM Growth ---\n";
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        
        // Loop through all 10 channels, adding 1 state to each
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

            // Lock in the winner
            basis.push_back(best_candidate);

            std::cout << "\rAdded State " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> E = " 
                      << std::fixed << std::setprecision(5) << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV    " << std::flush;
        }

        // ONE massive polish per cycle
        std::cout << "\n - Sweeping Cycle " << cycle+1 << " basis -\n";
        sweep_optimize_basis(basis, b_form, S, relativistic);
    }
    
    std::cout << "\n";
    return evaluate_basis_energy(basis, b_form, S, relativistic);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n\n";

    // Run with both kinetic energy models

    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY\n";
    ld E_classic = run_deuteron_svm(false);

    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY\n";
    ld E_relativistic = run_deuteron_svm(true);
    

    ld diff = E_relativistic - E_classic;

    // --- COMPARISON & INTERPRETATION ---
    std::cout << "========================================\n";
    std::cout << "  FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Classic Energy:      " << E_classic << " MeV\n";
    std::cout << "Relativistic Energy: " << E_relativistic << " MeV\n";
    std::cout << "Difference (rel-cl): " << diff << " MeV\n";
    std::cout << "========================================\n";
    std::cout << "\nInterpretation:\n";
    std::cout << "  Target experimental value: E = -2.224 MeV\n";
    std::cout << "  Difference shows relativistic correction (typically 3-10% binding)\n";
    std::cout << "  If results too high: increase S parameter and re-run\n";
    std::cout << "  If results too low: decrease S parameter and re-run\n";
    std::cout << "========================================\n";

    return 0;
}