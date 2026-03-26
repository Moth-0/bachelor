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

ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    return solve_ground_state_energy(H, N);
}

void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
    
    ld previous_E = 999999.0;
    ld sweep_tolerance = 1e-5; 
    int max_sweeps = 500;
    int sweep = 0;

    while (sweep < max_sweeps && std::abs(previous_E - current_E) > sweep_tolerance) {
        previous_E = current_E;
        
        for (size_t k = 0; k < basis.size(); ++k) {
            
            // NO SKIPPING! Every state (bare and dressed) must be optimized!
            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                unpack_wavefunction(basis[k].psi, p_test);
                
                // --- THE PHYSICS BOUNCER ---
                bool is_physical = true;

                for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
                    // CRITICAL FIX: The floor is now 0.02! 
                    // This prevents the nucleus from expanding beyond ~7 fm.
                    if (basis[k].psi.A(i, i) <= 0.02) is_physical = false;
                }
                if (basis[k].psi.A.determinant() <= ZERO_LIMIT) is_physical = false;

                for (size_t i = 0; i < basis[k].psi.s.size1(); ++i) {
                    for (size_t col = 0; col < 3; ++col) {
                        // Tighten the shift limit to 5.0 fm. Nucleons don't jump further than this!
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

// --- THE "SKELETON + COMPETITIVE SVM" WRAPPER ---
ld run_deuteron_svm(bool relativistic) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;  
    ld m_pi0 = 134.97, m_pic = 139.57; 
    
    ld b_range = 1.4, b_form = 1.4;
    // TUNE THIS 'S' PARAMETER TO HIT -2.224 MeV!
    ld S = 200.0;      

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    ld iso_c = std::sqrt(2.0L); 

    rmat dummy_bare_A = zeros<ld>(1, 1);
    rmat dummy_bare_s = zeros<ld>(1, 3);
    rmat dummy_dress_A = zeros<ld>(2, 2);
    rmat dummy_dress_s = zeros<ld>(2, 3);

    // ALL 10 channels defined cleanly
    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(dummy_bare_A, dummy_bare_s, 1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_mc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_mc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(dummy_dress_A, dummy_dress_s, -1), Channel::PI_mc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic}
    };

    // -------------------------------------------------------------
    // PHASE 1: PLANT THE 10-STATE SKELETON
    // -------------------------------------------------------------
    std::cout << "--- 1. Planting 10-State Skeleton ---\n";
    for (size_t t = 0; t < channel_templates.size(); ++t) {
        BasisState seed = channel_templates[t];
        seed.psi.randomize(seed.jac, b_range);
        basis.push_back(seed);
    }
    
    // Let Nelder-Mead settle the initial skeleton into the best possible configuration
    sweep_optimize_basis(basis, b_form, S, relativistic);
    std::cout << "Skeleton Energy: " << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV\n\n";

    // -------------------------------------------------------------
    // PHASE 2: COMPETITIVE SVM GROWTH
    // -------------------------------------------------------------
    int num_cycles = 1; // Will add more states per channel
    int num_candidates_per_step = 10;

    std::cout << "--- 2. Competitive SVM Growth ---\n";
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        for (size_t t = 0; t < channel_templates.size(); ++t) {
            
            BasisState best_candidate = channel_templates[t];
            ld best_E = 999999.0;

            // --> OPENMP PARALLEL CANDIDATE SEARCH <--
            #pragma omp parallel 
            {
                // Thread-local variables
                BasisState local_best_candidate = channel_templates[t];
                ld local_best_E = 999999.0;
                
                // Each thread needs its own copy of the basis so they don't crash into each other
                std::vector<BasisState> local_basis = basis; 
                
                #pragma omp for
                for (int c = 0; c < num_candidates_per_step; ++c) {
                    BasisState test_candidate = channel_templates[t];
                    test_candidate.psi.randomize(test_candidate.jac, b_range);
                    
                    local_basis.push_back(test_candidate);
                    ld E = evaluate_basis_energy(local_basis, b_form, S, relativistic);
                    local_basis.pop_back();

                    if (E < local_best_E) {
                        local_best_E = E;
                        local_best_candidate = test_candidate;
                    }
                }

                // Only 1 thread at a time is allowed to update the global best!
                #pragma omp critical
                {
                    if (local_best_E < best_E) {
                        best_E = local_best_E;
                        best_candidate = local_best_candidate;
                    }
                }
            }

            // Now push the ultimate winner into the real, shared basis!
            basis.push_back(best_candidate);
            sweep_optimize_basis(basis, b_form, S, relativistic);

            std::cout << "\rTotal States: " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> Energy: " 
                      << std::fixed << std::setprecision(5) << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV    " << std::flush;
        }
        std::cout << "\n"; // New line after a full 10-channel cycle
    }
    
    std::cout << "\n";
    return evaluate_basis_energy(basis, b_form, S, relativistic);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (COMPETITIVE SVM)\n";
    std::cout << "========================================\n\n";

    // --- RUN 1: CLASSIC ---
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY\n";
    ld E_classic = run_deuteron_svm(false);
    
    // --- RUN 2: RELATIVISTIC ---
    std::cout << ">>> RUNNING RELATIVISTIC KINETIC ENERGY\n";
    ld E_relativistic = run_deuteron_svm(true);

    // --- COMPARISON ---
    std::cout << "========================================\n";
    std::cout << "  FINAL RESULTS COMPARISON\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Classic Energy:      " << E_classic << " MeV\n";
    std::cout << "Relativistic Energy: " << E_relativistic << " MeV\n";
    
    ld diff = std::abs(E_classic - E_relativistic);
    ld percent_dev = (diff / std::abs(E_classic)) * 100.0;
    
    std::cout << "Difference:          " << diff << " MeV\n";
    std::cout << "Deviation:           " << percent_dev << " %\n";
    std::cout << "========================================\n";

    return 0;
}