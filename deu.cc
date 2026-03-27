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

// The tightened, high-speed sweep optimizer
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
    
    ld previous_E = 999999.0;
    ld sweep_tolerance = 1e-4; // Loosened slightly for tuning speed
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

// --- THE FAST CYCLE SVM WRAPPER ---
ld run_deuteron_svm(bool relativistic) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;  
    ld m_pi0 = 134.97, m_pic = 139.57; 
    
    ld b_range = 1.4, b_form = 1.4;
    // TUNE THIS 'S' PARAMETER TO HIT -2.224 MeV!
    ld S = 140.0;      

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    ld iso_c = std::sqrt(2.0L); 

    rmat dummy_bare_A = zeros<ld>(1, 1);
    rmat dummy_bare_s = zeros<ld>(1, 3);
    rmat dummy_dress_A = zeros<ld>(2, 2);
    rmat dummy_dress_s = zeros<ld>(2, 3);

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
    // PHASE 1: SKELETON WITH GEOMETRIC GRID
    // -------------------------------------------------------------
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
        seed.psi.randomize(seed.jac, b_range);
        basis.push_back(seed);
    }
    
    std::cout << "Skeleton Size: " << basis.size() << " states.\n";
    sweep_optimize_basis(basis, b_form, S, relativistic);
    std::cout << "\nSkeleton Energy: " << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV\n\n";

    // -------------------------------------------------------------
    // PHASE 2: COMPETITIVE SVM GROWTH (SWEEP PER CYCLE)
    // -------------------------------------------------------------
    int num_cycles = 3; 
    int num_candidates_per_step = 50;

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
                    // IMPORTANT: Ensure your randomize function has the thread_local Van der Corput!
                    test_candidate.psi.randomize(test_candidate.jac, b_range);
                    
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

            // Lock in the winner, but DO NOT SWEEP YET!
            basis.push_back(best_candidate);
            std::cout << "\n" << "\rAdded State " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> E = " 
                      << std::fixed << std::setprecision(5) << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV    " << std::flush;
        }

        // ONE massive polish per cycle
        std::cout << "\n - Sweeping Cycle " << cycle+1 << " basis -\n";
        sweep_optimize_basis(basis, b_form, S, relativistic);
        //std::cout << "  -> Cycle " << cycle+1 << " Polished Energy: " << evaluate_basis_energy(basis, b_form, S, relativistic) << " MeV\n";
    }
    
    std::cout << "\n";
    return evaluate_basis_energy(basis, b_form, S, relativistic);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n\n";

    // --- RUN 1: CLASSIC ---
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY\n";
    ld E_classic = run_deuteron_svm(false);
    
    // --- RUN 2: RELATIVISTIC ---
    // Commented out during tuning to prevent Gauss-Legendre aliasing explosions!
    
    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY\n";
    ld E_relativistic = run_deuteron_svm(true);
    

    ld diff = E_relativistic - E_classic;

    // --- COMPARISON ---
    std::cout << "========================================\n";
    std::cout << "  FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Classic Energy:      " << E_classic << " MeV\n";
    std::cout << "Relativistic Energy  " << E_relativistic << " MeV\n";
    std::cout << "Difference           " << diff << " Mev\n";
    std::cout << "========================================\n";

    return 0;
}