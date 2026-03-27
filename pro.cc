#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h" // Assuming nelder_mead is in here or matrix.h
#include "proton.h" 

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

// --- NEW WRAPPER FUNCTION ---
// Runs the entire competitive construction and returns the final energy
ld run_proton_svm(bool relativistic) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;  
    ld m_pi0 = 134.97, m_pic = 139.57; 
    ld b_range = 1.4, b_form = 1.4, S = 140.0;      

    Jacobian jac_bare({m_p});
    Jacobian jac_dressed_0({m_p, m_pi0});
    Jacobian jac_dressed_c({m_n, m_pic});

    std::vector<BasisState> basis;
    
    // 1. INITIALIZE THE BARE STATE
    rmat empty_A = zeros<ld>(0, 0);
    rmat empty_s = zeros<ld>(0, 3);
    SpatialWavefunction psi_bare(empty_A, empty_s, 1);
    
    basis.push_back({psi_bare, Channel::P, NO_FLIP, 1.0, jac_bare, 0.0});

    // 2. COMPETITIVE BASIS CONSTRUCTION
    int num_dressed_per_channel = 6; 
    int num_candidates_per_step = 20; 
    ld iso_c = std::sqrt(2.0L); 

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(zeros<ld>(1, 1), zeros<ld>(1, 3), -1), Channel::P_PI0_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(zeros<ld>(1, 1), zeros<ld>(1, 3), -1), Channel::P_PI0_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(zeros<ld>(1, 1), zeros<ld>(1, 3), -1), Channel::N_PIP_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(zeros<ld>(1, 1), zeros<ld>(1, 3), -1), Channel::N_PIP_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic}
    };

    for (int i = 0; i < num_dressed_per_channel; ++i) {
        for (size_t t = 0; t < channel_templates.size(); ++t) {
            
            BasisState best_candidate = channel_templates[t];
            ld best_E = 999999.0;

            for (int c = 0; c < num_candidates_per_step; ++c) {
                BasisState test_candidate = channel_templates[t];
                test_candidate.psi.randomize(test_candidate.jac, b_range);
                
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

    // --- RUN 1: CLASSIC ---
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY (T = p^2 / 2m)\n";
    ld E_classic = run_proton_svm(false);
    
    // --- RUN 2: RELATIVISTIC ---
    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY (T = sqrt(p^2 + m^2) - m)\n";
    ld E_relativistic = run_proton_svm(true);

    // --- COMPARISON ---
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