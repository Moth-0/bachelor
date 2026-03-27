#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h" 
#include "sigma.h" 

using namespace qm;

ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    return solve_ground_state_energy(H, N);
}

void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
    ld previous_E = 999999.0;
    ld sweep_tolerance = 1e-4; 
    int max_sweeps = 50;
    int sweep = 0;

    while (sweep < max_sweeps && std::abs(previous_E - current_E) > sweep_tolerance) {
        previous_E = current_E;
        
        for (size_t k = 0; k < basis.size(); ++k) {
            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                unpack_wavefunction(basis[k].psi, p_test);
                
                bool is_physical = true;
                for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
                    if (basis[k].psi.A(i, i) <= 0.02) is_physical = false;
                }
                if (basis[k].psi.A.determinant() <= ZERO_LIMIT) is_physical = false;

                for (size_t i = 0; i < basis[k].psi.s.size1(); ++i) {
                    for (size_t col = 0; col < 3; ++col) {
                        if (std::abs(basis[k].psi.s(i, col)) > 6.0) is_physical = false;
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

ld run_sigma_svm(bool relativistic) {
    // Physical Constants from Fedorov (2020)
    ld m_n = 939.0, m_p = 939.0;    // 
    ld m_sigma = 500.0;             // [cite: 156, 157]
    ld b_range = 3.0, S = 20.35;    // 

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed({m_p, m_n, m_sigma});

    std::vector<BasisState> basis;
    
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

    BasisState dressed_template = {SpatialWavefunction(zeros<ld>(2, 2), zeros<ld>(2, 3), 1), Channel::PN_SIGMA, jac_dressed, m_sigma};

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
                test_candidate.psi.randomize(test_candidate.jac, b_range);
                
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
        
        // Sweep every 5 states to keep it fast
        if (i % 5 == 0 || i == num_dressed - 1) {
            sweep_optimize_basis(basis, b_range, S, relativistic);
        }
        
        std::cout << "\r" << "Added Dressed State " << i+1 << "/" << num_dressed << " -> Energy: " 
                  << std::fixed << std::setprecision(5) << evaluate_basis_energy(basis, b_range, S, relativistic) << " MeV    " << std::flush;
    }
    
    std::cout << "\n";
    return evaluate_basis_energy(basis, b_range, S, relativistic);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  FEDOROV SIGMA-MESON DEUTERON MODEL\n";
    std::cout << "========================================\n\n";

    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY (T = p^2 / 2m)\n";
    ld E_classic = run_sigma_svm(false);
    
    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY (T = sqrt(p^2 + m^2) - m)\n";
    ld E_relativistic = run_sigma_svm(true);

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