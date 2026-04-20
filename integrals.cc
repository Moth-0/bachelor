#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <omp.h>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "deuterium.h" 
#include "SVM.h"

using namespace qm;

// Lightweight energy evaluator for the competitive search phase
ld evaluate_energy(const std::vector<BasisState>& basis, ld b, ld S, const std::vector<bool>& rel, Integrator method) {
    auto [H, N] = build_matrices(basis, b, S, rel, method);
    cmat L = N.cholesky();
    
    if (L.size1() == 0) return 999999.0;
    for (size_t i = 0; i < L.size1(); ++i) {
        if (std::abs(L(i, i)) < 1e-6) return 999999.0; // Linear dependence check
    }
    return solve_ground_state_energy(H, N);
}

int main() {
    // Default parameters
    ld b_range = 2.0;
    ld b_form = 0.8;
    ld S = 55.4;
    std::vector<bool> rel_flags = {true, true}; // Test relativistic mode

    std::cout << "\n=================================================================\n";
    std::cout << "           DYNAMIC BASIS INTEGRATOR ACCURACY TEST                \n";
    std::cout << "=================================================================\n";

    // 1. Setup physical constants and jacobians
    ld m_p = 938.27, m_n = 939.56, m_pi0 = 134.97, m_pic = 139.57;
    ld iso_c = std::sqrt(2.0L);
    
    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});
    
    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        
        {SpatialWavefunction(-1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        
        {SpatialWavefunction(-1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        
        {SpatialWavefunction(-1), Channel::PI_mc_0f, NO_FLIP, -iso_c, jac_dressed_c, m_pic}, 
        {SpatialWavefunction(-1), Channel::PI_mc_1f, FLIP_PARTICLE_1, -iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_2f, FLIP_PARTICLE_2, -iso_c, jac_dressed_c, m_pic}
    };

    std::vector<BasisState> dynamic_basis;

    // --- PHASE 1: Build the optimized basis using Simpson's Rule ---
    std::cout << "Building optimized test basis...\n";
    
    // Plant Geometric PN Grid
    std::vector<ld> deterministic_widths = {0.05, 0.3, 1.2, 2.0};
    for (ld width : deterministic_widths) {
        SpatialWavefunction psi_fixed(eye<ld>(1) * width, zeros<ld>(1, 3), 1);
        dynamic_basis.push_back({psi_fixed, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    int candidates = 200;
    int cycle = 5;
    for (int c=0; c<cycle; c++){
    for (size_t t = 0; t < channel_templates.size(); ++t) {
        BasisState best_cand = channel_templates[t];
        ld best_E = 999999.0;

        #pragma omp parallel
        {
            BasisState local_best = channel_templates[t];
            ld local_E = 999999.0;
            std::vector<BasisState> local_basis = dynamic_basis;

            #pragma omp for
            for (int c = 0; c < candidates; ++c) {
                BasisState test_cand = channel_templates[t];
                Gaussian g; g.randomize(test_cand.jac, b_range, b_form);
                test_cand.psi.set_from_gaussian(g);
                
                local_basis.push_back(test_cand);
                // Use Simpson's rule for the search
                ld E = evaluate_energy(local_basis, b_form, S, rel_flags, Integrator::GAUSS_LEGENDRE);
                local_basis.pop_back();

                if (E < local_E) {
                    local_E = E;
                    local_best = test_cand;
                }
            }
            #pragma omp critical
            {
                if (local_E < best_E) {
                    best_E = local_E;
                    best_cand = local_best;
                }
            }
        }
        dynamic_basis.push_back(best_cand);
    }}

    std::cout << "Basis generated: " << dynamic_basis.size() << " states spanning all channels.\n\n";

    // --- PHASE 2: Evaluate the exact same frozen basis with all integrators ---
    std::vector<std::pair<std::string, Integrator>> methods = {
        {"Gauss-Legendre (64-pt)", Integrator::GAUSS_LEGENDRE},
        {"Simpson's 1/3 (4000-pt)", Integrator::SIMPSON},
        {"Adaptive Recursive", Integrator::ADAPTIVE_RECURSIVE}
    };

    std::cout << std::left << std::setw(26) << "Integration Method" 
              << " | " << std::setw(12) << "Total E (MeV)" 
              << " | " << std::setw(12) << "<T> (MeV)" 
              << " | " << std::setw(12) << "Rc (fm)"
              << " | " << "Time (ms)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& method : methods) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Build matrices
        auto [H, N] = build_matrices(dynamic_basis, b_form, S, rel_flags, method.second);
        
        // Solve GEVP
        SvmResult result = evaluate_observables(dynamic_basis, b_form, S, rel_flags);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << std::left << std::setw(26) << method.first 
                  << " | " << std::setw(13) << std::fixed << std::setprecision(6) << result.energy 
                  << " | " << std::setw(12) << result.avg_kinetic_energy
                  << " | " << std::setw(12) << result.charge_radius
                  << " | " << duration.count() << " ms\n";
    }
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}