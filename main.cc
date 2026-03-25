#include "qm/matrix.h"
#include "qm/solver.h"
#include "hamiltonian.h"
#include <iostream>
#include <random>

using namespace qm;

ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    // 1. Build the H and N matrices
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    
    // 2. Solve the Generalized Eigenvalue Problem (Hc = E Nc)
    ld E_0 = solve_ground_state_energy(H, N);
    
    return E_0;
}

void optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic, int iterations) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);
    std::cout << "Starting Energy: " << current_E << " MeV\n";

    std::mt19937 rng(42); // Standard random number generator
    std::uniform_real_distribution<ld> dist_shift(-1.0, 1.0); // For tweaking s
    std::uniform_real_distribution<ld> dist_width(0.9, 1.1);  // For scaling A

    for (int iter = 0; iter < iterations; ++iter) {
        
        // Pick a random state in your basis to optimize
        size_t target_idx = rng() % basis.size();
        BasisState backup_state = basis[target_idx];

        // --- STOCHASTIC TWEAKING ---
        // 1. Maybe tweak the A matrix (scaling the widths slightly)
        for (size_t i = 0; i < basis[target_idx].psi.A.size1(); ++i) {
            basis[target_idx].psi.A(i, i) *= dist_width(rng);
        }

        // 2. Maybe tweak the shift vector (moving the Gaussians around)
        for (size_t i = 0; i < basis[target_idx].psi.s.size1(); ++i) {
            for (size_t col = 0; col < 3; ++col) {
                basis[target_idx].psi.s(i, col) += dist_shift(rng);
            }
        }

        // --- EVALUATE ---
        ld new_E = evaluate_basis_energy(basis, b, S, relativistic);

        // --- ACCEPT OR REJECT ---
        if (new_E < current_E) {
            // The random tweak lowered the energy! Keep it.
            current_E = new_E;
            std::cout << "\r" << "Iteration " << iter << " | New Min Energy: " << current_E << " MeV" << std::flush;
        } else {
            // The tweak made it worse (or caused linear dependence). Revert!
            basis[target_idx] = backup_state;
        }
    }
    
    std::cout << "Optimization Complete. Final Energy: " << current_E << " MeV\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  FULL 10-CHANNEL DEUTERON SVM \n";
    std::cout << "========================================\n\n";

    // 1. Physical Constants (MeV and fm)
    ld m_p = 938.27;  
    ld m_n = 939.56;  
    ld m_pi0 = 134.97; // Neutral pion mass
    ld m_pic = 139.57; // Charged pion mass
    
    ld b_range = 1.4;  // Physics range for basis generation
    ld b_form = 1.4;   // Form factor b parameter
    ld S = 10.0;      // Starting guess for coupling strength
    bool relativistic = false; 

    // Jacobians for bare (2-body) and dressed (3-body)
    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    int num_states_per_channel = 5; // Start small! 5 * 10 = 50x50 matrix
    
    std::cout << "Initializing Basis: " << num_states_per_channel << " states per channel...\n";

    rmat empty_A, empty_s; // Dummies to pass to constructor before randomizing

    // 2. Build the Full 10-Channel System
    for (int i = 0; i < num_states_per_channel; ++i) {
        
        // -------------------------------------------------------------
        // CHANNEL 1: Bare pn State (Parity +1)
        // -------------------------------------------------------------
        SpatialWavefunction psi_bare(empty_A, empty_s, 1);
        psi_bare.randomize(jac_bare, b_range);
        basis.push_back({psi_bare, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});

        // -------------------------------------------------------------
        // CHANNELS 2-4: Neutral Pions (pi^0) | Isospin = 1.0
        // -------------------------------------------------------------
        SpatialWavefunction psi_pi0_0f(empty_A, empty_s, -1);
        psi_pi0_0f.randomize(jac_dressed_0, b_range);
        basis.push_back({psi_pi0_0f, Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0});

        SpatialWavefunction psi_pi0_1f(empty_A, empty_s, -1);
        psi_pi0_1f.randomize(jac_dressed_0, b_range);
        basis.push_back({psi_pi0_1f, Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0});

        SpatialWavefunction psi_pi0_2f(empty_A, empty_s, -1);
        psi_pi0_2f.randomize(jac_dressed_0, b_range);
        basis.push_back({psi_pi0_2f, Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0});

        // -------------------------------------------------------------
        // CHANNELS 5-7: Positive Pions (pi^+) | Isospin = sqrt(2)
        // -------------------------------------------------------------
        ld iso_c = std::sqrt(2.0L);

        SpatialWavefunction psi_pip_0f(empty_A, empty_s, -1);
        psi_pip_0f.randomize(jac_dressed_c, b_range);
        basis.push_back({psi_pip_0f, Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic});

        SpatialWavefunction psi_pip_1f(empty_A, empty_s, -1);
        psi_pip_1f.randomize(jac_dressed_c, b_range);
        basis.push_back({psi_pip_1f, Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic});

        SpatialWavefunction psi_pip_2f(empty_A, empty_s, -1);
        psi_pip_2f.randomize(jac_dressed_c, b_range);
        basis.push_back({psi_pip_2f, Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic});

        // -------------------------------------------------------------
        // CHANNELS 8-10: Negative Pions (pi^-) | Isospin = sqrt(2)
        // -------------------------------------------------------------
        SpatialWavefunction psi_pim_0f(empty_A, empty_s, -1);
        psi_pim_0f.randomize(jac_dressed_c, b_range);
        basis.push_back({psi_pim_0f, Channel::PI_mc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic});

        SpatialWavefunction psi_pim_1f(empty_A, empty_s, -1);
        psi_pim_1f.randomize(jac_dressed_c, b_range);
        basis.push_back({psi_pim_1f, Channel::PI_mc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic});

        SpatialWavefunction psi_pim_2f(empty_A, empty_s, -1);
        psi_pim_2f.randomize(jac_dressed_c, b_range);
        basis.push_back({psi_pim_2f, Channel::PI_mc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic});
    }

    // 3. Run the Optimizer!
    int iterations = 10000;
    std::cout << "Starting optimization on " << basis.size() << "x" << basis.size() << " Hamiltonian with " << iterations << " steps...\n";
    
    // Make sure optimize_basis is updated to use the tuple-returning build_matrices!
    optimize_basis(basis, b_form, S, relativistic, iterations);

    std::cout << "\n========================================\n";
    std::cout << "  OPTIMIZATION COMPLETE\n";
    std::cout << "========================================\n";

    return 0;
}