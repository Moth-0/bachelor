#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#include "qm/matrix.h"
#include "qm/particle.h"
#include "qm/jacobian.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/eigen.h"

using namespace qm;

int main() {
    std::cout << "=== Hydrogen Atom SVM Test (MeV & fm) ===\n";

    // 1. Define Particles & Constants in MeV and fm
    Particle proton("Proton", 938.272, 1, 0.5, 0.5, 0.5, 0.5);
    Particle electron("Electron", 0.51099895, -1, 0.5, 0.5, 0.0, 0.0);
    
    Jacobian jac({proton, electron});
    hamiltonian H_op;
    H_op.hbar_c = 197.3269804; // Ensure this is set to MeV * fm
    
    long double alpha = 1.0 / 137.035999;
    long double coulomb_const = -alpha * H_op.hbar_c;

    // 2. Setup Basis Parameters
    size_t basis_size = 15;
    std::vector<gaus> basis;
    
    // In fm, Bohr radius is ~52900 fm. A ~ 1/r^2. 
    // We search extremely small A values.
    long double A_min = 1e-12; 
    long double A_max = 1e-6;  

    std::cout << "Generating basis of size " << basis_size << "...\n";

    for (size_t k = 0; k < basis_size; ++k) {
        gaus best_g;
        long double best_E = 1e9; 
        bool found_valid = false;

        for (int trial = 0; trial < 500; ++trial) {
            gaus g_trial(jac.dim(), A_min, A_max);
            
            // Force s-wave (no shift)
            for(size_t i=0; i<jac.dim(); i++) {
                g_trial.s(i, 0) = 0.0; g_trial.s(i, 1) = 0.0; g_trial.s(i, 2) = 0.0;
            }

            // --- LINEAR DEPENDENCE CHECK ---
            // Ensure the new Gaussian is not too similar to existing ones
            bool too_similar = false;
            long double norm_trial = overlap(g_trial, g_trial);
            
            for (const auto& b : basis) {
                long double norm_b = overlap(b, b);
                long double ov = std::abs(overlap(g_trial, b));
                long double normalized_overlap = ov / std::sqrt(norm_trial * norm_b);
                
                if (normalized_overlap > 0.95) { // 95% similarity threshold
                    too_similar = true;
                    break;
                }
            }
            if (too_similar) continue; // Skip if it causes ill-conditioning

            // Evaluate isolated energy
            long double k_val = H_op.K_cla(g_trial, g_trial, jac.c(0), jac.mu(0));
            long double v_val = coulomb_const * H_op.V_cou(g_trial, g_trial, jac.c(0));
            long double e_trial = (k_val + v_val) / norm_trial;
            
            if (e_trial < best_E) {
                best_E = e_trial;
                best_g = g_trial;
                found_valid = true;
            }
        }
        
        if (found_valid) {
            basis.push_back(best_g);
        } else {
            std::cerr << "Warning: Could not find a distinct Gaussian for basis index " << k << "\n";
            basis_size = k; // Shrink basis size to what we successfully found
            break;
        }
    }

    // 3. Construct H and N Matrices
    std::cout << "Building Hamiltonian and Overlap matrices...\n";
    matrix H_mat(basis_size, basis_size);
    matrix N_mat(basis_size, basis_size);

    for (size_t i = 0; i < basis_size; ++i) {
        for (size_t j = 0; j < basis_size; ++j) {
            long double k_val = H_op.K_cla(basis[i], basis[j], jac.c(0), jac.mu(0));
            long double v_val = coulomb_const * H_op.V_cou(basis[i], basis[j], jac.c(0));
            
            H_mat(i, j) = k_val + v_val;
            N_mat(i, j) = overlap(basis[i], basis[j]);
        }
    }

    // 4. Solve the Generalized Eigenvalue Problem
    std::cout << "Solving H*c = E*N*c...\n";
    vector energies = solve_generalized_eigenvalue(H_mat, N_mat);

    // 5. Output Results
    if (energies.size() > 0) {
        std::cout << "--------------------------------------------------\n";
        // Convert the target from eV to MeV
        std::cout << "Analytic Ground State Energy : -0.000013598 MeV\n";
        std::cout << "Simulated Ground State Energy: " << std::fixed << std::setprecision(9) << energies[0] << " MeV\n";
        std::cout << "--------------------------------------------------\n";
    } else {
        std::cout << "Solver failed to find eigenvalues.\n";
    }

    return 0;
}