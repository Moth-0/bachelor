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
    std::cout << "=== Deuteron + Sigma Meson SVM Test ===\n";

    // 1. Define Particles & Constants
    Particle neutron("Neutron", 939.0, 0, 0.5, 0.5, 0.5, -0.5);
    Particle proton("Proton", 939.0, 1, 0.5, 0.5, 0.5, 0.5);
    Particle sigma_meson("Sigma", 500.0, 0, 0.0, 0.0, 0.0, 0.0);
    
    Jacobian jac_2body({neutron, proton});
    Jacobian jac_3body({neutron, proton, sigma_meson});
    
    hamiltonian H_op;
    H_op.hbar_c = 197.3269804;
    
    // Physics Parameters for the W coupling
    long double b_sigma = 3.0; // fm
    long double S_sigma = -20.35; // MeV
    long double w = 1.0 / (b_sigma * b_sigma);

    // Build the general Omega kernel matrix for the W-operator
    matrix Omega(jac_3body.dim(), jac_3body.dim());
    Omega(0, 0) = w; // Acts on r_np
    Omega(1, 1) = w; // Acts on r_sigma_np
    Omega(0, 1) = 0.0;
    Omega(1, 0) = 0.0;

    // 2. Setup Basis Parameters
    size_t n_d = 25; 
    size_t n_sigma = 80;
    size_t total_basis = n_d + n_sigma;
    
    long double A_min = 0.001; 
    long double A_max = 10.0;  

    std::cout << "Generating basis (n_d = " << n_d << ", n_sigma = " << n_sigma << ")...\n";

    std::vector<gaus> basis_d;
    std::vector<gaus> basis_sigma;

    for (size_t k = 0; k < n_d; ++k) {
        gaus g(jac_2body.dim(), A_min, A_max);
        for(size_t i=0; i<jac_2body.dim(); i++) { g.s(i, 0) = 0; g.s(i, 1) = 0; g.s(i, 2) = 0; }
        basis_d.push_back(g);
    }
    
    for (size_t k = 0; k < n_sigma; ++k) {
        gaus g(jac_3body.dim(), A_min, A_max);
        for(size_t i=0; i<jac_3body.dim(); i++) { g.s(i, 0) = 0; g.s(i, 1) = 0; g.s(i, 2) = 0; }
        basis_sigma.push_back(g);
    }

    // 3. Construct the Block Matrices
    std::cout << "Building Block Hamiltonian and Overlap matrices...\n";
    matrix H_mat(total_basis, total_basis);
    matrix N_mat(total_basis, total_basis); 

    // Top-Left Block (2-body vs 2-body)
    for (size_t i = 0; i < n_d; ++i) {
        for (size_t j = 0; j < n_d; ++j) {
            H_mat(i, j) = H_op.K_cla(basis_d[i], basis_d[j], jac_2body.c(0), jac_2body.mu(0));
            N_mat(i, j) = overlap(basis_d[i], basis_d[j]);
        }
    }

    // Bottom-Right Block (3-body vs 3-body)
    for (size_t i = 0; i < n_sigma; ++i) {
        for (size_t j = 0; j < n_sigma; ++j) {
            long double K_np = H_op.K_cla(basis_sigma[i], basis_sigma[j], jac_3body.c(0), jac_3body.mu(0));
            long double K_sigma_np = H_op.K_cla(basis_sigma[i], basis_sigma[j], jac_3body.c(1), jac_3body.mu(1));
            long double ov = overlap(basis_sigma[i], basis_sigma[j]);
            
            H_mat(n_d + i, n_d + j) = K_np + K_sigma_np + (sigma_meson.mass * ov);
            N_mat(n_d + i, n_d + j) = ov;
        }
    }

    // Off-Diagonal Blocks (W-coupling)
    for (size_t i = 0; i < n_d; ++i) {
        for (size_t j = 0; j < n_sigma; ++j) {
            // Use the generalized W_transition. For the Sigma meson, we just take the scalar part.
            W_Result w_res = H_op.W_transition(basis_d[i], basis_sigma[j], Omega);
            long double w_val = S_sigma * w_res.scalar_val;
            
            H_mat(i, n_d + j) = w_val;         
            H_mat(n_d + j, i) = w_val;         
        }
    }

    // 4. Solve the Generalized Eigenvalue Problem
    std::cout << "Solving H*c = E*N*c...\n";
    vector energies = solve_generalized_eigenvalue(H_mat, N_mat);

    if (energies.size() > 0) {
        std::cout << "--------------------------------------------------\n";
        std::cout << "Target Deuteron Energy: -2.2000 MeV\n";
        std::cout << "Calculated Energy     : " << std::fixed << std::setprecision(4) << energies[0] << " MeV\n";
        std::cout << "--------------------------------------------------\n";
    } else {
        std::cout << "Solver failed to find eigenvalues.\n";
    }

    return 0;
}