#include <iostream>
#include <vector>
#include <random>
#include "matrix.h"
#include "eigen.h"
#include "gaussian.h"
#include "hamiltonian.h"
#include "jacobian.h"

// General Solver: Normalizes matrices to prevent floating-point collapse, then solves safely
long double get_ground_state_energy(const qm::matrix& H, const qm::matrix& N) {
    // ... (Your normalization loop stays exactly the same) ...
    
    // 2. Cholesky-factorization 
    qm::matrix L = qm::cholesky(N);
    
    // Catch the silent failure! If Cholesky returned an empty matrix, reject the step.
    if (L.size1() == 0) {
        return std::numeric_limits<long double>::quiet_NaN();
    }

    // Since L is valid, we can safely invert it
    qm::matrix L_inv = L.inverse(); 
    qm::matrix H_prime = L_inv * H * L_inv.transpose();

    // 3. Solve for eigenvalues
    qm::vector energies = qm::jacobi_eigenvalues(H_prime);

    // 4. Find the lowest energy
    long double E_0 = energies[0];
    for (size_t i = 1; i < energies.size(); i++) {
        if (energies[i] < E_0) E_0 = energies[i];
    }
    
    return E_0;
}

int main() {
    // Particle masses in MeV
    long double m_p = 938.27208816; 
    long double m_e = 0.51099895; 
    bool USE_REL = true;

    // System: 1 Proton (+1), 1 Electron (-1)
    qm::jacobian J_hyd({m_p, m_e}, {1, -1});
    qm::hamiltonian h_calc;

    size_t n = 10; // Number of Gaussians
    std::vector<qm::gaus> basis(n);

    long double E_best = std::numeric_limits<long double>::quiet_NaN();
    qm::matrix H(n, n);
    qm::matrix N(n, n);

    std::cout << "Searching for a valid initial basis..." << std::endl;

    // --- The "Keep Rolling" Loop ---
    while (std::isnan(E_best)) {
        // Generate a totally random basis
        for(size_t i = 0; i < n; i++) basis[i] = qm::gaus(1, 1e-5, 4e-5);

        // Build and Test
        N = h_calc.overlap_matrix(basis);
        H = h_calc.hamiltonian_matrix(basis, J_hyd, USE_REL); // false = Classical Kinetic Energy
        E_best = get_ground_state_energy(H, N);
    }

    std::cout << "Initial Energy: " << E_best << " MeV\n" << std::endl;

    // Random setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_n(0, n - 1);

    int max_iterations = 10000; // Let it run a bit longer for fine-tuning

    // --- The Stochastic Loop ---
    for (int step = 1; step <= max_iterations; step++) {
        int k = dist_n(gen);
        qm::gaus old_g = basis[k];
        
        basis[k] = qm::gaus(1, 1e-5, 4e-5); // Mutate 1D channel
        
        N = h_calc.overlap_matrix(basis);
        H = h_calc.hamiltonian_matrix(basis, J_hyd, USE_REL);
        long double E_trial = get_ground_state_energy(H, N);
        
        if (E_trial < E_best && !std::isnan(E_trial)) {
            E_best = E_trial;
            std::cout << "Step " << step << " | New Best Energy: " << E_best << " MeV" << std::endl;
        } else {
            basis[k] = old_g; // Revert
        }
    }

    std::cout << "\nOptimization Complete.\nFinal Ground State Energy: " << E_best << " MeV\n";
    return 0;
}