#include<iostream>
#include<vector>
#include<random>
#include"qm/matrix.h"
#include"qm/eigen.h"
#include"qm/gaussian.h"
#include"qm/hamiltonian.h"
#include"qm/jacobian.h"

// System Builder: Fills the total H and N block matrices for the coupled pn / pns system
void build_coupled_matrices(
    const std::vector<qm::gaus>& basis_pn, 
    const std::vector<qm::gaus>& basis_pns, 
    qm::hamiltonian& h_calc, 
    const qm::jacobian& J_pn, 
    const qm::jacobian& J_pns,
    qm::matrix& H_tot, 
    qm::matrix& N_tot,
    long double S,
    long double b,
    long double m_s,
    bool relativistic) 
{
    size_t n1 = basis_pn.size();
    size_t n2 = basis_pns.size();

    // --- 1. Fill Block 1,1 (Deuterium Channel) ---
    qm::matrix N_pn = h_calc.overlap_matrix(basis_pn);
    qm::matrix H_pn = h_calc.hamiltonian_matrix(basis_pn, J_pn, false); // false = classic
    FOR_MAT(H_pn) {
        N_tot(i, j) = N_pn(i, j);
        H_tot(i, j) = H_pn(i, j);
    }

    // --- 2. Fill Block 2,2 (Pion-Deuteron Channel) ---
    qm::matrix N_pns = h_calc.overlap_matrix(basis_pns);
    qm::matrix H_pns = h_calc.hamiltonian_matrix(basis_pns, J_pns, relativistic); 
    FOR_MAT(H_pns) {
        N_tot(n1 + i, n1 + j) = N_pns(i, j);
        H_tot(n1 + i, n1 + j) = H_pns(i, j) + (m_s * N_pns(i, j));
    }

    // --- 3. Fill Blocks 1,2 and 2,1 (The Coupling W) ---
    for(size_t i = 0; i < n1; i++) {
        for(size_t j = 0; j < n2; j++) {
            size_t row_2 = j + n1; 
            long double w_val = h_calc.W_couple(basis_pn[i], basis_pns[j], S, b);
            H_tot(i, row_2) = w_val;
            H_tot(row_2, i) = w_val;
        }
    }
}

// General Solver: Takes ANY H and N matrices and returns the lowest eigenvalue safely
long double get_ground_state_energy(const qm::matrix& H, const qm::matrix& N) {
    // 1. Cholesky-factorization
    qm::matrix L = qm::cholesky(N);
    
    // SAFETY CATCH: If matrix is bad, reject step instantly
    if (L.size1() == 0) {
        return std::numeric_limits<long double>::quiet_NaN();
    }

    qm::matrix L_inv = L.inverse(); 
    qm::matrix H_prime = L_inv * H * L_inv.transpose();

    // 2. Solve for eigenvalues
    qm::vector energies = qm::jacobi_eigenvalues(H_prime);

    // 3. Find the lowest energy
    long double E_0 = energies[0];
    for (size_t i = 1; i < energies.size(); i++) {
        if (energies[i] < E_0) E_0 = energies[i];
    }
    
    return E_0;
}

int main() {
    // Particle masses in MeV
    long double m_n  = 939.0; 
    long double m_p  = 939.0; 
    long double m_s = 500.0;

    // Coupeling variables 
    long double S = 20.35;  // MeV
    long double b = 1.5;    // fm

    // Relativistic? 
    bool r = true;

    qm::jacobian J_pn({m_p, m_n}, {1, 0});
    qm::jacobian J_pns({m_p, m_n, m_s}, {1, 0, 0});
    qm::hamiltonian h_calc;

    // --- The Search Range Variables ---
    long double min_A = 1e-2;
    long double max_A = 1e2;

    size_t n1 = 8;
    size_t n2 = 10;
    size_t N_tot = n1 + n2;

    std::vector<qm::gaus> basis_pn(n1);
    std::vector<qm::gaus> basis_pns(n2);
    for(size_t i = 0; i < n1; i++) basis_pn[i] = qm::gaus(1, min_A, max_A);
    for(size_t i = 0; i < n2; i++) basis_pns[i] = qm::gaus(2, min_A, max_A);

    // Build Hamiltonian and overlap matrices
    qm::matrix H(N_tot, N_tot);
    qm::matrix N(N_tot, N_tot);

    long double E_best = std::numeric_limits<long double>::quiet_NaN();

    std::cout << "Searching for a valid initial basis..." << std::endl;

    // --- The "Keep Rolling" Loop ---
    while (std::isnan(E_best)) {
        for(size_t i = 0; i < n1; i++) basis_pn[i] = qm::gaus(1, min_A, max_A);
        for(size_t i = 0; i < n2; i++) basis_pns[i] = qm::gaus(2, min_A, max_A);

        build_coupled_matrices(basis_pn, basis_pns, h_calc, J_pn, J_pns, H, N, S, b, m_s, r);
        E_best = get_ground_state_energy(H, N);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_channel(1, 2);
    std::uniform_int_distribution<int> dist_n1(0, n1 - 1);
    std::uniform_int_distribution<int> dist_n2(0, n2 - 1);

    int max_iterations = 10000;

    // --- The Stochastic Loop ---
    for (int step = 1; step <= max_iterations; step++) {
        int channel = dist_channel(gen);
        
        if (channel == 1) {
            int k = dist_n1(gen);
            qm::gaus old_g = basis_pn[k];
            basis_pn[k] = qm::gaus(1, min_A, max_A); 
            
            build_coupled_matrices(basis_pn, basis_pns, h_calc, J_pn, J_pns, H, N, S, b, m_s, r);
            long double E_trial = get_ground_state_energy(H, N);
            
            if (E_trial < E_best && !std::isnan(E_trial)) {
                E_best = E_trial;
                //std::cout << "Step " << step << " | New Best (pn mutated): " << E_best << " MeV\n";
            } else {
                basis_pn[k] = old_g; 
            }
        } 
        else {
            int k = dist_n2(gen);
            qm::gaus old_g = basis_pns[k];
            basis_pns[k] = qm::gaus(2, min_A, max_A); 
            
            build_coupled_matrices(basis_pn, basis_pns, h_calc, J_pn, J_pns, H, N, S, b, m_s, r);
            long double E_trial = get_ground_state_energy(H, N);
            
            if (E_trial < E_best && !std::isnan(E_trial)) {
                E_best = E_trial;
                //std::cout << "Step " << step << " | New Best (pns mutated): " << E_best << " MeV\n";
            } else {
                basis_pns[k] = old_g; 
            }
        }
    }

    std::cout << "\nOptimization Complete.\nFinal Ground State Energy: " << E_best << " MeV\n";
    return 0;
}