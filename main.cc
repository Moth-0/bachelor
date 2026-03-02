#include<iostream>
#include<vector>
#include<random>
#include<fstream>
#include<cmath>
#include"qm/matrix.h"
#include"qm/eigen.h"
#include"qm/gaussian.h"
#include"qm/hamiltonian.h"
#include"qm/jacobian.h"

// System Builder: Fills the total H and N block matrices for the coupled pn / pnπ system
void build_coupled_matrices(
    const std::vector<qm::gaus>& basis_pn, 
    const std::vector<qm::gaus>& basis_pnπ, 
    qm::hamiltonian& h_calc, 
    const qm::jacobian& J_pn, 
    const qm::jacobian& J_pnπ,
    qm::matrix& H_tot, 
    qm::matrix& N_tot,
    long double S,
    long double b,
    long double m_π,
    bool relativistic) 
{
    size_t n1 = basis_pn.size();
    size_t n2 = basis_pnπ.size();

    // --- 1. Fill Block 1,1 (Deuterium Channel) ---
    qm::matrix N_pn = h_calc.overlap_matrix(basis_pn);
    qm::matrix H_pn = h_calc.hamiltonian_matrix(basis_pn, J_pn, false); // false = classic
    FOR_MAT(H_pn) {
        N_tot(i, j) = N_pn(i, j);
        H_tot(i, j) = H_pn(i, j);
    }

    // --- 2. Fill Block 2,2 (Pion-Deuteron Channel) ---
    qm::matrix N_pnπ = h_calc.overlap_matrix(basis_pnπ);
    qm::matrix H_pnπ = h_calc.hamiltonian_matrix(basis_pnπ, J_pnπ, relativistic); 
    FOR_MAT(H_pnπ) {
        N_tot(n1 + i, n1 + j) = N_pnπ(i, j);
        H_tot(n1 + i, n1 + j) = H_pnπ(i, j) + (m_π * N_pnπ(i, j));
    }

    // --- 3. Fill Blocks 1,2 and 2,1 (The Coupling W) ---
    for(size_t i = 0; i < n1; i++) {
        for(size_t j = 0; j < n2; j++) {
            size_t row_2 = j + n1; 
            long double w_val = h_calc.W_couple(basis_pn[i], basis_pnπ[j], S, b);
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
    std::ofstream file("step_energies.txt");

    // Particle masses in MeV
    long double m_n = 939.0; 
    long double m_p = 939.0; 
    long double m_π = 139.57039;

    // Search Variables 
    long double target = -2.225; // MeV
    bool relativistic = false;
    // Coupeling variables - Initial guess
    long double S = 15.0;  // MeV 
    long double b = 2.0;    // fm 

    // Make jacobians with particle mass and charges 
    qm::jacobian J_pn({m_p, m_n}, {1, 0});
    qm::jacobian J_pnπ({m_p, m_n, m_π}, {1, 0, -1});
    qm::hamiltonian h_calc;

    // --- The Search Range Variables ---
    long double min_A = 1e-2;
    long double max_A = 1e2;

    size_t n1 = 10;
    size_t n2 = 20;
    size_t N_tot = n1 + n2;

    long double E_best = std::numeric_limits<long double>::quiet_NaN();


    // Loop to find optimal S for Classic energy 
    for (int b_step=1; b_step<=1; b_step++) {
    for (int S_step=1; S_step<=1; S_step++) {
        std::cout << "\n--- TUNING STEP " << b_step * S_step << " | Current S: " << S << " MeV, b: " << b << " fm ---" << std::endl;

        // Build starting gaussians 
        std::vector<qm::gaus> basis_pn(n1);
        std::vector<qm::gaus> basis_pnπ(n2);
        for(size_t i = 0; i < n1; i++) basis_pn[i] = qm::gaus(1, min_A, max_A);
        for(size_t i = 0; i < n2; i++) basis_pnπ[i] = qm::gaus(2, min_A, max_A);

        // Build Hamiltonian and overlap matrices
        qm::matrix H(N_tot, N_tot);
        qm::matrix N(N_tot, N_tot);

        // Loop for finding a starting basis that is valid 
        E_best = std::numeric_limits<long double>::quiet_NaN();
        file << "Searching for a valid initial basis..." << std::endl;
        while (std::isnan(E_best)) {
            for(size_t i = 0; i < n1; i++) basis_pn[i] = qm::gaus(1, min_A, max_A);
            for(size_t i = 0; i < n2; i++) basis_pnπ[i] = qm::gaus(2, min_A, max_A);

            build_coupled_matrices(basis_pn, basis_pnπ, h_calc, J_pn, J_pnπ, H, N, S, b, m_π, relativistic);
            E_best = get_ground_state_energy(H, N);
        }

        // Random number generator for choosing which system to change 
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist_channel(1, 2);
        std::uniform_int_distribution<size_t> dist_n1(0, n1 - 1);
        std::uniform_int_distribution<size_t> dist_n2(0, n2 - 1);

        int max_iterations = 1000;
        int num_candidates = 20; // Size of the competitive pool

        long double overlap_threshold = 0.999;

        for (int step = 1; step <= max_iterations; step++) {
            // 1. Pick a channel and a specific basis function to "refine"
            int channel = dist_channel(gen);
            size_t k = (channel == 1) ? dist_n1(gen) : dist_n2(gen);
            std::vector<qm::gaus>& current_basis = (channel == 1) ? basis_pn : basis_pnπ;
            
            // Store the original so we can revert if no candidate wins
            qm::gaus original_g = current_basis[k];
            long double E_best_this_round = E_best;
            qm::gaus winning_candidate = original_g;
            bool improved = false;

            // 2. THE COMPETITION: Try multiple random candidates for this specific slot 'k'
            for (int c = 0; c < num_candidates; c++) {
                qm::gaus trial_g = original_g; // Copy structure
                trial_g.randomize(min_A, max_A); // Generate new parameters using the new engine
                
                // --- LINEAR INDEPENDENCE CHECK ---
                bool is_linearly_dependent = false;
                for (size_t i = 0; i < current_basis.size(); i++) {
                    // Skip the slot we are currently refining
                    if (i == k) continue; 

                    long double N_ii = qm::overlap(current_basis[i], current_basis[i]);
                    long double N_trial = qm::overlap(trial_g, trial_g);
                    long double N_i_trial = qm::overlap(current_basis[i], trial_g);

                    long double normalized_overlap = std::abs(N_i_trial) / std::sqrt(N_ii * N_trial);

                    if (normalized_overlap > overlap_threshold) {
                        is_linearly_dependent = true;
                        break;
                    }
                }

                if (is_linearly_dependent) {
                    continue; // Discard this candidate and try the next one
                }
                // ---------------------------------

                // Place candidate in the basis
                current_basis[k] = trial_g;

                // 3. Evaluate the candidate
                build_coupled_matrices(basis_pn, basis_pnπ, h_calc, J_pn, J_pnπ, H, N, S, b, m_π, relativistic);
                long double E_trial = get_ground_state_energy(H, N);

                // 4. Update the "Leaderboard"
                if (!std::isnan(E_trial) && E_trial < E_best_this_round) {
                    E_best_this_round = E_trial;
                    winning_candidate = trial_g;
                    improved = true;
                }
            }

            // 5. FINAL SELECTION: The winner of the pool replaces the old Gaussian
            if (improved) {
                current_basis[k] = winning_candidate;
                
                E_best = E_best_this_round;
                file << "Step " << step << " | Slot " << k << " improved to: " << E_best << " MeV\n";
            } else {
                // No candidate was better than the original; revert
                current_basis[k] = original_g;
            }
        }
        std::cout << "Best Energy at S=" << S << ", b=" << b << " : " << E_best << " MeV" << std::endl;

        // Adjust S for the next tuning step
        long double error = target - E_best; 
        S -= error;
    }
        std::cout << "Best Energy at S=" << S << ", b=" << b << " : " << E_best << " MeV" << std::endl;

        // Adjust b
        long double error = std::pow(target - E_best, 2); 
        b -= error * 0.1;
    }

    return 0;

}