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
void build_coupled_matrix(
    const std::vector<qm::gaus>& basis_bare, 
    const std::vector<qm::gaus>& basis_pi0, 
    const std::vector<qm::gaus>& basis_charged, 
    qm::hamiltonian& h_calc, 
    const qm::jacobian& J_pn, 
    const qm::jacobian& J_π0,
    const qm::jacobian& J_πc,
    qm::matrix& H_tot, 
    qm::matrix& N_tot,
    long double S_w, long double b_w,
    long double m_pi0, long double m_charged,
    bool relativistic) 
{
    size_t n1 = basis_bare.size();
    size_t n2 = basis_pi0.size();
    size_t n3 = basis_charged.size();
    
    // Total matrix size is n1 + n2 + n3
    size_t offset_pi0 = n1;
    size_t offset_charged = n1 + n2;

    // Initialize to zero to be safe
    H_tot = qm::matrix(H_tot.size1(), H_tot.size2());
    N_tot = qm::matrix(N_tot.size1(), N_tot.size2());

    // ==========================================
    // 1. DIAGONAL BLOCKS (The H_bare, H_pi0, H_charged)
    // ==========================================
    
    // Block 1,1: Bare |pn>
    qm::matrix H_bare = h_calc.hamiltonian_matrix(basis_bare, J_pn, false); 
    qm::matrix N_bare = h_calc.overlap_matrix(basis_bare);
    for (size_t i = 0; i < n1; i++) {
        for (size_t j = 0; j < n1; j++) {
            H_tot(i, j) = H_bare(i, j);
            N_tot(i, j) = N_bare(i, j);
        }
    }

    // Block 2,2: Clothed |pn, pi0>
    qm::matrix H_pi0 = h_calc.hamiltonian_matrix(basis_pi0, J_π0, relativistic); 
    qm::matrix N_pi0 = h_calc.overlap_matrix(basis_pi0);
    for (size_t i = 0; i < n2; i++) {
        for (size_t j = 0; j < n2; j++) {
            // Add the rest mass of the pion to the energy of the clothed state
            H_tot(offset_pi0 + i, offset_pi0 + j) = H_pi0(i, j) + (m_pi0 * N_pi0(i, j));
            N_tot(offset_pi0 + i, offset_pi0 + j) = N_pi0(i, j);
        }
    }

    // Block 3,3: Clothed |Charged>
    qm::matrix H_charged = h_calc.hamiltonian_matrix(basis_charged, J_πc, relativistic); 
    qm::matrix N_charged = h_calc.overlap_matrix(basis_charged);
    for (size_t i = 0; i < n3; i++) {
        for (size_t j = 0; j < n3; j++) {
            // Add the rest mass of the charged pion
            H_tot(offset_charged + i, offset_charged + j) = H_charged(i, j) + (m_charged * N_charged(i, j));
            N_tot(offset_charged + i, offset_charged + j) = N_charged(i, j);
        }
    }

    // ==========================================
    // 2. OFF-DIAGONAL BLOCKS (The W-Operators)
    // ==========================================
    
    for(size_t i = 0; i < n1; i++) {
        
        // A. Neutral Pion Coupling (Weight = 1.0)
        for(size_t j = 0; j < n2; j++) {
            size_t row_pi0 = offset_pi0 + j; 
            
            // Note: Remember to update W_couple in hamiltonian.h to calculate the p-wave vector displacement!
            long double w_val_neutral = 1.0 * h_calc.W_couple(basis_bare[i], basis_pi0[j], S_w, b_w);
            
            H_tot(i, row_pi0) = w_val_neutral;
            H_tot(row_pi0, i) = w_val_neutral;
        }
        
        // B. Charged Pion Coupling (Weight = sqrt(2.0))
        for(size_t k = 0; k < n3; k++) {
            size_t row_charged = offset_charged + k; 
            
            long double w_val_charged = std::sqrt(2.0) * h_calc.W_couple(basis_bare[i], basis_charged[k], S_w, b_w);
            
            H_tot(i, row_charged) = w_val_charged;
            H_tot(row_charged, i) = w_val_charged;
        }
    }
}

// General Solver: Takes ANY H and N matrices and returns the lowest eigenvalue safely
long double get_ground_state_energy(const qm::matrix& H, const qm::matrix& N) {
    qm::matrix L = qm::cholesky(N);
    
    if (L.size1() == 0) return std::numeric_limits<long double>::quiet_NaN();

    for(size_t i = 0; i < L.size1(); i++) {
        if (std::abs(L(i,i)) < ZERO_LIMIT) {
            return std::numeric_limits<long double>::quiet_NaN();
        }
    }

    qm::matrix L_inv = L.inverse_lower(); 
    qm::matrix H_prime = L_inv * H * L_inv.transpose();
    qm::vector energies = qm::jacobi_eigenvalues(H_prime);

    long double E_0 = energies[0];
    for (size_t i = 1; i < energies.size(); i++) {
        if (energies[i] < E_0) E_0 = energies[i];
    }
    
    return E_0;
}

int main() {
    std::ofstream file("step_energies.txt");

    // Particle masses in MeV
    long double m_n = 939.5654; 
    long double m_p = 938.2721; 
    long double m_π0 = 134.9768; // Swapped this to correct neutral mass
    long double m_πc = 139.5704; // Swapped this to correct charged mass

    long double target = -2.225; // MeV
    bool relativistic = false;
    long double S = 15.0;  // MeV 
    long double b = 2.0;    // fm 

    qm::jacobian J_pn({m_p, m_n}, {1, 0});
    qm::jacobian J_π0({m_p, m_n, m_π0}, {1, 0, 0});
    qm::jacobian J_πc({m_n, m_n, m_πc}, {0, 0, 0}); // Using n,n for the charged state

    qm::hamiltonian h_calc;

    long double min_A = 1e-2;
    long double max_A = 1e2;

    size_t n1 = 10;
    size_t n2 = 20;
    size_t n3 = n2; // Define n3 so it matches the charged basis size
    size_t N_tot = n1 + n2 + n3; // FIXED: N_tot is now the sum of all three blocks

    long double E_best = std::numeric_limits<long double>::quiet_NaN();

    long double overlap_threshold = 0.99;
    int max_iterations = 500;
    int num_candidates = 20;

    for (int b_step=1; b_step<=1; b_step++) {
    for (int S_step=1; S_step<=1; S_step++) {
        std::cout << "\n--- TUNING STEP " << b_step * S_step << " | Current S: " << S << " MeV, b: " << b << " fm ---" << std::endl;

        std::vector<qm::gaus> basis_pn(n1);
        std::vector<qm::gaus> basis_π0(n2);
        std::vector<qm::gaus> basis_πc(n3);

        for(size_t i = 0; i < n1; i++) basis_pn[i] = qm::gaus(1, min_A, max_A);
        for(size_t i = 0; i < n2; i++) basis_π0[i] = qm::gaus(2, min_A, max_A);
        for(size_t i = 0; i < n3; i++) basis_πc[i] = qm::gaus(2, min_A, max_A);

        qm::matrix H(N_tot, N_tot);
        qm::matrix N(N_tot, N_tot);

        file << "Building linearly independent initial basis..." << std::endl;

        auto is_independent = [](const qm::gaus& g, const std::vector<qm::gaus>& basis, size_t current_size, long double threshold) {
            long double N_trial = qm::overlap(g, g);
            for(size_t i = 0; i < current_size; i++) {
                long double N_ii = qm::overlap(basis[i], basis[i]);
                long double N_i_trial = qm::overlap(basis[i], g);
                long double normalized_overlap = std::abs(N_i_trial) / std::sqrt(N_ii * N_trial);
                if (normalized_overlap > threshold) return false; 
            }
            return true;
        };

        E_best = std::numeric_limits<long double>::quiet_NaN();
        
        while (std::isnan(E_best)) {
            for (size_t i = 0; i < n1; ) {
                qm::gaus trial_g(1, min_A, max_A);
                if (is_independent(trial_g, basis_pn, i, overlap_threshold)) { basis_pn[i] = trial_g; i++; }
            }
            for (size_t i = 0; i < n2; ) {
                qm::gaus trial_g(2, min_A, max_A);
                if (is_independent(trial_g, basis_π0, i, overlap_threshold)) { basis_π0[i] = trial_g; i++; }
            }
            for (size_t i = 0; i < n3; ) {
                qm::gaus trial_g(2, min_A, max_A);
                if (is_independent(trial_g, basis_πc, i, overlap_threshold)) { basis_πc[i] = trial_g; i++; }
            }

            build_coupled_matrix(basis_pn, basis_π0, basis_πc, h_calc, J_pn, J_π0, J_πc, H, N, S, b, m_π0, m_πc, relativistic);
            E_best = get_ground_state_energy(H, N);
        }
        
        std::cout << "Initial valid energy: " << E_best << " MeV\n";

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist_channel(1, 3);
        std::uniform_int_distribution<size_t> dist_n1(0, n1 - 1);
        std::uniform_int_distribution<size_t> dist_n2(0, n2 - 1);

        for (int step = 1; step <= max_iterations; step++) {
            int channel = dist_channel(gen);
            
            // FIXED: Ensure we target the right basis and index based on channel
            size_t k;
            std::vector<qm::gaus>* current_basis_ptr;
            
            if (channel == 1) {
                k = dist_n1(gen);
                current_basis_ptr = &basis_pn;
            } else if (channel == 2) {
                k = dist_n2(gen);
                current_basis_ptr = &basis_π0;
            } else {
                k = dist_n2(gen); // n3 is the same size as n2, so this is safe
                current_basis_ptr = &basis_πc;
            }
            
            std::vector<qm::gaus>& current_basis = *current_basis_ptr;
            
            qm::gaus original_g = current_basis[k];
            long double E_best_this_round = E_best;
            qm::gaus winning_candidate = original_g;
            bool improved = false;

            for (int c = 0; c < num_candidates; c++) {
                qm::gaus trial_g = original_g; 
                trial_g.randomize(min_A, max_A); 
                
                bool is_linearly_dependent = false;
                for (size_t i = 0; i < current_basis.size(); i++) {
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

                if (is_linearly_dependent) continue;

                current_basis[k] = trial_g;

                build_coupled_matrix(basis_pn, basis_π0, basis_πc, h_calc, J_pn, J_π0, J_πc, H, N, S, b, m_π0, m_πc, relativistic);
                long double E_trial = get_ground_state_energy(H, N);

                if (!std::isnan(E_trial) && E_trial < E_best_this_round) {
                    E_best_this_round = E_trial;
                    winning_candidate = trial_g;
                    improved = true;
                }
            }

            if (improved) {
                current_basis[k] = winning_candidate;
                E_best = E_best_this_round;
                std::cout << "Step " << step << " | Channel " << channel << " Slot " << k << " improved to: " << E_best << " MeV\n";
            } else {
                current_basis[k] = original_g;
            }
        }
        std::cout << "Best Energy at S=" << S << ", b=" << b << " : " << E_best << " MeV" << std::endl;

        long double error = target - E_best; 
        S -= error;
    }
        long double error = std::pow(target - E_best, 2); 
        b -= error * 0.1;
    }

    return 0;
}