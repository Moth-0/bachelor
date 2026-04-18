/*
 * test_suite.cc - MASTER ANALYTICAL VERIFICATION SUITE
 * Contains deterministic, by-hand verified tests for 2-body and 3-body 
 * quantum mechanical operators using Stochastic Variational Gaussians.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// ========================================================================
// TEST 1: 2-PARTICLE SYSTEM (Unshifted, S-Wave)
// ========================================================================
bool test_2_particle_system() {
    std::cout << "----------------------------------------------------\n";
    std::cout << " RUNNING TEST 1: 2-Particle System (Unshifted)\n";
    std::cout << "----------------------------------------------------\n";

    ld m_p = 939.0, m_n = 939.0;
    ld mu = (m_p * m_n) / (m_p + m_n);
    ld hbarc_sq_over_2mu = (HBARC * HBARC) / (2.0 * mu);

    Jacobian jac({m_p, m_n});

    ld A_val = 0.5;
    rmat A = eye<ld>(1) * A_val;
    rmat s = zeros<ld>(1, 3);
    SpatialWavefunction psi(A, s, 1); // Positive parity (+1)

    // 1. Overlap
    ld N_overlap = spactial_overlap(psi, psi);
    ld expected_overlap = std::pow(M_PI, 1.5) * 2.0;

    // 2. Kinetic Energy
    std::vector<bool> rel_flags = {false}; 
    ld T_matrix_element = total_kinetic_energy(psi, psi, jac, rel_flags);
    ld calculated_T_exp = T_matrix_element / N_overlap;
    ld expected_T_exp = hbarc_sq_over_2mu * 3.0 * A_val;

    // 3. Charge Radius
    rvec charges = {1.0, 0.0}; 
    ld R2_matrix_element = charge_radius_operator(psi, psi, jac, charges);
    ld calculated_R2_exp = R2_matrix_element / N_overlap;
    ld expected_R2_exp = 0.375;

    // Print Results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Overlap <N>:    Exp = " << expected_overlap << " | Calc = " << N_overlap << "\n";
    std::cout << " Kinetic <T>:    Exp = " << expected_T_exp << " | Calc = " << calculated_T_exp << " MeV\n";
    std::cout << " Radius <R^2>:   Exp = " << expected_R2_exp << " | Calc = " << calculated_R2_exp << " fm^2\n";

    bool success = (std::abs(calculated_T_exp - expected_T_exp) < 1e-4) && 
                   (std::abs(calculated_R2_exp - expected_R2_exp) < 1e-4);
    
    if (success) std::cout << " -> [PASS] \n\n";
    else std::cout << " -> [FAIL] 2-Particle Math mismatch!\n\n";

    return success;
}

// ========================================================================
// TEST 2: 3-PARTICLE SYSTEM (Shifted, Parity Interference)
// ========================================================================
bool test_3_particle_shifted_system() {
    std::cout << "----------------------------------------------------\n";
    std::cout << " RUNNING TEST 2: 3-Particle System (Shifted)\n";
    std::cout << "----------------------------------------------------\n";

    ld m_p = 900.0, m_n = 900.0, m_pi = 200.0;
    ld mu_1 = 450.0;
    ld mu_2 = 180.0;

    ld C1 = (HBARC * HBARC) / (2.0 * mu_1);
    ld C2 = (HBARC * HBARC) / (2.0 * mu_2);

    Jacobian jac({m_p, m_n, m_pi});

    ld A_val = 0.5;
    rmat A = zeros<ld>(2, 2);
    A(0, 0) = A_val;
    A(1, 1) = A_val;
    
    rmat s = zeros<ld>(2, 3);
    s(0, 2) = 1.0; // Shift coordinate 1 by 1.0 fm in Z

    SpatialWavefunction psi(A, s, 1); // Positive parity (+1)
    ld e1 = std::exp(1.0); 

    // 1. Overlap
    ld N_overlap = spactial_overlap(psi, psi);
    ld expected_overlap = std::pow(M_PI, 3.0) * (e1 + 1.0);

    // 2. Kinetic Energy
    std::vector<bool> rel_flags = {false, false}; 
    ld T_matrix_element = total_kinetic_energy(psi, psi, jac, rel_flags);
    ld calculated_T_exp = T_matrix_element / N_overlap;
    
    ld T1_exp = C1 * (1.5 - (4.0 / (e1 + 1.0)));
    ld T2_exp = C2 * 1.5; 
    ld expected_T_exp = T1_exp + T2_exp;

    // 3. Charge Radius
    rvec charges = {1.0, 0.0, 1.0}; 
    ld R2_matrix_element = charge_radius_operator(psi, psi, jac, charges);
    ld calculated_R2_exp = R2_matrix_element / N_overlap;
    
    ld shift_inflation = 0.25 * (e1 / (e1 + 1.0));
    ld expected_R2_exp = 1.605 + shift_inflation; 

    // Print Results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Overlap <N>:    Exp = " << expected_overlap << " | Calc = " << N_overlap << "\n";
    std::cout << " Kinetic <T>:    Exp = " << expected_T_exp << " | Calc = " << calculated_T_exp << " MeV\n";
    std::cout << " Radius <R^2>:   Exp = " << expected_R2_exp << " | Calc = " << calculated_R2_exp << " fm^2\n";

    bool success = (std::abs(calculated_T_exp - expected_T_exp) < 1e-4) && 
                   (std::abs(calculated_R2_exp - expected_R2_exp) < 1e-4);
    
    if (success) std::cout << " -> [PASS] \n\n";
    else std::cout << " -> [FAIL] 3-Particle Math mismatch!\n\n";

    return success;
}

// ========================================================================
// TEST 3: OFF-DIAGONAL HAMILTONIAN (Pion Coupling Operator W)
// ========================================================================
bool test_w_coupling() {
    std::cout << "----------------------------------------------------\n";
    std::cout << " RUNNING TEST 3: Hamiltonian Cross-Term (W Operator)\n";
    std::cout << "----------------------------------------------------\n";

    // 1. Setup Parameters
    ld b_form = 1.0; 
    ld S_strength = 10.0;
    ld iso_factor = std::sqrt(2.0L); // e.g., PN -> PN + pi^+

    // Transition coordinate vector (Selects the pion Jacobi coordinate)
    rvec c_transition = {0.0, 1.0}; 

    // 2. Setup the Bare State (1D)
    rmat A_bare = eye<ld>(1) * 0.5L;
    rmat s_bare = zeros<ld>(1, 3);
    SpatialWavefunction psi_bare(A_bare, s_bare, 1); // P = +1

    // 3. Setup the Dressed State (2D)
    // Matches the bare core (0.5) and matches the form factor width (1.0 / b^2 = 1.0)
    rmat A_dressed = zeros<ld>(2, 2);
    A_dressed(0, 0) = 0.5;
    A_dressed(1, 1) = 1.0; 
    
    // Shift the pion by 1.0 fm in the Z direction
    rmat s_dressed = zeros<ld>(2, 3);
    s_dressed(1, 2) = 1.0; 

    SpatialWavefunction psi_dressed(A_dressed, s_dressed, -1); // P = -1

    // 4. Calculate W using the engine
    // We test NO_FLIP, which strictly evaluates the Z-coordinate projection
    cld W_calc_cld = total_w_coupling(psi_bare, psi_dressed, c_transition, 
                                      b_form, S_strength, iso_factor, NO_FLIP);
    ld W_calculated = std::real(W_calc_cld); // Imaginary part should be 0

    // 5. Analytical Hand-Calculation
    ld M_term = (std::pow(M_PI, 3.0) / (2.0 * std::sqrt(2.0))) * std::exp(0.125);
    ld W_spatial = M_term * 0.5; // Includes the parity normalization!
    
    ld norm_sq = (3.0 * std::pow(M_PI, 1.5)) / (8.0 * std::sqrt(2.0));
    ld W_expected = W_spatial * S_strength * iso_factor * (1.0 / std::sqrt(norm_sq));

    // Print Results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Transition Matrix Element <PN | W | PN pi>\n";
    std::cout << " Expected:   " << W_expected << " MeV\n";
    std::cout << " Calculated: " << W_calculated << " MeV\n\n";

    bool success = (std::abs(W_calculated - W_expected) < 1e-4);
    
    if (success) std::cout << " -> [PASS] \n\n";
    else std::cout << " -> [FAIL] W Operator mismatch!\n\n";

    return success;
}

void test_relativistic_explosion() {
    std::cout << "\n=== TESTING RELATIVISTIC KINETIC ENERGY EXPLOSION ===\n";

    // 1. Setup a simple 1D-like environment
    ld mass = 938.0; 
    rmat A = eye<ld>(1) * 1.0L; // 1.0 fm width
    
    // Gaussian 1: Sitting perfectly at the origin
    Gaussian g1;
    g1.A = A;
    g1.s = zeros<ld>(1, 3); 

    // Gaussian 2: We will push this one further and further away
    Gaussian g2;
    g2.A = A;
    g2.s = zeros<ld>(1, 3); 

    rmat R = eye<ld>(1) * 0.5L; // Simple inverse matrix
    rvec c = {1.0};            // Simple coordinate transform
    ld M_overlap = 1.0;        // Assume normalized for the test

    std::cout << std::left << std::setw(15) << "Shift (fm)" 
              << std::setw(20) << "gamma * eta^2" 
              << std::setw(25) << "Classic T (MeV)" 
              << std::setw(25) << "Relativistic T (MeV)" << "\n";
    std::cout << std::string(85, '-') << "\n";

    // 2. Push the second Gaussian away from 0 fm to 20 fm
    for (ld shift = 0.0; shift <= 20.0; shift += 2.0) {
        g2.s(0, 0) = shift; // Shift it along the x-axis

        // Calculate the exponent term manually to see it grow
        rvec diff = (g1.A * (R * g2.s[0])) - (g2.A * (R * g1.s[0]));
        ld eta = std::sqrt(dot_no_conj(diff, diff) * 4.0);
        
        rvec A_ket_c = g2.A * c;
        ld inv_gamma = 4.0 * dot_no_conj(c, g1.A * (R * A_ket_c));
        ld gamma = 1.0 / inv_gamma;
        
        ld exponent_term = gamma * eta * eta;

        // Calculate both energies (without the safety fallback!)
        ld t_class = classic_kinetic_energy(g1, g2, M_overlap, R, c, mass);
        ld t_rel   = relativistic_kinetic_energy(g1, g2, M_overlap, R, c, mass);

        std::cout << std::left << std::setw(15) << shift 
                  << std::setw(20) << exponent_term 
                  << std::setw(25) << t_class 
                  << std::setw(25) << t_rel << "\n";
    }
    std::cout << "=====================================================\n";
}

// ========================================================================
// MAIN EXECUTION
// ========================================================================
int main() {
    std::cout << "====================================================\n";
    std::cout << "        STARTING SVM OPERATOR TEST SUITE\n";
    std::cout << "====================================================\n\n";

    bool t1 = test_2_particle_system();
    bool t2 = test_3_particle_shifted_system();
    bool t3 = test_w_coupling();
    test_relativistic_explosion();

    std::cout << "====================================================\n";
    if (t1 && t2 && t3) {
        std::cout << " [ALL TESTS PASSED] \n";
    } else {
        std::cout << " [WARNING] One or more tests failed. \n";
    }
    std::cout << "====================================================\n";

    return 0;
}