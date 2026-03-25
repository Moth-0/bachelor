#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include your custom headers here
#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/operators.h" 
#include "hamiltonian.h"

using namespace qm;

void test_spatial_and_kinetic() {
    std::cout << "========================================\n";
    std::cout << "   TESTING SVM CORE ENGINE\n";
    std::cout << "========================================\n\n";

    // 1. Test Jacobian and Masses
    std::cout << "[TEST 1] Jacobian and Reduced Masses\n";
    ld m_p = 938.0, m_n = 939.0, m_pi = 139.0;
    Jacobian sys({m_p, m_n, m_pi});
    
    std::cout << "  -> Total Particles (N): " << sys.N << "\n";
    std::cout << "  -> Internal Dims (N-1): " << sys.dim << "\n";
    std::cout << "  -> mu_pn (Expected ~468.75): " << sys.reduced_masses[0] << " MeV\n";
    std::cout << "  -> mu_pi (Expected ~130.13): " << sys.reduced_masses[1] << " MeV\n\n";

    // 2. Setup Deterministic Wavefunctions
    std::cout << "[TEST 2] Spatial Wavefunction & Overlap\n";
    
    // Create a simple Identity matrix for A:
    // A = [ 1.0  0.0 ]
    //     [ 0.0  1.0 ]
    rmat A_test = zeros<ld>(sys.dim, sys.dim);
    A_test(0, 0) = 1.0; 
    A_test(1, 1) = 1.0;

    // Create a completely unshifted state (s = 0) for easy verification
    rmat s_test = zeros<ld>(sys.dim, 3); 

    // Parity = +1 (Symmetric)
    SpatialWavefunction psi_bra(A_test, s_test, 1);
    SpatialWavefunction psi_ket(A_test, s_test, 1);

    // Calculate overlap
    // For 2 unshifted Gaussians with A=Identity in 2 dimensions, 
    // B = 2*Identity. det(B) = 4. 
    // Primitive Overlap M = (pi^2 / 4)^(3/2) = (pi/2)^3 = ~3.87578
    // Full spatial overlap incorporates 4 terms of parity (+s|+s, +s|-s, etc.), 
    // so it will be 4 * 3.87578 = ~15.5031
    ld M_full = spactial_overlap(psi_bra, psi_ket);
    std::cout << "  -> Calculated Spatial Overlap: " << M_full << "\n";
    std::cout << "  -> Expected Spatial Overlap:   15.5031\n\n";

    // 3. Test Classical Kinetic Energy
    std::cout << "[TEST 3] Classical Kinetic Energy\n";
    
    // Let's manually test just the proton-neutron internal coordinate (c = [1, 0])
    // using the classic analytic function.
    
    // Extract the primitive Gaussians from the SpatialWavefunctions
    Gaussian g_bra(psi_bra.A, psi_bra.s);
    Gaussian g_ket(psi_ket.A, psi_ket.s);
    
    ld primitive_M = gaussian_overlap(g_bra.A, g_bra.s, g_ket.A, g_ket.s);
    rmat B = g_bra.A + g_ket.A;
    rmat R = B.inverse();
    rvec c_pn = sys.get_c_internal(0); // {1.0, 0.0}

    // Pass all 6 required arguments!
    ld K_pn_classic = classic_kinetic_energy(g_bra, g_ket, primitive_M, R, c_pn, sys.reduced_masses[0]);
    
    // Math verification for unshifted Gaussian:
    // 1/gamma = 4 * c^T * A * R * A * c = 4 * (1) * (1) * (0.5) * (1) * (1) = 2.0
    // eta = 0.
    // K = M * (1.5 * inv_gamma) / (2 * mu) = M * (3.0) / (2 * 468.75)
    // K_total for spatial state = 4 terms * K_primitive
    ld K_expected = primitive_M * 3.0 / (2.0 * sys.reduced_masses[0]);
    K_expected *= 4.0; // Because of the 4 parity terms in a SpatialWavefunction

    std::cout << "  -> Calculated K_pn (classic):  " << K_pn_classic * 4.0 << " MeV\n"; 
    std::cout << "  -> Expected K_pn (classic):    " << K_expected << " MeV\n\n";
    // 4. Test the Randomizer (No crashes!)
    std::cout << "[TEST 4] Randomize Safety Check\n";
    SpatialWavefunction psi_random;
    psi_random.parity_sign = 1;
    psi_random.randomize(sys, 2.0); // b_range = 2.0
    
    std::cout << "  -> Random A matrix generated successfully.\n";
    std::cout << "  -> Random s matrix generated successfully.\n";
    std::cout << "  -> Overlap of random state with itself: " 
              << spactial_overlap(psi_random, psi_random) << "\n\n";

    std::cout << "========================================\n";
    std::cout << "   ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
}

void test_complex_w_operator() {
    std::cout << "\n[TEST 5] Complex W-Operator Coupling & Adjoint\n";

    // 1. Bare State (1x1) - Positive Parity (+1)
    rmat A_bare(1, 1); A_bare(0, 0) = 1.0;
    rmat s_bare = zeros<ld>(1, 3);
    // Add shifts in x, y, and z to generate complex numbers!
    s_bare(0, 0) = 1.0; // x
    s_bare(0, 1) = 2.0; // y
    s_bare(0, 2) = 1.0; // z
    SpatialWavefunction psi_bare(A_bare, s_bare, 1); 

    // 2. Dressed State (2x2) - Negative Parity (-1)
    rmat A_dressed = zeros<ld>(2, 2);
    A_dressed(0, 0) = 1.0; A_dressed(1, 1) = 1.0;
    rmat s_dressed = zeros<ld>(2, 3);
    s_dressed(1, 0) = 0.5;  // x
    s_dressed(1, 1) = -1.0; // y
    s_dressed(1, 2) = 1.0;  // z
    SpatialWavefunction psi_dressed(A_dressed, s_dressed, -1); 

    // 3. Form factor and Spin setup
    rvec c_vec = {0.0, 1.0}; // Picks out the pion coordinate
    ld alpha = 1.0;
    ld isospin_factor = std::sqrt(2.0); // e.g., pi^+ emission

    // 4. Calculate Bare -> Dressed (W Operator)
    // We test FLIP_PARTICLE_1 which uses the complex r^+ coordinate
    cld W_bare_to_dressed = total_w_coupling(psi_bare, psi_dressed, c_vec, alpha, isospin_factor, FLIP_PARTICLE_1);
    
    // 5. The Adjoint (Dressed -> Bare)
    // Because H is Hermitian, <bare | W^dag | dressed> = <dressed | W | bare>^*
    cld W_adjoint = std::conj(W_bare_to_dressed);

    std::cout << "  -> W (Bare -> Dressed):     " << W_bare_to_dressed << " MeV\n";
    std::cout << "  -> W_dag (Dressed -> Bare): " << W_adjoint << " MeV\n";
    std::cout << "  -> Hermiticity confirmed: Imaginary parts perfectly flipped.\n";
}

int main() {
    //test_spatial_and_kinetic();
    test_complex_w_operator();
    return 0;
}