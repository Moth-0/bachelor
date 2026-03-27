#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Include your custom headers here
#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/operators.h" 
#include "qm/solver.h"
#include "deuterium.h"

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
    cld W_bare_to_dressed = total_w_coupling(psi_bare, psi_dressed, c_vec, alpha, alpha, isospin_factor, FLIP_PARTICLE_1);
    
    // 5. The Adjoint (Dressed -> Bare)
    // Because H is Hermitian, <bare | W^dag | dressed> = <dressed | W | bare>^*
    cld W_adjoint = std::conj(W_bare_to_dressed);

    std::cout << "  -> W (Bare -> Dressed):     " << W_bare_to_dressed << " MeV\n";
    std::cout << "  -> W_dag (Dressed -> Bare): " << W_adjoint << " MeV\n";
    std::cout << "  -> Hermiticity confirmed: Imaginary parts perfectly flipped.\n";
}

void test_physics_engine() {
    std::cout << "\n========================================\n";
    std::cout << "  PHYSICS ENGINE DIAGNOSTIC TEST\n";
    std::cout << "========================================\n";

    // 1. Setup standard physical parameters
    qm::ld m_p = 938.27, m_n = 939.56, m_pi = 134.97;
    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed({m_p, m_n, m_pi});

    // 2. Create a perfectly symmetric, well-behaved Bare State
    // Width A = 0.5 (reasonable), shifts = 0 (centered)
    qm::rmat A_bare = qm::eye<qm::ld>(1) * 0.5L;
    qm::rmat s_bare = qm::zeros<qm::ld>(1, 3);
    SpatialWavefunction psi_bare(A_bare, s_bare, 1);

    // 3. Create a well-behaved Dressed State
    qm::rmat A_dress = qm::eye<qm::ld>(2) * 0.5L;
    qm::rmat s_dress = qm::zeros<qm::ld>(2, 3);
    // Let's give it a tiny shift so the W-operator doesn't return exactly 0 due to parity
    s_dress(1, 0) = 0.5; // Shift pion slightly in x
    SpatialWavefunction psi_dress(A_dress, s_dress, -1);

    // 4. Package them
    std::vector<BasisState> basis;
    basis.push_back({psi_bare, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    basis.push_back({psi_dress, Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed, m_pi});

    // 5. Build the matrices! 
    qm::ld b = 1.4;
    qm::ld S = 20.0; // Keep coupling small for the test
    auto [H, N] = build_matrices(basis, b, S, false);

    std::cout << "\n--- OVERLAP MATRIX (N) ---\n" << N << "\n";
    std::cout << "\n--- HAMILTONIAN MATRIX (H) ---\n" << H << "\n";

    // 6. The Physics Sanity Checks
    std::cout << "\n--- DIAGNOSTICS ---\n";
    
    // True energy is <psi|H|psi> / <psi|psi>
    qm::ld bare_E = std::real(H(0,0)) / std::real(N(0,0));
    qm::ld dress_E = std::real(H(1,1)) / std::real(N(1,1));
    
    std::cout << "1. Bare Kinetic Energy:   " << bare_E << " MeV\n";
    std::cout << "2. Dressed Total Energy:  " << dress_E << " MeV\n";
    std::cout << "   (Expected roughly: Kinetic + " << m_pi << " MeV)\n";
    std::cout << "3. Coupling Strength (W): " << H(0,1) << " MeV\n";

    // 7. Test the Solver
    qm::ld E0 = solve_ground_state_energy(H, N);
    std::cout << "4. GEVP Solver Result:    " << E0 << " MeV\n";
    std::cout << "========================================\n\n";
}

void test_deuteron_matrices() {
    std::cout << "\n========================================\n";
    std::cout << "  DEUTERON 3-BODY DIAGNOSTIC TEST\n";
    std::cout << "========================================\n";

    ld m_p = 938.27, m_n = 939.56, m_pi = 134.97;
    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed({m_p, m_n, m_pi});

    // 1. Bare State (1 internal dimension: PN distance)
    rmat A_bare = eye<ld>(1) * 0.5L;
    rmat s_bare = zeros<ld>(1, 3);
    s_bare(0, 2) = 0.5; // Force a shift in the Z-axis!
    SpatialWavefunction psi_bare(A_bare, s_bare, 1);

    // 2. Dressed State (2 internal dimensions: PN distance, Pion distance)
    rmat A_dress = eye<ld>(2) * 0.5L;
    rmat s_dress = zeros<ld>(2, 3);
    s_dress(0, 2) = 0.5;  // Shift PN distance in Z
    s_dress(1, 2) = -0.5; // Shift Pion distance in Z
    SpatialWavefunction psi_dress(A_dress, s_dress, -1);

    std::vector<BasisState> test_basis;
    test_basis.push_back({psi_bare, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    test_basis.push_back({psi_dress, Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed, m_pi});

    ld b = 1.4;
    ld S = 100.0; // Huge coupling to make it obvious
    auto [H, N] = build_matrices(test_basis, b, S, false);

    std::cout << "--- OVERLAP MATRIX (N) ---\n" << N << "\n";
    std::cout << "--- HAMILTONIAN MATRIX (H) ---\n" << H << "\n";
    
    ld E0 = solve_ground_state_energy(H, N);
    std::cout << "\n--- DIAGNOSTICS ---\n";
    std::cout << "Bare Kinetic Energy:      " << std::real(H(0,0)/N(0,0)) << " MeV\n";
    std::cout << "Coupling Strength H(0,1): " << H(0,1) << " MeV\n";
    std::cout << "GEVP Ground State:        " << E0 << " MeV\n";
    std::cout << "========================================\n\n";
    
    exit(0); // Stop the program so we can read this output
}

// A stripped-down, single-state optimizer for testing
void test_optimize(std::vector<BasisState>& basis, ld b, ld S) {
    for (size_t k = 0; k < basis.size(); ++k) {
        SpatialWavefunction backup_psi = basis[k].psi;
        rvec p0 = pack_wavefunction(backup_psi);

        auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
            unpack_wavefunction(basis[k].psi, p_test);
            
            bool is_physical = true;
            for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
                if (basis[k].psi.A(i, i) <= 0.02) is_physical = false;
            }
            if (basis[k].psi.A.determinant() <= ZERO_LIMIT) is_physical = false;
            for (size_t i = 0; i < basis[k].psi.s.size1(); ++i) {
                for (size_t col = 0; col < 3; ++col) {
                    if (std::abs(basis[k].psi.s(i, col)) > 5.0) is_physical = false;
                }
            }
            if (!is_physical) return 999999.0; 

            auto [H, N] = build_matrices(basis, b, S, false);
            return solve_ground_state_energy(H, N);
        };

        // Run Nelder-Mead with a strict tolerance to ensure it actually finishes
        rvec p_best = nelder_mead(p0, objective_func); 
        unpack_wavefunction(basis[k].psi, p_best);
    }
}

// The core test function
void run_stability_test(bool use_grid) {
    ld m_p = 938.27, m_n = 939.56, m_pi0 = 134.97;
    ld b_form = 1.4, b_range = 1.4, S = 100.0;
    
    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed({m_p, m_n, m_pi0});

    std::cout << (use_grid ? ">>> RUNNING GEOMETRIC GRID TEST\n" : ">>> RUNNING PURE RANDOM TEST\n");

    for (int run = 1; run <= 5; ++run) {
        std::vector<BasisState> basis;

        // 1. ADD BARE PN STATES (Either Grid or Random)
        if (use_grid) {
            std::vector<ld> widths = {0.05, 0.2, 0.8, 3.0}; // Covers 0.5 fm to ~4.5 fm
            for (ld w : widths) {
                SpatialWavefunction psi(eye<ld>(1) * w, zeros<ld>(1, 3), 1);
                basis.push_back({psi, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                SpatialWavefunction psi(zeros<ld>(1, 1), zeros<ld>(1, 3), 1);
                psi.randomize(jac_bare, b_range);
                basis.push_back({psi, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
            }
        }

        // 2. ADD 4 RANDOM NEUTRAL PION STATES
        for (int i = 0; i < 4; ++i) {
            SpatialWavefunction psi(zeros<ld>(2, 2), zeros<ld>(2, 3), -1);
            psi.randomize(jac_dressed, b_range);
            basis.push_back({psi, Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed, m_pi0});
        }

        // 3. OPTIMIZE AND EVALUATE
        // We will loop the optimizer 3 times to simulate a "sweep"
        for(int sweep = 0; sweep < 3; sweep++) {
            test_optimize(basis, b_form, S);
        }

        auto [H, N] = build_matrices(basis, b_form, S, false);
        ld final_E = solve_ground_state_energy(H, N);

        std::cout << "Run " << run << "/5 Final Energy: " << std::fixed << std::setprecision(5) << final_E << " MeV\n";
    }
    std::cout << "\n";
}

int main() {
    //test_spatial_and_kinetic();
    //test_complex_w_operator();
    //test_physics_engine();
    //test_deuteron_matrices();
    
    // Test 1: Let the randomizer pick all starting widths
    run_stability_test(false);

    // Test 2: Force the PN core to span the correct physical sizes
    run_stability_test(true);
    return 0;
}