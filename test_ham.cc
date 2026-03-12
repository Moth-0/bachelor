// test_ham.cc -- Unit test for the W matrix element and Hermitian properties
#include <iostream>
#include <iomanip>
#include <complex>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/particle.h"
#include "qm/hamiltonian.h"

using namespace qm;

int main() {
    std::cout << "========================================================\n";
    std::cout << " TESTING HAMILTONIAN: W = <bare | W | clothed>          \n";
    std::cout << "========================================================\n\n";

    hamiltonian H;
    H.hbar_c = 197.3269804L; 

    // 1. Build a bare Gaussian (dim=1) - S-wave, no shifts
    gaus g_bare(1, 1.0, 0.0);
    g_bare.A(0,0) = 0.5L; 
    g_bare.zero_shifts();

    // 2. Build a clothed Gaussian (dim=2) - P-wave, shifted in Z (Channel A)
    gaus g_cloth_Z(2, 1.0, 0.0);
    g_cloth_Z.A(0,0) = 0.5L; g_cloth_Z.A(0,1) = 0.0L;
    g_cloth_Z.A(1,0) = 0.0L; g_cloth_Z.A(1,1) = 0.5L;
    g_cloth_Z.zero_shifts();
    g_cloth_Z.s(1, 2) = 1.0L; // Shift in Z direction

    // 3. Build a clothed Gaussian (dim=2) - P-wave, shifted in X (Channel B)
    gaus g_cloth_X = g_cloth_Z;
    g_cloth_X.zero_shifts();
    g_cloth_X.s(1, 0) = 1.0L; // Shift in X direction

    // 4. Set up the pion emission coupling (Proton -> Proton + pi0)
    // We use a simple projection vector c = [0.5, 1.0] and b_pion = 1.4, S_pion = 20.0
    long double b_pion = 1.4L;
    long double S_pion = 20.0L;
    vector c_coord = {0.5L, 1.0L}; 
    
    // Build the Omega form factor matrix: Omega_ij = (c_i * c_j) / b^2
    matrix Om(2, 2);
    long double inv_b2 = 1.0L / (b_pion * b_pion);
    for(int i=0; i<2; ++i)
        for(int j=0; j<2; ++j)
            Om(i,j) = inv_b2 * c_coord[i] * c_coord[j];

    // Build the coupling structure (sz=+0.5 -> sz=+0.5 emits pi0)
    NucleonCoupling coupling;
    coupling.c_coord = c_coord;
    coupling.strength = S_pion / b_pion;
    coupling.terms = apply_vertex(Nucleon::Proton(+0.5), Pion::PiZero());

    std::vector<NucleonCoupling> couplings = { coupling };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Test parameters: S_pion = " << S_pion << ", b_pion = " << b_pion << " fm\n\n";

    // --- TEST 1: Z-Shift (Channel A) ---
    cld W_Z = H.W(g_bare, g_cloth_Z, Om, couplings);
    std::cout << "--- TEST 1: Z-Shifted Clothed Gaussian (Channel A) ---\n";
    std::cout << "  Expected: Purely REAL matrix element.\n";
    std::cout << "  W_Z       = " << std::showpos << W_Z.real() << " + " << W_Z.imag() << "i MeV\n";
    std::cout << "  Conj(W_Z) = " << std::conj(W_Z).real() << " + " << std::conj(W_Z).imag() << "i MeV\n\n";

    // --- TEST 2: X-Shift (Channel B) ---
    // In this state, the spin operator sigma_- or sigma_+ acts on X. 
    // We expect the matrix element to reflect the complex coordinate mapping.
    // However, our initial state is sz=+0.5, and sigma_z acts on Z. 
    // Since this Gaussian is ONLY shifted in X, the Z-overlap is zero.
    cld W_X = H.W(g_bare, g_cloth_X, Om, couplings);
    std::cout << "--- TEST 2: X-Shifted Clothed Gaussian (Channel B, sz=+0.5 to sz=+0.5) ---\n";
    std::cout << "  Expected: ZERO (sigma_z operator has no X-spatial component).\n";
    std::cout << "  W_X       = " << std::showpos << W_X.real() << " + " << W_X.imag() << "i MeV\n\n";

    // --- TEST 3: X-Shift (Channel B) with Spin Flip ---
    // Let's create a coupling where sz=-0.5 transitions to sz=+0.5 via sigma_+
    NucleonCoupling coupling_flip;
    coupling_flip.c_coord = c_coord;
    coupling_flip.strength = S_pion / b_pion;
    coupling_flip.terms = apply_vertex(Nucleon::Proton(-0.5), Pion::PiZero());
    std::vector<NucleonCoupling> couplings_flip = { coupling_flip };

    cld W_flip = H.W(g_bare, g_cloth_X, Om, couplings_flip);
    std::cout << "--- TEST 3: X-Shifted Gaussian with Spin Flip (sz=-0.5 to sz=+0.5) ---\n";
    std::cout << "  Expected: Non-zero COMPLEX or REAL component from sigma_+.\n";
    std::cout << "  W_flip    = " << std::showpos << W_flip.real() << " + " << W_flip.imag() << "i MeV\n";
    std::cout << "  Conj(W)   = " << std::conj(W_flip).real() << " + " << std::conj(W_flip).imag() << "i MeV\n\n";

    // --- TEST 4: Magnitude Check (Is the explosion here?) ---
    std::cout << "--- TEST 4: Magnitude Check ---\n";
    std::cout << "  Checking if W scales to absurd values (e.g. > 10,000 MeV)...\n";
    if (std::abs(W_Z) > 1000.0L || std::abs(W_flip) > 1000.0L) {
        std::cout << "  [WARNING] W matrix elements are unusually large! Check 'strength' or B_inv scaling.\n";
    } else {
        std::cout << "  [OK] Magnitudes are physically reasonable ( < 1000 MeV ).\n";
    }

    return 0;
}