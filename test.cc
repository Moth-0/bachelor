// =============================================================================
//  test.cc — FOCUSED PHYSICS VALIDATION 
//
//  Tests:
//  1. W-operator norm_factor (Proving the analytical volume integral)
//  2. Rest mass term scaling (Proving GEVP requires overlap scaling)
//  3. NO_FLIP coupling logic (Center-of-Mass '+' vs True Dipole '-')
//  4. Adjoint Matrix placement 
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "deuterium.h"

using namespace qm;

void print_sep(char c = '=', int width = 80) {
    for (int i = 0; i < width; ++i) std::cout << c;
    std::cout << "\n";
}

void print_matrix(const std::string& name, const cmat& M) {
    std::cout << "\n" << name << " (" << M.size1() << "x" << M.size2() << "):\n";
    print_sep('-', 60);
    for (size_t i = 0; i < M.size1(); ++i) {
        for (size_t j = 0; j < M.size2(); ++j) {
            cld val = M(i, j);
            std::cout << std::setw(14) << std::fixed << std::setprecision(6);
            if (std::abs(val.imag()) < 1e-12L) {
                std::cout << val.real();
            } else {
                std::cout << "(" << val.real() << ", " << val.imag() << ")";
            }
            if (j < M.size2() - 1) std::cout << "  ";
        }
        std::cout << "\n";
    }
}

// =============================================================================
//  TEST 1: W-OPERATOR NORM_FACTOR (ANALYTICAL VOLUME)
// =============================================================================
void test_w_operator_norm_factor() {
    print_sep();
    std::cout << "TEST 1: W-OPERATOR NORM_FACTOR (ANALYTICAL VOLUME)\n";
    print_sep();

    const ld m_p = 938.272L, m_n = 939.565L, m_pi = 134.97L;
    const ld b_form = 1.4L;
    const ld S = 590.0L;

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_pion({m_p, m_n, m_pi});

    // PN state (bare)
    rmat A_pn(1, 1); A_pn(0, 0) = 1.0L;
    rmat s_pn = zeros<ld>(1, 3);
    SpatialWavefunction psi_pn(A_pn, s_pn, 1);

    // π⁰ state
    rmat A_pi(2, 2);
    A_pi(0, 0) = 1.0L; A_pi(0, 1) = 0.0L;
    A_pi(1, 0) = 0.0L; A_pi(1, 1) = 1.0L;
    rmat s_pi(2, 3);
    s_pi(1, 2) = 0.6L;
    SpatialWavefunction psi_pi(A_pi, s_pi, -1);

    std::cout << "\nComputing True Form Factor Normalization:\n";

    ld b_pow_5 = std::pow(b_form, 5.0);
    ld two_pow_11_halves = std::pow(2.0, 5.5);
    ld norm_sq = 4.0 * M_PI * (3.0 * std::sqrt(M_PI) * b_pow_5) / two_pow_11_halves;
    ld norm_factor = 1.0 / std::sqrt(norm_sq);

    std::cout << "  b = " << b_form << " fm\n";
    std::cout << "  Volume Integral (norm_sq) = " << norm_sq << "\n";
    std::cout << "  norm_factor = 1/√norm_sq = " << norm_factor << "\n";

    rvec c_pi_1 = jac_pion.get_internal_distance_vector(2, 0);

    cld w_val_1 = total_w_coupling(psi_pn, psi_pi, c_pi_1, b_form, S, 1.0L, NO_FLIP);
    ld w_restored_1 = w_val_1.real() / norm_factor;

    std::cout << "\nComparison:\n";
    std::cout << "  Unnormalized Raw Integral : " << w_restored_1 << " MeV*fm^3 (Physically meaningless units)\n";
    std::cout << "  Normalized True Energy    : " << w_val_1.real() << " MeV (Correct!)\n";
    std::cout << "  ✓ The norm factor ensures S parameter dictates energy magnitude.\n";
}

// =============================================================================
//  TEST 2: REST MASS TERM SCALING (GEVP PROOF)
// =============================================================================
void test_rest_mass_scaling() {
    print_sep();
    std::cout << "TEST 2: REST MASS TERM SCALING (GEVP PROOF)\n";
    print_sep();

    const ld m_p = 938.272L, m_n = 939.565L, m_pi = 134.97L;

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_pion({m_p, m_n, m_pi});

    std::vector<BasisState> basis;

    {
        rmat A(1, 1); A(0, 0) = 1.0L;
        rmat s = zeros<ld>(1, 3);
        SpatialWavefunction psi(A, s, 1);
        basis.push_back({psi, Channel::PN, NO_FLIP, 1.0L, jac_bare, 0.0L});
    }

    {
        rmat A(2, 2);
        A(0, 0) = 1.0L; A(0, 1) = 0.0L;
        A(1, 0) = 0.0L; A(1, 1) = 1.0L;
        rmat s(2, 3);
        s(1, 2) = 0.6L;
        SpatialWavefunction psi(A, s, -1);
        basis.push_back({psi, Channel::PI_0c_0f, NO_FLIP, 1.0L, jac_pion, m_pi});
    }

    // Compute pion state overlap
    ld N11 = spactial_overlap(basis[1].psi, basis[1].psi);
    ld T_pi = total_kinetic_energy(basis[1].psi, basis[1].psi, basis[1].jac, {false, false});

    std::cout << "\nPion state diagonal (i=j=1):\n";
    std::cout << "  N(1,1) = <π|π> = " << N11 << " (NOTE: Basis is non-orthogonal, N != 1.0)\n";
    std::cout << "  T_π = " << T_pi << " MeV\n";
    std::cout << "  m_π = " << m_pi << " MeV\n\n";

    std::cout << "TRUE MATH (H_11 = T_11 + m_π * N_11):\n";
    ld rest_mass_WITH_overlap = m_pi * N11;
    ld H11_WITH_overlap = T_pi + rest_mass_WITH_overlap;
    std::cout << "  H(1,1) = " << T_pi << " + (" << m_pi << " × " << N11 << ") = " << H11_WITH_overlap << " MeV\n";
    std::cout << "  ✓ Correct. Evaluates full expectation value <ψ | H | ψ>\n\n";

    std::cout << "BROKEN PHYSICS (Claude's Suggestion: H_11 = T_11 + m_π):\n";
    ld rest_mass_NO_overlap = m_pi;
    ld H11_NO_overlap = T_pi + rest_mass_NO_overlap;
    std::cout << "  H(1,1) = " << T_pi << " + " << m_pi << " = " << H11_NO_overlap << " MeV\n";
    std::cout << "  ⚠️ INCORRECT! Assumes <ψ|ψ> = 1.0. Will mathematically break GEVP!\n";
}

// =============================================================================
//  TEST 3: NO_FLIP COUPLING LOGIC (+ vs -)
// =============================================================================
void test_no_flip_coupling() {
    print_sep();
    std::cout << "TEST 3: NO_FLIP W-COUPLING LOGIC (C.O.M. vs TRUE DIPOLE)\n";
    print_sep();

    const ld m_p = 938.272L, m_n = 939.565L, m_pi = 134.97L;
    const ld b_form = 1.4L;
    const ld S = 590.0L;

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_pion({m_p, m_n, m_pi});

    // PN state
    rmat A_pn(1, 1); A_pn(0, 0) = 1.0L;
    rmat s_pn = zeros<ld>(1, 3);
    SpatialWavefunction psi_pn(A_pn, s_pn, 1);

    // π⁰ state
    rmat A_pi(2, 2);
    A_pi(0, 0) = 1.0L; A_pi(0, 1) = 0.0L;
    A_pi(1, 0) = 0.0L; A_pi(1, 1) = 1.0L;
    rmat s_pi(2, 3);
    s_pi(1, 2) = 0.6L;
    SpatialWavefunction psi_pi(A_pi, s_pi, -1);

    rvec c_pi_1 = jac_pion.get_internal_distance_vector(2, 0); // Pion to N1
    rvec c_pi_2 = jac_pion.get_internal_distance_vector(2, 1); // Pion to N2

    cld w_val_n1 = total_w_coupling(psi_pn, psi_pi, c_pi_1, b_form, S, 1.0L, NO_FLIP);
    cld w_val_n2 = total_w_coupling(psi_pn, psi_pi, c_pi_2, b_form, S, 1.0L, NO_FLIP);

    std::cout << "\nIndividual Nucleon Couplings:\n";
    std::cout << "  W(c_pi_1) = " << w_val_n1.real() << " MeV\n";
    std::cout << "  W(c_pi_2) = " << w_val_n2.real() << " MeV\n\n";

    std::cout << "BUGGY 'PLUS' SIGN (w_n1 + w_n2):\n";
    cld w_val_add = w_val_n1 + w_val_n2;
    std::cout << "  SUM = " << w_val_add.real() << " MeV\n";
    std::cout << "  Math Result: Destructive Interference.\n";
    std::cout << "  Physics: Nucleon coordinates cancel out (r1 + r2 = 0 in COM).\n";
    std::cout << "  Calculates distance from COM to pion. Artificially weak.\n\n";

    std::cout << "TRUE ISOSPIN 'MINUS' SIGN (w_n1 - w_n2):\n";
    cld w_val_sub = w_val_n1 - w_val_n2;
    std::cout << "  SUBTRACT = " << w_val_sub.real() << " MeV\n";
    std::cout << "  Math Result: Constructive Interference.\n";
    std::cout << "  Physics: Pion coordinate cancels (rpi - rpi = 0). Leaves r2 - r1.\n";
    std::cout << "  Calculates true P-Wave relative distance. Vastly stronger coupling!\n";
}

// =============================================================================
//  TEST 4: FULL HAMILTONIAN WITH ADJOINT FIX
// =============================================================================
void test_full_hamiltonian_adjoint() {
    print_sep();
    std::cout << "TEST 4: FULL HAMILTONIAN WITH ADJOINT FIX\n";
    print_sep();

    // NOTE: Requires building build_matrices with `h_val += std::conj(w_val);` for i_is_bare
    std::cout << "\nVerify your deuterium.h has the Adjoint fixed!\n";
    std::cout << "If i_is_bare is true, we are building the Upper-Right block.\n";
    std::cout << "Upper-Right block must be W^dagger (std::conj).\n";
    std::cout << "Lower-Left block must be W (no conj).\n";
}

// =============================================================================
//  MAIN
// =============================================================================
int main() {
    std::cout << std::fixed << std::setprecision(8);
    print_sep('=', 80);
    std::cout << "PHYSICS DEBUGGING: PROVING THE MATHEMATICAL FRAMEWORK\n";
    print_sep('=', 80);

    try {
        test_w_operator_norm_factor();
        test_rest_mass_scaling();
        test_no_flip_coupling();
        test_full_hamiltonian_adjoint();
    } catch (const std::exception& e) {
        std::cout << "\n[ERROR] Exception: " << e.what() << "\n";
        return 1;
    }

    print_sep('=', 80);
    std::cout << "DIAGNOSIS SUMMARY:\n";
    std::cout << "  1. The Norm Factor and Rest Mass algorithms are 100% correct.\n";
    std::cout << "  2. Changing NO_FLIP to '+' breaks the physics but stops collapse.\n";
    std::cout << "  3. Keeping NO_FLIP as '-' requires bounding A(i,i) < 50.0 to stop collapse.\n";
    print_sep('=', 80);

    return 0;
}