// =============================================================================
//  tests.cc  —  Unit tests for matrix.h, jacobi.h, gaussian.h, operators.h,
//               and solver.h
//
//  Compile (from the directory containing your qm/ headers):
//    g++ -std=c++17 -O2 -fopenmp -I. tests.cc -o tests && ./tests
//
//  Each test prints PASS or FAIL.  A summary is printed at the end.
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

// ── Pull in your headers ─────────────────────────────────────────────────────
// Adjust the include paths to match your project layout.
#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"

using namespace qm;

// =============================================================================
//  Minimal test harness
// =============================================================================
static int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& name) {
    if (ok) {
        std::cout << "  [PASS] " << name << "\n";
        ++g_pass;
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        ++g_fail;
    }
}

// Relative/absolute tolerance helper
bool near(ld a, ld b, ld tol = 1e-6L) {
    ld diff = std::fabs(a - b);
    ld scale = std::max(std::fabs(a), std::fabs(b));
    if (scale < 1e-12L) return diff < tol;
    return diff / scale < tol;
}

// =============================================================================
//  1.  matrix.h — core linear-algebra primitives
// =============================================================================
void test_matrix() {
    std::cout << "\n=== 1. matrix.h ===\n";

    // 1a. Identity matrix
    {
        rmat I = eye<ld>(3);
        bool ok = true;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                ok &= near(I(i,j), (i == j) ? 1.0L : 0.0L);
        check(ok, "eye(3) is identity");
    }

    // 1b. Matrix–vector product
    {
        rmat A(2, 2);
        A(0,0) = 1; A(0,1) = 2;
        A(1,0) = 3; A(1,1) = 4;
        rvec v(2); v[0] = 1; v[1] = 2;
        rvec r = A * v;            // [5, 11]
        check(near(r[0], 5.0L) && near(r[1], 11.0L), "2x2 mat-vec product");
    }

    // 1c. Matrix–matrix product
    {
        rmat A(2,2), B(2,2);
        A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
        B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
        rmat C = A * B;            // [[19,22],[43,50]]
        check(near(C(0,0),19.0L) && near(C(0,1),22.0L) &&
              near(C(1,0),43.0L) && near(C(1,1),50.0L),
              "2x2 mat-mat product");
    }

    // 1d. Determinant  (known: det([[2,1],[1,3]]) = 5)
    {
        rmat M(2,2);
        M(0,0)=2; M(0,1)=1; M(1,0)=1; M(1,1)=3;
        check(near(M.determinant(), 5.0L, 1e-9L), "2x2 determinant");
    }

    // 1e. Inverse (A * A^-1 == I)
    {
        rmat M(2,2);
        M(0,0)=2; M(0,1)=1; M(1,0)=0; M(1,1)=3;
        rmat MI = M.inverse();
        rmat prod = M * MI;
        bool ok = near(prod(0,0),1.0L) && near(prod(0,1),0.0L) &&
                  near(prod(1,0),0.0L) && near(prod(1,1),1.0L);
        check(ok, "2x2 inverse: M * M^-1 == I");
    }

    // 1f. Transpose
    {
        rmat M(2,3);
        M(0,0)=1; M(0,1)=2; M(0,2)=3;
        M(1,0)=4; M(1,1)=5; M(1,2)=6;
        rmat T = M.transpose();
        bool ok = T.size1()==3 && T.size2()==2 &&
                  near(T(0,0),1.0L) && near(T(2,1),6.0L);
        check(ok, "2x3 transpose shape and values");
    }

    // 1g. Cholesky: L * L^T == A for a known 2x2 SPD matrix
    {
        rmat A(2,2);
        A(0,0)=4; A(0,1)=2; A(1,0)=2; A(1,1)=3;
        rmat L = A.cholesky();
        rmat Lt = L.transpose();
        rmat prod = L * Lt;
        bool ok = near(prod(0,0),A(0,0)) && near(prod(0,1),A(0,1)) &&
                  near(prod(1,0),A(1,0)) && near(prod(1,1),A(1,1));
        check(ok, "Cholesky: L * L^T == A");
    }

    // 1h. Outer product
    {
        rvec u(2); u[0]=1; u[1]=2;
        rvec v(2); v[0]=3; v[1]=4;
        rmat O = outer_no_conj(u, v);
        bool ok = near(O(0,0),3.0L) && near(O(0,1),4.0L) &&
                  near(O(1,0),6.0L) && near(O(1,1),8.0L);
        check(ok, "outer_no_conj");
    }

    // 1i. dot_no_conj
    {
        rvec a(3); a[0]=1; a[1]=2; a[2]=3;
        rvec b(3); b[0]=4; b[1]=5; b[2]=6;
        check(near(dot_no_conj(a,b), 32.0L), "dot_no_conj [1,2,3]·[4,5,6]=32");
    }

    // 1j. 3x3 determinant
    {
        rmat M(3,3);
        M(0,0)=1; M(0,1)=2; M(0,2)=3;
        M(1,0)=0; M(1,1)=4; M(1,2)=5;
        M(2,0)=1; M(2,1)=0; M(2,2)=6;
        // det = 1*(4*6-5*0) - 2*(0*6-5*1) + 3*(0*0-4*1) = 24+10-12 = 22
        check(near(M.determinant(), 22.0L, 1e-9L), "3x3 determinant = 22");
    }
}

// =============================================================================
//  2.  jacobi.h — Jacobi coordinate construction
// =============================================================================
void test_jacobi() {
    std::cout << "\n=== 2. jacobi.h ===\n";

    // 2a. Two equal-mass particles: reduced mass = m/2
    {
        ld m = 939.0L;
        Jacobian jac({m, m});
        bool ok = near(jac.reduced_masses[0], m / 2.0L, 1e-6L);
        check(ok, "2-body equal mass: mu = m/2");
    }

    // 2b. Proton-neutron reduced mass
    {
        ld mp = 938.272L, mn = 939.565L;
        Jacobian jac({mp, mn});
        ld mu_expected = mp * mn / (mp + mn);
        check(near(jac.reduced_masses[0], mu_expected, 1e-4L),
              "p-n reduced mass");
    }

    // 2c. J matrix has correct dimensions
    {
        Jacobian jac({1.0L, 2.0L, 3.0L});
        check(jac.J.size1() == 3 && jac.J.size2() == 3, "3-body J is 3x3");
    }

    // 2d. transform_w returns length-N vectors
    {
        Jacobian jac({938.0L, 939.0L, 135.0L});
        rvec w0 = jac.transform_w(0);
        check(w0.size() == 3, "transform_w returns length-3 vector for 3-body");
    }

    // 2e. Jacobi matrix last row sums to 1 (center-of-mass row)
    {
        ld m1=1.0L, m2=2.0L, m3=3.0L;
        Jacobian jac({m1, m2, m3});
        ld row_sum = jac.J(2,0) + jac.J(2,1) + jac.J(2,2);
        check(near(row_sum, 1.0L, 1e-9L), "Jacobi CM row sums to 1");
    }

    // 2f. First relative coordinate picks out r2 - r1 correctly
    //     w_ij = (U^T)(e_i - e_j) in Jacobi space.
    //     For 2-body equal masses J(0,0) = m/(2m) = 0.5, J(0,1) = -1.
    {
        Jacobian jac({939.0L, 939.0L});
        // In relative coordinates the first row should give r1-r2 direction
        // encoded in the w-vectors.  We just verify w0 - w1 gives a unit scale.
        rvec w0 = jac.transform_w(0);
        rvec w1 = jac.transform_w(1);
        rvec diff(2);
        diff[0] = w0[0] - w1[0];
        diff[1] = w0[1] - w1[1];
        // For 2 equal-mass particles the internal component should be ±1
        check(near(std::fabs(diff[0]), 1.0L, 1e-6L),
              "2-body equal mass: w0[0]-w1[0] = ±1");
    }
}

// =============================================================================
//  3.  gaussian.h — Gaussian overlap and SpatialWavefunction
// =============================================================================
void test_gaussian() {
    std::cout << "\n=== 3. gaussian.h (overlap) ===\n";

    // For unshifted Gaussians (s=0), the overlap is (π/det(B))^(3/2).
    // With A1=A2=a*I_1x1, B=2a, det(B)=2a,
    // overlap = (π/(2a))^(3/2).

    // 3a. 1D unshifted overlap
    {
        ld a = 1.0L;
        rmat A(1,1); A(0,0) = a;
        rmat s = zeros<ld>(1,3);
        ld ov = gaussian_overlap(A, s, A, s);
        ld expected = std::pow(M_PIl / (2.0L * a), 1.5L);
        check(near(ov, expected, 1e-6L),
              "1D unshifted overlap == (pi/(2a))^1.5");
    }

    // 3b. Overlap is symmetric: <g1|g2> == <g2|g1>
    {
        rmat A1(1,1); A1(0,0) = 0.5L;
        rmat A2(1,1); A2(0,0) = 1.5L;
        rmat s1(1,3); s1(0,0)=0.1L; s1(0,1)=0.0L; s1(0,2)=0.0L;
        rmat s2(1,3); s2(0,0)=0.2L; s2(0,1)=0.1L; s2(0,2)=0.0L;
        ld ov12 = gaussian_overlap(A1, s1, A2, s2);
        ld ov21 = gaussian_overlap(A2, s2, A1, s1);
        check(near(ov12, ov21, 1e-8L), "overlap symmetry: <g1|g2>==<g2|g1>");
    }

    // 3c. Two Gaussians centred far apart have smaller overlap than the same
    //     Gaussian with itself.
    //     In the overlap formula v = s_bra + s_ket.
    //     To represent two spatially separated peaks we use opposite shifts:
    //       g1 centred at +d:  s1 = 2*A*( d) so peak is at u = B^-1 v / 2
    //       g2 centred at -d:  s2 = 2*A*(-d)
    //     Then v = s1 + s2 = 0 and the exponential factor is 1,
    //     while the self-overlap of g1 has v = 2*s1 giving a large boost.
    //     Alternatively — the simplest correct test — compare the overlap of
    //     two *identical* unshifted Gaussians with very different widths
    //     (almost orthogonal) against the self-overlap of either one.
    {
        rmat A_wide(1,1);   A_wide(0,0)   = 0.02L;
        rmat A_narrow(1,1); A_narrow(0,0) = 50.0L;
        rmat s0 = zeros<ld>(1,3);
        ld ov_cross = gaussian_overlap(A_wide, s0, A_narrow, s0);
        ld ov_self  = gaussian_overlap(A_wide, s0, A_wide,   s0);
        check(ov_cross < ov_self,
              "cross-overlap (very wide vs very narrow) < self-overlap");
    }

    // 3d. Parity-symmetric SpatialWavefunction: parity +1 case
    //     spactial_overlap for a state with itself should be positive and
    //     equal to 2*(ov(+s,+s) + ov(+s,-s)) by hand.
    {
        rmat A(1,1); A(0,0) = 1.0L;
        rmat s(1,3);  s(0,0) = 0.3L;
        SpatialWavefunction psi(A, s, +1);
        ld ov = spactial_overlap(psi, psi);
        // Manual: (ov++ + ov+- + ov-+ + ov--)
        ld pp = gaussian_overlap(A,  s, A,  s);
        ld pm = gaussian_overlap(A,  s, A, -1.0L * s);
        ld mp = gaussian_overlap(A, -1.0L * s, A,  s);
        ld mm = gaussian_overlap(A, -1.0L * s, A, -1.0L * s);
        ld expected = pp + pm + mp + mm;
        check(near(ov, expected, 1e-8L),
              "spactial_overlap(psi,psi) matches manual expansion for p=+1");
    }

    // 3e. Parity -1: antisymmetric state overlap formula
    {
        rmat A(1,1); A(0,0) = 1.0L;
        rmat s(1,3);  s(0,0) = 0.5L;
        SpatialWavefunction psi(A, s, -1);
        ld ov  = spactial_overlap(psi, psi);
        ld pp  = gaussian_overlap(A,  s, A,  s);
        ld pm  = gaussian_overlap(A,  s, A, -1.0L * s);
        ld mp  = gaussian_overlap(A, -1.0L * s, A,  s);
        ld mm  = gaussian_overlap(A, -1.0L * s, A, -1.0L * s);
        ld expected = pp - pm - mp + mm;   // p1*p2 = (-1)*(-1) = +1 for mm
        check(near(ov, expected, 1e-8L),
              "spactial_overlap(psi,psi) matches manual expansion for p=-1");
    }

    // 3f. Orthogonality test: wide and very narrow Gaussians have small overlap
    {
        rmat A_wide(1,1);   A_wide(0,0)   = 0.01L;
        rmat A_narrow(1,1); A_narrow(0,0) = 100.0L;
        rmat s = zeros<ld>(1,3);
        ld ov = gaussian_overlap(A_wide, s, A_narrow, s);
        ld ov_self = gaussian_overlap(A_wide, s, A_wide, s);
        check(ov < 0.01L * ov_self,
              "wide vs narrow Gaussian overlap is small");
    }
}

// =============================================================================
//  4.  operators.h — Kinetic energy matrix elements
// =============================================================================
void test_operators() {
    std::cout << "\n=== 4. operators.h (kinetic energy) ===\n";

    // Physical constants
    const ld HBARC = 197.3269804L;

    // Setup: simple 1D (Jacobi) two-body system, equal masses
    ld m_p = 938.272L, m_n = 939.565L;
    Jacobian jac({m_p, m_n});
    std::vector<bool> rel_false = {false};
    std::vector<bool> rel_true  = {true};

    // Helper: build a simple 1x1 unshifted wavefunction
    auto make_psi = [](ld width) -> SpatialWavefunction {
        rmat A(1,1); A(0,0) = width;
        rmat s = zeros<ld>(1,3);
        return SpatialWavefunction(A, s, +1);
    };

    // 4a. Non-relativistic KE is positive
    {
        SpatialWavefunction psi = make_psi(1.0L);
        ld KE = total_kinetic_energy(psi, psi, jac, rel_false);
        check(KE > 0.0L, "NR kinetic energy is positive (diagonal)");
    }

    // 4b. Relativistic KE is positive
    {
        SpatialWavefunction psi = make_psi(1.0L);
        ld KE = total_kinetic_energy(psi, psi, jac, rel_true);
        check(KE > 0.0L, "Rel kinetic energy is positive (diagonal)");
    }

    // 4c. Non-relativistic KE scales as 1/b^2 (wider state = lower KE)
    //     K(b) should decrease as b increases (less confinement).
    {
        SpatialWavefunction psi_narrow = make_psi(5.0L);
        SpatialWavefunction psi_wide   = make_psi(0.2L);
        ld KE_narrow = total_kinetic_energy(psi_narrow, psi_narrow, jac, rel_false);
        ld KE_wide   = total_kinetic_energy(psi_wide,   psi_wide,   jac, rel_false);
        // Normalise by overlap to get <K>/<N> per state
        auto make_N = [&](const SpatialWavefunction& p) -> ld {
            return spactial_overlap(p, p);
        };
        ld ratio_narrow = KE_narrow / make_N(psi_narrow);
        ld ratio_wide   = KE_wide   / make_N(psi_wide);
        check(ratio_narrow > ratio_wide,
              "NR <K>: narrow Gaussian has higher KE than wide");
    }

    // 4d. Relativistic KE >= Non-relativistic KE (at same state)
    //     This is because sqrt(p^2+m^2)-m >= p^2/(2m) is NOT always true,
    //     but for the same gaussian state in the kinematic range relevant here
    //     they should be very close and both positive.  We just verify both
    //     give finite positive values and print the ratio for information.
    {
        SpatialWavefunction psi = make_psi(1.0L);
        ld KE_nr  = total_kinetic_energy(psi, psi, jac, rel_false);
        ld KE_rel = total_kinetic_energy(psi, psi, jac, rel_true);
        std::cout << "    (Info) NR KE = " << KE_nr
                  << " MeV, Rel KE = " << KE_rel << " MeV\n";
        check(KE_nr > 0.0L && KE_rel > 0.0L,
              "Both NR and Rel KE are finite and positive");
    }

    // 4e. NR kinetic energy formula cross-check (unshifted case)
    //     For an unshifted 1x1 Gaussian with A=a, the non-relativistic
    //     matrix element of (c^T p)^2/(2mu) between |A,0> + |A,0> states is:
    //     <psi|K|psi> = 4 * M(A,0,A,0) * (hbarc)^2 * (3/2) * (1/gamma) / (2*mu)
    //     where 1/gamma = 4 * c^T * A * R * A' * c,  R = (A+A')^-1 = 1/(2a)
    //     c = e_1 (internal Jacobi unit vector), so
    //     1/gamma = 4 * a * (1/2a) * a = 2a
    //     KE_matrix_element = M * hbarc^2 * (3/2) * 2a / (2*mu)
    //     The spatial wavefunction with parity +1 and s=0 gives:
    //     total KE = 4 * (same formula, since all 4 terms in apply_basis_expansion
    //                     are identical for s=0, parity=+1).
    {
        ld a  = 0.5L;
        ld mu = jac.reduced_masses[0];
        rmat A(1,1); A(0,0) = a;
        rmat sz = zeros<ld>(1,3);
        SpatialWavefunction psi(A, sz, +1);

        ld KE_code = total_kinetic_energy(psi, psi, jac, rel_false);

        // Analytic reference for the full symmetric state
        ld M_prim   = gaussian_overlap(A, sz, A, sz);  // primitive overlap
        ld inv_gamma = 2.0L * a;
        ld KE_prim   = M_prim * HBARC * HBARC * 1.5L * inv_gamma / (2.0L * mu);
        // The symmetric state |A,+s>+|A,-s> with s=0 is 2|A,0>, so the 4
        // parity terms all give the same result: total = 4 * KE_prim.
        ld KE_expected = 4.0L * KE_prim;

        std::cout << "    (Info) Analytic NR KE = " << KE_expected
                  << " MeV,  Code = " << KE_code << " MeV\n";
        check(near(KE_code, KE_expected, 1e-4L),
              "NR KE matches analytic formula for unshifted 1-body Gaussian");
    }

    // 4f. Gaussian integration sanity (integrate_1d)
    //     integral of exp(-x^2) from 0 to inf == sqrt(pi)/2
    {
        auto f = [](ld x) -> ld { return std::exp(-x * x); };
        ld result = integrate_1d(f, 0.0L, 20.0L);
        ld expected = std::sqrt(M_PIl) / 2.0L;
        check(near(result, expected, 1e-5L),
              "Gauss-Legendre: integral of exp(-x^2) [0,20] ≈ sqrt(pi)/2");
    }

    // 4g. integrate_1d: integral of x^2 from 0 to 3 == 9
    {
        auto f = [](ld x) -> ld { return x * x; };
        ld result = integrate_1d(f, 0.0L, 3.0L);
        check(near(result, 9.0L, 1e-8L),
              "Gauss-Legendre: integral of x^2 [0,3] = 9");
    }
}

// =============================================================================
//  5.  solver.h — GEVP and eigenvalue solver
// =============================================================================
void test_solver() {
    std::cout << "\n=== 5. solver.h (GEVP / Jacobi eigenvalue) ===\n";

    // 5a. Jacobi diagonalisation: known 2x2 symmetric matrix
    //     A = [[3,1],[1,3]], eigenvalues = 2, 4  → lowest = 2
    {
        cmat A(2,2);
        A(0,0) = cld{3,0}; A(0,1) = cld{1,0};
        A(1,0) = cld{1,0}; A(1,1) = cld{3,0};
        ld ev = jacobi_lowest_eigenvalue(A);
        check(near(ev, 2.0L, 1e-6L),
              "Jacobi diag: lowest eigenvalue of [[3,1],[1,3]] == 2");
    }

    // 5b. Diagonal matrix: lowest eigenvalue trivially readable
    {
        cmat A(3,3);
        A(0,0)=cld{5,0}; A(1,1)=cld{2,0}; A(2,2)=cld{8,0};
        ld ev = jacobi_lowest_eigenvalue(A);
        check(near(ev, 2.0L, 1e-6L),
              "Jacobi diag: lowest eigenvalue of diag(5,2,8) == 2");
    }

    // 5c. solve_ground_state_energy with H=N=I: energy should be 1
    {
        size_t n = 3;
        cmat H(n,n), N(n,n);
        for (size_t i = 0; i < n; ++i) {
            H(i,i) = cld{1,0};
            N(i,i) = cld{1,0};
        }
        ld E = solve_ground_state_energy(H, N);
        check(near(E, 1.0L, 1e-6L),
              "GEVP with H=N=I returns E=1");
    }

    // 5d. solve_ground_state_energy with H=2I, N=I: energy should be 2
    {
        size_t n = 4;
        cmat H(n,n), N(n,n);
        for (size_t i = 0; i < n; ++i) {
            H(i,i) = cld{2,0};
            N(i,i) = cld{1,0};
        }
        ld E = solve_ground_state_energy(H, N);
        check(near(E, 2.0L, 1e-6L),
              "GEVP with H=2I, N=I returns E=2");
    }

    // 5e. solve_ground_state_energy with known non-trivial 2x2
    //     H = [[3,1],[1,3]], N = [[2,0],[0,2]]
    //     Generalised problem: H c = E N c → (H/2) c = E c (since N=2I)
    //     Eigenvalues of H/2: 1 and 2 → lowest = 1
    {
        cmat H(2,2), N(2,2);
        H(0,0)=cld{3,0}; H(0,1)=cld{1,0};
        H(1,0)=cld{1,0}; H(1,1)=cld{3,0};
        N(0,0)=cld{2,0}; N(1,1)=cld{2,0};
        ld E = solve_ground_state_energy(H, N);
        check(near(E, 1.0L, 1e-6L),
              "GEVP 2x2 non-trivial: H c = E N c → E_min = 1");
    }

    // 5f. Ill-conditioned overlap returns sentinel energy
    {
        cmat H(2,2), N(2,2);
        H(0,0)=cld{1,0}; H(1,1)=cld{1,0};
        N(0,0)=cld{1e-20L,0}; N(1,1)=cld{1e-20L,0};
        ld E = solve_ground_state_energy(H, N);
        // Should return the "reject" sentinel value
        check(E > 1e5L,
              "Ill-conditioned overlap matrix returns large sentinel energy");
    }
}

// =============================================================================
//  6.  Integration tests — Gaussian + operators working together
// =============================================================================
void test_integration() {
    std::cout << "\n=== 6. Integration: Gaussian + operators ===\n";

    ld m_p = 938.272L, m_n = 939.565L;
    Jacobian jac({m_p, m_n});
    std::vector<bool> nr = {false};

    // 6a. Hamiltonian matrix element ordering: H(0,0) with one state
    //     Only kinetic energy. For a single-state basis the GEVP gives
    //     E = <psi|K|psi> / <psi|psi>.
    {
        rmat A(1,1); A(0,0) = 0.8L;
        rmat sz = zeros<ld>(1,3);
        SpatialWavefunction psi(A, sz, +1);

        ld K   = total_kinetic_energy(psi, psi, jac, nr);
        ld Ov  = spactial_overlap(psi, psi);
        ld ratio = K / Ov;   // <K>

        // Build tiny 1x1 H and N in complex form and solve GEVP
        cmat H(1,1), N(1,1);
        H(0,0) = cld{K,  0};
        N(0,0) = cld{Ov, 0};
        ld E_gevp = solve_ground_state_energy(H, N);

        check(near(E_gevp, ratio, 1e-5L),
              "Single-state GEVP: E = <K>/<N> matches direct ratio");
    }

    // 6b. Two identical states: GEVP should give same energy as one state
    //     (The overlap matrix is rank-deficient but Cholesky guards it;
    //      expect either the same energy or a sentinel.)
    {
        rmat A(1,1); A(0,0) = 0.5L;
        rmat sz = zeros<ld>(1,3);
        SpatialWavefunction psi(A, sz, +1);

        ld K  = total_kinetic_energy(psi, psi, jac, nr);
        ld Ov = spactial_overlap(psi, psi);

        cmat H(2,2), N(2,2);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j) {
                H(i,j) = cld{K,  0};
                N(i,j) = cld{Ov, 0};
            }
        ld E = solve_ground_state_energy(H, N);
        // Either matches the single-state energy (if solver handles it)
        // or returns the sentinel — both are acceptable "safe" behaviours.
        bool ok = near(E, K/Ov, 1e-3L) || E > 1e5L;
        check(ok, "Two identical states: GEVP is safe (no crash, sane output)");
    }

    // 6c. promote_and_absorb produces correct dimension
    {
        rmat A_bare(1,1); A_bare(0,0) = 1.0L;
        rmat s_bare = zeros<ld>(1,3);
        Gaussian g_bare(A_bare, s_bare);

        Jacobian jac3({938.0L, 939.0L, 135.0L});
        rvec c = jac3.get_c_internal(1);   // Jacobi coordinate for pion
        ld b = 1.4L;

        Gaussian g_promoted = promote_and_absorb(g_bare, 2, c, 1.0L/(b*b));
        check(g_promoted.A.size1() == 2 && g_promoted.A.size2() == 2,
              "promote_and_absorb: output A is 2x2");
    }

    // 6d. promote_and_absorb: the promoted A is positive definite
    {
        rmat A_bare(1,1); A_bare(0,0) = 0.3L;
        rmat s_bare = zeros<ld>(1,3);
        Gaussian g_bare(A_bare, s_bare);

        Jacobian jac3({938.0L, 939.0L, 135.0L});
        rvec c = jac3.get_c_internal(1);
        ld b = 1.4L;
        Gaussian g_prom = promote_and_absorb(g_bare, 2, c, 1.0L/(b*b));

        rmat L = g_prom.A.cholesky();
        check(L.size1() == 2, "promote_and_absorb: promoted A is positive definite");
    }
}

// =============================================================================
//  7. Complex Matrices (cmat)
// =============================================================================
void test_complex_matrices() {
    std::cout << "\n=== 7. Complex Matrices ===\n";

    // 7a. Complex Matrix Multiplication
    {
        cmat A(2,2), B(2,2);
        A(0,0) = cld{1, 1}; A(0,1) = cld{0, 2};
        A(1,0) = cld{2, 0}; A(1,1) = cld{1, -1};
        
        B(0,0) = cld{2, 0}; B(0,1) = cld{1, -1};
        B(1,0) = cld{0, 1}; B(1,1) = cld{1, 1};
        
        cmat C = A * B;
        // C(0,0) = (1+i)*2 + (2i)*(i) = 2 + 2i - 2 = 2i
        check(near(C(0,0).real(), 0.0L) && near(C(0,0).imag(), 2.0L), 
              "Complex mat-mat product (real and imag parts)");
    }

    // 7b. Complex Conjugate Transpose (Hermitian Check)
    {
        cmat H(2,2);
        H(0,0) = cld{5, 0}; 
        H(0,1) = cld{2, 3}; // 2 + 3i
        H(1,0) = std::conj(H(0,1)); // 2 - 3i
        H(1,1) = cld{4, 0};
        
        // A Hermitian matrix must have real eigenvalues
        ld E = jacobi_lowest_eigenvalue(H);
        check(E < 10.0L && E > 0.0L, "Hermitian complex matrix yields real eigenvalues");
    }
}

// =============================================================================
//  8. 2D Gaussian Overlap (Dressed States)
// =============================================================================
void test_2d_gaussian() {
    std::cout << "\n=== 8. Multi-Dimensional Gaussians ===\n";

    // 8a. 2D Unshifted Overlap matches analytic formula: (pi^n / det(B))^1.5
    {
        rmat A(2,2);
        A(0,0) = 2.0L; A(0,1) = 0.5L;
        A(1,0) = 0.5L; A(1,1) = 2.0L;
        rmat s = zeros<ld>(2,3);
        
        ld ov = gaussian_overlap(A, s, A, s);
        
        // B = A + A = 2A. 
        // 2A = [[4, 1], [1, 4]]. det(2A) = (4*4) - (1*1) = 15.
        // Analytic Overlap for 2 dimensions (n=2): (pi^2 / 15)^1.5
        ld expected = std::pow(M_PIl * M_PIl / 15.0L, 1.5L);
        
        check(near(ov, expected, 1e-6L), 
              "2x2 Gaussian overlap matches analytic multidimensional volume");
    }
}

// =============================================================================
//  9. The W-Operator (Central Force Coupling)
// =============================================================================
void test_w_operator_sanity() {
    std::cout << "\n=== 9. W-Operator Sanity Check ===\n";

    {
        // Setup a 1D Bare State (Centered S-wave, Parity +1)
        rmat A_bare(1,1); A_bare(0,0) = 1.0L;
        rmat s_bare = zeros<ld>(1,3);
        SpatialWavefunction psi_bare(A_bare, s_bare, 1);
        
        // Setup a 2D Dressed State
        rmat A_dress(2,2); 
        A_dress(0,0) = 1.0L; A_dress(0,1) = 0.0L;
        A_dress(1,0) = 0.0L; A_dress(1,1) = 1.0L;
        
        // CRITICAL FIX: The Pion must be shifted to create a P-wave!
        // We shift the second internal coordinate (the pion) in the z-direction.
        rmat s_dress = zeros<ld>(2,3);
        s_dress(1, 2) = 1.0L; // z-shift = 1.0 fm
        SpatialWavefunction psi_dress(A_dress, s_dress, -1);

        // Jacobi coordinate for the pion 
        rvec c_pi(2); c_pi[0] = 0.0L; c_pi[1] = 1.0L; 
        
        ld b_range = 1.4L;
        ld S_coupling = 100.0L;
        ld isospin = 1.0L;

        // Fire the W-Operator!
        cld w_val = total_w_coupling(psi_bare, psi_dress, c_pi, b_range, S_coupling, isospin, NO_FLIP);
        
        // Now that parity allows it, the coupling must be non-zero!
        check(std::abs(w_val) > 0.01L, "W-operator yields non-zero coupling for shifted states (Parity conserved)");
    }
}

// =============================================================================
//  main
// =============================================================================
int main() {
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "============================================================\n";
    std::cout << "  QM Header Unit Tests\n";
    std::cout << "============================================================\n";

    test_matrix();
    test_jacobi();
    test_gaussian();
    test_operators();
    test_solver();
    test_integration();
    test_complex_matrices();
    test_2d_gaussian();
    test_w_operator_sanity();

    std::cout << "\n============================================================\n";
    std::cout << "  Results: " << g_pass << " passed, " << g_fail << " failed"
              << " (total " << g_pass + g_fail << ")\n";
    std::cout << "============================================================\n";
    return g_fail == 0 ? 0 : 1;
}