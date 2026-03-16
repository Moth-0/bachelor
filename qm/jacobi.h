#pragma once

#include "matrix.h"
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// jacobi.h  — N-body Jacobi coordinate system
//
// Implements the standard "chain" Jacobi transformation for N particles:
//   x[0] = r[0] - r[1]                            (relative coord, pair 0-1)
//   x[1] = CM(0,1) - r[2]                         (relative coord, add particle 2)
//   ...
//   x[N-2] = CM(0..N-2) - r[N-1]
//   x[N-1] = CM(all)                               (CM — discarded in practice)
//
// All matrix elements and vectors needed by hamiltonian.h are pre-computed:
//   - Reduced masses  mu[α]  for kinetic energy
//   - A_kin matrix           for total kinetic energy expression
//   - w_particle[i]          to extract particle i's position in Jacobi space
//   - w_rel(i,j)             to extract r[j]-r[i] in Jacobi space
//   - k_particle[i]          to extract particle i's momentum in Jacobi space
//   - promote(), build_A_tilde()  for W-operator matrix element construction
//
// Reference:  Thesis ch.1 (Jacobi matrix eq.1)
//             SVM book eq. 2.5, 2.10, 2.11
//             Fedorov "Analytic matrix elements"
//
// Units:  masses in MeV,  lengths in fm,  hbar*c = 197.3269804 MeV·fm
// ─────────────────────────────────────────────────────────────────────────────

namespace qm {

// ── Physical constants ───────────────────────────────────────────────────────
namespace phys {
    constexpr ld hbar_c = 197.3269804L;  // MeV · fm
    constexpr ld hbar_c2 = hbar_c * hbar_c;
}

// ─────────────────────────────────────────────────────────────────────────────
// JacobiSystem
//
// Construction:
//   JacobiSystem sys({m_p, m_n, m_pi}, {"p","n","pi"});
//
// For a 3-body system the standard Jacobi ordering is:
//   Particle 0 = proton  (or nucleon A)
//   Particle 1 = neutron (or nucleon B)
//   Particle 2 = pion    (or meson)
//
// The same class is used for 4-body (e.g. tritium core + meson) by simply
// passing 4 masses — all formulas are N-body general.
// ─────────────────────────────────────────────────────────────────────────────
class JacobiSystem {
public:
    // ── Public data (all pre-computed in build()) ─────────────────────────────
    size_t              N;          // number of particles
    rvec                m;          // masses[i]  (MeV)
    std::vector<std::string> names; // particle labels, e.g. {"p","n","pi"}
    ld                  M_total;    // sum of all masses
    rvec                M_cum;      // M_cum[k] = m[0]+...+m[k]

    // Full N×N Jacobi matrix J  (last row = CM)
    // J[α][j]:  m[j]/M_cum[α]  for j <= α
    //           -1              for j == α+1
    //            0              for j >  α+1
    // (last row α=N-1 is pure CM, all entries m[j]/M_total)
    rmat J;

    // Inverse U = J^{-1}.
    // r[i] = sum_j  U(i,j) * x[j]   (recovers particle position from Jacobi coords)
    rmat U;

    // Reduced masses  μ[α]  for Jacobi coordinate α = 0..N-2
    //   μ[α] = M_cum[α] * m[α+1] / M_cum[α+1]
    // The kinetic energy of mode α is  T[α] = π[α]² / (2 μ[α])
    rvec  mu;

    // Kinetic energy matrix  (N-1)×(N-1)  in Jacobi momentum space.
    //   A_kin[α][β] = sum_{k=0}^{N-1}  J(α,k) * J(β,k) / m[k]
    //   T_int = (1/2) sum_{α,β}  A_kin[α][β]  π[α]·π[β]
    // For the standard chain Jacobi ordering, A_kin is diagonal:
    //   A_kin[α][α] = 1/μ[α]   (verifiable from the formula)
    rmat A_kin;

    // w_particle[i]  = (N-1)-dim vector in Jacobi position space.
    //   Extracts particle i's position:  r[i] = w_particle[i] · x   (in CM frame)
    //   = first N-1 elements of row i of U
    rmat w_particle;

    // k_particle[i]  = (N-1)-dim vector in Jacobi momentum space.
    //   Momentum of particle i in original space expressed via Jacobi momenta:
    //   p[i] = k_particle[i] · π
    //   = first N-1 elements of column i of J   = J[0..N-2, i]
    rmat k_particle;

    // c_vec[α]  = e_α  unit vector in (N-1)-dim Jacobi momentum space.
    //   Used in kinetic energy integrals: f(c_α^T π) = f(π[α])
    rmat c_vec;

    // ── Constructors ──────────────────────────────────────────────────────────
    JacobiSystem() = default;

    // masses  : particle masses in MeV, ordered as the Jacobi chain
    // pnames  : optional labels for debug output
    JacobiSystem(rvec masses,
                 std::vector<std::string> pnames = {})
        : N(masses.size()), m(masses), names(pnames), M_total(0.0), M_cum(masses.size())
    {
        assert(N >= 2 && "Need at least 2 particles");
        if (names.empty())
            for (size_t i = 0; i < N; i++)
                names.push_back("p" + std::to_string(i));
        build();
    }

    // ── Relative-coordinate extraction (the key physics accessors) ───────────

    // w(i, j)  →  vector extracting  r[j] - r[i]  in Jacobi space
    // e.g.  w(0,2)  gives the vector for r_pion - r_proton
    rvec w_rel(size_t from_idx, size_t to_idx) const {
        assert(from_idx < N && to_idx < N);
        // w_particle is stored as an N×(N-1) rmat (rows=particles, cols=Jacobi coords).
        // operator[](j) returns the j-th COLUMN, so we must extract rows explicitly.
        rvec w_to(N-1), w_from(N-1);
        for (size_t alpha = 0; alpha < N-1; alpha++) {
            w_to[alpha]   = w_particle(to_idx,   alpha);
            w_from[alpha] = w_particle(from_idx, alpha);
        }
        return w_to - w_from;
    }

    // 3-body convenience aliases (assumes ordering: 0=proton, 1=neutron, 2=meson)
    rvec w_meson_proton()  const { return w_rel(0, 2); }  // r_π - r_p
    rvec w_meson_neutron() const { return w_rel(1, 2); }  // r_π - r_n
    rvec w_proton_neutron() const { return w_rel(0, 1); } // r_n - r_p  (= -x[0] direction)

    // ── Promotion of a lower-dimensional Gaussian matrix ─────────────────────
    //
    // The bare pn state lives in a 1D Jacobi space (only x[0] = r_p - r_n).
    // To compute overlap with a 3-body dressed state, we promote A_bare (1×1)
    // to the full (N-1)×(N-1) space by padding with zeros:
    //
    //   A_promoted = | A_bare  0 |
    //                |   0     0 |
    //
    // This reflects that the bare state has no correlation involving the pion
    // coordinate x[1] — the pion is "absent".
    //
    // coord_dim  = dimension of A_small  (e.g. 1 for a 2-body pn state)
    rmat promote(const rmat& A_small, size_t coord_dim) const {
        assert(A_small.size1() == coord_dim);
        assert(A_small.size2() == coord_dim);
        assert(coord_dim <= N-1);
        rmat A_full(N-1, N-1);   // zero-initialised
        for (size_t i = 0; i < coord_dim; i++)
            for (size_t j = 0; j < coord_dim; j++)
                A_full(i, j) = A_small(i, j);
        return A_full;
    }

    // ── Absorb the pion form factor into the Gaussian matrix ─────────────────
    //
    // The W-operator contains  f(r_πN) = S · exp(-r_πN² / b²).
    // In the Gaussian framework, multiplying by this form factor is equivalent
    // to adding its correlation matrix to A:
    //
    //   A_tilde = A_promoted + (1/b²) · w_πN · w_πN^T
    //
    // where w_πN is the Jacobi-space vector for r_π - r_N (2D for N=3).
    // See thesis §"Evaluation of Transition Matrix Elements".
    //
    // Parameters:
    //   A_promoted : promoted bare-state Gaussian matrix (N-1 × N-1)
    //   w_piN      : relative coordinate vector r_π - r_N in Jacobi space
    //   b          : form factor range in fm
    //
    // Returns: Ã  (N-1 × N-1), the effective Gaussian for the matrix element
    rmat build_A_tilde(const rmat& A_promoted,
                       const rvec& w_piN,
                       ld b) const
    {
        assert(A_promoted.size1() == N-1);
        assert(w_piN.size()       == N-1);
        // outer_no_conj(w, w) = w * w^T  (no conjugation needed — all real)
        rmat ww = outer_no_conj(w_piN, w_piN);
        return A_promoted + ww * (ld{1} / (b * b));
    }

    // ── Verification helpers ──────────────────────────────────────────────────

    // Check  J * U ≈ I  (should be identity to machine precision)
    ld jacobi_inverse_error() const {
        rmat JU = J * U;
        ld err = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++) {
                ld expected = (i == j) ? ld{1} : ld{0};
                ld diff = JU(i,j) - expected;
                err += diff * diff;
            }
        return std::sqrt(err);
    }

    // Check that A_kin is diagonal with A_kin[α][α] ≈ 1/mu[α]
    ld A_kin_diagonal_error() const {
        ld err = 0;
        for (size_t alpha = 0; alpha < N-1; alpha++) {
            ld diff = A_kin(alpha, alpha) - ld{1}/mu[alpha];
            err += diff * diff;
            for (size_t beta = alpha+1; beta < N-1; beta++)
                err += A_kin(alpha,beta)*A_kin(alpha,beta)
                     + A_kin(beta,alpha)*A_kin(beta,alpha);
        }
        return std::sqrt(err);
    }

    // ── Debug output ──────────────────────────────────────────────────────────
    void print() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "╔══ JacobiSystem (" << N << " particles) ══════════════╗\n";

        std::cout << "  Particles:  ";
        for (size_t i = 0; i < N; i++)
            std::cout << names[i] << " (" << m[i] << " MeV)  ";
        std::cout << "\n  M_total = " << M_total << " MeV\n\n";

        std::cout << "  Reduced masses:\n";
        for (size_t alpha = 0; alpha < N-1; alpha++)
            std::cout << "    μ[" << alpha << "]  (x_" << alpha << ")  =  "
                      << mu[alpha] << " MeV\n";

        std::cout << "\n  Jacobi matrix J (last row = CM):\n";
        for (size_t i = 0; i < N; i++) {
            std::cout << "    [";
            for (size_t j = 0; j < N; j++)
                std::cout << std::setw(10) << J(i,j);
            std::cout << " ]";
            if (i == N-1) std::cout << "  ← CM (discarded)";
            std::cout << "\n";
        }

        std::cout << "\n  A_kin (should be diag[1/μ]):\n";
        std::cout << "  " << A_kin << "\n";

        std::cout << "\n  Verification — J*U = I error: "
                  << jacobi_inverse_error() << "\n";
        std::cout << "  Verification — A_kin diagonal error: "
                  << A_kin_diagonal_error() << "\n";

        std::cout << "\n  Position extraction vectors  w_particle[i]  (first N-1 rows of U):\n";
        for (size_t i = 0; i < N; i++) {
            rvec wp(N-1); for (size_t a=0;a<N-1;a++) wp[a]=w_particle(i,a);
            std::cout << "    w_" << names[i] << " = " << wp << "\n";
        }

        std::cout << "\n  Momentum extraction vectors  k_particle[i]  (first N-1 cols of J):\n";
        for (size_t i = 0; i < N; i++) {
            rvec kp(N-1); for (size_t a=0;a<N-1;a++) kp[a]=k_particle(i,a);
            std::cout << "    k_" << names[i] << " = " << kp << "\n";
        }

        if (N == 3) {
            std::cout << "\n  W-operator relative vectors:\n";
            std::cout << "    w(r_π - r_p) = " << w_meson_proton()  << "\n";
            std::cout << "    w(r_π - r_n) = " << w_meson_neutron() << "\n";
        }
        std::cout << "╚══════════════════════════════════════════════╝\n";
    }

private:
    // ── Build all quantities from the mass array ───────────────────────────────
    void build() {
        // ── 1. Cumulative masses ──────────────────────────────────────────────
        M_cum.resize(N);
        M_cum[0] = m[0];
        for (size_t i = 1; i < N; i++) M_cum[i] = M_cum[i-1] + m[i];
        M_total = M_cum[N-1];

        // ── 2. Jacobi matrix J  (N × N) ──────────────────────────────────────
        // From thesis eq.(1):
        //   J[α][j] = m[j] / M_cum[α]   for j <= α
        //           = -1                 for j == α+1
        //           =  0                 for j >  α+1
        // Last row (α = N-1): all entries = m[j]/M_total  (CM row)
        J.resize(N, N);
        for (size_t alpha = 0; alpha < N; alpha++) {
            for (size_t j = 0; j < N; j++) {
                if (j <= alpha)
                    J(alpha, j) = m[j] / M_cum[alpha];
                else if (j == alpha + 1)
                    J(alpha, j) = ld{-1};
                else
                    J(alpha, j) = ld{0};
            }
        }

        // ── 3. Inverse  U = J^{-1} ────────────────────────────────────────────
        U = J.inverse();

        // ── 4. Reduced masses ─────────────────────────────────────────────────
        //   μ[α] = M_cum[α] * m[α+1] / M_cum[α+1]
        mu.resize(N-1);
        for (size_t alpha = 0; alpha < N-1; alpha++)
            mu[alpha] = M_cum[alpha] * m[alpha+1] / M_cum[alpha+1];

        // ── 5. Kinetic energy matrix  A_kin  (N-1 × N-1) ─────────────────────
        //   A_kin[α][β] = sum_{k=0}^{N-1}  J(α,k) * J(β,k) / m[k]
        //
        //   T_int = (1/2) sum_{α,β}  A_kin[α][β]  π_α · π_β
        //
        // For the standard Jacobi chain this is diagonal:
        //   A_kin[α][α] = 1/μ[α]
        //   (verified by the Python script — see comments in repo)
        A_kin.resize(N-1, N-1);
        for (size_t alpha = 0; alpha < N-1; alpha++) {
            for (size_t beta = 0; beta < N-1; beta++) {
                ld s = 0;
                for (size_t k = 0; k < N; k++)
                    s += J(alpha, k) * J(beta, k) / m[k];
                A_kin(alpha, beta) = s;
            }
        }

        // ── 6. Position extraction vectors  w_particle[i] ────────────────────
        //   w_particle[i] = first (N-1) elements of row i of U
        //   Gives:  r[i] = w_particle[i] · x   (in CM frame)
        //
        //   For relative distances:
        //   r[j] - r[i] = (w_particle[j] - w_particle[i]) · x
        w_particle.resize(N, N-1);
        for (size_t i = 0; i < N; i++) {
            for (size_t alpha = 0; alpha < N-1; alpha++)
                w_particle(i, alpha) = U(i, alpha);
        }

        // ── 7. Momentum extraction vectors  k_particle[i] ────────────────────
        //   k_particle[i] = first (N-1) elements of column i of J
        //   Gives:  p[i] = k_particle[i] · π
        //
        //   From the thesis: k_i → J k_i under the Jacobi transform.
        //   For unit vector e_i:  J e_i = column i of J.
        k_particle.resize(N, N-1);
        for (size_t i = 0; i < N; i++) {
            for (size_t alpha = 0; alpha < N-1; alpha++)
                k_particle(i, alpha) = J(alpha, i);
        }

        // ── 8. c_vec[α] = e_α  (unit vectors in (N-1)-dim Jacobi momentum space)
        //   Used as the 'c' argument in the kinetic energy integrals:
        //   ⟨g'| f(c·π) |g⟩  with  f(p) = p²/(2μ_α)  or  sqrt(p²+m²)-m
        c_vec.resize(N-1, N-1);
        for (size_t alpha = 0; alpha < N-1; alpha++) {
            c_vec(alpha, alpha) = ld{1};
        }
    }
};

} // namespace qm