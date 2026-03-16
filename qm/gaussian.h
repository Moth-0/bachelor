#pragma once

#include "matrix.h"
#include "jacobi.h"
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
// gaussian.h  —  Correlated Shifted Gaussian basis functions
//
// Implements the wavefunction basis from thesis ch.2 and ch.5:
//
//   ⟨r|g⟩ = exp(-r^T A r + s^T r)
//
// where:
//   A   — positive-definite (dim × dim) correlation matrix
//   s   — dim-dimensional shift vector  (encodes angular momentum)
//   dim — number of active Jacobi coordinates
//         = 1    for the bare pn state  (only x_0 = r_p - r_n)
//         = N-1  for dressed pion states (all relative coordinates)
//
// Key geometric quantities derived from a pair (g', g):
//   B   = A' + A
//   v   = s' + s
//   M   = (π^dim / det B)^{3/2} exp(1/4 v^T B^{-1} v)   [overlap]
//   u   = (1/2) B^{-1} v                                 [⟨g'|r|g⟩ = u M]
//
// Kinetic energy parameters (γ, η) are physics quantities and live in
// hamiltonian.h, not here.
//
// Random generation (thesis §5.1, Fedorov eq.19):
//   A = Σ_{i<j}  w_{ij} w_{ij}^T / b_{ij}^2
//   b_{ij} = -ln(u) * b0    (u ~ Uniform(0,1))
//   s[k]   ~ Uniform(-s_max, s_max)
//
// Units: masses in MeV, lengths in fm,  hbar*c = 197.3269804 MeV·fm
// ─────────────────────────────────────────────────────────────────────────────

namespace qm {

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian  —  a single basis function  |g⟩ = |A, s⟩
// ─────────────────────────────────────────────────────────────────────────────
struct Gaussian {
    rmat   A;    // correlation matrix  (dim × dim), positive definite
    rvec   s;    // shift vector        (dim)
    size_t dim;  // number of active Jacobi coordinates

    Gaussian() : dim(0) {}

    Gaussian(const rmat& A_, const rvec& s_)
        : A(A_), s(s_), dim(s_.size())
    {
        assert(A_.size1() == dim);
        assert(A_.size2() == dim);
    }

    // Convenience: build from scalars (1D bare state)
    Gaussian(ld a_scalar, ld s_scalar)
        : A(1, 1), s(1), dim(1)
    {
        A(0,0) = a_scalar;
        s[0]   = s_scalar;
    }

    void print(const std::string& label = "") const {
        std::cout << std::fixed << std::setprecision(5);
        if (!label.empty()) std::cout << label << "\n";
        std::cout << "  dim = " << dim << "\n";
        std::cout << "  A =\n  " << A << "\n";
        std::cout << "  s = " << s << "\n";
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// GaussianPair  —  geometric intermediates for the pair (g', g)
//
// Caches B, B^{-1}, det(B), v, M and u — quantities shared by every
// matrix element between the same pair (overlap, KE, potential, W-operator).
// Construct once per (i,j) pair and pass to all hamiltonian functions.
//
// Usage:
//   GaussianPair gp(g_bra, g_ket);
//   ld N_ij    = gp.M;          // overlap matrix element
//   rvec r_ij  = gp.u * gp.M;  // ⟨g'|r|g⟩  (thesis eq.4)
// ─────────────────────────────────────────────────────────────────────────────
struct GaussianPair {
    Gaussian bra;   // stored by VALUE — avoids dangling-reference issues
    Gaussian ket;   // storing copies is safe and cheap for our small matrices

    size_t dim;

    rmat B;       // A_bra + A_ket
    rvec v;       // s_bra + s_ket
    rmat Binv;    // B^{-1}
    ld   detB;    // det(B)
    ld   M;       // overlap ⟨g'|g⟩
    rvec u;       // (1/2) B^{-1} v   →   ⟨g'|r|g⟩ = u * M

    GaussianPair(const Gaussian& bra_, const Gaussian& ket_)
        : bra(bra_), ket(ket_)
    {
        assert(bra.dim == ket.dim);
        dim  = bra.dim;

        B    = bra.A + ket.A;
        v    = bra.s + ket.s;
        Binv = B.inverse();
        detB = B.determinant();

        // M = (π^dim / det B)^{3/2} * exp(1/4 * v^T B^{-1} v)
        ld pi_over_det = std::pow(static_cast<ld>(M_PI),
                                  static_cast<ld>(dim)) / detB;
        ld exponent    = ld{0.25L} * dot(v, Binv * v);
        M = std::pow(pi_over_det, ld{1.5L}) * std::exp(exponent);

        // u = (1/2) B^{-1} v
        u = (Binv * v) * ld{0.5L};
    }

    GaussianPair() = delete;  // always needs real Gaussians
};


// ─────────────────────────────────────────────────────────────────────────────
// Standalone overlap  ⟨g'|g⟩
//
// Convenience wrapper — builds GaussianPair and returns M.
// For repeated calls between the same pair, build GaussianPair directly.
// ─────────────────────────────────────────────────────────────────────────────
inline ld overlap(const Gaussian& bra, const Gaussian& ket) {
    assert(bra.dim == ket.dim);
    return GaussianPair(bra, ket).M;
}

inline ld self_overlap(const Gaussian& g) {
    return overlap(g, g);
}


// ─────────────────────────────────────────────────────────────────────────────
// ⟨g'|r|g⟩  —  position expectation value  (thesis eq.4)
//
//   ⟨g'|r|g⟩ = u * M    where  u = (1/2) B^{-1} v
//
// Returns the full dim-vector.  Used for the W-operator coupling in
// hamiltonian.h:  the spatial factor is (σ·r) with r extracted via u.
// ─────────────────────────────────────────────────────────────────────────────
inline rvec r_expectation(const Gaussian& bra, const Gaussian& ket) {
    GaussianPair gp(bra, ket);
    return gp.u * gp.M;
}

inline ld r_expectation_component(const Gaussian& bra,
                                   const Gaussian& ket,
                                   size_t alpha)
{
    GaussianPair gp(bra, ket);
    return gp.u[alpha] * gp.M;
}


// ─────────────────────────────────────────────────────────────────────────────
// Random Gaussian generation  (thesis §5.1, Fedorov eq.19)
//
//   A = Σ_{i<j}  w_{ij}(dim) w_{ij}(dim)^T / b_{ij}^2
//   b_{ij} = -ln(u) * b0,     u ~ Uniform(0,1)
//   s[k]   ~ Uniform(-s_max, s_max)
//
// Parameters:
//   sys    — JacobiSystem providing w_rel vectors
//   dim    — 1 = bare pn state,  N-1 = dressed pion state
//   b0     — length scale in fm              (thesis: ~1.4 fm)
//   s_max  — shift bound in fm^{-1}
//             15–30 MeV / (ħc = 197.3 MeV·fm) ≈ 0.076–0.152 fm^{-1}
//   rng    — std::mt19937 or compatible engine (passed by reference)
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
Gaussian random_gaussian(const JacobiSystem& sys,
                         size_t              dim,
                         ld                  b0,
                         ld                  s_max,
                         RNG&                rng)
{
    assert(dim >= 1 && dim <= sys.N - 1);

    std::uniform_real_distribution<long double> uniform01(
        std::numeric_limits<long double>::epsilon(), ld{1});
    std::uniform_real_distribution<long double> uniform_s(-s_max, s_max);

    // ── A = Σ_{i<j} w_{ij} w_{ij}^T / b_{ij}^2 ──────────────────────────────
    rmat A(dim, dim);

    for (size_t i = 0; i < sys.N; i++) {
        for (size_t j = i + 1; j < sys.N; j++) {
            ld u_rand = uniform01(rng);
            ld b_ij   = -std::log(u_rand) * b0;

            rvec w_full = sys.w_rel(i, j);
            rvec w_ij(dim);
            for (size_t k = 0; k < dim; k++) w_ij[k] = w_full[k];

            A += (outer_no_conj(w_ij, w_ij) * (ld{1} / (b_ij * b_ij)));
        }
    }

    // ── s[k] ~ Uniform(-s_max, s_max) ────────────────────────────────────────
    rvec s(dim);
    for (size_t k = 0; k < dim; k++) s[k] = uniform_s(rng);

    return Gaussian(A, s);
}

template<typename RNG>
Gaussian random_gaussian_bare(const JacobiSystem& sys,
                               ld b0, ld s_max, RNG& rng)
{
    return random_gaussian(sys, 1, b0, s_max, rng);
}

template<typename RNG>
Gaussian random_gaussian_dressed(const JacobiSystem& sys,
                                  ld b0, ld s_max, RNG& rng)
{
    return random_gaussian(sys, sys.N - 1, b0, s_max, rng);
}


// ─────────────────────────────────────────────────────────────────────────────
// Positive-definiteness check  (Sylvester's criterion on leading minors)
//
// Guaranteed true by construction for random_gaussian.
// Useful as a sanity check for manually constructed Gaussians.
// ─────────────────────────────────────────────────────────────────────────────
inline bool is_positive_definite(const rmat& A) {
    size_t n = A.size1();
    assert(n == A.size2());
    for (size_t k = 1; k <= n; k++) {
        rmat sub(k, k);
        for (size_t i = 0; i < k; i++)
            for (size_t j = 0; j < k; j++)
                sub(i,j) = A(i,j);
        if (sub.determinant() <= ld{0}) return false;
    }
    return true;
}

} // namespace qm