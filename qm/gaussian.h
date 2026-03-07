#pragma once
#include <cmath>
#include <numbers>
#include <random>
#include <cassert>
#include "matrix.h"

// ============================================================
//  gaussian.h
//
//  Defines the correlated shifted Gaussian basis function:
//
//    phi(x) = exp( -x^T A x + s^T x )
//
//  where:
//    x : (dim x 3) matrix of Jacobi coordinates (each row
//        is one Jacobi coordinate vector in 3D space)
//    A : (dim x dim) symmetric positive-definite matrix
//        encoding spatial correlations between coordinates
//    s : (dim x 3) matrix of shift vectors
//
//  Physical interpretation (key):
//    The Gaussian peaks at mean position u = (1/2) A^{-1} s,
//    i.e.  s = 2 A u.
//    The RMS width along Jacobi coordinate i is
//      sigma_i ~ 1 / sqrt(2 * A_{ii})
//    so  A_{ii} ~ 1 / (2 * r_i^2)  where r_i is the typical
//    inter-particle distance for that coordinate.
//
//  Key design decision: gaus is a PURE MATH object.
//  It has no knowledge of particles, masses, or physics.
//  All particle/physics information lives in jacobian.h
//  and hamiltonian.h.
//
//  For S-wave channels (e.g., bare deuteron), keep s = 0.
//  For P-wave clothed channels (e.g., pion sector), the SVM
//  optimizes both A and s — do NOT zero out the shifts.
// ============================================================

namespace qm {

const long double pi = std::numbers::pi_v<long double>;

inline long double random_ld(long double lo, long double hi) {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<long double> dist(lo, hi);
    return dist(rng);
}

// ------------------------------------------------------------
struct gaus {
    matrix A; // (dim x dim) positive-definite exponent matrix
    matrix s; // (dim x 3)  shift vectors

    gaus() = default;

    // Construct and immediately randomize.
    //   mean_r [fm] : typical inter-particle distance
    //                 (controls width of A; default 2 fm is
    //                  a reasonable nuclear scale)
    //   mean_R [fm] : typical displacement of the Gaussian
    //                 peak from the origin in Jacobi space
    //                 (zero for S-wave; ~1-2 fm for P-wave)
    explicit gaus(size_t dim,
                  long double mean_r = 2.0L,
                  long double mean_R = 0.0L) {
        A = matrix(dim, dim);
        s = matrix(dim, 3);
        randomize(mean_r, mean_R);
    }

    // Construct from explicit matrices (e.g., for unit tests)
    gaus(const matrix& A_in, const matrix& s_in) : A(A_in), s(s_in) {
        assert(A_in.size1() == A_in.size2());
        assert(A_in.size1() == s_in.size1());
        assert(s_in.size2() == 3);
    }

    size_t dim() const { return A.size1(); }

    // --------------------------------------------------------
    //  randomize(mean_r, mean_R)
    //
    //  Generates a physically grounded Gaussian with:
    //
    //  A — built via Cholesky A = L Lᵀ so it is guaranteed
    //      positive-definite (Varga & Suzuki, Ch. 4).
    //      Diagonal entries of L are log-uniformly drawn in
    //
    //        [ 1/(sqrt(2) * spread * mean_r),
    //          1/(sqrt(2) * mean_r / spread) ]
    //
    //      which maps to A_{ii} in
    //
    //        [ 1/(2*(spread*mean_r)^2),  1/(2*(mean_r/spread)^2) ]
    //
    //      with spread = 3, so the basis spans one decade
    //      around the physical scale A ~ 1/(2*mean_r^2).
    //
    //      Off-diagonal entries of L are drawn uniformly in
    //      [-corr * L_{jj}, +corr * L_{jj}] with corr = 0.3.
    //      Small correlation keeps A well-conditioned while
    //      still allowing genuine inter-coordinate coupling.
    //
    //  s — derived from a random peak position u:
    //        u_i ~ Uniform(-mean_R, +mean_R)  for each
    //              Jacobi coordinate i and spatial direction d.
    //        s = 2 * A * u   (exact completion-of-square)
    //
    //      This guarantees the Gaussian is peaked at u and
    //      that s scales consistently with A.
    //      For S-wave channels call zero_shifts() afterwards.
    //
    //  Parameters:
    //    mean_r [fm]  — typical inter-particle distance
    //    mean_R [fm]  — typical Gaussian peak displacement
    //                   (set 0 for S-wave / bare sectors)
    // --------------------------------------------------------
    void randomize(long double mean_r, long double mean_R = 0.0L) {
        size_t d = A.size1();

        // --- Build A = L Lᵀ ---
        // L_{ii} drawn log-uniformly so that A_{ii} spans
        //   [1/(2*(3*mean_r)^2),  1/(2*(mean_r/3)^2)]
        // i.e. one order of magnitude each side of 1/(2*mean_r^2).
        const long double spread = 3.0L;
        const long double L_lo   = 1.0L / (std::sqrt(2.0L) * spread * mean_r);
        const long double L_hi   = 1.0L / (std::sqrt(2.0L) * mean_r / spread);
        const long double log_lo = std::log(L_lo);
        const long double log_hi = std::log(L_hi);
        // Off-diagonal correlation fraction (keep small for conditioning)
        const long double corr = 0.3L;

        matrix L(d, d);
        for (size_t i = 0; i < d; ++i) {
            // Diagonal: log-uniform, centred on 1/(sqrt(2)*mean_r)
            L(i, i) = std::exp(random_ld(log_lo, log_hi));
            // Lower triangle: bounded fraction of the column diagonal
            for (size_t j = 0; j < i; ++j)
                L(i, j) = random_ld(-corr * L(j,j), +corr * L(j,j));
        }

        // A = L Lᵀ  (positive-definite by construction)
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) {
                long double sum = 0.0L;
                for (size_t k = 0; k <= std::min(i,j); ++k)
                    sum += L(i,k) * L(j,k);
                A(i,j) = sum;
            }

        // --- Build s = 2 A u ---
        // u: peak position in Jacobi space, drawn per coordinate
        //    and per spatial direction from Uniform(-mean_R, +mean_R).
        // u is stored as a (dim x 3) matrix (row = Jacobi coord, col = xyz).
        // If mean_R == 0 all shifts are zero (S-wave).
        matrix u(d, 3);
        for (size_t i = 0; i < d; ++i)
            for (size_t k = 0; k < 3; ++k)
                u(i, k) = (mean_R > 0.0L)
                           ? random_ld(-mean_R, +mean_R)
                           : 0.0L;

        // s = 2 A u  (per spatial direction independently)
        for (size_t i = 0; i < d; ++i)
            for (size_t k = 0; k < 3; ++k) {
                long double val = 0.0L;
                for (size_t j = 0; j < d; ++j)
                    val += A(i,j) * u(j,k);
                s(i, k) = 2.0L * val;
            }
    }

    // Zero out all shift vectors (call this for S-wave channels).
    // The Gaussian peak is then exactly at the origin.
    void zero_shifts() {
        for (size_t i = 0; i < dim(); ++i)
            for (size_t k = 0; k < 3; ++k)
                s(i, k) = 0.0L;
    }
};

// ------------------------------------------------------------
//  Promote a lower-dimensional Gaussian to a higher-dimensional
//  space by padding A with zeros in the new rows/cols and s
//  with zeros in the new rows. Used in the W operator when the
//  bare (2-body) basis must be embedded into the clothed (3-body)
//  space.
// ------------------------------------------------------------
inline gaus promote(const gaus& g, size_t new_dim) {
    size_t old_dim = g.dim();
    assert(new_dim >= old_dim);

    matrix A_new(new_dim, new_dim);
    matrix s_new(new_dim, 3);

    // Copy existing block into top-left corner
    for (size_t i = 0; i < new_dim; ++i)
        for (size_t j = 0; j < new_dim; ++j)
            A_new(i,j) = (i < old_dim && j < old_dim) ? g.A(i,j) : 0.0L;

    for (size_t i = 0; i < new_dim; ++i)
        for (size_t k = 0; k < 3; ++k)
            s_new(i,k) = (i < old_dim) ? g.s(i,k) : 0.0L;

    return gaus(A_new, s_new);
}

// ------------------------------------------------------------
//  Overlap integral:
//
//    <phi_a | phi_b> = integral exp(-x^T(A_a+A_b)x + (s_a+s_b)^T x) dx
//
//  Analytic result (for dim Jacobi coordinates, each 3D):
//
//    N_ab = (pi^dim / det(B))^(3/2) * exp(1/4 * v^T B^{-1} v)
//
//  where B = A_a + A_b and the exponent is summed over 3D directions.
// ------------------------------------------------------------
inline long double overlap(const gaus& a, const gaus& b) {
    assert(a.dim() == b.dim());
    size_t d = a.dim();

    matrix B = a.A + b.A;
    matrix B_inv = B.inverse();
    long double detB = B.determinant();

    if (detB <= 1e-30L) return 0.0L;

    // Exponent: sum over spatial directions independently (x,y,z)
    // v_k = s_a[:,k] + s_b[:,k]  (a dim-vector for direction k)
    long double vBv = 0.0L;
    for (size_t k = 0; k < 3; ++k) {
        for (size_t i = 0; i < d; ++i) {
            long double vi = a.s(i,k) + b.s(i,k);
            for (size_t j = 0; j < d; ++j) {
                long double vj = a.s(j,k) + b.s(j,k);
                vBv += B_inv(i,j) * vi * vj;
            }
        }
    }

    // pi^{3d/2} / det(B)^{3/2}
    long double front = std::pow(pi, 1.5L * (long double)d) / std::pow(detB, 1.5L);
    return front * std::exp(0.25L * vBv);
}

} // namespace qm