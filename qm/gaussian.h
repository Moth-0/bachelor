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

    // Construct and immediately randomize
    explicit gaus(size_t dim, long double min_A = 0.01, long double max_A = 10.0) {
        A = matrix(dim, dim);
        s = matrix(dim, 3);
        randomize(min_A, max_A);
    }

    // Construct from explicit matrices (e.g., for unit tests)
    gaus(const matrix& A_in, const matrix& s_in) : A(A_in), s(s_in) {
        assert(A_in.size1() == A_in.size2());
        assert(A_in.size1() == s_in.size1());
        assert(s_in.size2() == 3);
    }

    size_t dim() const { return A.size1(); }

    // Randomize in-place without reallocating.
    // A is built as L*L^T (Cholesky) to guarantee positive-definiteness.
    // Diagonal entries of L are log-uniformly distributed so the basis
    // spans many length scales simultaneously.
    void randomize(long double min_A, long double max_A) {
        size_t d = A.size1();
        matrix L(d, d);
        long double log_lo = std::log(std::sqrt(min_A));
        long double log_hi = std::log(std::sqrt(max_A));

        for (size_t i = 0; i < d; ++i) {
            // Diagonal: strictly positive, log-uniform
            L(i, i) = std::exp(random_ld(log_lo, log_hi));
            // Lower triangle: bounded by smaller diagonal to keep conditioning
            for (size_t j = 0; j < i; ++j) {
                long double bound = std::min(L(i,i), L(j,j)) * 0.5L;
                L(i, j) = random_ld(-bound, bound);
            }
        }

        // A = L * L^T  (always positive-definite)
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) {
                long double sum = 0.0;
                for (size_t k = 0; k <= std::min(i,j); ++k)
                    sum += L(i,k) * L(j,k);
                A(i,j) = sum;
            }

        // Shifts: small random values.
        // For S-wave channels the caller should zero these out after construction.
        // For P-wave channels the SVM will optimize them.
        for (size_t i = 0; i < d; ++i)
            for (size_t k = 0; k < 3; ++k)
                s(i, k) = random_ld(-0.1L, 0.1L);
    }

    // Zero out all shift vectors (call this for S-wave channels)
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
//    N_ab = (pi^dim / det(B))^(3/2) * exp(1/4 * v^T B^{-1} v_dot)
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

    // Correct: pi^{3d/2} / det(B)^{3/2}
    // The det exponent is ALWAYS 3/2 (one factor per spatial dimension x,y,z).
    // Only the pi exponent scales with d. Note: for d=1 both forms are equal,
    // which is why the hydrogen (1-body Jacobi) test passed with the old formula.
    long double front = std::pow(pi, 1.5L * (long double)d) / std::pow(detB, 1.5L);
    return front * std::exp(0.25L * vBv);
}

} // namespace qm