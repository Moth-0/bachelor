#pragma once
// gaussian.h  --  correlated shifted Gaussian basis functions.
//
// A Gaussian is:  g(x; A, s) = exp(-x^T A x + s^T x)
//
// where x is a (dim x 3) matrix of Jacobi coordinates (one row per coordinate,
// one column per Cartesian direction), A is a (dim x dim) positive-definite
// correlation matrix, and s is a (dim x 3) shift matrix.
//
// Equivalently: g = exp(-(x - u)^T A (x - u)) * const, where u = (1/2) A^{-1} s
// is the centre of the Gaussian in position space.  Keeping s as the primary
// parameter makes the analytic overlap formula simpler.
//
// All quantities here are REAL.  The complex structure enters only in the W
// matrix elements via the spherical decomposition of sigma.r (see hamiltonian.h).

#include <cmath>
#include <cassert>
#include <random>
#include <functional>
#include <vector>
#include "matrix.h"

namespace qm {

static constexpr long double pi            = 3.14159265358979323846L;
static constexpr long double LINEAR_DEP_TOL = 0.95L;

// ---------------------------------------------------------------------------
// Gaussian basis function
// ---------------------------------------------------------------------------
struct gaus {
    matrix A;   // (dim x dim) correlation matrix
    matrix s;   // (dim x 3)  shift matrix  (s[i][d] = s_{i,d})

    gaus() = default;
    gaus(const matrix& A_, const matrix& s_) : A(A_), s(s_) {}

    // Construct with random A (exponential distribution) and random/zero shifts.
    // mean_r:  scale for the NN / pion relative Gaussian width
    // mean_R:  scale for the pion-CM shift (0 for S-wave)
    gaus(size_t dim, long double mean_r, long double mean_R) {
        A = matrix(dim, dim);
        s = matrix(dim, 3);

        // Random number engine (thread_local for OMP safety)
        thread_local static std::mt19937_64 rng(std::random_device{}());
        auto random_ld = [&](long double lo, long double hi) -> long double {
            return std::uniform_real_distribution<long double>(lo, hi)(rng);
        };
        auto rand_exp = [&](long double scale) -> long double {
            // exponential: -scale * ln(u),  u uniform in (0,1)
            long double u = random_ld(1e-9L, 1.0L);
            return -scale * std::log(u);
        };

        // Fill A as a random positive-definite matrix via random diagonal in a rotated basis
        for (size_t i = 0; i < dim; ++i)
            for (size_t j = 0; j < dim; ++j)
                A(i,j) = (i == j) ? rand_exp(1.0L / mean_r) : 0.0L;

        // Shifts: set as  s = 2 A u  where u is a random position
        matrix u(dim, 3);
        for (size_t i = 0; i < dim; ++i)
            for (size_t k = 0; k < 3; ++k)
                u(i,k) = (mean_R > 0.0L && i == dim-1) ? random_ld(-mean_R, +mean_R) : 0.0L;

        for (size_t i = 0; i < dim; ++i)
            for (size_t k = 0; k < 3; ++k) {
                long double val = 0.0L;
                for (size_t j = 0; j < dim; ++j) val += A(i,j) * u(j,k);
                s(i,k) = 2.0L * val;
            }
    }

    size_t dim() const { return A.size1(); }

    void zero_shifts() {
        for (size_t i = 0; i < dim(); ++i)
            for (size_t k = 0; k < 3; ++k)
                s(i,k) = 0.0L;
    }
};

// ---------------------------------------------------------------------------
// promote: embed a lower-dim Gaussian into a higher-dim space by zero-padding.
// Used to match dimensions before computing W matrix elements.
// ---------------------------------------------------------------------------
inline gaus promote(const gaus& g, size_t new_dim) {
    size_t old_dim = g.dim();
    assert(new_dim >= old_dim);
    matrix A_new(new_dim, new_dim);
    matrix s_new(new_dim, 3);
    for (size_t i = 0; i < new_dim; ++i) {
        for (size_t j = 0; j < new_dim; ++j)
            A_new(i,j) = (i < old_dim && j < old_dim) ? g.A(i,j) : 0.0L;
        for (size_t k = 0; k < 3; ++k)
            s_new(i,k) = (i < old_dim) ? g.s(i,k) : 0.0L;
    }
    return gaus(A_new, s_new);
}

// ---------------------------------------------------------------------------
// overlap: <phi_a | phi_b>
//
//   N_ab = (pi^dim / det B)^{3/2}  exp(1/4 v^T B^{-1} v)
//
// where B = A_a + A_b  and  v[:,d] = s_a[:,d] + s_b[:,d].
// ---------------------------------------------------------------------------
inline long double overlap(const gaus& a, const gaus& b) {
    assert(a.dim() == b.dim());
    size_t d = a.dim();
    matrix B     = a.A + b.A;
    long double detB = B.determinant();
    if (detB <= ZERO_LIMIT) return 0.0L;
    matrix B_inv = B.inverse();

    long double vBv = 0.0L;
    for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < d; ++i) {
            long double vi = a.s(i,k) + b.s(i,k);
            for (size_t j = 0; j < d; ++j)
                vBv += B_inv(i,j) * vi * (a.s(j,k) + b.s(j,k));
        }

    long double front = std::pow(pi, 1.5L * (long double)d) / std::pow(detB, 1.5L);
    return front * std::exp(0.25L * vBv);
}

// ---------------------------------------------------------------------------
// is_linearly_dependent: cosine-similarity filter for SVM basis selection.
//
// Returns true if |<trial|g_i>| / sqrt(<trial|trial><g_i|g_i>) > LINEAR_DEP_TOL
// for any basis function g_i (excluding index skip).
// ---------------------------------------------------------------------------
inline bool is_linearly_dependent(const gaus& trial,
                                  const std::vector<gaus>& basis,
                                  size_t skip = static_cast<size_t>(-1)) {
    long double n_tt = overlap(trial, trial);
    for (size_t i = 0; i < basis.size(); ++i) {
        if (i == skip) continue;
        long double n_ii = overlap(basis[i], basis[i]);
        long double n_ti = std::abs(overlap(trial, basis[i]));
        if (n_ti / std::sqrt(n_tt * n_ii) > LINEAR_DEP_TOL) return true;
    }
    return false;
}

} // namespace qm