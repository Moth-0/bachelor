#pragma once
#include <cmath>
#include <numbers>
#include <random>
#include <vector>
#include <cassert>
#include "matrix.h"

namespace qm {

const long double pi = std::numbers::pi_v<long double>;

// Cosine similarity threshold for linear dependence rejection.
// Candidates with |<trial|g_i>| / sqrt(<trial|trial><g_i|g_i>) > LINEAR_DEP_TOL are rejected.
constexpr long double LINEAR_DEP_TOL = 0.80;

long double random_ld(long double lo, long double hi) {
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<long double> dist(lo, hi);
    return dist(rng);
}

// Correlated shifted Gaussian basis function:
//   phi(x) = exp(-x^T A x + s^T x)
// where x is a (dim x 3) matrix of Jacobi coordinates,
// A is (dim x dim) symmetric positive-definite, s is (dim x 3).
// The Gaussian peaks at u = (1/2) A^{-1} s,  i.e. s = 2 A u.
struct gaus {
    matrix A;  // (dim x dim) exponent matrix
    matrix s;  // (dim x 3)  shift vectors

    gaus() = default;

    explicit gaus(size_t dim, long double mean_r = 2.0, long double mean_R = 0.0) {
        A = matrix(dim, dim);
        s = matrix(dim, 3);
        randomize(mean_r, mean_R);
    }

    gaus(const matrix& A_in, const matrix& s_in) : A(A_in), s(s_in) {
        assert(A_in.size1() == A_in.size2());
        assert(A_in.size1() == s_in.size1());
        assert(s_in.size2() == 3);
    }

    size_t dim() const { return A.size1(); }

    // Build A = L L^T with L_{ii} log-uniform in
    //   [1/(sqrt(2)*spread*mean_r),  1/(sqrt(2)*mean_r/spread)],  spread=3
    // so A_{ii} ~ 1/(2 mean_r^2) over one decade each side.
    // Off-diagonals: L_{ij} in [-corr*L_{jj}, +corr*L_{jj}], corr=0.3.
    // Shifts: s = 2 A u,  u_ik ~ Uniform(-mean_R, +mean_R).
    void randomize(long double mean_r, long double mean_R = 0.0) {
        size_t d = A.size1();

        const long double spread = 3.0;
        const long double L_lo   = 1.0 / (std::sqrt(2.0) * spread * mean_r);
        const long double L_hi   = 1.0 / (std::sqrt(2.0) * mean_r / spread);
        const long double log_lo = std::log(L_lo);
        const long double log_hi = std::log(L_hi);
        const long double corr   = 0.3;

        matrix L(d, d);
        for (size_t i = 0; i < d; ++i) {
            L(i, i) = std::exp(random_ld(log_lo, log_hi));
            for (size_t j = 0; j < i; ++j)
                L(i, j) = random_ld(-corr * L(j, j), +corr * L(j, j));
        }

        // A = L L^T
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) {
                long double s = 0.0;
                for (size_t k = 0; k <= std::min(i, j); ++k) s += L(i, k) * L(j, k);
                A(i, j) = s;
            }

        // s = 2 A u,  u drawn per coordinate and direction
        matrix u(d, 3);
        for (size_t i = 0; i < d; ++i)
            for (size_t k = 0; k < 3; ++k)
                u(i, k) = (mean_R > 0.0) ? random_ld(-mean_R, +mean_R) : 0.0;

        for (size_t i = 0; i < d; ++i)
            for (size_t k = 0; k < 3; ++k) {
                long double val = 0.0;
                for (size_t j = 0; j < d; ++j) val += A(i, j) * u(j, k);
                s(i, k) = 2.0 * val;
            }
    }

    void zero_shifts() {
        for (size_t i = 0; i < dim(); ++i)
            for (size_t k = 0; k < 3; ++k)
                s(i, k) = 0.0;
    }
};

// Pad A and s with zeros to embed a lower-dim Gaussian into a higher-dim space
gaus promote(const gaus& g, size_t new_dim) {
    size_t old_dim = g.dim();
    assert(new_dim >= old_dim);
    matrix A_new(new_dim, new_dim);
    matrix s_new(new_dim, 3);
    for (size_t i = 0; i < new_dim; ++i) {
        for (size_t j = 0; j < new_dim; ++j)
            A_new(i, j) = (i < old_dim && j < old_dim) ? g.A(i, j) : 0.0;
        for (size_t k = 0; k < 3; ++k)
            s_new(i, k) = (i < old_dim) ? g.s(i, k) : 0.0;
    }
    return gaus(A_new, s_new);
}

// Overlap integral <phi_a|phi_b>:
//   N_ab = (pi^d / det B)^{3/2} exp(1/4 v^T B^{-1} v)
// where B = A_a + A_b,  v^k = s_a[:,k] + s_b[:,k].
long double overlap(const gaus& a, const gaus& b) {
    assert(a.dim() == b.dim());
    size_t d = a.dim();
    matrix B     = a.A + b.A;
    matrix B_inv = B.inverse();
    long double detB = B.determinant();
    if (detB <= 1e-30) return 0.0;

    long double vBv = 0.0;
    for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < d; ++i) {
            long double vi = a.s(i, k) + b.s(i, k);
            for (size_t j = 0; j < d; ++j)
                vBv += B_inv(i, j) * vi * (a.s(j, k) + b.s(j, k));
        }

    long double front = std::pow(pi, 1.5 * (long double)d) / std::pow(detB, 1.5);
    return front * std::exp(0.25 * vBv);
}

// Returns true if trial is nearly linearly dependent on basis.
// Computes cosine similarity |<trial|g_i>| / sqrt(<trial|trial><g_i|g_i>).
// skip: index within basis to exclude (used during refinement to ignore the replaced state).
bool is_linearly_dependent(const gaus& trial, const std::vector<gaus>& basis,
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