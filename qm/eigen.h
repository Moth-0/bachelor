#pragma once
#include <cmath>
#include <algorithm>
#include "matrix.h"

namespace qm {

struct EigenResult {
    vector evals;
    matrix evecs;  // columns are eigenvectors
};

// Cholesky decomposition A = L L^T for symmetric positive-definite A.
// Returns L (lower triangular), or a 0x0 matrix if A is ill-conditioned.
matrix cholesky(const matrix& A) {
    size_t n = A.size1();
    matrix L(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            long double s = 0.0;
            for (size_t k = 0; k < j; k++) s += L(i, k) * L(j, k);
            if (i == j) {
                long double d = A(i, i) - s;
                if (d <= ZERO_LIMIT) return matrix(0, 0);
                L(i, j) = std::sqrt(d);
            } else {
                L(i, j) = (A(i, j) - s) / L(j, j);
            }
        }
    }
    return L;
}

// Jacobi diagonalization of symmetric matrix M.
// Iteratively zeroes off-diagonal elements via Givens rotations.
// Returns eigenvalues (diagonal of rotated M) and eigenvector matrix V.
EigenResult jacobi_eigensystem(const matrix& M) {
    matrix A = M;
    size_t n = A.size1();
    int max_iter = 20 * (int)(n * n);

    matrix V(n, n);
    for (size_t i = 0; i < n; ++i) V(i, i) = 1.0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Find largest off-diagonal element
        long double max_val = 0.0;
        size_t p = 0, q = 1;
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++)
                if (std::abs(A(i, j)) > max_val) { max_val = std::abs(A(i, j)); p = i; q = j; }
        if (max_val < ZERO_LIMIT) break;

        // Givens rotation angle: tan(2θ) = 2A_{pq} / (A_{qq} - A_{pp})
        long double theta = 0.5 * std::atan2(2.0 * A(p, q), A(q, q) - A(p, p));
        long double c = std::cos(theta);
        long double s = std::sin(theta);

        long double App = c*c*A(p,p) - 2.0*s*c*A(p,q) + s*s*A(q,q);
        long double Aqq = s*s*A(p,p) + 2.0*s*c*A(p,q) + c*c*A(q,q);
        A(p, q) = A(q, p) = 0.0;

        for (size_t i = 0; i < n; i++) {
            if (i != p && i != q) {
                long double Aip = A(i, p), Aiq = A(i, q);
                A(i, p) = A(p, i) = c*Aip - s*Aiq;
                A(i, q) = A(q, i) = s*Aip + c*Aiq;
            }
            long double Vip = V(i, p), Viq = V(i, q);
            V(i, p) = c*Vip - s*Viq;
            V(i, q) = s*Vip + c*Viq;
        }
        A(p, p) = App;
        A(q, q) = Aqq;
    }

    EigenResult res;
    res.evals = vector(n);
    res.evecs = V;
    for (size_t i = 0; i < n; i++) res.evals[i] = A(i, i);
    return res;
}

// Generalized eigenvalue problem H c = E N c.
// Reduction via Cholesky N = L L^T:
//   H' = L^{-1} H (L^{-1})^T,  c = (L^{-1})^T v
// Returns eigenvalues and eigenvectors sorted ascending.
EigenResult solve_generalized_eigensystem(const matrix& H, const matrix& N) {
    matrix L = cholesky(N);
    if (L.size1() == 0) return {vector(), matrix()};

    matrix Li = L.inverse_lower();
    if (Li.size1() == 0) return {vector(), matrix()};

    matrix Hp = Li * H * Li.transpose();
    EigenResult sys = jacobi_eigensystem(Hp);
    sys.evecs = Li.transpose() * sys.evecs;

    // Sort by eigenvalue (ascending)
    size_t n = sys.evals.size();
    for (size_t i = 0; i < n - 1; i++)
        for (size_t j = 0; j < n - i - 1; j++)
            if (sys.evals[j] > sys.evals[j + 1]) {
                std::swap(sys.evals[j], sys.evals[j + 1]);
                for (size_t k = 0; k < n; k++) std::swap(sys.evecs(k, j), sys.evecs(k, j + 1));
            }
    return sys;
}

} // namespace qm