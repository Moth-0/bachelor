#ifndef QM_EIGEN_TYPES_H
#define QM_EIGEN_TYPES_H

#include <Eigen/Dense>
#include <complex>

// Basic type definitions for quantum mechanical computations
using ld = double;              // Precision: double (15-17 digits)
using cld = std::complex<double>;

// ─────────────────────────────────────────────────────────────────────────────
// Macros for compatibility with custom matrix library
// ─────────────────────────────────────────────────────────────────────────────
#define ZERO_LIMIT     1e-4
#define FORV(i,v)      for (size_t i = 0; i < (v).size(); i++)
#define FOR_COLS(i,A)  for (size_t i = 0; i < (A).cols(); i++)
#define FOR_MAT(M)     for (size_t i = 0; i < (M).cols(); i++) \
                           for (size_t j = 0; j < (M).rows(); j++)

namespace qm {

// Matrix types
using ld_mat = Eigen::MatrixXcd;           // Complex double matrix
using ld_vec = Eigen::VectorXcd;           // Complex double vector
using rvec = Eigen::VectorXd;              // Real double vector
using rmat = Eigen::MatrixXd;              // Real double matrix

// Aliases for compatibility with existing code
using cmat = ld_mat;                       // Keep old name for minimal changes
using cvec = ld_vec;

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions for creating matrices
// ─────────────────────────────────────────────────────────────────────────────

// Create zero matrix
template <typename T>
auto zeros(size_t rows, size_t cols) {
    if constexpr (std::is_same_v<T, cld>) {
        return Eigen::MatrixXcd::Zero(rows, cols);
    } else if constexpr (std::is_same_v<T, ld>) {
        return Eigen::MatrixXd::Zero(rows, cols);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return Eigen::MatrixXcd::Zero(rows, cols);
    } else {
        return Eigen::MatrixXd::Zero(rows, cols);
    }
}

// Create identity matrix
template <typename T>
auto eye(size_t n) {
    if constexpr (std::is_same_v<T, cld>) {
        return Eigen::MatrixXcd::Identity(n, n);
    } else if constexpr (std::is_same_v<T, ld>) {
        return Eigen::MatrixXd::Identity(n, n);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return Eigen::MatrixXcd::Identity(n, n);
    } else {
        return Eigen::MatrixXd::Identity(n, n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix operations helpers (inline for performance)
// ─────────────────────────────────────────────────────────────────────────────

inline ld_mat conjugate_transpose(const ld_mat& M) {
    return M.adjoint();
}

inline double frobenius_norm(const ld_mat& M) {
    return M.norm();
}

inline ld_mat cholesky_factor(const ld_mat& M) {
    return M.llt().matrixL();
}

}  // namespace qm

#endif  // QM_EIGEN_TYPES_H
