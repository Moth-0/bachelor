/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                           matrix.h - LINEAR ALGEBRA LIBRARY                    ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Custom templated matrix and vector classes supporting both real              ║
║   (long double) and complex (std::complex<long double>) arithmetic.            ║
║   Specifically optimized for quantum mechanical calculations.                  ║
║                                                                                ║
║ KEY CLASSES:                                                                   ║
║   • vector<T>: 1D array with +, -, *, /, dot, norm operations                  ║
║   • matrix<T>: 2D column-major matrix with inverse, determinant, Cholesky      ║
║                                                                                ║
║ WHY CUSTOM?                                                                    ║
║   • Explicit control over numerical thresholds (ZERO_LIMIT)                    ║
║   • Specialized methods: Cholesky (for overlap matrix N)                       ║
║   • Inverse of lower-triangular (for GEVP transform)                           ║
║   • Complex support for W-operator coupling elements                           ║
║   • Column-major layout: natural for Fortran-style QM code                     ║
║                                                                                ║
║ USAGE EXAMPLE:                                                                 ║
║   rmat A = eye<ld>(3) * 2.0;      // 3×3 identity scaled by 2                  ║
║   rvec v(3); v[0] = 1.0;           // vector with 3 elements                   ║
║   rvec u = A * v;                  // matrix-vector multiplication             ║
║   ld det = A.determinant();        // compute determinant                      ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include <vector>
#include <initializer_list>
#include <functional>
#include <cmath>
#include <complex>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <type_traits>

// ─────────────────────────────────────────────────────────────────────────────
// Convenience macros
// ─────────────────────────────────────────────────────────────────────────────
#define FORV(i,v)      for (size_t i = 0; i < (v).size(); i++)
#define FOR_COLS(i,A)  for (size_t i = 0; i < (A).size2(); i++)
#define FOR_MAT(M)     for (size_t i = 0; i < (M).size2(); i++) \
                           for (size_t j = 0; j < (M).size1(); j++)
#define SELF           (*this)
#define ZERO_LIMIT     1e-4

namespace qm {

// ─────────────────────────────────────────────────────────────────────────────
// Scalar trait helpers
//   scalar_conj(x)   : complex conjugate (identity for real types)
//   scalar_real(x)   : real part (identity for real types)
//   is_complex<T>    : type trait
// ─────────────────────────────────────────────────────────────────────────────
template<typename T> struct is_complex_t                   : std::false_type {};
template<typename T> struct is_complex_t<std::complex<T>>  : std::true_type  {};
template<typename T> constexpr bool is_complex = is_complex_t<T>::value;

// conj for real types (no-op)
template<typename T>
inline T scalar_conj(const T& x,
    typename std::enable_if<!is_complex<T>>::type* = nullptr)
{ return x; }

// conj for complex types
template<typename T>
inline T scalar_conj(const T& x,
    typename std::enable_if<is_complex<T>>::type* = nullptr)
{ return std::conj(x); }

// abs2 = |x|^2 for norm calculations
template<typename T>
inline auto scalar_abs2(const T& x) -> decltype(std::real(x * scalar_conj(x)))
{ return std::real(x * scalar_conj(x)); }

// ─────────────────────────────────────────────────────────────────────────────
// Templated vector<T>
//   T = long double               for real calculations
//   T = std::complex<long double> for complex (W-operator blocks, P-wave)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T = long double>
struct vector {
    using value_type = T;
    std::vector<T> data;

    vector() = default;
    explicit vector(size_t n) : data(n, T{0}) {}
    vector(std::initializer_list<T> list) : data(list.begin(), list.end()) {}

    vector(const vector&)            = default;
    vector(vector&&)                 = default;
    vector& operator=(const vector&) = default;
    vector& operator=(vector&&)      = default;
    ~vector()                        = default;

    size_t size() const { return data.size(); }
    void   resize(size_t n) { data.resize(n, T{0}); }

    T&       operator[](size_t i)       { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    vector& operator+=(const vector& o) { FORV(i, SELF) data[i] += o.data[i]; return SELF; }
    vector& operator-=(const vector& o) { FORV(i, SELF) data[i] -= o.data[i]; return SELF; }
    vector& operator*=(T x)            { FORV(i, SELF) data[i] *= x;           return SELF; }
    vector& operator/=(T x)            { FORV(i, SELF) data[i] /= x;           return SELF; }

    vector& push_back(T x) { data.push_back(x); return SELF; }

    // Euclidean norm: sqrt( sum |x_i|^2 )
    auto norm() const -> decltype(std::sqrt(scalar_abs2(T{}))) {
        decltype(scalar_abs2(T{})) s = 0;
        FORV(i, SELF) s += scalar_abs2(data[i]);
        return std::sqrt(s);
    }

    // Element-wise complex conjugate
    vector conj() const {
        vector r = SELF;
        FORV(i, r) r[i] = scalar_conj(r[i]);
        return r;
    }

    vector map(std::function<T(T)> f) const {
        vector r = SELF;
        FORV(i, r) r[i] = f(r[i]);
        return r;
    }
};

// ── Free operators for vector<T> ─────────────────────────────────────────────
template<typename T>
vector<T> operator+(const vector<T>& v, const vector<T>& u)
{ vector<T> r = v; r += u; return r; }

template<typename T>
vector<T> operator-(const vector<T>& v)
{ vector<T> r = v; FORV(i, r) r[i] = -r[i]; return r; }

template<typename T>
vector<T> operator-(const vector<T>& v, const vector<T>& u)
{ vector<T> r = v; r -= u; return r; }

template<typename T>
vector<T> operator*(const vector<T>& v, T a) { vector<T> r = v; r *= a; return r; }

template<typename T>
vector<T> operator*(T a, const vector<T>& v) { return v * a; }

template<typename T>
vector<T> operator/(const vector<T>& v, T a) { vector<T> r = v; r /= a; return r; }

// Dot product: sum_i conj(v_i) * u_i  (inner product, works for real & complex)
template<typename T>
T dot(const vector<T>& v, const vector<T>& u) {
    T s{0};
    FORV(i, v) s += scalar_conj(v[i]) * u[i];
    return s;
}

// Real dot (no conjugation) — useful for gradient/shift calculations
template<typename T>
T dot_no_conj(const vector<T>& v, const vector<T>& u) {
    T s{0};
    FORV(i, v) s += v[i] * u[i];
    return s;
}

// Approximate equality (real types only)
inline bool approx(long double x, long double y,
                   long double acc = 1e-6, long double eps = 1e-6) {
    if (std::fabs(x - y) < acc) return true;
    if (std::fabs(x - y) < eps * (std::fabs(x) + std::fabs(y))) return true;
    return false;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
    os << "(";
    for (size_t i = 0; i + 1 < v.size(); i++) os << v[i] << ", ";
    if (!v.data.empty()) os << v[v.size() - 1];
    os << ")";
    return os;
}


// ─────────────────────────────────────────────────────────────────────────────
// Templated matrix<T>  (column-major: cols[j][i] = M(i,j))
//   T = long double               for real matrices
//   T = std::complex<long double> for complex matrices
// ─────────────────────────────────────────────────────────────────────────────
template<typename T = long double>
struct matrix {
    using value_type = T;
    std::vector<vector<T>> cols;

    // ── Constructors ─────────────────────────────────────────────────────────
    matrix() = default;

    matrix(size_t rows, size_t ncols) {
        cols.resize(ncols);
        for (size_t j = 0; j < ncols; j++) cols[j].resize(rows);
    }

    // Initialiser list: each inner list is a COLUMN
    matrix(std::initializer_list<std::initializer_list<T>> list) {
        for (auto& c : list) cols.push_back(vector<T>(c));
    }

    matrix(const matrix&)            = default;
    matrix(matrix&&)                 = default;
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&)      = default;
    ~matrix()                        = default;

    // ── Size ─────────────────────────────────────────────────────────────────
    size_t size1() const { return cols.empty() ? 0 : cols[0].size(); } // rows
    size_t size2() const { return cols.size(); }                       // cols

    void resize(size_t rows, size_t ncols) {
        cols.resize(ncols);
        for (size_t j = 0; j < ncols; j++) cols[j].resize(rows);
    }

    // Set to identity (square matrices only)
    void setid() {
        assert(size1() == size2());
        for (size_t i = 0; i < size1(); i++)
            for (size_t j = 0; j < size2(); j++)
                SELF(i,j) = (i == j) ? T{1} : T{0};
    }

    // Zero out all elements
    void zero() {
        FOR_COLS(j, SELF) FORV(i, cols[j]) cols[j][i] = T{0};
    }

    // ── Element access ───────────────────────────────────────────────────────
    T&       operator()(size_t i, size_t j)        { return cols[j][i]; }
    const T& operator()(size_t i, size_t j) const  { return cols[j][i]; }
    vector<T>&       operator[](size_t j)          { return cols[j]; }
    const vector<T>& operator[](size_t j) const    { return cols[j]; }

    // ── Transpose: M^T ───────────────────────────────────────────────────────
    matrix transpose() const {
        matrix R(size2(), size1());
        for (size_t i = 0; i < size2(); i++)
            for (size_t j = 0; j < size1(); j++)
                R(i,j) = SELF(j,i);
        return R;
    }

    // ── Conjugate: element-wise conj ─────────────────────────────────────────
    matrix conj() const {
        matrix R = SELF;
        FOR_COLS(j, R) FORV(i, R.cols[j]) R.cols[j][i] = scalar_conj(R.cols[j][i]);
        return R;
    }

    // ── Adjoint (Hermitian conjugate): M† = (M^T)* ───────────────────────────
    // For real matrices this is the same as transpose.
    // For complex matrices this is transpose + conjugate.
    matrix adjoint() const { return SELF.transpose().conj(); }
    matrix dag()     const { return SELF.adjoint(); }

    // ── Arithmetic ───────────────────────────────────────────────────────────
    matrix& operator+=(const matrix& o) { FOR_COLS(j, SELF) SELF[j] += o[j]; return SELF; }
    matrix& operator-=(const matrix& o) { FOR_COLS(j, SELF) SELF[j] -= o[j]; return SELF; }
    matrix& operator*=(T x)             { FOR_COLS(j, SELF) SELF[j] *= x;    return SELF; }
    matrix& operator/=(T x)             { FOR_COLS(j, SELF) SELF[j] /= x;    return SELF; }
    matrix& operator-(const matrix& o)  { FOR_COLS(j, SELF) SELF[j] - o[j];  return SELF; }

    // ── Determinant (LU via Gaussian elimination with partial pivoting) ───────
    // Works for both real and complex T.
    T determinant() const {
        size_t n = size1();
        assert(n == size2());
        matrix tmp = SELF;
        T det{1};
        for (size_t i = 0; i < n; i++) {
            // Partial pivot: find row with largest |element|
            size_t pivot = i;
            auto best = scalar_abs2(tmp(i,i));
            for (size_t row = i+1; row < n; row++) {
                auto val = scalar_abs2(tmp(row,i));
                if (val > best) { best = val; pivot = row; }
            }
            if (pivot != i) {
                for (size_t k = 0; k < n; k++) std::swap(tmp(i,k), tmp(pivot,k));
                det *= T{-1};
            }
            if (scalar_abs2(tmp(i,i)) < 1e-28) return T{0};
            det *= tmp(i,i);
            for (size_t row = i+1; row < n; row++) {
                T f = tmp(row,i) / tmp(i,i);
                for (size_t col = i+1; col < n; col++)
                    tmp(row,col) -= f * tmp(i,col);
            }
        }
        return det;
    }

    // ── Inverse (Gauss-Jordan with partial pivoting) ──────────────────────────
    // Works for both real and complex T.
    matrix inverse() const {
        size_t n = size1();
        assert(n == size2());
        matrix tmp = SELF;
        matrix res(n,n);
        res.setid();
        for (size_t i = 0; i < n; i++) {
            // Partial pivot
            size_t pivot = i;
            auto best = scalar_abs2(tmp(i,i));
            for (size_t row = i+1; row < n; row++) {
                auto val = scalar_abs2(tmp(row,i));
                if (val > best) { best = val; pivot = row; }
            }
            if (pivot != i)
                for (size_t k = 0; k < n; k++) {
                    std::swap(tmp(i,k), tmp(pivot,k));
                    std::swap(res(i,k), res(pivot,k));
                }
            T d = tmp(i,i);
            if (scalar_abs2(d) < 1e-28) {
                std::cerr << "[matrix::inverse] singular matrix\n";
                return matrix(n,n); // return zero matrix on failure
            }
            for (size_t k = 0; k < n; k++) { tmp(i,k) /= d; res(i,k) /= d; }
            for (size_t row = 0; row < n; row++) {
                if (row == i) continue;
                T f = tmp(row,i);
                for (size_t k = 0; k < n; k++) {
                    tmp(row,k) -= f * tmp(i,k);
                    res(row,k) -= f * res(i,k);
                }
            }
        }
        return res;
    }

    // ── Inverse of lower-triangular matrix (exact back-substitution) ──────────
    // Used by solver.h for Cholesky L^{-1}.
    //
    // Returns matrix(0,0) if any diagonal entry |L(i,i)| < ZERO_LIMIT.
    // This threshold matches the one used by cholesky() below, so raising
    // ZERO_LIMIT in one place tightens both rejection criteria together.
    matrix inverse_lower() const {
        size_t n = size1();
        assert(n == size2());
        matrix res(n,n);
        for (size_t i = 0; i < n; i++) {
            if (scalar_abs2(SELF(i,i)) < ZERO_LIMIT * ZERO_LIMIT) {
                return matrix(0,0);  // signal failure silently; SVM rejects the candidate
            }
            res(i,i) = T{1} / SELF(i,i);
            for (size_t j = 0; j < i; j++) {
                T s{0};
                for (size_t k = j; k < i; k++) s += SELF(i,k) * res(k,j);
                res(i,j) = -s / SELF(i,i);
            }
        }
        return res;
    }

    // ── Cholesky decomposition: A = L L†  (A must be Hermitian pos-definite) ──
    // Returns lower-triangular L, or matrix(0,0) on failure.
    //
    // Failure threshold: diag_val < ZERO_LIMIT * ZERO_LIMIT
    
    matrix cholesky() const {
        size_t n = size1();
        assert(n == size2());
        matrix L(n,n);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j <= i; j++) {
                T s{0};
                for (size_t k = 0; k < j; k++)
                    s += L(i,k) * scalar_conj(L(j,k));
                if (i == j) {
                    auto diag_val = std::real(SELF(i,i) - s);
                    if (diag_val < ZERO_LIMIT * ZERO_LIMIT) {
                        return matrix(0,0);  // not positive-definite or too ill-conditioned
                    }
                    L(i,i) = T{std::sqrt(diag_val)};
                } else {
                    L(i,j) = (SELF(i,j) - s) / L(j,j);
                }
            }
        }
        return L;
    }

    // ── Trace ─────────────────────────────────────────────────────────────────
    T trace() const {
        assert(size1() == size2());
        T s{0};
        for (size_t i = 0; i < size1(); i++) s += SELF(i,i);
        return s;
    }

    // ── Check if (approximately) Hermitian ───────────────────────────────────
    bool is_hermitian(double tol = 1e-10) const {
        if (size1() != size2()) return false;
        for (size_t i = 0; i < size1(); i++)
            for (size_t j = i+1; j < size2(); j++) {
                auto diff = SELF(i,j) - scalar_conj(SELF(j,i));
                if (scalar_abs2(diff) > tol*tol) return false;
            }
        return true;
    }
};

// ── Free operators for matrix<T> ─────────────────────────────────────────────
template<typename T>
matrix<T> operator+(const matrix<T>& A, const matrix<T>& B)
{ matrix<T> R = A; R += B; return R; }

template<typename T>
matrix<T> operator-(const matrix<T>& A, const matrix<T>& B)
{ matrix<T> R = A; R -= B; return R; }

template<typename T>
matrix<T> operator-(const matrix<T>& A)
{ matrix<T> R(A.size1(), A.size2()); R -= A; return R; }

template<typename T>
matrix<T> operator*(const matrix<T>& A, T x) { matrix<T> R = A; R *= x; return R; }

template<typename T>
matrix<T> operator*(T x, const matrix<T>& A) { return A * x; }

template<typename T>
matrix<T> operator/(const matrix<T>& A, T x) { matrix<T> R = A; R /= x; return R; }

// Matrix-vector product: M * v
template<typename T>
vector<T> operator*(const matrix<T>& M, const vector<T>& v) {
    vector<T> r(M.size1());
    for (size_t i = 0; i < M.size1(); i++) {
        T s{0};
        for (size_t j = 0; j < v.size(); j++) s += M(i,j) * v[j];
        r[i] = s;
    }
    return r;
}

// Matrix-matrix product: A * B
template<typename T>
matrix<T> operator*(const matrix<T>& A, const matrix<T>& B) {
    assert(A.size2() == B.size1());
    matrix<T> R(A.size1(), B.size2());
    // k-j-i loop order for better cache performance (column-major layout)
    for (size_t k = 0; k < A.size2(); k++)
        for (size_t j = 0; j < B.size2(); j++)
            for (size_t i = 0; i < A.size1(); i++)
                R(i,j) += A(i,k) * B(k,j);
    return R;
}

// Outer product: v * u^T  (result is a matrix)
template<typename T>
matrix<T> outer(const vector<T>& v, const vector<T>& u) {
    matrix<T> R(v.size(), u.size());
    for (size_t i = 0; i < v.size(); i++)
        for (size_t j = 0; j < u.size(); j++)
            R(i,j) = v[i] * scalar_conj(u[j]); // outer product with conjugation
    return R;
}

// Outer product without conjugation: v * u^T (useful for A += w*w^T/b^2)
template<typename T>
matrix<T> outer_no_conj(const vector<T>& v, const vector<T>& u) {
    matrix<T> R(v.size(), u.size());
    for (size_t i = 0; i < v.size(); i++)
        for (size_t j = 0; j < u.size(); j++)
            R(i,j) = v[i] * u[j];
    return R;
}

// ── Printing ─────────────────────────────────────────────────────────────────
template<typename T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& M) {
    os << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < M.size1(); i++) {
        os << "[";
        for (size_t j = 0; j < M.size2(); j++) {
            os << std::setw(10) << std::right << M(i,j);
            if (j + 1 < M.size2()) os << " ";
        }
        os << (i + 1 < M.size1() ? "]\n" : "]");
    }
    return os;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience type aliases
// ─────────────────────────────────────────────────────────────────────────────
using ld      = long double;
using cld     = std::complex<long double>;

using rvec    = vector<ld>;   // real vector
using cvec    = vector<cld>;  // complex vector
using rmat    = matrix<ld>;   // real matrix
using cmat    = matrix<cld>;  // complex matrix

// ─────────────────────────────────────────────────────────────────────────────
// Block matrix utilities
//   Useful for assembling the 9×9 block Hamiltonian where each block
//   is a K×K submatrix inside a (9K × 9K) full matrix.
// ─────────────────────────────────────────────────────────────────────────────

// Copy a (rows×cols) block from src into dst at offset (row_off, col_off)
template<typename T>
void set_block(matrix<T>& dst,
               const matrix<T>& src,
               size_t row_off, size_t col_off)
{
    for (size_t i = 0; i < src.size1(); i++)
        for (size_t j = 0; j < src.size2(); j++)
            dst(row_off + i, col_off + j) = src(i, j);
}

// Read a (nrows×ncols) block from src starting at (row_off, col_off)
template<typename T>
matrix<T> get_block(const matrix<T>& src,
                    size_t row_off, size_t col_off,
                    size_t nrows,   size_t ncols)
{
    matrix<T> R(nrows, ncols);
    for (size_t i = 0; i < nrows; i++)
        for (size_t j = 0; j < ncols; j++)
            R(i,j) = src(row_off + i, col_off + j);
    return R;
}

// Add contribution to a block: dst[block] += src
template<typename T>
void add_block(matrix<T>& dst,
               const matrix<T>& src,
               size_t row_off, size_t col_off)
{
    for (size_t i = 0; i < src.size1(); i++)
        for (size_t j = 0; j < src.size2(); j++)
            dst(row_off + i, col_off + j) += src(i, j);
}

// Promote a real matrix to complex
inline cmat to_complex(const rmat& A) {
    cmat R(A.size1(), A.size2());
    for (size_t i = 0; i < A.size1(); i++)
        for (size_t j = 0; j < A.size2(); j++)
            R(i,j) = cld{A(i,j), 0.0L};
    return R;
}

// Make a zero matrix of given size
template<typename T>
matrix<T> zeros(size_t rows, size_t cols) {
    return matrix<T>(rows, cols); // default-initialised to zero
}

// Make identity matrix of size n
template<typename T = long double>
matrix<T> eye(size_t n) {
    matrix<T> M(n, n);
    M.setid();
    return M;
}

} // namespace qm