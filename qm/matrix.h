#pragma once
// matrix.h  --  generic column-major matrix and vector templated on scalar type T.
// Provides:
//   tvector<T>, tmatrix<T>           (generic)
//   vector  = tvector<long double>   (real alias used everywhere in spatial/KE code)
//   matrix  = tmatrix<long double>
//   cvector = tvector<cld>           (complex alias used for W matrix elements)
//   cmatrix = tmatrix<cld>
//
// Design goal: header-only, no external dependencies, works for both T=long double
// and T=complex<long double>.  All member functions are templated so nothing needs
// to be duplicated.

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <functional>
#include <initializer_list>

#define FORV(i,v)     for (size_t i = 0; i < (v).size();  ++i)
#define FOR_COLS(i,A) for (size_t i = 0; i < (A).size2(); ++i)
#define SELF          (*this)
#define ZERO_LIMIT    1e-10

namespace qm {

// ---------------------------------------------------------------------------
// Scalar helpers that work for both real and complex T
// ---------------------------------------------------------------------------
inline long double              scalar_abs (long double x)                   { return std::abs(x); }
inline long double              scalar_abs (std::complex<long double> x)     { return std::abs(x); }
inline long double              scalar_conj(long double x)                   { return x; }
inline std::complex<long double> scalar_conj(std::complex<long double> x)   { return std::conj(x); }

// ---------------------------------------------------------------------------
// tvector<T>  --  simple heap vector with arithmetic and norm
// ---------------------------------------------------------------------------
template<typename T>
struct tvector {
    std::vector<T> data;

    tvector()                        = default;
    explicit tvector(size_t n)       : data(n, T(0)) {}
    tvector(std::initializer_list<T> l) : data(l.begin(), l.end()) {}
    tvector(const tvector&)          = default;
    tvector(tvector&&)               = default;
    tvector& operator=(const tvector&) = default;
    tvector& operator=(tvector&&)    = default;

    size_t size() const { return data.size(); }
    void   resize(size_t n) { data.resize(n, T(0)); }
    void   push_back(const T& x) { data.push_back(x); }

    T&       operator[](size_t i)       { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    tvector& operator+=(const tvector& o) { FORV(i,SELF) data[i] += o[i]; return SELF; }
    tvector& operator-=(const tvector& o) { FORV(i,SELF) data[i] -= o[i]; return SELF; }
    tvector& operator*=(T x)              { FORV(i,SELF) data[i] *= x;     return SELF; }
    tvector& operator/=(T x)              { FORV(i,SELF) data[i] /= x;     return SELF; }

    long double norm() const {
        long double s = 0.0;
        FORV(i,SELF) s += std::norm(SELF[i]);  // std::norm(x) = |x|^2
        return std::sqrt(s);
    }
};

// Free-function mixed scalar multiply: vec * ld and ld * vec for any T
// (needed when T=complex<ld> and scaling factor is plain ld)
template<typename T>
tvector<T> scale(const tvector<T>& v, long double x) {
    tvector<T> r = v;
    FORV(i,r) r[i] *= x;
    return r;
}

template<typename T> tvector<T> operator+(const tvector<T>& a, const tvector<T>& b) { auto r=a; r+=b; return r; }
template<typename T> tvector<T> operator-(const tvector<T>& a, const tvector<T>& b) { auto r=a; r-=b; return r; }
template<typename T> tvector<T> operator-(const tvector<T>& a)                      { auto r=a; FORV(i,r) r[i]=-r[i]; return r; }
template<typename T> tvector<T> operator*(const tvector<T>& v, long double s)       { auto r=v; r*=s; return r; }
template<typename T> tvector<T> operator*(long double s, const tvector<T>& v)       { return v*s; }
template<typename T> tvector<T> operator/(const tvector<T>& v, long double s)       { auto r=v; r/=s; return r; }

// Hermitian inner product: sum_i conj(a[i]) * b[i]
template<typename T>
T dot(const tvector<T>& a, const tvector<T>& b) {
    T s(0); FORV(i,a) s += scalar_conj(a[i]) * b[i]; return s;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const tvector<T>& v) {
    os << "(";
    FORV(i,v) { os << v[i]; if (i+1 < v.size()) os << ", "; }
    return os << ")";
}

// ---------------------------------------------------------------------------
// tmatrix<T>  --  column-major matrix: cols[j][i] == M(i,j)
// ---------------------------------------------------------------------------
template<typename T>
struct tmatrix {
    std::vector<tvector<T>> cols;

    tmatrix() = default;
    tmatrix(size_t n, size_t m) {
        cols.resize(m);
        for (size_t j = 0; j < m; ++j) cols[j].resize(n);
    }
    tmatrix(const tmatrix&)            = default;
    tmatrix(tmatrix&&)                 = default;
    tmatrix& operator=(const tmatrix&) = default;
    tmatrix& operator=(tmatrix&&)      = default;

    size_t size1() const { return cols.empty() ? 0 : cols[0].size(); }
    size_t size2() const { return cols.size(); }

    void resize(size_t n, size_t m) {
        cols.resize(m);
        for (size_t j = 0; j < m; ++j) cols[j].resize(n);
    }

    void setid() {
        assert(size1() == size2());
        for (size_t i = 0; i < size1(); ++i) {
            SELF(i,i) = T(1);
            for (size_t j = i+1; j < size1(); ++j) SELF(i,j) = SELF(j,i) = T(0);
        }
    }

    T&            operator()(size_t i, size_t j)       { return cols[j][i]; }
    const T&      operator()(size_t i, size_t j) const { return cols[j][i]; }
    tvector<T>&       operator[](size_t j)             { return cols[j]; }
    const tvector<T>& operator[](size_t j) const       { return cols[j]; }

    // Plain transpose (no conjugation)
    tmatrix transpose() const {
        tmatrix R(size2(), size1());
        for (size_t i = 0; i < R.size1(); ++i)
            for (size_t j = 0; j < R.size2(); ++j)
                R(i,j) = SELF(j,i);
        return R;
    }
    tmatrix T_() const { return transpose(); }

    // Conjugate transpose  H^† = conj(H^T)
    tmatrix adjoint() const {
        tmatrix R(size2(), size1());
        for (size_t i = 0; i < R.size1(); ++i)
            for (size_t j = 0; j < R.size2(); ++j)
                R(i,j) = scalar_conj(SELF(j,i));
        return R;
    }

    tmatrix& operator+=(const tmatrix& o) { FOR_COLS(j,SELF) SELF[j] += o[j]; return SELF; }
    tmatrix& operator-=(const tmatrix& o) { FOR_COLS(j,SELF) SELF[j] -= o[j]; return SELF; }
    tmatrix& operator*=(long double x)    { FOR_COLS(j,SELF) SELF[j] *= x;     return SELF; }
    tmatrix& operator/=(long double x)    { FOR_COLS(j,SELF) SELF[j] /= x;     return SELF; }

    // LU determinant with partial pivoting
    T determinant() const {
        size_t n = size1(); assert(n == size2());
        tmatrix tmp = SELF;
        T det(1);
        for (size_t i = 0; i < n; ++i) {
            size_t piv = i;
            for (size_t k = i+1; k < n; ++k)
                if (scalar_abs(tmp(k,i)) > scalar_abs(tmp(piv,i))) piv = k;
            if (piv != i) {
                for (size_t k = 0; k < n; ++k) std::swap(tmp(i,k), tmp(piv,k));
                det *= T(-1);
            }
            if (scalar_abs(tmp(i,i)) < 1e-15) return T(0);
            for (size_t r = i+1; r < n; ++r) {
                T f = tmp(r,i) / tmp(i,i);
                for (size_t c = i+1; c < n; ++c) tmp(r,c) -= f * tmp(i,c);
            }
            det *= tmp(i,i);
        }
        return det;
    }

    // Gauss-Jordan inverse with partial pivoting
    tmatrix inverse() const {
        size_t n = size1(); assert(n == size2());
        tmatrix tmp = SELF, res(n,n); res.setid();
        for (size_t i = 0; i < n; ++i) {
            size_t piv = i;
            for (size_t k = i+1; k < n; ++k)
                if (scalar_abs(tmp(k,i)) > scalar_abs(tmp(piv,i))) piv = k;
            if (piv != i)
                for (size_t k = 0; k < n; ++k) {
                    std::swap(tmp(i,k), tmp(piv,k));
                    std::swap(res(i,k), res(piv,k));
                }
            T d = tmp(i,i);
            if (scalar_abs(d) < 1e-18) { std::cerr << "[matrix] singular\n"; return SELF; }
            for (size_t k = 0; k < n; ++k) { tmp(i,k) /= d; res(i,k) /= d; }
            for (size_t r = 0; r < n; ++r) {
                if (r == i) continue;
                T f = tmp(r,i);
                for (size_t k = 0; k < n; ++k) { tmp(r,k) -= f*tmp(i,k); res(r,k) -= f*res(i,k); }
            }
        }
        return res;
    }

    // Lower-triangular inverse
    tmatrix inverse_lower() const {
        size_t n = size1(); tmatrix res(n,n);
        for (size_t i = 0; i < n; ++i) {
            if (scalar_abs(SELF(i,i)) < ZERO_LIMIT) return tmatrix(0,0);
            res(i,i) = T(1) / SELF(i,i);
            for (size_t j = 0; j < i; ++j) {
                T s(0);
                for (size_t k = j; k < i; ++k) s += SELF(i,k) * res(k,j);
                res(i,j) = -s / SELF(i,i);
            }
        }
        return res;
    }
};

template<typename T> tmatrix<T> operator+(const tmatrix<T>& A, const tmatrix<T>& B) { auto R=A; R+=B; return R; }
template<typename T> tmatrix<T> operator-(const tmatrix<T>& A, const tmatrix<T>& B) { auto R=A; R-=B; return R; }
template<typename T> tmatrix<T> operator*(const tmatrix<T>& A, long double x)       { auto R=A; R*=x; return R; }
template<typename T> tmatrix<T> operator*(long double x, const tmatrix<T>& A)       { return A*x; }
template<typename T> tmatrix<T> operator/(const tmatrix<T>& A, long double x)       { auto R=A; R/=x; return R; }

template<typename T>
tvector<T> operator*(const tmatrix<T>& M, const tvector<T>& v) {
    tvector<T> r(M.size1());
    for (size_t i = 0; i < r.size(); ++i) {
        T s(0);
        for (size_t j = 0; j < v.size(); ++j) s += M(i,j) * v[j];
        r[i] = s;
    }
    return r;
}

template<typename T>
tmatrix<T> operator*(const tmatrix<T>& A, const tmatrix<T>& B) {
    tmatrix<T> R(A.size1(), B.size2());
    for (size_t k = 0; k < A.size2(); ++k)
        for (size_t j = 0; j < B.size2(); ++j)
            for (size_t i = 0; i < A.size1(); ++i)
                R(i,j) += A(i,k) * B(k,j);
    return R;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const tmatrix<T>& M) {
    os << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < M.size1(); ++i) {
        os << "[";
        for (size_t j = 0; j < M.size2(); ++j) {
            os << std::setw(12) << M(i,j);
            if (j+1 < M.size2()) os << " ";
        }
        os << (i+1 < M.size1() ? "]\n" : "]");
    }
    return os;
}

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------
using ld      = long double;
using cld     = std::complex<long double>;
using vector  = tvector<ld>;
using matrix  = tmatrix<ld>;
using cvector = tvector<cld>;
using cmatrix = tmatrix<cld>;

// Promote real → complex
inline cmatrix to_complex(const matrix& A) {
    cmatrix C(A.size1(), A.size2());
    for (size_t i = 0; i < A.size1(); ++i)
        for (size_t j = 0; j < A.size2(); ++j)
            C(i,j) = cld(A(i,j), 0.0L);
    return C;
}
inline cvector to_complex(const vector& v) {
    cvector c(v.size());
    for (size_t i = 0; i < v.size(); ++i) c[i] = cld(v[i], 0.0L);
    return c;
}

// Legacy scalar approx used in gaussian.h
inline bool approx(ld x, ld y, ld acc=1e-6L, ld eps=1e-6L) {
    if (std::fabs(x-y) < acc) return true;
    if (std::fabs(x-y) < eps*(std::fabs(x)+std::fabs(y))) return true;
    return false;
}
inline bool approx(const vector& v, const vector& u, ld acc=1e-6L, ld eps=1e-6L) {
    if (u.size()!=v.size()) return false;
    for (size_t i=0;i<u.size();i++) if(!approx(u[i],v[i],acc,eps)) return false;
    return true;
}

} // namespace qm