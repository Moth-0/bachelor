#pragma once

#include <vector>
#include <initializer_list>
#include <functional>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <ostream>

#define FORV(i,v)      for (size_t i = 0; i < v.size(); i++)
#define FOR_COLS(i,A)  for (size_t i = 0; i < A.size2(); i++)
#define FOR_MAT(M)     for (size_t i = 0; i < M.size1(); i++) for (size_t j = 0; j < M.size2(); j++)
#define SELF           (*this)
#define ZERO_LIMIT     1e-10

namespace qm {

struct vector {
    std::vector<long double> data;

    vector(size_t n) : data(n, 0.0) {}
    vector(std::initializer_list<long double> list) : data(list.begin(), list.end()) {}

    vector()                            = default;
    vector(const vector&)               = default;
    vector(vector&&)                    = default;
    ~vector()                           = default;
    vector& operator=(const vector&)    = default;
    vector& operator=(vector&&)         = default;

    size_t size() const { return data.size(); }
    void   resize(size_t n) { data.resize(n); }

    long double&       operator[](size_t i)       { return data[i]; }
    const long double& operator[](size_t i) const { return data[i]; }

    vector& operator+=(const vector& o) { FORV(i, SELF) data[i] += o.data[i]; return SELF; }
    vector& operator-=(const vector& o) { FORV(i, SELF) data[i] -= o.data[i]; return SELF; }
    vector& operator*=(long double x)   { FORV(i, SELF) data[i] *= x;          return SELF; }
    vector& operator/=(long double x)   { FORV(i, SELF) data[i] /= x;          return SELF; }

    vector& push_back(long double x) { data.push_back(x); return SELF; }

    long double norm() const {
        long double s = 0.0;
        FORV(i, SELF) s += SELF[i] * SELF[i];
        return std::sqrt(s);
    }

    vector map(std::function<long double(long double)> f) const {
        vector r = SELF;
        FORV(i, r) r[i] = f(r[i]);
        return r;
    }
};

vector operator+(const vector& v, const vector& u) { vector r = v; r += u; return r; }
vector operator-(const vector& v)                  { vector r = v; FORV(i, r) r[i] = -r[i]; return r; }
vector operator-(const vector& v, const vector& u) { vector r = v; r -= u; return r; }
vector operator*(const vector& v, long double a)   { vector r = v; r *= a; return r; }
vector operator*(long double a, const vector& v)   { return v * a; }
vector operator/(const vector& v, long double a)   { vector r = v; r /= a; return r; }

vector pow(const vector& v, size_t x) {
    vector r = v;
    FORV(i, r) r[i] = std::pow(v[i], x);
    return r;
}

long double dot(const vector& v, const vector& u) {
    long double s = 0.0;
    FORV(i, v) s += v[i] * u[i];
    return s;
}

bool approx(long double x, long double y, long double acc = 1e-6, long double eps = 1e-6) {
    if (std::fabs(x - y) < acc) return true;
    if (std::fabs(x - y) < eps * (std::fabs(x) + std::fabs(y))) return true;
    return false;
}

bool approx(const vector& v, const vector& u, long double acc = 1e-6, long double eps = 1e-6) {
    if (u.size() != v.size()) return false;
    for (size_t i = 0; i < u.size(); i++) if (!approx(u[i], v[i], acc, eps)) return false;
    return true;
}

std::ostream& operator<<(std::ostream& os, const vector& v) {
    os << "(";
    for (size_t i = 0; i < v.size() - 1; i++) os << v[i] << ", ";
    os << v[v.size() - 1] << ")";
    return os;
}


// Column-major matrix: cols[j][i] = M(i,j)
struct matrix {
    std::vector<vector> cols;

    matrix(size_t n, size_t m) {
        cols.resize(m);
        for (size_t i = 0; i < m; i++) cols[i].resize(n);
    }
    matrix(std::initializer_list<std::initializer_list<long double>> list) {
        for (auto c : list) cols.push_back(vector(c));
    }

    matrix()                            = default;
    matrix(const matrix&)               = default;
    matrix(matrix&&)                    = default;
    ~matrix()                           = default;
    matrix& operator=(const matrix&)    = default;
    matrix& operator=(matrix&&)         = default;

    size_t size1() const { return cols.empty() ? 0 : cols[0].size(); }
    size_t size2() const { return cols.size(); }

    void resize(size_t n, size_t m) {
        cols.resize(m);
        for (size_t i = 0; i < m; ++i) cols[i].resize(n);
    }

    void setid() {
        assert(size1() == size2());
        for (size_t i = 0; i < size1(); i++) {
            SELF(i, i) = 1.0;
            for (size_t j = i + 1; j < size1(); j++) SELF(i, j) = SELF(j, i) = 0.0;
        }
    }

    long double&       operator()(size_t i, size_t j)       { return cols[j][i]; }
    const long double& operator()(size_t i, size_t j) const { return cols[j][i]; }
    vector&            operator[](size_t i)                  { return cols[i]; }
    const vector&      operator[](size_t i) const            { return cols[i]; }

    matrix transpose() const {
        matrix R(size2(), size1());
        for (size_t i = 0; i < R.size1(); i++)
            for (size_t j = 0; j < R.size2(); j++)
                R(i, j) = SELF(j, i);
        return R;
    }

    matrix T() const { return SELF.transpose(); }

    matrix& operator+=(const matrix& o) { FOR_COLS(i, SELF) SELF[i] += o[i]; return SELF; }
    matrix& operator-=(const matrix& o) { FOR_COLS(i, SELF) SELF[i] -= o[i]; return SELF; }
    matrix& operator*=(long double x)   { FOR_COLS(i, SELF) SELF[i] *= x;    return SELF; }
    matrix& operator/=(long double x)   { FOR_COLS(i, SELF) SELF[i] /= x;    return SELF; }

    // LU determinant via Gaussian elimination with partial pivoting
    long double determinant() const {
        size_t n = size1();
        assert(n == size2());
        matrix tmp = SELF;
        long double det = 1.0;
        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            for (size_t j = i + 1; j < n; ++j)
                if (std::abs(tmp(j, i)) > std::abs(tmp(pivot, i))) pivot = j;
            if (pivot != i) {
                for (size_t k = 0; k < n; ++k) std::swap(tmp(i, k), tmp(pivot, k));
                det *= -1.0;
            }
            if (std::abs(tmp(i, i)) < 1e-15) return 0.0;
            for (size_t row = i + 1; row < n; ++row) {
                long double f = tmp(row, i) / tmp(i, i);
                for (size_t col = i + 1; col < n; ++col)
                    tmp(row, col) -= f * tmp(i, col);
            }
            det *= tmp(i, i);
        }
        return det;
    }

    // Gauss-Jordan inverse with partial pivoting
    matrix inverse() const {
        size_t n = size1();
        assert(n == size2());
        matrix tmp = SELF;
        matrix res(n, n);
        res.setid();
        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            for (size_t j = i + 1; j < n; ++j)
                if (std::abs(tmp(j, i)) > std::abs(tmp(pivot, i))) pivot = j;
            if (pivot != i)
                for (size_t k = 0; k < n; ++k) {
                    std::swap(tmp(i, k), tmp(pivot, k));
                    std::swap(res(i, k), res(pivot, k));
                }
            long double d = tmp(i, i);
            if (std::abs(d) < 1e-18) {
                std::cerr << "Matrix is singular\n";
                return SELF;
            }
            for (size_t k = 0; k < n; ++k) { tmp(i, k) /= d; res(i, k) /= d; }
            for (size_t row = 0; row < n; ++row) {
                if (row == i) continue;
                long double f = tmp(row, i);
                for (size_t k = 0; k < n; ++k) {
                    tmp(row, k) -= f * tmp(i, k);
                    res(row, k) -= f * res(i, k);
                }
            }
        }
        return res;
    }

    // Exact inversion of a lower-triangular matrix
    matrix inverse_lower() const {
        size_t n = size1();
        matrix res(n, n);
        for (size_t i = 0; i < n; i++) {
            if (std::abs(SELF(i, i)) < ZERO_LIMIT) return matrix(0, 0);
            res(i, i) = 1.0 / SELF(i, i);
            for (size_t j = 0; j < i; j++) {
                long double s = 0.0;
                for (size_t k = j; k < i; k++) s += SELF(i, k) * res(k, j);
                res(i, j) = -s / SELF(i, i);
            }
        }
        return res;
    }
};

matrix operator/(const matrix& A, long double x)   { matrix R = A; R /= x; return R; }
matrix operator*(const matrix& A, long double x)   { matrix R = A; R *= x; return R; }
matrix operator*(long double x, const matrix& A)   { return A * x; }
matrix operator+(const matrix& A, const matrix& B) { matrix R = A; R += B; return R; }
matrix operator-(const matrix& A, const matrix& B) { matrix R = A; R -= B; return R; }

vector operator*(const matrix& M, const vector& v) {
    vector r(M.size1());
    for (size_t i = 0; i < r.size(); i++) {
        long double s = 0.0;
        for (size_t j = 0; j < v.size(); j++) s += M(i, j) * v[j];
        r[i] = s;
    }
    return r;
}

matrix operator*(const matrix& A, const matrix& B) {
    matrix R(A.size1(), B.size2());
    for (size_t k = 0; k < A.size2(); k++)
        for (size_t j = 0; j < B.size2(); j++)
            for (size_t i = 0; i < A.size1(); i++)
                R(i, j) += A(i, k) * B(k, j);
    return R;
}

matrix pow(const matrix& M, size_t x) {
    matrix R = M;
    FOR_COLS(i, R) R[i] = pow(M[i], x);
    return R;
}

std::ostream& operator<<(std::ostream& os, const matrix& M) {
    os << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < M.size1(); ++i) {
        os << "[";
        for (size_t j = 0; j < M.size2(); ++j) {
            os << std::setw(6) << std::right << M(i, j);
            if (j < M.size2() - 1) os << " ";
        }
        os << (i != M.size1() - 1 ? "]\n" : "]");
    }
    return os;
}

} // namespace qm