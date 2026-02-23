#pragma once 

#include<cmath>
#include<numbers>
#include<initializer_list>
#include<ostream>
#include"matrix.h"

#define SELF (*this)
const long double pi = std::numbers::pi_v<long double>;

namespace qm{
struct gaus {
    matrix A; 
    matrix s;
    
    gaus() = default;
    ~gaus() = default; 
    gaus(const matrix& A_in, const matrix& s_in) : A(A_in), s(s_in) {}

    // long double M () {
    //     size_t n = SELF.A.size1();
    //     matrix A_inv = SELF.A.inverse();
    //     long double sAs = 0;
    //     for(size_t i=0; i<n; i++) for(size_t j=0; j<n; j++) {
    //     sAs += A_inv(i, j)* dot(s[i], s[j]);
    //     }
        
    //     return std::pow((std::pow(pi, n)/A.determinat()), (3.0/2.0)) * exp(1.0/4.0*sAs);
    // }

    

    long double operator()(const matrix& r) const {
        long double rAr = 0; 
        long double sr =0; 
        for(size_t i=0; i<A.size1(); i++){
            sr += dot(s[i], r[i]);
            for(size_t j=0; j<A.size2(); j++) {
                rAr += A(i, j)* dot(r[i], r[j]);
            }} 
        
        return std::exp(- rAr + sr);
    }
    
    void update(matrix& A_new, matrix& s_new) {
        A = A_new; s = s_new;
    }
};

long double overlap(const gaus& a, const gaus& b) {
    matrix v = a.s + b.s;
    matrix B = a.A + b.A;
    size_t n = B.size1();

    long double vBv = 0; 
    matrix B_inv = B.inverse();
    for(size_t i=0; i<n; i++) for(size_t j=0; j<n; j++) {
        vBv += B_inv(i, j)* dot(v[i], v[j]);
    }

    return std::pow((std::pow(pi, n)/B.determinat()), (3.0/2.0)) * exp(1.0/4.0*vBv);
}

}