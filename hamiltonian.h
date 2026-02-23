#pragma once

#include<cmath>
#include"matrix.h"
#include"gaussian.h"

namespace qm {
struct hamiltonian {
    long double m_n, m_p, m_e; // Masses 
    long double hbar_c = 1973.27; // eV

    matrix overlap_matrix (const std::vector<gaus>& basis) {
        size_t n = basis.size();
        matrix N(n, n);
        for(size_t i=0;i<n;i++) for(size_t j=0;j<n;j++) {
            N(i, j) = overlap(basis[i], basis[j]); // eq(6)
        }
        return N;
    }

    long double calc_gamma (const gaus& gi, const gaus& gj, const vector& c) {
        matrix R = (gj.A + gi.A).inverse();
        size_t n = gj.A.size1();
        long double cRc = 0;
        for(size_t i=0;i<n;i++) for(size_t j=0;j<n;j++) {
            cRc += c[i]*R(i, j)*c[j];
        }
        return 0.25/cRc;
    }

    vector calc_eta (const gaus& gi, const gaus& gj, const vector& c) {
        matrix R = (gj.A.inverse() + gi.A.inverse()).inverse();
        size_t n = gj.A.size1();
        
        vector Ra(n), Rb(n);
        for(size_t i=0;i<n;i++) for(size_t j=0;j<n;j++) {
            Ra[i] += R(i, j) * gi.s[j].norm(); 
            Rb[i] += R(i, j) * gj.s[j].norm();
        }

        vector ARb = gi.A * Rb;
        vector BRa = gj.A * Ra;

        long double eta = 0; 
        vector diff = ARb - BRa;
        for(size_t i=0; i<n; i++){
            eta += c[i] * diff[i];
        }

        return eta;
    }

    long double K_rel (const gaus& gi, const gaus& gj, const vector& c) {
        long double ov = overlap(gi, gj);
        long double gamma = calc_gamma(gi, gj, c);
        vector eta = calc_eta(gi, gj, c);

        
    }
};
}