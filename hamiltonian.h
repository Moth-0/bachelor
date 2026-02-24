#pragma once

#include<cmath>
#include"matrix.h"
#include"gaussian.h"
#include"jacobian.h"

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

    matrix hamiltonian_matrix(const std::vector<gaus>& basis, const jacobian& J, bool relativistic) {
        size_t n_g = basis.size();
        size_t n_jacobi = J.dim(); // N-1 coordinates
        matrix H(n_g, n_g);

        for (size_t i = 0; i < n_g; ++i) {
            for (size_t k = 0; k < n_g; ++k) {
                long double K_total = 0;
                long double V_total = 0;

                // Sum kinetic energy over all Jacobi coordinates [cite: 481, 482]
                for (size_t d = 0; d < n_jacobi; ++d) {
                    vector c = J.c(d);
                    long double mass_eff = J.mu(d);
                    
                    if (relativistic) {
                        K_total += K_rel(basis[i], basis[k], c, mass_eff);
                    } else {
                        K_total += K_cla(basis[i], basis[k], c, mass_eff);
                    }
                    
                    // Example: Add Coulomb potential for each coordinate
                    V_total += V_cou(basis[i], basis[k], c); 
                }

                H(i, k) = K_total + V_total;
            }
        }
        return H;
    }

    long double calc_beta (const gaus& gi, const gaus& gj, const vector& c) {
        matrix R = (gj.A + gi.A).inverse();
        size_t n = gj.A.size1();
        long double cRc = 0;
        for(size_t i=0;i<n;i++) for(size_t j=0;j<n;j++) {
            cRc += c[i]*R(i, j)*c[j];
        }
        return cRc;
    }
    long double calc_gamma (const gaus& gi, const gaus& gj, const vector& c) {
        return 0.25/calc_beta(gi, gj, c);
    }

    long double calc_q (const gaus& gi, const gaus& gj, const vector& c) {
        matrix R = (gi.A + gj.A).inverse();
        size_t n = R.size1();

        vector v(n);
        for(size_t i = 0; i < n; i++) {
            // v_i is the sum of shift vectors for particle i
            v[i] = (gi.s[i] + gj.s[i]).norm(); 
        }

        vector u = (R * v) * 0.5;

        long double q_val = 0;
        for(size_t i=0;i<n;i++) {
            q_val += c[i] * u[i];
        }

        return q_val;
    }

    long double calc_eta (const gaus& gi, const gaus& gj, const vector& c) {
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

    long double calculate_integrand(long double x, long double gamma, long double eta, long double mass) {
        long double f_val = sqrt(std::pow(hbar_c,2) * std::pow(x,2) + std::pow(mass,2));
        long double base = x * f_val * std::exp(-gamma * x * x);

        // Handling the limit as eta -> 0 to avoid division by zero 
        if (std::abs(eta) < 1e-15) {
            // eq (A.26)
            return 2.0 * x * base;
        } else {
            // General formula eq (A.25)
            return std::exp(gamma * eta * eta) / (gamma * eta) * base * std::sin(2.0 * gamma * eta * x);
        }
    }

    long double solve_J_rel(long double gamma, long double eta, long double mass) { // eq(A.25.1)
        const int n = 2000; // Must be even for Simpson's Rule
        long double x_max = 6.0 / std::sqrt(gamma); 
        long double h = x_max / n;
        long double sum = 0;

        for (int i = 0; i <= n; ++i) {
            long double x = i * h;
            long double val = calculate_integrand(x, gamma, eta, mass);

            if (i == 0 || i == n) {
                sum += val;
            } else if (i % 2 == 1) {
                sum += 4.0 * val;
            } else {
                sum += 2.0 * val;
            }
        }

        // Constants from your notes: (gamma/pi)^1.5 * 2 * pi * exp(-gamma * eta^2) [cite: 104, 736]
        long double front_factor = std::pow(gamma / pi, 1.5) * 2.0 * pi;
        
        return (h / 3.0) * sum * front_factor;
    }

    long double K_rel (const gaus& gi, const gaus& gj, const vector& c, const long double& mass) { //eq(A.25)
        long double ov = overlap(gi, gj);
        long double gamma = calc_gamma(gi, gj, c);
        long double eta = calc_eta(gi, gj, c);
        
        return ov * solve_J_rel(gamma, eta, mass);
    }

    long double K_cla (const gaus& gi, const gaus& gj, const vector& c, const long double& mass) { //eq(A.23)
        long double ov = overlap(gi, gj); 
        long double gamma = calc_gamma(gi, gj, c);
        long double eta = calc_eta(gi, gj, c);

        return ov * (3.0/2.0 * 1.0/gamma - eta * eta);
    }

    long double V_cou (const gaus& gi, const gaus& gj, const vector& c) { //eq(22-24)
        long double ov = overlap(gi, gj);
        long double beta = calc_beta(gi, gj, c);
        long double q = calc_q(gi, gj, c);

        long double J;
        // Handling the limit as q -> 0 to avoid division by zero 
        if (std::abs(q) < 1e-15) {
            // q -> 0
            J = 2 * std::sqrt(beta) / std::sqrt(pi);
        } else {
            // General formula 
            J = std::erf(std::sqrt(beta) * q) / q;
        }

        return ov * J;
    }
};
}