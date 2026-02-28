#pragma once

#include<cmath>
#include<omp.h>

#include"matrix.h"
#include"gaussian.h"
#include"jacobian.h"

namespace qm {
struct hamiltonian {
    long double m_n, m_p, m_e; // Masses 
    long double hbar_c = 197.3; // MeV * fm
    long double alpha = 1.0 / 137.035999; // Fine structure

    matrix overlap_matrix (const std::vector<gaus>& basis) {
        size_t n = basis.size();
        matrix N(n, n);
        vector norms(n);

        for(size_t i=0; i<n; i++) {
            norms[i] = std::sqrt(overlap(basis[i], basis[i]));
        }

        #pragma omp parallel for num_threads(8)
        FOR_MAT(N) N(i, j) = overlap(basis[i], basis[j]) / (norms[i] * norms[j]); // eq(6)
        return N;
    }

    matrix hamiltonian_matrix(const std::vector<gaus>& basis, const jacobian& J, bool relativistic) {
        size_t n = basis.size();
        size_t n_jacobi = J.dim(); 
        matrix H(n, n);
        vector norms(n);

        for(size_t i=0; i<n; i++) {
            norms[i] = std::sqrt(overlap(basis[i], basis[i]));
        }

        #pragma omp parallel for num_threads(8)
        FOR_MAT(H) {
            long double K_total = 0;
            long double V_total = 0;

            // 1. Kinetic energy (Sums over Jacobi coordinates)
            for (size_t d = 0; d < n_jacobi; d++) {
                vector c = J.c(d);
                long double mass_eff = J.mu(d);
                
                if (relativistic) {
                    K_total += K_rel(basis[i], basis[j], c, mass_eff);
                } else {
                    K_total += K_cla(basis[i], basis[j], c, mass_eff);
                }
            }

            // 2. Potential energy (Sums over all particle pairs)
            for (size_t p1 = 0; p1 < J.num_particles(); p1++) {
                for (size_t p2 = p1 + 1; p2 < J.num_particles(); p2++) {
                    
                    // Check if both particles have a charge
                    if (J.charges[p1] != 0 && J.charges[p2] != 0) {
                        vector c_pair = J.w(p1, p2);
                        long double charge_factor = J.charges[p1] * J.charges[p2];
                        
                        // Multiply the integral result by the physical constants
                        V_total += charge_factor * alpha * hbar_c * V_cou(basis[i], basis[j], c_pair);
                    }
                    
                    // NOTE: If you add a Nuclear strong force potential (e.g. Gaussian potential) 
                    // between the proton and neutron, you would add it right here!
                }
            }

            H(i, j) = (K_total + V_total) / (norms[i] * norms[j]);
        }
        return H;
    }

    

    // Generalized Coupling Operator W for any N -> N+1 system
    long double W_couple(const gaus& g_n, const gaus& g_n1, const long double& S, const long double& b) {
        size_t d1 = g_n.A.size1();   // Dimension of the smaller system
        size_t d2 = g_n1.A.size1();  // Dimension of the larger system

        // Safety check: The ket must have exactly one more Jacobi coordinate than the bra
        if (d2 != d1 + 1) {
            std::cerr << "Error: W_couple requires the ket to have exactly one more dimension than the bra." << std::endl;
            return 0.0;
        }

        long double w = 1.0 / (b * b);

        // 1. Create the temporary "upgraded" Gaussian |A'> with the larger dimension
        gaus g_prime(d2); 

        // 2. Embed A_n into the top-left of A', and apply the coupling 'w'
        #pragma omp parallel for num_threads(8)
        FOR_MAT(g_prime.A) {
            if (i < d1 && j < d1) {
                // Top-left block: Copy A_n 
                g_prime.A(i, j) = g_n.A(i, j);
                
                // Add the coupling term to the diagonal of the existing coordinates
                if (i == j) g_prime.A(i, j) += w; 
            } 
            else if (i == d1 && j == d1) {
                // The brand new dimension's diagonal gets the w term
                g_prime.A(i, j) = w;
            } 
            else {
                // The off-diagonals connecting the old system to the new particle are 0
                g_prime.A(i, j) = 0.0;
            }
        }

        // 3. Set the shift vectors for |A'>
        for (size_t i = 0; i < d2; ++i) {
            for (int d = 0; d < 3; ++d) {
                if (i < d1) {
                    g_prime.s(i, d) = g_n.s(i, d); // Copy existing spatial shifts
                } else {
                    g_prime.s(i, d) = 0.0;         // The new coordinate has no shift
                }
            }
        }

        // Normalize using the individual norms of the bra and ket
        long double norm_n = std::sqrt(overlap(g_n, g_n));
        long double norm_n1 = std::sqrt(overlap(g_n1, g_n1));

        // 4. Calculate the analytical integral in the larger space
        return S * overlap(g_prime, g_n1) / (norm_n * norm_n1);
    }

    long double calc_beta (const gaus& gi, const gaus& gj, const vector& c) {
        matrix R = (gj.A + gi.A).inverse();
        long double cRc = 0;
        FOR_MAT(R) cRc += c[i]*R(i, j)*c[j];
        return 1.0/cRc;
    }

    long double calc_gamma (const gaus& gi, const gaus& gj, const vector& c) {
        matrix R = (gj.A + gi.A).inverse();
        matrix BRA = gj.A * R * gi.A;
        long double cBRAc = 0;
        FOR_MAT(BRA) cBRAc += c[i]*BRA(i, j)*c[j];
        return 0.25/cBRAc;
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
        FOR_MAT(R) {
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
        long double p = hbar_c * x;
        // long double f_val = sqrt(p*p + mass*mass) - mass;
        // Use Conjugate trict to eliminate "-"
        long double f_val = (p*p) / (std::sqrt(p*p + mass*mass) + mass);
        long double base = x * f_val * std::exp(-gamma * x * x);

        // Handling the limit as eta -> 0 to avoid division by zero 
        if (std::abs(gamma * eta) < ZERO_LIMIT) {
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

    long double K_cla(const gaus& gi, const gaus& gj, const vector& c, const long double& mass) { 
        long double ov = overlap(gi, gj); 
        long double eta = calc_eta(gi, gj, c);

        long double prefactor = std::pow(hbar_c, 2) / (2.0 * mass);

        if (std::abs(eta) < ZERO_LIMIT) {
            // η -> 0
            matrix R = (gi.A + gj.A).inverse();
            matrix BRA = gj.A * R * gi.A;
            long double cBRAc = 0;
            FOR_MAT(BRA) {
                cBRAc += c[i] * BRA(i, j) * c[j];
            }

            return ov * prefactor * 6.0 * cBRAc;
        } else {
            long double gamma = calc_gamma(gi, gj, c);

            return ov * prefactor * (3.0/2.0 * 1.0/gamma - eta * eta);
        }
    }

    long double V_cou (const gaus& gi, const gaus& gj, const vector& c) { //eq(22-24)
        long double ov = overlap(gi, gj);
        long double beta = calc_beta(gi, gj, c);
        long double q = calc_q(gi, gj, c);

        long double J;
        // Handling the limit as q -> 0 to avoid division by zero 
        if (std::abs(q) < ZERO_LIMIT) {
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