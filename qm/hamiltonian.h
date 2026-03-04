#pragma once

#include <cmath>
#include <numbers>

#include "matrix.h"
#include "gaussian.h"
#include "jacobian.h"

namespace qm {

struct hamiltonian {
    long double hbar_c = 197.3269804; // MeV * fm 

    // ---------------------------------------------------------
    // Helper: Calculate gamma (momentum space width parameter)
    // gamma = 1 / (4 * c^T * (A_i * B^{-1} * A_j) * c)
    // ---------------------------------------------------------
    long double calc_gamma(const gaus& gi, const gaus& gj, const vector& c) const {
        matrix R = (gi.A + gj.A).inverse();
        matrix BRA = gj.A * R * gi.A;
        
        long double cBRAc = 0.0;
        for(size_t i = 0; i < BRA.size1(); ++i) {
            for(size_t j = 0; j < BRA.size2(); ++j) {
                cBRAc += c[i] * BRA(i, j) * c[j];
            }
        }
        return 0.25 / cBRAc;
    }

    // ---------------------------------------------------------
    // Helper: Calculate eta (momentum space shift magnitude)
    // Evaluates the projected 3D shift vector and returns its magnitude
    // ---------------------------------------------------------
    long double calc_eta(const gaus& gi, const gaus& gj, const vector& c) const {
        matrix B_inv = (gi.A + gj.A).inverse();
        
        // Calculate the full N x 3 momentum shift matrix: A_j * B^{-1} * s_i - A_i * B^{-1} * s_j
        matrix term1 = gj.A * B_inv * gi.s; 
        matrix term2 = gi.A * B_inv * gj.s; 
        matrix eta_mat = term1 - term2;
        
        // Project down to a single 3D vector using the Jacobi extraction vector 'c'
        long double eta_x = 0.0, eta_y = 0.0, eta_z = 0.0;
        for(size_t i = 0; i < c.size(); ++i) {
            eta_x += c[i] * eta_mat(i, 0);
            eta_y += c[i] * eta_mat(i, 1);
            eta_z += c[i] * eta_mat(i, 2);
        }

        // Return the magnitude |eta|
        return std::sqrt(eta_x * eta_x + eta_y * eta_y + eta_z * eta_z);
    }

    // ---------------------------------------------------------
    // Classical Kinetic Energy (1D Projection Method)
    // ---------------------------------------------------------
    long double K_cla(const gaus& gi, const gaus& gj, const vector& c, const long double& mass) const { 
        long double ov = overlap(gi, gj); 
        if (std::abs(ov) < ZERO_LIMIT) return 0.0;

        long double eta = calc_eta(gi, gj, c);
        long double prefactor = (hbar_c * hbar_c) / (2.0 * mass);

        // If eta is near 0, compute without gamma to avoid precision loss
        if (std::abs(eta) < ZERO_LIMIT) {
            matrix B_inv = (gi.A + gj.A).inverse();
            matrix BRA = gj.A * B_inv * gi.A;
            long double cBRAc = 0.0;
            for(size_t i = 0; i < BRA.size1(); ++i) {
                for(size_t j = 0; j < BRA.size2(); ++j) {
                    cBRAc += c[i] * BRA(i, j) * c[j];
                }
            }
            return ov * prefactor * 6.0 * cBRAc;

        } else {
            long double gamma = calc_gamma(gi, gj, c);
            return ov * prefactor * (1.5 / gamma - (eta * eta));
        }
    }

    // ---------------------------------------------------------
    // Relativistic Kinetic Energy (Numerical Integral Method)
    // ---------------------------------------------------------
    long double calculate_integrand(long double x, long double gamma, long double eta, long double mass) const {
        long double p = hbar_c * x;
        // Conjugate trick to eliminate floating point cancellation for sqrt(p^2 + m^2) - m
        long double f_val = (p * p) / (std::sqrt(p * p + mass * mass) + mass);
        long double base = x * f_val * std::exp(-gamma * x * x);

        if (std::abs(gamma * eta) < ZERO_LIMIT) {
            return 2.0 * x * base;
        } else {
            return std::exp(gamma * eta * eta) / (gamma * eta) * base * std::sin(2.0 * gamma * eta * x);
        }
    }

    long double solve_J_rel(long double gamma, long double eta, long double mass) const { 
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

        long double front_factor = std::pow(gamma / pi, 1.5) * 2.0 * pi;
        return (h / 3.0) * sum * front_factor;
    }

    long double K_rel(const gaus& gi, const gaus& gj, const vector& c, const long double& mass) const { 
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0;

        long double gamma = calc_gamma(gi, gj, c);
        long double eta = calc_eta(gi, gj, c);
        
        return ov * solve_J_rel(gamma, eta, mass);
    }

    // Add this inside the hamiltonian struct in hamiltonian.h
    long double V_cou(const gaus& gi, const gaus& gj, const vector& c) const {
        long double ov = overlap(gi, gj);
        if (std::abs(ov) < ZERO_LIMIT) return 0.0;

        // Calculate beta
        matrix B_inv = (gi.A + gj.A).inverse();
        long double cBc = 0.0;
        for(size_t i = 0; i < B_inv.size1(); ++i) {
            for(size_t j = 0; j < B_inv.size2(); ++j) {
                cBc += c[i] * B_inv(i, j) * c[j];
            }
        }
        long double beta = 1.0 / cBc;

        // Calculate q (effective shift)
        matrix term1 = gj.A * B_inv * gi.s; 
        matrix term2 = gi.A * B_inv * gj.s; 
        matrix u_mat = term1 + term2; // Notice this is + for q, unlike - for eta
        
        long double q_x = 0, q_y = 0, q_z = 0;
        for(size_t i = 0; i < c.size(); ++i) {
            q_x += c[i] * u_mat(i, 0); q_y += c[i] * u_mat(i, 1); q_z += c[i] * u_mat(i, 2);
        }
        long double q = std::sqrt(q_x*q_x + q_y*q_y + q_z*q_z);

        long double J;
        if (std::abs(q) < ZERO_LIMIT) {
            J = 2.0 * std::sqrt(beta / pi);
        } else {
            J = std::erf(std::sqrt(beta) * q) / q;
        }

        return ov * J;
    }

    // ---------------------------------------------------------
    // W_vec: For P-wave transitions (e.g., Pion meson)
    // Evaluates < g_n | (c^T r) * exp(-r^2/b^2) | g_n1 >
    // ---------------------------------------------------------
    long double W_vec(const gaus& g_n, const gaus& g_n1, const matrix& Omega, const vector& c) const {
        size_t d1 = g_n.dim();
        size_t d2 = g_n1.dim(); 

        // 1. Promote g_n to d2
        gaus g_prime(d2);
        for (size_t i = 0; i < d2; ++i) {
            for (size_t j = 0; j < d2; ++j) {
                if (i < d1 && j < d1) g_prime.A(i, j) = g_n.A(i, j);
                else g_prime.A(i, j) = 0.0;
            }
            for (int d = 0; d < 3; ++d) {
                if (i < d1) g_prime.s(i, d) = g_n.s(i, d);
                else g_prime.s(i, d) = 0.0;
            }
        }

        // 2. Apply the W-kernel (Omega)
        g_prime.A = g_prime.A + Omega;

        // 3. Scalar overlap (M)
        long double M_val = overlap(g_prime, g_n1);
        if (std::abs(M_val) < ZERO_LIMIT) return 0.0;

        // 4. Calculate u = 0.5 * B_inv * v
        matrix B_inv = (g_prime.A + g_n1.A).inverse();
        matrix v = g_prime.s + g_n1.s;
        matrix u_mat = B_inv * v * 0.5;

        // 5. Evaluate the projection: c^T * u
        // The vector 'c' contains the spatial weights determined by the spin evaluation
        long double c_dot_u = 0.0;
        for (size_t i = 0; i < c.size(); ++i) {
            // Assuming 'c' projects the coordinates of the new meson
            c_dot_u += c[i] * u_mat(d1, i); 
        }

        // Return the final scalar energy
        return c_dot_u * M_val;
    }
};
} // namespace qm