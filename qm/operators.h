#pragma once

#include <cmath>
#include <functional>
#include <array>
#include "matrix.h"
#include "gaussian.h"

namespace qm {

// 32-point Gauss-Legendre Quadrature
// Computes roots dynamically at startup for perfect machine precision!
inline ld integrate_1d(const std::function<ld(ld)>& func, ld lower_bound, ld upper_bound) {
    constexpr int N = 32;
    
    // C++11 static initialization is guaranteed to be thread-safe.
    // This lambda runs exactly ONCE, caches the math, and never runs again.
    static const auto [nodes, weights] = []() {
        std::array<ld, N> x;
        std::array<ld, N> w;
        int m = (N + 1) / 2;
        
        for (int i = 0; i < m; i++) {
            // Initial guess using high-precision PI
            ld z = std::cos(3.14159265358979323846L * (i + 0.75) / (N + 0.5)); 
            ld z1 = 0;
            ld pp = 0;
            
            // Newton-Raphson root finding
            do {
                ld p1 = 1.0;
                ld p2 = 0.0;
                for (int j = 1; j <= N; j++) {
                    ld p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                }
                pp = N * (z * p1 - p2) / (z * z - 1.0);
                z1 = z;
                z = z1 - p1 / pp;
            } while (std::abs(z - z1) > 1e-15); // Loop until machine precision is hit
            
            // Nodes are perfectly symmetric
            x[i] = -z;
            x[N - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[N - 1 - i] = w[i];
        }
        return std::make_pair(x, w);
    }();

    // The actual high-speed integration loop
    ld half_width = 0.5 * (upper_bound - lower_bound);
    ld midpoint   = 0.5 * (upper_bound + lower_bound);
    ld sum = 0.0;

    for (int i = 0; i < N; ++i) {
        ld x = half_width * nodes[i] + midpoint;
        sum += weights[i] * func(x);
    }

    return sum * half_width;
}

// Define the global physical constant for hbar * c (in MeV * fm)
constexpr ld HBARC = 197.3269804;

// --- Classic Kinetic Energy ---
ld classic_kinetic_energy(const Gaussian& g_bra, const Gaussian& g_ket, 
                          ld M_overlap, const rmat& R, const rvec& c, ld mass) 
{
    // 1/gamma = 4 * c^T * A * R * A' * c (Units: fm^-2)
    rvec ARs_ket = g_ket.A * c;
    ld inv_gamma = 4.0 * dot_no_conj(c, g_bra.A * (R * ARs_ket));
    
    // eta calculation (Units: fm^-2)
    rvec eta_vec(3);
    for (size_t col = 0; col < 3; ++col) {
        rvec diff = (g_bra.A * (R * g_ket.s[col])) - (g_ket.A * (R * g_bra.s[col]));
        eta_vec[col] = dot_no_conj(c, diff);
    }
    ld eta_sq = dot_no_conj(eta_vec, eta_vec);

    if (eta_sq < ZERO_LIMIT * ZERO_LIMIT) {
        eta_sq = 0.0;
    }

    // Convert fm^-2 to MeV^2 using (hbar*c)^2, then divide by 2*mass (MeV)
    // Result is strictly in MeV!
    ld hbarc_sq = HBARC * HBARC;
    return M_overlap * hbarc_sq * (1.5 * inv_gamma - eta_sq) / (2.0 * mass);
}

// --- Relativistic Kinetic Energy ---
ld relativistic_kinetic_energy(const Gaussian& g_bra, const Gaussian& g_ket, 
                                ld M_overlap, const rmat& R, const rvec& c, ld mass) 
{
    // Calculate gamma (Units: fm^2)
    rvec A_ket_c = g_ket.A * c;
    rvec R_A_ket_c = R * A_ket_c;
    rvec A_bra_R_A_ket_c = g_bra.A * R_A_ket_c;
    ld inv_gamma = 4.0 * dot_no_conj(c, A_bra_R_A_ket_c);
    ld gamma = 1.0 / inv_gamma;

    // Calculate the shift magnitude eta (Units: fm^-1)
    rvec eta_vec(3);
    for (size_t col = 0; col < 3; ++col) {
        rvec Rs_ket = R * g_ket.s[col];
        rvec Rs_bra = R * g_bra.s[col];
        rvec ARs_ket = g_bra.A * Rs_ket;
        rvec ARs_bra = g_ket.A * Rs_bra;
        rvec diff = ARs_ket - ARs_bra;
        eta_vec[col] = dot_no_conj(c, diff);
    }
    ld eta = std::sqrt(dot_no_conj(eta_vec, eta_vec)); 

    // Define the 1D integrand function f(x)
    // 'x' here is the wavenumber 'k' in units of fm^-1
    auto integrand = [gamma, eta, mass](ld x) -> ld {
        
        // --- UNIT FIX ---
        // Convert wavenumber k (x) to momentum p (MeV) using p = hbar * c * k
        ld p = x * HBARC; 
        
        // Now it is perfectly safe to add p^2 (MeV^2) to mass^2 (MeV^2)
        ld kinetic_term = std::sqrt(p * p + mass * mass) - mass;
        
        if (eta < ZERO_LIMIT) {
            return x * x * std::exp(-gamma * x * x) * kinetic_term; 
        } else {
            return x * std::exp(-gamma * x * x) * std::sin(2.0 * gamma * eta * x) * kinetic_term; 
        }
    };

    // Perform numerical integration 
    ld x_max = std::sqrt(20.0 / gamma); 
    ld integral_result = integrate_1d(integrand, 0.0, x_max);

    // Apply the prefactors
    ld prefactor;
    if (eta < ZERO_LIMIT) {
        prefactor = 4.0 * M_PI * std::pow(gamma / M_PI, 1.5);
    } else {
        prefactor = 2.0 * M_PI * std::pow(gamma / M_PI, 1.5) * std::exp(gamma * eta * eta) / (gamma * eta);
    }

    return M_overlap * prefactor * integral_result;
}

ld total_kinetic_energy(const SpatialWavefunction& psi_bra, const SpatialWavefunction& psi_ket, 
                        const Jacobian& jac, const std::vector<bool>& relativistic) 
{
    // Ensure the relativistic flags match the internal dimensions!
    if (relativistic.size() != jac.dim) {
        std::cerr << "Fatal: 'relativistic' flags do not match Jacobi dimensions.\n";
        return 0.0;
    }

    // It will handle the B.inverse(), the M_term, and the 4 parity loops automatically.
    return apply_basis_expansion(psi_bra, psi_ket, [&](const Gaussian& g_b, const Gaussian& g_k, ld M_term, const rmat& R) {
        
        ld K_term = 0.0;
        
        // Loop over all internal Jacobi coordinates for THIS specific Gaussian pair
        for (size_t i = 0; i < jac.dim; ++i) {
            rvec c_i = jac.get_c_internal(i);
            ld mu_i  = jac.reduced_masses[i];

            if (relativistic[i]) {
                K_term += relativistic_kinetic_energy(g_b, g_k, M_term, R, c_i, mu_i);
            } else {
                K_term += classic_kinetic_energy(g_b, g_k, M_term, R, c_i, mu_i);
            }
        }
        
        return K_term; 
    });
}


// Enum to make your code super readable when calling this function
enum SpinChannel { NO_FLIP = 0, FLIP_PARTICLE_1 = 1, FLIP_PARTICLE_2 = 2 };

cld total_w_coupling(const SpatialWavefunction& psi_bare, const SpatialWavefunction& psi_dressed, 
                     const rvec& c, ld b, ld S, ld isospin_factor, SpinChannel spin_chan) 
{
    // 1. Calculate the Gaussian width (alpha) from the physical interaction range (b)
    ld alpha = 1.0L / (b * b);

    // 2. Promote the bare state up to the dressed dimension
    size_t target_dim = psi_dressed.A.size1();
    Gaussian g_bare_prim(psi_bare.A, psi_bare.s);
    Gaussian g_tilde = promote_and_absorb(g_bare_prim, target_dim, c, alpha);
    
    SpatialWavefunction psi_tilde(g_tilde.A, g_tilde.s, psi_bare.parity_sign);

    // Notice we use cld here, because the wrapper will sum up complex terms!
    return apply_basis_expansion(psi_tilde, psi_dressed, 
            [&](const Gaussian& g_t, const Gaussian& g_d, ld M_term, const rmat& R) -> cld {
        
        // 3. Calculate the real Cartesian vector exactly as before
        rmat v = g_t.s + g_d.s; 
        rvec spatial_vec(3); // [x, y, z]
        for (size_t col = 0; col < 3; ++col) {
            spatial_vec[col] = M_term * 0.5 * dot_no_conj(c, R * v[col]);
        }
        
        // 4. Map Cartesian [x, y, z] to Spherical [r^0, r^+, r^-]
        cld r_0 = cld(spatial_vec[2], 0.0); // z
        cld r_plus = cld(spatial_vec[0], spatial_vec[1]) / std::sqrt(2.0L); // (x + iy)/sqrt(2)

        // 5. Apply the specific spin operator from Section 3.6
        cld W_term = 0.0;
        if (spin_chan == NO_FLIP) {
            // No flip: f(w^T r) * w^T r^0
            W_term = r_0; 
        } 
        else if (spin_chan == FLIP_PARTICLE_1) {
            // Flip 1: f(w^T r) * sqrt(2) * w^T r^+
            W_term = std::sqrt(2.0L) * r_plus;
        }
        else if (spin_chan == FLIP_PARTICLE_2) {
            // Flip 2: -f(w^T r) * sqrt(2) * w^T r^+
            W_term = -std::sqrt(2.0L) * r_plus;
        }

        // 6. Multiply by S (strength) and the isospin constant (e.g., 1 for pi^0, sqrt(2) for pi^+)
        return W_term * S * isospin_factor;
    });
}

} // namespace qm