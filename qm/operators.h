/*
╔════════════════════════════════════════════════════════════════════════════════╗
║         operators.h - KINETIC ENERGY & PION EXCHANGE OPERATORS                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Computes matrix elements of the kinetic energy operator T and the            ║
║   pion exchange coupling operator W between Gaussian basis states.             ║
║                                                                                ║
║   <ψ_i | T | ψ_j> = Hamiltonian diagonal (kinetic part)                        ║
║   <ψ_i | W | ψ_j> = W-operator coupling (pion exchange)                        ║
║                                                                                ║
║ KINETIC ENERGY MATRIX ELEMENTS:                                                ║
║                                                                                ║
║   1. CLASSICAL (non-relativistic):                                             ║
║      T_classical = ℏ²/2μ × <ψ | ∇² | ψ'>                                       ║
║      For Gaussians: known analytical form (3/2 × inv_gamma - eta²)             ║
║                                                                                ║
║      where inv_gamma = 4 c^T A_bra R A_ket c   (R = (A+A')⁻¹)                  ║
║      and eta measures shift mismatch between bra and ket                       ║
║                                                                                ║
║   2. RELATIVISTIC:                                                             ║
║      T_rel = <ψ | (√(p²+m²) - m) | ψ'>                                         ║
║      No closed form → solved via Gauss-Legendre quadrature in momentum space   ║
║                                                                                ║
║      Integrand: x² exp(-γx²) × kinetic_term(x) [γ and kinetic_term from data]  ║
║      Why k-space? Exponential factors → fast convergence at x_max = √(20/γ)    ║
║                                                                                ║
║ W-OPERATOR (PION EXCHANGE):                                                    ║
║   Couples 2-body (bare) and 3-body (dressed+pion) states.                      ║
║                                                                                ║
║   Physical process:                                                            ║
║     Nucleon pair (PN) ←→ Nucleon pair + virtual pion (PN + π)                  ║
║                                                                                ║
║   Mathematical form (Section 3.6):                                             ║
║     W_ij = S × g(b) × f(w_π^T r) × [pion tensor operator]                      ║
║                                                                                ║
║   where:                                                                       ║
║     • S = coupling strength (tuning parameter per deu.cc)                      ║
║     • g(b) = form factor strength                                              ║
║     • f(w_π^T r) = spatial form factor exp(-(w_π^T r)²/b²)                     ║
║     • w_π = pion coordinate direction in Jacobi space                          ║
║     • Tensor operator ~ rxr spin structure (3 types: no flip, flip1, flip2)    ║
║                                                                                ║
║   Spin operators encode which nucleon's spin is flipped (affects amplitude)    ║
║                                                                                ║
║ NUMERICAL TECHNIQUES:                                                          ║
║   • Gaussian overlaps: Analytical via Cholesky ingredients                     ║
║   • Classica TKE: Rational function of A matrices (fast, exact)                ║
║   • Relativistic TKE: 32-point Gauss-Legendre quadrature (>10 digit accuracy)  ║
║   • apply_basis_expansion(): Handles all 4 parity term combinations auto       ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

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
    
    // This lambda runs exactly ONCE, caches the math, and never runs again.
    static const auto [nodes, weights] = []() {
        std::array<ld, N> x;
        std::array<ld, N> w;
        int m = (N + 1) / 2;
        
        for (int i = 0; i < m; i++) {
            // Initial guess using PI
            ld z = std::cos(M_PI * (i + 0.75) / (N + 0.5)); 
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

    // The high-speed integration loop
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
    // Result is in MeV
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
        
        // Convert wavenumber k (x) to momentum p (MeV) using p = hbar * c * k
        ld p = x * HBARC; 
        
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


// Coupeling operator using spin flip 
enum SpinChannel { NO_FLIP = 0, FLIP_PARTICLE_1 = 1, FLIP_PARTICLE_2 = 2 };

cld total_w_coupling(const SpatialWavefunction& psi_bare, const SpatialWavefunction& psi_dressed, 
                     const rvec& c, ld b, ld S, ld isospin_factor, SpinChannel spin_chan) 
{
    // Calculate the Gaussian width (alpha) from the physical interaction range (b)
    ld alpha = 1.0 / (b * b);

    // Calculate the normalization scaling factor from Eq. 10
    ld b_pow_5 = std::pow(b, 5.0);
    ld two_pow_11_halves = std::pow(2.0, 5.5); 
    ld norm_sq = 4.0 * M_PI * (3.0 * std::sqrt(M_PI) * b_pow_5) / two_pow_11_halves;
    ld norm_factor = 1.0 / std::sqrt(norm_sq);

    // Promote the bare state up to the dressed dimension
    size_t target_dim = psi_dressed.A.size1();
    Gaussian g_bare_prim(psi_bare.A, psi_bare.s);
    Gaussian g_tilde = promote_and_absorb(g_bare_prim, target_dim, c, alpha);
    
    SpatialWavefunction psi_tilde(g_tilde.A, g_tilde.s, psi_bare.parity_sign);

    return apply_basis_expansion(psi_tilde, psi_dressed, 
            [&](const Gaussian& g_t, const Gaussian& g_d, ld M_term, const rmat& R) -> cld {
        
        // Calculate the real Cartesian vector
        rmat v = g_t.s + g_d.s; 
        rvec spatial_vec(3); // [x, y, z]
        for (size_t col = 0; col < 3; ++col) {
            spatial_vec[col] = M_term * 0.5 * dot_no_conj(c, R * v[col]);
        }
        
        // Map Cartesian [x, y, z] to Spherical [r^0, r^+, r^-]
        cld r_0 = cld(spatial_vec[2], 0.0); // z
        cld r_plus = cld(spatial_vec[0], spatial_vec[1]) / std::sqrt(2.0L); // (x + iy)/sqrt(2)

        // Apply the specific spin operator
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

        // Multiply by S (strength) and the isospin constant (e.g., 1 for pi^0, sqrt(2) for pi^+)
        return W_term * S * isospin_factor * norm_factor;
    });
}

// --- Charge Radius Operator r² ---
// Computes <ψ_bra | r² | ψ_ket> for charge radius calculations
// For PN pair: uses only the relative coordinate (first Jacobi coordinate)
ld charge_radius_operator(const SpatialWavefunction& psi_bra, const SpatialWavefunction& psi_ket,
                          const Jacobian& jac)
{
    return apply_basis_expansion(psi_bra, psi_ket,
            [&](const Gaussian& g_b, const Gaussian& g_k, ld M_term, const rmat& R) -> ld {

        // For PN pair, only the first Jacobi coordinate matters (relative coordinate)
        // Higher coordinates are for pion degrees of freedom
        if (jac.dim < 1) return 0.0;  // Safety check

        ld r_squared = 0.0;
        rvec c_rel = jac.get_c_internal(0);  // Only PN relative coordinate

        // For Gaussians g(r) = exp(-r·A·r + s·r), compute <r²> analytically
        // Key insight: r² = Σ_i r_i² where r_i is displacement in direction c_i

        // Method: For each spatial direction (x,y,z), compute <coord²>
        // In Jacobi space with correlation A, the spread in direction c is:
        // <(c·r)²> ≈ (trace of A⁻¹ component in direction c) / (correlation strength)

        rmat A_eff = g_b.A + g_k.A;  // Effective correlation from sum
        rmat R_eff = A_eff.inverse();  // Inverse gives spatial extent

        // Compute <r²> as sum of spatial variances
        // For direction c: variance ∝ c^T A_eff^-1 c
        ld c_variance = dot_no_conj(c_rel, R_eff * c_rel);

        // The coefficient 1.0 (not 0.75 or other empirical factors)
        // This directly gives <r²> contribution from this coordinate
        r_squared = c_variance;

        // Add shift contribution from displaced Gaussian centers
        // <r_shift²> from mismatch in center positions
        rvec shift_total = g_b.s[0] + g_k.s[0];  // Use only spatial part (first row)
        ld shift_sq = dot_no_conj(c_rel, R_eff * shift_total);
        shift_sq *= shift_sq;  // Square it

        r_squared += shift_sq * 0.5;  // Reduced contribution from shifts

        return M_term * r_squared;
    });
}

} // namespace qm