#pragma once

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// Simplified Enum for the Sigma Model
enum class Channel { 
    PN,         // Bare Proton-Neutron Core (2 bodies)
    PN_SIGMA    // Proton-Neutron + Sigma Meson (3 bodies)
};

struct BasisState {
    SpatialWavefunction psi;
    Channel type;
    Jacobian jac;
    ld meson_mass;
};

// --- PURE SCALAR COUPLING ---
// Matches Equation (24) of Fedorov (2020)
inline cld scalar_w_coupling(const SpatialWavefunction& psi_bare, const SpatialWavefunction& psi_dressed, 
                             const rvec& c, ld b, ld S) 
{
    // 1. Calculate the Gaussian width (alpha) from the physical interaction range (b)
    ld alpha = 1.0L / (b * b);

    // 2. Promote the bare state up to the dressed dimension (A_tilde)
    size_t target_dim = psi_dressed.A.size1();
    Gaussian g_bare_prim(psi_bare.A, psi_bare.s);
    Gaussian g_tilde = promote_and_absorb(g_bare_prim, target_dim, c, alpha);
    
    SpatialWavefunction psi_tilde(g_tilde.A, g_tilde.s, psi_bare.parity_sign);

    // 3. For a purely scalar meson, it's just the spatial overlap multiplied by S!
    return cld(S * spactial_overlap(psi_tilde, psi_dressed), 0.0);
}

std::tuple<cmat, cmat> build_matrices(const std::vector<BasisState>& basis, const ld b, const ld S, bool relativistic) 
{
    size_t size = basis.size();
    cmat H = zeros<cld>(size, size);
    cmat N = zeros<cld>(size, size);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            
            cld h_val = 0.0;
            cld n_val = 0.0;
            
            const auto& state_i = basis[i];
            const auto& state_j = basis[j];

            // ---------------------------------------------------------
            // 1. OVERLAP MATRIX (N)
            // ---------------------------------------------------------
            if (state_i.type == state_j.type) {
                // Both Bare (1D) and Dressed (2D) are spatial Gaussians, evaluate normally
                n_val = static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi)); 
            }

            // ---------------------------------------------------------
            // 2. HAMILTONIAN MATRIX (H)
            // ---------------------------------------------------------
            if (state_i.type == state_j.type) {
                
                // --- KINETIC ENERGY & MASS ---
                ld T_total = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, {relativistic, relativistic});
                
                ld rest_mass_term = 0.0;
                if (state_i.type == Channel::PN_SIGMA) {
                    rest_mass_term = state_i.meson_mass * std::real(n_val); 
                }

                h_val += cld(T_total + rest_mass_term, 0.0);
            }
            // --- SYMMETRIC W-OPERATOR COUPLING ---
            else if ((state_i.type == Channel::PN && state_j.type == Channel::PN_SIGMA) || 
                     (state_i.type == Channel::PN_SIGMA && state_j.type == Channel::PN)) {
                
                bool i_is_bare = (state_i.type == Channel::PN);
                const auto& state_bare  = i_is_bare ? state_i : state_j;
                const auto& state_dress = i_is_bare ? state_j : state_i;

                // The sigma meson is the SECOND internal Jacobi coordinate (index 1)
                rvec c_sigma = state_dress.jac.get_c_internal(1); 
                
                cld w_val = scalar_w_coupling(state_bare.psi, state_dress.psi, c_sigma, b, S);

                if (i_is_bare) {
                    h_val += w_val;
                } else {
                    h_val += std::conj(w_val);
                }
            }

            // ---------------------------------------------------------
            // 3. APPLY AND MIRROR 
            // ---------------------------------------------------------
            H(i, j) = h_val;
            N(i, j) = n_val;
            
            if (i != j) {
                H(j, i) = std::conj(h_val);
                N(j, i) = std::conj(n_val);
            }
        }
    }

    return std::tuple<cmat, cmat> {H, N};
}