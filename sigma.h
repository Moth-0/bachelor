/*
╔════════════════════════════════════════════════════════════════════════════════╗
║          sigma.h - DEUTERON WITH SIGMA MESON EXCHANGE COUPLING                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Fedorov sigma-meson model: deuteron as proton-neutron with scalar σ(500)     ║
║   exchange. Simplified compared to pion model (only one meson type, scalar     ║
║   coupling). Demonstrates how different meson exchanges affect binding.        ║
║                                                                                ║
║ CHANNEL ENUM:                                                                  ║
║   • PN:          Bare proton-neutron (2-body state, parity +1)                 ║
║   • PN_SIGMA:    PN + sigma meson (3-body state, parity -1)                    ║
║                                                                                ║
║ BASISSTATE STRUCTURE:                                                          ║
║   Bundles spatial wavefunction + physical metadata:                            ║
║     • psi:              SpatialWavefunction (A, s, parity ±)                   ║
║     • type:             Channel (PN or PN_SIGMA)                              ║
║     • jac:              Jacobian (reduced masses, transformations)             ║
║     • meson_mass:       Sigma rest mass energy offset (~500 MeV)               ║
║                                                                                ║
║ HAMILTONIAN CONSTRUCTION:                                                      ║
║   build_matrices() computes H[i,j] and N[i,j] for all basis states:            ║
║                                                                                ║
║   Case 1: Same channel (both PN or both PN_SIGMA)                              ║
║     • N[i,j] = <ψ_i | ψ_j>  (spatial overlap)                                  ║
║     • H[i,j] = T[i,j] + (meson_mass if dressed)                                ║
║                                                                                ║
║   Case 2: PN and PN_SIGMA states (different channels)                          ║
║     • N[i,j] = 0  (orthogonal channels)                                        ║
║     • H[i,j] = <ψ_bare | W_scalar | ψ_dressed>  (sigma exchange)               ║
║                                                                                ║
║   Matrix properties:                                                           ║
║     • Hermitian: H† = H (ensures real eigenvalues)                             ║
║     • Positive definite N: ensures GEVP well-conditioned                       ║
║     • Block diagonal structure: PN block + PN_SIGMA block + coupling           ║
║                                                                                ║
║ SCALAR vs VECTOR COUPLING:                                                     ║
║   Unlike pion exchange (vector), sigma is pure scalar coupling.                ║
║   This simplifies the W-operator: no spin-flip channels, no isospin factors.   ║
║   Equation (24)-(25) from Fedorov (2020): W = S · <A_tilde | A_sigma>         ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// Enum for the Sigma Model: bare PN or dressed with sigma meson
enum class Channel {
    PN,         // Bare Proton-Neutron Core (2-body state)
    PN_SIGMA    // Proton-Neutron + Sigma Meson (3-body state)
};

// Wrapper holding wavefunction + physical properties
struct BasisState {
    SpatialWavefunction psi;       // Gaussian basis function (A, s, parity)
    Channel type;                  // Which physics channel (PN or PN_SIGMA)
    Jacobian jac;                  // Reduced mass / coordinate transformations
    ld meson_mass = 500.0;                 // Rest mass of sigma meson (~500 MeV)
};

// --- PURE SCALAR COUPLING ---
// Implements Equation (24) and (25) from Fedorov (2020).
//
// W(r) = S · overlap in modified coordinate system where:
//   - A_tilde combines PN system with sigma form factor
//   - Same coordinate shifts as PN state
//
// Parameters:
//   - psi_bare, psi_dressed: bare PN and sigma-dressed wavefunctions
//   - b: interaction range parameter (form factor scale)
//   - S: coupling strength (determines binding energy)
//
// Returns: Complex coupling matrix element (typically real for scalar)
inline cld scalar_w_coupling(const SpatialWavefunction& psi_bare, const SpatialWavefunction& psi_dressed,
                             ld b, ld S)
{
    ld alpha = 1.0L / (b * b);

    // Build A_tilde exactly as shown in Eq (25)
    rmat A_tilde = zeros<ld>(2, 2);
    A_tilde(0, 0) = psi_bare.A(0, 0) + alpha; // Top-left: Nucleon coupling
    A_tilde(1, 1) = alpha;                    // Bottom-right: Meson coupling

    // Carry over the shifts from the 1D bare state to the top row of the 2D state
    rmat s_tilde = zeros<ld>(2, 3);
    for (int col = 0; col < 3; ++col) {
        s_tilde(0, col) = psi_bare.s(0, col);
    }

    SpatialWavefunction psi_tilde(A_tilde, s_tilde, psi_bare.parity_sign);

    // Eq 24: S * < A_tilde | A_sigma >
    return cld(S * spactial_overlap(psi_tilde, psi_dressed), 0.0);
}

/// Constructs full Hamiltonian H and overlap N matrices from basis states.
///
/// For each basis state pair (i,j), evaluates:
///   1. Overlap matrix N: spatial overlap <ψ_i | ψ_j>
///   2. Hamiltonian matrix H: kinetic energy T + potential couplings
///
/// The matrices encode the 2-channel physics:
///   • Diagonal PN blocks: kinetic energy only (T = p²/2m)
///   • Diagonal PN_SIGMA blocks: kinetic energy + meson rest mass
///   • Off-diagonal coupling: scalar W-operator between PN and PN_SIGMA
///
/// Parameters:
///   - basis: vector of BasisState (wavefunction + metadata)
///   - b: form factor range (fm) - controls meson interaction softness
///   - S: coupling strength (MeV) - main tuning parameter
///   - relativistic: if true, use T_rel = √(p²+m²)-m; else T = p²/2m
///
/// Returns: Pair (H, N) of complex matrices
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
                
                // --- CRITICAL FIX: DYNAMIC RELATIVISTIC FLAGS ---
                // Creates a vector of the exact right size (1 or 2), filled with the flag
                std::vector<bool> rel_flags(state_i.jac.dim, relativistic);
                
                // --- KINETIC ENERGY & MASS ---
                ld T_total = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, rel_flags);
                
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
                
                cld w_val = scalar_w_coupling(state_bare.psi, state_dress.psi, b, S);

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