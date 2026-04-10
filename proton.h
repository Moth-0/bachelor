/*
╔════════════════════════════════════════════════════════════════════════════════╗
║       proton.h - SINGLE PROTON WITH PION EXCHANGE DEFINITIONS                  ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Physical system for a single proton dressed with virtual pion clouds.        ║
║   Similar to deuteron but with one nucleon (no 2-body core), testing          ║
║   pion exchange physics in a simpler system.                                   ║
║                                                                                ║
║ CHANNEL ENUM:                                                                  ║
║   Proton couples via pion-nucleon interactions:                                ║
║     • P:          Bare proton (1-body state, parity +1)                        ║
║     • P_PI0_0f:   Proton + π⁰ no spin flip (parity -1)                         ║
║     • P_PI0_1f:   Proton + π⁰ with spin flip (parity -1)                       ║
║     • N_PIP_0f:   Neutron + π⁺ no spin flip (parity -1)                        ║
║     • N_PIP_1f:   Neutron + π⁺ with spin flip (parity -1)                      ║
║                                                                                ║
║   Note: Proton can transition to neutron via π⁺ absorption (isospin dynamics)  ║
║                                                                                ║
║ BASISSTATE STRUCTURE:                                                          ║
║   Bundles spatial wavefunction + physical metadata:                            ║
║     • psi:              SpatialWavefunction (A, s, parity ±)                   ║
║     • type:             Channel (which physics configuration)                  ║
║     • flip:             SpinChannel (spin flip or not)                         ║
║     • isospin_factor:   Pion-type weighting (π⁰: 1, π⁺: √2)                    ║
║     • jac:              Jacobian (reduced masses, transformations)             ║
║     • pion_mass:        Pion rest mass energy offset                           ║
║                                                                                ║
║ HAMILTONIAN CONSTRUCTION:                                                      ║
║   build_matrices() computes H[i,j] and N[i,j] for all basis states:            ║
║                                                                                ║
║   Case 1: Bare proton (i = j = P)                                              ║
║     • N[i,j] = 1.0  (single particle, normalized by assumption)                ║
║     • H[i,j] = 0.0  (bare mass is energy reference)                            ║
║                                                                                ║
║   Case 2: Same dressed channel (both π or both N+π)                            ║
║     • N[i,j] = <ψ_i | ψ_j>  (spatial overlap)                                  ║
║     • H[i,j] = T[i,j] + pion_mass (kinetic energy + meson mass)                ║
║                                                                                ║
║   Case 3: Bare proton coupling to dressed (pion cloud)                         ║
║     • N[i,j] = 0  (orthogonal channels)                                        ║
║     • H[i,j] = <P | W | P+π>  (pion exchange transition)                       ║
║                                                                                ║
║   Matrix properties:                                                           ║
║     • Hermitian: H† = H (ensures real eigenvalues)                             ║
║     • Positive definite N: ensures GEVP well-conditioned                       ║
║     • Sparse: many H[i,j] = 0 by parity/isospin selection rules                ║
║                                                                                ║
║ PROTON MASS SELF-ENERGY:                                                       ║
║   The bare proton state |P> formally has infinite mass (one point particle).   ║
║   In practice, H(P,P) = 0 (reference point). Pion dressing lowers the energy.  ║
║   Result: proton gains a "constituent" mass from interactions.                 ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// Enum for 1+1 and 1+2 body states: single nucleon dressed with pion cloud
enum class Channel {
    P,            // Bare Proton (1-body, reference state)
    P_PI0_0f,     // Proton + Neutral Pion (No Flip)
    P_PI0_1f,     // Proton + Neutral Pion (Spin Flip)
    N_PIP_0f,     // Neutron + Positive Pion (No Flip)
    N_PIP_1f      // Neutron + Positive Pion (Spin Flip)
};

// Wrapper holding wavefunction + physical properties
struct BasisState {
    SpatialWavefunction psi;       // Gaussian basis function (A, s, parity)
    Channel type;                  // Which physics channel (P, P+π, N+π, etc.)
    SpinChannel flip;              // Spin flip type (if pion dressing)
    ld isospin_factor;             // Iso-weighting (π⁰: 1, π⁺: √2)
    Jacobian jac;                  // Reduced mass / coordinate transformations
    ld pion_mass;                  // Rest mass of associated pion (if dressed)
};

/// Constructs full Hamiltonian H and overlap N matrices from basis states.
///
/// For each basis state pair (i,j), evaluates:
///   1. Overlap matrix N: spatial overlap <ψ_i | ψ_j>
///   2. Hamiltonian matrix H: kinetic energy T + potential couplings
///
/// Special handling for bare proton:
///   • N(P,P) = 1.0 (point particle, exactly normalized)
///   • H(P,P) = 0.0 (reference energy point)
///
/// Dressed states (P+π, N+π):
///   • N[i,j] = Gaussian overlaps in relative coordinates
///   • H[i,j] = kinetic energy T + pion rest mass
///   • Coupling between bare and dressed: W-operator (pion exchange)
///
/// Parameters:
///   - basis: vector of BasisState (wavefunction + metadata)
///   - b: form factor range (fm) - controls pion interaction softness
///   - S: coupling strength (MeV) - main tuning parameter
///   - relativistic: if true, use T_rel = √(p²+m²)-m; else T = p²/2m
///
/// Returns: Pair (H, N) of complex matrices
std::tuple<cmat, cmat> build_matrices(const std::vector<BasisState>& basis, const ld b, const ld S, bool relativistic)
{
    size_t size = basis.size();
    cmat H = zeros<cld>(size, size);
    cmat N = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            
            cld h_val = 0.0;
            cld n_val = 0.0;
            
            const auto& state_i = basis[i];
            const auto& state_j = basis[j];

            // ---------------------------------------------------------
            // 1. OVERLAP MATRIX (N) & DIAGONAL BARE PHYSICS
            // ---------------------------------------------------------
            if (state_i.type == state_j.type) {
                if (state_i.type == Channel::P) {
                    // A single particle has no internal volume to integrate over.
                    // It is perfectly normalized by definition!
                    n_val = 1.0; 
                    h_val = 0.0; // The bare mass is our reference 0.0 energy
                } 
                else {
                    // Dressed states are 1D Gaussians, evaluate normally
                    n_val = static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi)); 
                }
            }

            // ---------------------------------------------------------
            // 2. HAMILTONIAN MATRIX (H)
            // ---------------------------------------------------------
            if (state_i.type == state_j.type && state_i.type != Channel::P) {
                
                // --- DRESSED KINETIC ENERGY & MASS ---
                ld T_total = 0.0;
                // state_i.jac only has 1 internal coordinate now!
                T_total += total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, {relativistic});
                
                ld rest_mass_term = state_i.pion_mass * std::real(n_val); 
                h_val += cld(T_total + rest_mass_term, 0.0);
                
            }
            // CRITICAL FIX: Symmetric W-Operator Check!
            else if ((state_i.type == Channel::P && state_j.type != Channel::P) || 
                     (state_i.type != Channel::P && state_j.type == Channel::P)) {
                
                // Identify which index holds the Bare state
                bool i_is_bare = (state_i.type == Channel::P);
                const auto& state_bare  = i_is_bare ? state_i : state_j;
                const auto& state_dress = i_is_bare ? state_j : state_i;

                // The pion is the ONLY internal coordinate (index 0)
                rvec c_pi = state_dress.jac.get_c_internal(0); 
                
                cld w_val = total_w_coupling(state_bare.psi, state_dress.psi, 
                                             c_pi, b, S, 
                                             state_dress.isospin_factor, state_dress.flip);

                // Apply symmetry
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