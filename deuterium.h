/*
╔════════════════════════════════════════════════════════════════════════════════╗
║            deuterium.h - DEUTERON SYSTEM DEFINITIONS & HAMILTONIAN             ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Physical system definitions for deuteron (proton-neutron) with pion          ║
║   exchange coupling. Defines the 10 physics channels and constructs the        ║
║   full Hamiltonian and overlap matrices from basis states.                     ║
║                                                                                ║
║ CHANNEL ENUM:                                                                  ║
║   Deuteron couples between:                                                    ║
║     • PN: bare proton-neutron (2-body state, parity +1)                        ║
║     • π⁰: PN + neutral pion (3 spin-flip variants, parity -1 each)             ║
║     • π⁺: PN + charged pion (3 spin-flip variants, parity -1 each)             ║
║     • π⁻: PN + charged pion (3 spin-flip variants, parity -1 each)             ║
║                                                                                ║
║   Spin flips:                                                                  ║
║     • NO_FLIP (0f):                                                            ║
║     • FLIP_PARTICLE_1:                                                         ║
║     • FLIP_PARTICLE_2:                                                         ║
║                                                                                ║
║   Isospin factors:                                                             ║
║     • π⁰: factor = 1                                                           ║
║     • π⁺, π⁻: factor = √2                                                      ║
║                                                                                ║
║ BASISSTATE STRUCTURE:                                                          ║
║   Bundles spatial wavefunction + physical metadata:                            ║
║     • psi:              SpatialWavefunction (A, s, parity ±)                   ║
║     • type:             Channel (which physics coupling)                       ║
║     • flip:             SpinChannel (which nucleon(s) flip)                    ║
║     • isospin_factor:   Iso-coupling weight                                    ║
║     • jac:              Jacobian (reduced masses, transformations)             ║
║     • pion_mass:        Rest mass energy offset (if dressed)                   ║
║                                                                                ║
║   Example: PN bare state (ground channel)                                      ║
║     BasisState {psi, Channel::PN, NO_FLIP, 1.0, jac_bare(m_p,m_n), 0.0}        ║
║                                                                                ║
║   Example: π⁰ with particle 1 flipped                                          ║
║     BasisState {psi, Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_3body, m_π}  ║
║                                                                                ║
║ HAMILTONIAN CONSTRUCTION:                                                      ║
║   build_matrices() computes H[i,j] and N[i,j] for all basis states:            ║
║                                                                                ║
║   Case 1: Same channel (i.j both PN or both pi-X)                              ║
║     • N[i,j] = <ψ_i | ψ_j>  (spatial overlap)                                  ║
║     • H[i,j] = T[i,j] + (pion_mass if dressed)                                 ║
║                                                                                ║
║   Case 2: PN and pi-dressed states (different channels)                        ║
║     • N[i,j] = 0  (orthogonal channels)                                        ║
║     • H[i,j] = <ψ_bare | W | ψ_dressed>  (pion exchange coupling)              ║
║                                                                                ║
║   Matrix properties:                                                           ║
║     • Hermitian: H† = H (ensures real eigenvalues)                             ║
║     • Positive definite N: ensures GEVP well-conditioned                       ║
║     • Sparse structure: many H[i,j] = 0 by selection rules                     ║
║                                                                                ║
║ PARITY CONSIDERATIONS:                                                         ║
║   Deuteron has J^PC = 1^++ (total angular momentum 1, positive parity)         ║
║   PN pairs:            even parity   (+1 spatial parity)                       ║
║   PN + pion:           odd parity    (-1 spatial parity, one body is particle) ║
║                                                                                ║
║   All basis functions constructed to respect parity conservation:              ║
║     • PN: psi.parity_sign = +1  (symmetric under PN ↔)                         ║
║     • π*: psi.parity_sign = -1  (antisymmetric, includes pion)                 ║
║                                                                                ║
║ NUMERICAL STABILITY:                                                           ║
║   • Cholesky of N fails if cond(N) > ZERO_LIMIT⁻²                              ║
║     → SVM rejects linearly dependent basis states automatically                ║
║   • H elements stay O(1-100 MeV) via proper normalization                      ║
║   • Careful W-operator branch selection prevents phase issues                  ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

# pragma once

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// Enum to easily track what physics apply to what state
enum class Channel { PN, 
                     PI_0c_0f, PI_0c_1f, PI_0c_2f, 
                     PI_pc_0f, PI_pc_1f, PI_pc_2f, 
                     PI_mc_0f, PI_mc_1f, PI_mc_2f };

// A wrapper that holds the state and its physical properties
struct BasisState {
    SpatialWavefunction psi;
    Channel type;
    SpinChannel flip;
    ld isospin_factor;
    Jacobian jac;
    ld pion_mass;
};

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

            // 1. OVERLAP MATRIX (N)
            if (state_i.type == state_j.type) {
                n_val = static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi)); 
            }

            // 2. HAMILTONIAN MATRIX (H)
            if (state_i.type == state_j.type && state_i.type == Channel::PN) {
                
                // --- BARE KINETIC ENERGY --
                ld T_pn = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, {false});
                h_val += cld(T_pn, 0.0);
                
            } 

            else if (state_i.type == state_j.type && state_i.type != Channel::PN) {
                
                // --- DRESSED KINETIC ENERGY ---
                ld T_total = 0.0;
                T_total += total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, {false, relativistic});
                
                ld rest_mass_term = state_i.pion_mass * std::real(n_val); 
                h_val += cld(T_total + rest_mass_term, 0.0);
                
            }

            else if ((state_i.type == Channel::PN && state_j.type != Channel::PN) || 
                     (state_i.type != Channel::PN && state_j.type == Channel::PN)) {
                
                // --- W OPERATOR ---
                // Identify which index holds the Bare state and which holds the Dressed state
                bool i_is_bare = (state_i.type == Channel::PN);
                const auto& state_bare  = i_is_bare ? state_i : state_j;
                const auto& state_dress = i_is_bare ? state_j : state_i;

                // Extract the exact Jacobi coordinate vectors
                rvec c_pi_1 = state_dress.jac.get_internal_distance_vector(2, 0); // Pion to N1
                rvec c_pi_2 = state_dress.jac.get_internal_distance_vector(2, 1); // Pion to N2
                
                cld w_val = 0.0;

                if (state_dress.flip == FLIP_PARTICLE_1) {
                    w_val = total_w_coupling(state_bare.psi, state_dress.psi, 
                                             c_pi_1, b, S, state_dress.isospin_factor, state_dress.flip);
                } 
                else if (state_dress.flip == FLIP_PARTICLE_2) {
                    w_val = total_w_coupling(state_bare.psi, state_dress.psi, 
                                             c_pi_2, b, S, state_dress.isospin_factor, state_dress.flip);
                } 
                else { // NO_FLIP
                    // Combine the emission from both Nucleon 1 and Nucleon 2!
                    cld w_val_n1 = total_w_coupling(state_bare.psi, state_dress.psi, 
                                                    c_pi_1, b, S, state_dress.isospin_factor, state_dress.flip);
                    
                    cld w_val_n2 = total_w_coupling(state_bare.psi, state_dress.psi, 
                                                    c_pi_2, b, S, state_dress.isospin_factor, state_dress.flip);
                    
                    // Isospin singlet dictates a relative minus sign between N1 and N2 emission
                    w_val = w_val_n1 - w_val_n2; 
                }
                
                // Because we are building the upper triangle H(i, j):
                // If 'i' is the bare state, H(i,j) = <Bare | W | Dressed>
                // If 'i' is the dressed state, H(i,j) = <Dressed | W | Bare> = conj(<Bare | W | Dressed>)
                if (i_is_bare) {
                    h_val += w_val;
                } else {
                    h_val += std::conj(w_val);
                }
            }

            // 3. APPLY AND MIRROR 
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

// Build the r² (charge radius) matrix for ALL diagonal basis states
cmat build_r2_matrix(const std::vector<BasisState>& basis)
{
    size_t size = basis.size();
    cmat R2 = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            // Radius operator preserves particle number and channel!
            // We MUST evaluate it for both bare (PN) and dressed (PN+pion) states.
            if (basis[i].type == basis[j].type) {
                ld r2_val = charge_radius_operator(basis[i].psi, basis[j].psi, basis[i].jac);
                R2(i, j) = cld(r2_val, 0.0);
                if (i != j) {
                    R2(j, i) = cld(r2_val, 0.0);
                }
            }
        }
    }
    return R2;
}