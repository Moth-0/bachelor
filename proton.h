#pragma once

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// Simplified Enum for the 5 Proton System States
enum class Channel { 
    P,            // Bare Proton
    P_PI0_0f,     // Proton + Neutral Pion (No Flip)
    P_PI0_1f,     // Proton + Neutral Pion (Spin Flip)
    N_PIP_0f,     // Neutron + Positive Pion (No Flip)
    N_PIP_1f      // Neutron + Positive Pion (Spin Flip)
};

// A wrapper that holds the state AND its physical properties
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