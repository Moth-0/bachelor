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

            else if (state_i.type == Channel::PN && state_j.type != Channel::PN) {
                
                // --- W OPERATOR ---
                // The transition strictly uses the 3-body dressed state's coordinate system
                // to locate the pion relative to the center of mass.
                rvec c_pi = state_j.jac.get_c_internal(1); 
                
                // Remember: total_w_coupling promotes the bare state inside!
                h_val += total_w_coupling(state_i.psi, state_j.psi, 
                                        c_pi, b, S, 
                                        state_j.isospin_factor, state_j.flip);
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