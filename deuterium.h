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
║     • pion_mass:        Rest mass energy offset (if dressed)                    ║
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
// --- NEW HELPER: Calculate a single Overlap matrix element ---
inline cld calc_N_elem(const BasisState& state_i, const BasisState& state_j) {
    if (state_i.type == state_j.type) {
        return cld(static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi)), 0.0);
    }
    return cld(0.0, 0.0); // Orthogonal channels
}

// --- NEW HELPER: Calculate a single Hamiltonian matrix element ---
inline cld calc_H_elem(const BasisState& state_i, const BasisState& state_j, const ld b, const ld S, 
                       const std::vector<bool>& relativistic, ld ho_k = 0.0, Integrator method = Integrator::SIMPSON) {
    cld h_val = 0.0;

    // Case 1: Same Channel
    if (state_i.type == state_j.type) {
        if (state_i.type == Channel::PN) {
            // Bare Kinetic Energy
            ld T_pn = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, {relativistic[0]}, method);
            h_val = cld(T_pn, 0.0);
        } else {
            // Dressed Kinetic + Rest Mass
            ld T_total = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, relativistic, method);
            ld n_val = static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi));
            ld rest_mass_term = state_i.pion_mass * n_val;
            h_val = cld(T_total + rest_mass_term, 0.0);
        }

        // =====================================================================
        // NEW: HARMONIC OSCILLATOR REGULARIZATION (THE "BOX")
        // If ho_k > 0, we apply an artificial confining potential.
        // =====================================================================
        if (ho_k > ZERO_LIMIT) {
            // Create a vector of 1.0s for all N particles in this specific channel
            size_t N_particles = state_i.jac.N;
            rvec box_weights(N_particles);
            for(size_t p = 0; p < N_particles; ++p) {
                box_weights[p] = 1.0; 
            }
            ld r2_val = charge_radius_operator(state_i.psi, state_j.psi, state_i.jac, box_weights);
            
            // Multiply by 4.0 to convert CM distance squared to relative distance squared
            ld v_box = ho_k * r2_val; 
            h_val += cld(v_box, 0.0);
        }
    } 
    // Case 2: W-Operator Coupling (PN <-> Pion)
    else if ((state_i.type == Channel::PN && state_j.type != Channel::PN) ||
             (state_i.type != Channel::PN && state_j.type == Channel::PN)) {

        bool i_is_bare = (state_i.type == Channel::PN);
        const auto& state_bare  = i_is_bare ? state_i : state_j;
        const auto& state_dress = i_is_bare ? state_j : state_i;

        rvec c_pi_1 = state_dress.jac.get_internal_distance_vector(2, 0);
        rvec c_pi_2 = state_dress.jac.get_internal_distance_vector(2, 1);
        rvec c_nn = state_dress.jac.get_internal_distance_vector(0, 1);  // Nucleon-nucleon distance
        cld w_val = 0.0;

        if (state_dress.flip == FLIP_PARTICLE_1) {
            w_val = total_w_coupling(state_bare.psi, state_dress.psi, c_pi_1, c_nn, b, S, state_dress.isospin_factor, state_dress.flip);
        } else if (state_dress.flip == FLIP_PARTICLE_2) {
            w_val = total_w_coupling(state_bare.psi, state_dress.psi, c_pi_2, c_nn, b, S, state_dress.isospin_factor, state_dress.flip);
        } else {
            cld w_val_n1 = total_w_coupling(state_bare.psi, state_dress.psi, c_pi_1, c_nn, b, S, state_dress.isospin_factor, state_dress.flip);
            cld w_val_n2 = total_w_coupling(state_bare.psi, state_dress.psi, c_pi_2, c_nn, b, S, state_dress.isospin_factor, state_dress.flip);
            w_val = w_val_n1 - w_val_n2;
        }

        // Apply conjugation if 'i' is the dressed state
        h_val = i_is_bare ? w_val : std::conj(w_val);
    }
    
    return h_val;
}

std::tuple<cmat, cmat> build_matrices(const std::vector<BasisState>& basis, const ld b, const ld S, 
                                      const std::vector<bool>& relativistic, ld ho_k = 0.0, Integrator method=Integrator::SIMPSON)
{
    size_t size = basis.size();
    cmat H = zeros<cld>(size, size);
    cmat N = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            
            cld h_val = calc_H_elem(basis[i], basis[j], b, S, relativistic, ho_k, method);
            cld n_val = calc_N_elem(basis[i], basis[j]);

            H(i, j) = h_val;
            N(i, j) = n_val;

            if (i != j) {
                H(j, i) = std::conj(h_val);
                N(j, i) = std::conj(n_val);
            }
        }
    }
    return {H, N};
}

// Helper to map Channel enum to physical Z_i charges
rvec get_channel_charges(Channel type) {
    switch (type) {
        case Channel::PN:
            return {1.0, 0.0}; // Proton, Neutron
        case Channel::PI_0c_0f: case Channel::PI_0c_1f: case Channel::PI_0c_2f:
            return {1.0, 0.0, 0.0}; // Proton, Neutron, pi0
        case Channel::PI_pc_0f: case Channel::PI_pc_1f: case Channel::PI_pc_2f:
            return {0.0, 0.0, 1.0}; // Neutron, Neutron, pi+ (Proton emitted pi+)
        case Channel::PI_mc_0f: case Channel::PI_mc_1f: case Channel::PI_mc_2f:
            return {1.0, 1.0, -1.0}; // Proton, Proton, pi- (Neutron emitted pi-)
        default:
            return {1.0, 0.0};
    }
}

// Build the r² (charge radius) matrix for ALL diagonal basis states
cmat build_r2_matrix(const std::vector<BasisState>& basis)
{
    size_t size = basis.size();
    cmat R2 = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            if (basis[i].type == basis[j].type) {
                // Fetch the correct charge distribution for this specific channel
                rvec charges = get_channel_charges(basis[i].type);

                ld r2_val = charge_radius_operator(basis[i].psi, basis[j].psi, basis[i].jac, charges);

                R2(i, j) = cld(r2_val, 0.0);
                if (i != j) {
                    R2(j, i) = cld(r2_val, 0.0);
                }
            }
        }
    }
    return R2;
}

// Build the T (Kinetic Energy) matrix for ALL diagonal basis states
inline cmat build_T_matrix(const std::vector<BasisState>& basis, const std::vector<bool>& relativistic, 
                           Integrator method=Integrator::GAUSS_LEGENDRE)
{
    size_t size = basis.size();
    cmat T_mat = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            if (basis[i].type == basis[j].type) {
                ld t_val = 0.0;
                if (basis[i].type == Channel::PN) {
                    t_val = total_kinetic_energy(basis[i].psi, basis[j].psi, basis[i].jac, {relativistic[0]}, method);
                } else {
                    t_val = total_kinetic_energy(basis[i].psi, basis[j].psi, basis[i].jac, relativistic, method);
                }
                T_mat(i, j) = cld(t_val, 0.0);
                if (i != j) {
                    T_mat(j, i) = cld(t_val, 0.0);
                }
            }
        }
    }
    return T_mat;
}
