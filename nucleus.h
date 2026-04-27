/*
╔════════════════════════════════════════════════════════════════════════════════╗
║           nucleus.h - SINGLE NUCLEON SYSTEM DEFINITIONS & HAMILTONIAN          ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Physical system definitions for a single nucleon (proton or neutron) with    ║
║   pion exchange coupling. Computes self-energy via the Explicit Meson Model.  ║
║                                                                                ║
║ CHANNEL ENUM:                                                                  ║
║   Single nucleon couples between:                                              ║
║     • BARE_NUCLEON: Bare proton or neutron (0-coordinate, point particle)      ║
║     • NUCLEON_PI0:  Nucleon + neutral pion (1-coordinate, parity -1)           ║
║     • NUCLEON_PIC:  Nucleon + charged pion (1-coordinate, parity -1)           ║
║                                                                                ║
║ PHYSICS OF 0-COORDINATE SYSTEM:                                               ║
║   • Bare nucleon: 1 particle → 1-1=0 internal Jacobi coordinates              ║
║   • No spatial structure, no internal kinetic energy                           ║
║   • Serves as mathematical anchor for W-operator coupling                      ║
║   • No Gaussian A matrix (0×0)                                                 ║
║                                                                                ║
║ CHANNEL TEMPLATES:                                                             ║
║   Proton system:  BARE_NUCLEON + p+π⁰ + n+π⁺                                  ║
║   Neutron system: BARE_NUCLEON + n+π⁰ + p+π⁻                                  ║
║                                                                                ║
║ HAMILTONIAN CONSTRUCTION:                                                      ║
║   Case 1: Same channel (both bare or both nucleon-pion)                        ║
║     • N[i,j] = <ψ_i | ψ_j>  (spatial overlap)                                  ║
║     • H[i,j] = T[i,j] + (pion_mass if dressed)                                 ║
║     • For bare: hardcoded baseline values (perfect self-overlap)                ║
║                                                                                ║
║   Case 2: Bare <-> Pion states (different channels)                            ║
║     • N[i,j] = 0  (orthogonal channels)                                        ║
║     • H[i,j] = <ψ_bare | W | ψ_dressed>  (pion exchange coupling)              ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

# pragma once

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/jacobi.h"

using namespace qm;

// Enum for single-nucleon channels
enum class Channel { BARE_NUCLEON,
                     NUCLEON_PI0,
                     NUCLEON_PIC };

// Wrapper that holds the state and its physical properties
struct BasisState {
    SpatialWavefunction psi;
    Channel type;
    SpinChannel flip;
    ld isospin_factor;
    Jacobian jac;
    ld pion_mass;
};

// --- HELPER: Calculate a single Overlap matrix element ---
inline cld calc_N_elem(const BasisState& state_i, const BasisState& state_j) {
    if (state_i.type == state_j.type) {
        // Same channel
        if (state_i.type == Channel::BARE_NUCLEON) {
            // Bare nucleon: point particle, perfect self-overlap
            return cld(1.0, 0.0);
        } else {
            // Nucleon-pion state: standard spatial overlap
            return cld(static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi)), 0.0);
        }
    }
    // Different channels: orthogonal
    return cld(0.0, 0.0);
}

// --- HELPER: Calculate a single Hamiltonian matrix element ---
inline cld calc_H_elem(const BasisState& state_i, const BasisState& state_j, const ld b, const ld S,
                       const std::vector<bool>& relativistic, ld ho_k = 0.0, Integrator method = Integrator::SIMPSON) {
    cld h_val = 0.0;

    // Case 1: Same Channel
    if (state_i.type == state_j.type) {
        if (state_i.type == Channel::BARE_NUCLEON) {
            // Bare nucleon: point particle baseline energy (nucleon rest mass)
            // This gets the rest mass from the Jacobian (single particle)
            h_val = cld(0.0, 0.0);
        } else {
            // Nucleon-pion state: Kinetic Energy + Rest Mass
            ld T_total = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, relativistic, method);
            ld n_val = static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi));
            ld rest_mass_term = state_i.pion_mass * n_val;
            h_val = cld(T_total + rest_mass_term, 0.0);
        }

        // Harmonic oscillator regularization (the "box")
        if (ho_k > ZERO_LIMIT && state_i.type != Channel::BARE_NUCLEON) {
            size_t N_particles = state_i.jac.N;
            rvec box_weights(N_particles);
            for (size_t p = 0; p < N_particles; ++p) {
                box_weights[p] = 1.0;
            }
            ld r2_val = charge_radius_operator(state_i.psi, state_j.psi, state_i.jac, box_weights);
            ld v_box = ho_k * r2_val;
            h_val += cld(v_box, 0.0);
        }
    }
    // Case 2: W-Operator Coupling (BARE_NUCLEON <-> NUCLEON_PION)
    else if ((state_i.type == Channel::BARE_NUCLEON && state_j.type != Channel::BARE_NUCLEON) ||
             (state_i.type != Channel::BARE_NUCLEON && state_j.type == Channel::BARE_NUCLEON)) {

        bool i_is_bare = (state_i.type == Channel::BARE_NUCLEON);
        const auto& state_bare = i_is_bare ? state_i : state_j;
        const auto& state_dress = i_is_bare ? state_j : state_i;

        // For 2-particle system (nucleon + pion), get the pion-nucleon distance
        rvec c_pion = state_dress.jac.get_internal_distance_vector(1, 0);  // Pion relative to nucleon
        rvec c_zero = rvec(state_dress.jac.dim);
        for (size_t i = 0; i < state_dress.jac.dim; ++i) {
            c_zero[i] = 0.0;
        }

        // Evaluate W operator
        cld w_val = total_w_coupling(state_bare.psi, state_dress.psi, c_pion, c_zero, b, S,
                                    state_dress.isospin_factor, state_dress.flip);

        // Apply conjugation if 'i' is the dressed state
        h_val = i_is_bare ? w_val : std::conj(w_val);
    }

    return h_val;
}

std::tuple<cmat, cmat> build_matrices(const std::vector<BasisState>& basis, const ld b, const ld S,
                                      const std::vector<bool>& relativistic, ld ho_k = 0.0, Integrator method = Integrator::SIMPSON)
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

// Helper to map Channel enum to physical charges
rvec get_channel_charges(Channel type) {
    switch (type) {
        case Channel::BARE_NUCLEON:
            return {1.0};  // Single nucleon (charge basis)
        case Channel::NUCLEON_PI0:
            return {1.0, 0.0};  // Nucleon, pi0
        case Channel::NUCLEON_PIC:
            return {1.0, 1.0};  // Nucleon, charged pion
        default:
            return {1.0};
    }
}

// Build the r² (charge radius) matrix for diagonal basis states
cmat build_r2_matrix(const std::vector<BasisState>& basis)
{
    size_t size = basis.size();
    cmat R2 = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            if (basis[i].type == basis[j].type) {
                rvec charges = get_channel_charges(basis[i].type);
                ld r2_val = 0.0;

                if (basis[i].type != Channel::BARE_NUCLEON) {
                    r2_val = charge_radius_operator(basis[i].psi, basis[j].psi, basis[i].jac, charges);
                }
                // Bare nucleon contributes 0 to radius (point particle)

                R2(i, j) = cld(r2_val, 0.0);
                if (i != j) {
                    R2(j, i) = cld(r2_val, 0.0);
                }
            }
        }
    }
    return R2;
}

// Build the T (Kinetic Energy) matrix for diagonal basis states
inline cmat build_T_matrix(const std::vector<BasisState>& basis, const std::vector<bool>& relativistic,
                           Integrator method = Integrator::GAUSS_LEGENDRE)
{
    size_t size = basis.size();
    cmat T_mat = zeros<cld>(size, size);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            if (basis[i].type == basis[j].type) {
                ld t_val = 0.0;

                if (basis[i].type != Channel::BARE_NUCLEON) {
                    t_val = total_kinetic_energy(basis[i].psi, basis[j].psi, basis[i].jac, relativistic, method);
                }
                // Bare nucleon: no kinetic energy (point particle)

                T_mat(i, j) = cld(t_val, 0.0);
                if (i != j) {
                    T_mat(j, i) = cld(t_val, 0.0);
                }
            }
        }
    }
    return T_mat;
}

// Create channel templates for proton system
inline std::vector<BasisState> create_proton_templates(const Jacobian& jac_bare, const Jacobian& jac_dressed_0,
                                                       const Jacobian& jac_dressed_c) {
    std::vector<BasisState> templates;

    // Bare proton (no pion)
    templates.push_back({SpatialWavefunction(1), Channel::BARE_NUCLEON, NO_FLIP, 1.0, jac_bare, 0.0});

    // Proton + π⁰ (neutral pion)
    templates.push_back({SpatialWavefunction(-1), Channel::NUCLEON_PI0, NO_FLIP, 1.0, jac_dressed_0, 134.97});

    // Neutron + π⁺ (from p → n + π⁺)
    templates.push_back({SpatialWavefunction(-1), Channel::NUCLEON_PIC, NO_FLIP, std::sqrt(2.0), jac_dressed_c, 139.57});

    return templates;
}

// Create channel templates for neutron system
inline std::vector<BasisState> create_neutron_templates(const Jacobian& jac_bare, const Jacobian& jac_dressed_0,
                                                        const Jacobian& jac_dressed_c) {
    std::vector<BasisState> templates;

    // Bare neutron (no pion)
    templates.push_back({SpatialWavefunction(1), Channel::BARE_NUCLEON, NO_FLIP, 1.0, jac_bare, 0.0});

    // Neutron + π⁰ (neutral pion)
    templates.push_back({SpatialWavefunction(-1), Channel::NUCLEON_PI0, NO_FLIP, 1.0, jac_dressed_0, 134.97});

    // Proton + π⁻ (from n → p + π⁻)
    templates.push_back({SpatialWavefunction(-1), Channel::NUCLEON_PIC, NO_FLIP, -std::sqrt(2.0), jac_dressed_c, 139.57});

    return templates;
}
