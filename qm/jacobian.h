#pragma once
//#include <vector>
#include <cassert>
#include "matrix.h"
#include "particle.h"

namespace qm {

// ============================================================
//  jacobian.h
//
//  Computes Jacobi coordinates and reduced masses for an
//  N-body system.
//
//  Convention: particles are ordered [1, 2, ..., N].
//  Jacobi coordinate x_i connects particle (i+1) to the
//  centre-of-mass of particles [1..i]:
//
//    x_0 = r_2 - r_1
//    x_1 = r_3 - CoM(r_1, r_2)
//    ...
//
//  The corresponding reduced masses are:
//    mu_i = m_{i+1} * M_i / (m_{i+1} + M_i)
//  where M_i = sum_{j=0}^{i} m_j
//
//  For a clothed 3-body sector {p1, p2, meson}, the meson
//  is always the LAST particle, so its Jacobi coordinate
//  index is dim()-1 = N-2.
// ============================================================

struct jacobian {
    std::vector<Particle> particles;

    jacobian() = default;

    explicit jacobian(const std::vector<Particle>& p) : particles(p) {
        assert(p.size() >= 2);
    }

    // Number of Jacobi coordinates = N - 1
    size_t dim() const {
        return particles.empty() ? 0 : particles.size() - 1;
    }

    // Reduced mass for the i-th Jacobi coordinate (in MeV/c^2)
    long double mu(size_t i) const {
        assert(i < dim());
        long double M_i = 0.0;
        for (size_t j = 0; j <= i; ++j)
            M_i += particles[j].mass;
        long double m_next = particles[i + 1].mass;
        return (m_next * M_i) / (m_next + M_i);
    }

    // Unit vector that selects Jacobi coordinate i.
    // c(i) is a dim()-length vector with a 1 at position i, 0 elsewhere.
    // Used to project the kinetic-energy operator onto coordinate i.
    vector c(size_t i) const {
        vector v(dim());
        if (i < dim()) v[i] = 1.0;
        return v;
    }

    // For a clothed sector {nucleon, nucleon, meson}, the meson
    // Jacobi coordinate is always the last one.
    size_t meson_index() const {
        assert(dim() >= 1);
        return dim() - 1;
    }

    // Convenience: total mass of the system
    long double total_mass() const {
        long double M = 0.0;
        for (const auto& p : particles) M += p.mass;
        return M;
    }
};

} // namespace qm