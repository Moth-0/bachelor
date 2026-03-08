#pragma once
#include <cassert>
#include <vector>
#include "matrix.h"
#include "particle.h"

namespace qm {

// Jacobi coordinates for an N-body system.
// x_0 = r_2 - r_1
// x_i = r_{i+1} - CoM(r_1,...,r_i)
// mu_i = m_{i+1} * M_i / (m_{i+1} + M_i),  M_i = sum_{j<=i} m_j
struct jacobian {
    std::vector<Particle> particles;

    jacobian() = default;
    explicit jacobian(const std::vector<Particle>& p) : particles(p) { assert(p.size() >= 2); }

    // N-1 Jacobi coordinates for N particles
    size_t dim() const { return particles.empty() ? 0 : particles.size() - 1; }

    // Reduced mass for coordinate i (MeV/c^2)
    long double mu(size_t i) const {
        assert(i < dim());
        long double M = 0.0;
        for (size_t j = 0; j <= i; ++j) M += particles[j].mass;
        long double m = particles[i + 1].mass;
        return (m * M) / (m + M);
    }

    // Unit vector e_i in Jacobi-coordinate space
    vector c(size_t i) const {
        vector v(dim());
        if (i < dim()) v[i] = 1.0;
        return v;
    }

    // For a clothed {N, N, meson} sector the meson is always the last coordinate
    size_t meson_index() const { assert(dim() >= 1); return dim() - 1; }

    long double total_mass() const {
        long double M = 0.0;
        for (const auto& p : particles) M += p.mass;
        return M;
    }
};

} // namespace qm