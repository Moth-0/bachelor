#pragma once

#include <vector>
#include "matrix.h"
#include "particle.h" // Includes our Particle, Nucleon, Pion structs

namespace qm {

struct Jacobian {
    std::vector<Particle> particles;

    // Constructor taking a vector of our generalized Particle objects
    Jacobian(const std::vector<Particle>& p) : particles(p) {}

    size_t num_particles() const { return particles.size(); }
    
    // The dimension of the Jacobi space is N - 1
    size_t dim() const { return particles.empty() ? 0 : particles.size() - 1; }

    // Calculates the reduced mass for the i-th Jacobi coordinate
    // Formula: mu_i = (M_i * m_{i+1}) / (M_i + m_{i+1}) where M_i is the cumulative mass
    long double mu(size_t i) const {
        if (i >= dim()) return 0.0;
        
        long double M_i = 0;
        for (size_t j = 0; j <= i; ++j) {
            M_i += particles[j].mass;
        }
        return (particles[i + 1].mass * M_i) / (particles[i + 1].mass + M_i);
    }

    // Returns a column vector w_i (or c_i) to extract the i-th coordinate
    // e.g., for a 3-body system, c(0) = {1, 0}, c(1) = {0, 1}
    vector c(size_t i) const {
        vector v(dim());
        if (i < dim()) {
            v[i] = 1.0;
        }
        return v;
    }
};

} // namespace qm