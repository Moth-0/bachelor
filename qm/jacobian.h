#pragma once
#include <vector>
#include "matrix.h"

namespace qm {
struct jacobian {
    std::vector<long double> masses;
    std::vector<int> charges; // e.g., +1 for proton, 0 for neutron, -1 for pion

    // Constructor now takes masses and charges
    jacobian(std::vector<long double> m, std::vector<int> q) : masses(m), charges(q) {}

    size_t num_particles() const { return masses.size(); }
    size_t dim() const { return masses.size() - 1; }

    long double mu(size_t i) const {
        long double M_i = 0;
        for (size_t j = 0; j <= i; ++j) M_i += masses[j];
        return (masses[i + 1] * M_i) / (masses[i + 1] + M_i);
    }

    vector c(size_t i) const {
        vector v(dim());
        v[i] = 1.0;
        return v;
    }

    // Calculates the 'c' vector for ANY pair of particles (p1, p2)
    vector w(size_t p1, size_t p2) const {
        vector c(dim());
        for (size_t k = 0; k < dim(); ++k) {
            long double M_k = 0; 
            for(size_t j = 0; j <= k; ++j) M_k += masses[j];
            long double M_k1 = M_k + masses[k+1];
            
            long double w_p1 = 0, w_p2 = 0;
            
            // Weight for particle 1
            if (p1 < k + 1) w_p1 = -masses[k+1] / M_k1;
            else if (p1 == k + 1) w_p1 = M_k / M_k1;
            
            // Weight for particle 2
            if (p2 < k + 1) w_p2 = -masses[k+1] / M_k1;
            else if (p2 == k + 1) w_p2 = M_k / M_k1;
            
            c[k] = w_p2 - w_p1;
        }
        return c;
    }
};
}