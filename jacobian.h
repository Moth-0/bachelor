#pragma once

#include<cmath>
#include<vector>
#include<initializer_list>
#include"matrix.h"

namespace qm {
struct jacobian {
    std::vector<long double> masses; // Masses 

    jacobian(std::initializer_list<long double> m) : masses(m) {}

    size_t N() const {return masses.size();}
    size_t dim() const {return masses.size() - 1;}

    // Calculates the reduced mass for the i-th Jacobi coordinate
    long double mu(size_t i) const {
        // M_i is the sum of masses from 0 to i
        long double M_i = 0;
        for (size_t j = 0; j <= i; ++j) M_i += masses[j];
        
        // mu_i = (m_{i+1} * M_i) / (m_{i+1} + M_i)
        return (masses[i + 1] * M_i) / (masses[i + 1] + M_i);
    }

    // Returns a selection vector for coordinate i (size N-1)
    vector c(size_t i) const {
        vector v(dim());
        v[i] = 1.0;
        return v;
    }
};
}