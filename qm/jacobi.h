#pragma once

#include <vector>
#include "matrix.h" // Your custom matrix library

namespace qm {

struct Jacobian {
    size_t N;            // Number of particles
    rvec masses;         // m_1, m_2, ..., m_N
    rvec reduced_masses; // μ_1, ..., μ_N-1
    rmat J;              // The Jacobi Matrix (N x N)
    rmat U_trans;        // The Transpose of the Inverse (J^-1)^T for coordinate vectors
    size_t dim;          // N-1 dimension

    // Constructor:
    Jacobian(const rvec& m_list) {
        N = m_list.size();
        dim = N - 1;
        masses = m_list;
        
        J = rmat(N, N); 
        
        // Calculate cumulative masses M and reduced masses mu 
        rvec M(N + 1); 
        M[0] = 0.0;
        reduced_masses.resize(dim); 

        for(size_t i = 0; i < N; ++i) {
            // 1. Accumulate the mass for the current particle
            M[i+1] = M[i] + m_list[i];
            
            // 2. Calculate the reduced mass for the Jacobi coordinate
            if (i < dim) {
                // M[i+1] is the cumulative mass so far (M_current)
                // m_list[i+1] is the mass of the next particle (m_next)
                reduced_masses[i] = (M[i+1] * m_list[i+1]) / (M[i+1] + m_list[i+1]);
            }
        }

        // Fill Jacobi Matrix J
        for (size_t i = 0; i < N; ++i) {
            if (i < N - 1) {
                // Relative coordinate rows
                for (size_t j = 0; j <= i; ++j) {
                    J(i, j) = m_list[j] / M[i+1];
                }
                J(i, i+1) = -1.0;
            } else {
                // Last row: Center of Mass coordinates
                for (size_t j = 0; j < N; ++j) {
                    J(i, j) = m_list[j] / M[N];
                }
            }
        }
        
        // Precompute U^T for physical space transformations: w_i -> U^T * w_i
        U_trans = J.inverse().transpose();

        
    }

    ~Jacobian() = default;

    rvec transform_w(size_t particle_index) const {
        rvec w(N); // Initializes with zeros
        w[particle_index] = 1.0;
        return U_trans * w; 
    }


    rvec transform_k(size_t particle_index) const {
        rvec k(N); // Initializes with zeros
        k[particle_index] = 1.0;
        return J * k;
    }

    // Gets the c-vector for an internal Jacobi coordinate.
    // In the decoupled N-1 space, this is just a standard basis vector!
    rvec get_c_internal(size_t jacobi_idx) const {
        rvec c(dim); // Initializes with zeros
        c[jacobi_idx] = 1.0;
        return c;
    }
};

} // namespace qm