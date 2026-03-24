#pragma once

#include <vector>
#include "matrix.h" // Your custom matrix library

namespace qm {

struct Jacobian {
    size_t N;         // Number of particles
    rvec masses;      // m_1, m_2, ..., m_N
    rmat J;           // The Jacobi Matrix (N x N)
    rmat U_trans;     // The Transpose of the Inverse (J^-1)^T for coordinate vectors
    size_t dim;       // N-1 dimension

    // Constructor:
    Jacobian(const rvec& m_list) {
        N = m_list.size();
        dim = N - 1;
        masses = m_list;
        
        J = rmat(N, N); 
        
        // Calculate cumulative masses M_n
        rvec M(N + 1); // Size N+1 so we can 1-index M mathematically
        M[0] = 0.0;
        for(size_t i = 0; i < N; ++i) {
            M[i+1] = M[i] + m_list[i];
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
};

} // namespace qm