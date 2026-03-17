#pragma once

#include "matrix.h"
#include "jacobi.h"
#include "gaussian.h"
#include "hamiltonian.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// solver.h  —  Generalised eigenvalue problem and SVM optimisation loop
// ─────────────────────────────────────────────────────────────────────────────

namespace qm {

// ─────────────────────────────────────────────────────────────────────────────
// jacobi_diag 
// ─────────────────────────────────────────────────────────────────────────────
inline cmat jacobi_diag(cmat& H, ld tol = ld{1e-12L}, int max_sweeps = 500)
{
    size_t n = H.size1();
    assert(n == H.size2());

    cmat V = eye<cld>(n); 

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        bool converged = true;

        for (size_t p = 0; p < n - 1; p++) {
            for (size_t q = p + 1; q < n; q++) {
                cld b = H(p, q);
                if (scalar_abs2(b) < tol * tol) continue;
                converged = false;

                ld a_pp = std::real(H(p, p));
                ld a_qq = std::real(H(q, q));
                ld r    = std::sqrt(scalar_abs2(b));
                ld phi  = std::arg(b);

                ld theta = ld{0.5L} * std::atan2(ld{2} * r, a_pp - a_qq);
                ld c     = std::cos(theta);
                ld s     = std::sin(theta);

                cld ephi_half     = std::exp(cld{ld{0}, phi / ld{2}});
                cld ephi_half_neg = std::conj(ephi_half);

                cld gpp =  ephi_half     * c;
                cld gpq = -ephi_half     * s;
                cld gqp =  ephi_half_neg * s;
                cld gqq =  ephi_half_neg * c;

                for (size_t i = 0; i < n; i++) {
                    cld hip = H(i, p);
                    cld hiq = H(i, q);
                    H(i, p) = gpp * hip + gqp * hiq;
                    H(i, q) = gpq * hip + gqq * hiq;
                }

                for (size_t j = 0; j < n; j++) {
                    cld hpj = H(p, j);
                    cld hqj = H(q, j);
                    H(p, j) = std::conj(gpp) * hpj + std::conj(gqp) * hqj;
                    H(q, j) = std::conj(gpq) * hpj + std::conj(gqq) * hqj;
                }

                for (size_t i = 0; i < n; i++) {
                    cld vip = V(i, p);
                    cld viq = V(i, q);
                    V(i, p) = gpp * vip + gqp * viq;
                    V(i, q) = gpq * vip + gqq * viq;
                }
            }
        }
        if (converged) break;
    }
    return V;
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_gevp 
// ─────────────────────────────────────────────────────────────────────────────
inline ld solve_gevp(const cmat& H, const cmat& N)
{
    assert(H.size1() == N.size1());
    size_t n = H.size1();

    cmat L = N.cholesky();
    if (L.size1() == 0) return std::numeric_limits<ld>::quiet_NaN();

    cmat Linv = L.inverse_lower();
    if (Linv.size1() == 0) return std::numeric_limits<ld>::quiet_NaN();

    cmat Linv_dag = Linv.adjoint();
    cmat Hp = Linv * H * Linv_dag;

    jacobi_diag(Hp); 

    ld E0 = std::numeric_limits<ld>::max();
    for (size_t i = 0; i < n; i++) {
        ld e = std::real(Hp(i, i));
        if (e < E0) E0 = e;
    }
    return E0;
}

inline std::vector<ld> solve_gevp_all(const cmat& H, const cmat& N)
{
    size_t n = H.size1();
    cmat L    = N.cholesky();
    if (L.size1() == 0) return {};
    cmat Linv = L.inverse_lower();
    if (Linv.size1() == 0) return {};
    cmat Hp   = Linv * H * Linv.adjoint();
    jacobi_diag(Hp);

    std::vector<ld> evals(n);
    for (size_t i = 0; i < n; i++) evals[i] = std::real(Hp(i, i));
    std::sort(evals.begin(), evals.end());
    return evals;
}

// ─────────────────────────────────────────────────────────────────────────────
// SvmParams & SvmState
// ─────────────────────────────────────────────────────────────────────────────
struct SvmParams {
    int  K_max;
    int  N_trial;
    int  refine_every;
    int  N_refine_trial;
    ld   b0;
    ld   s_max;
    ld   b_ff;
    ld   S_coupling;
    bool relativistic;
    bool verbose;

    SvmParams()
        : K_max(40), N_trial(50), refine_every(10), N_refine_trial(20),
          b0(ld{1.4L}), s_max(ld{0.12L}), b_ff(ld{1.4L}), S_coupling(ld{15}),
          relativistic(false), verbose(true)
    {}
};

struct SvmState {
    std::vector<Gaussian> basis_bare;
    std::vector<Gaussian> basis_dressed;

    cmat H;
    cmat N;
    ld   E0;

    std::vector<ld> energy_history;

    size_t K() const { return basis_bare.size(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// incremental_update  —  Generalized O(K) Magic
//
// Computes ONLY the matrix elements that involve the new/replaced Gaussian.
// Automatically scales to whatever number of channels are provided.
// ─────────────────────────────────────────────────────────────────────────────
inline void incremental_update(const SvmState& state,
                               const Gaussian& g_bare_new,
                               const Gaussian& g_dress_new,
                               int target_idx,
                               const JacobiSystem& sys,
                               const std::vector<Channel>& channels,
                               const SvmParams& params,
                               cmat& H_out, cmat& N_out)
{
    size_t K = state.K();
    bool is_append = (target_idx < 0);
    size_t K_new = is_append ? K + 1 : K;
    
    // Generalized Dimension Calculation
    size_t n_channels = channels.size();
    size_t dim_new = n_channels * K_new;

    H_out = zeros<cld>(dim_new, dim_new);
    N_out = zeros<cld>(dim_new, dim_new);

    // 1. Memory Layout Preparation
    if (is_append) {
        if (K > 0) {
            // Shift old K blocks into K+1 dimension spaces
            for(size_t a = 0; a < n_channels; ++a) {
                size_t oa = a * K;
                size_t na = a * K_new;
                for(size_t b = 0; b < n_channels; ++b) {
                    size_t ob = b * K;
                    size_t nb = b * K_new;
                    for(size_t i = 0; i < K; ++i) {
                        for(size_t j = 0; j < K; ++j) {
                            H_out(na + i, nb + j) = state.H(oa + i, ob + j);
                            N_out(na + i, nb + j) = state.N(oa + i, ob + j);
                        }
                    }
                }
            }
        }
    } else {
        // Replace Mode: Copy flatly, then strictly zero out the discarded rows/cols
        H_out = state.H;
        N_out = state.N;
        for (size_t a = 0; a < n_channels; ++a) {
            // Find the global index of the basis function being replaced
            size_t global_idx = a * K_new + target_idx;
            
            // Wipe the entire global row and global column for that index
            for (size_t k = 0; k < dim_new; ++k) {
                H_out(global_idx, k) = cld{0, 0};
                H_out(k, global_idx) = cld{0, 0};
                N_out(global_idx, k) = cld{0, 0};
                N_out(k, global_idx) = cld{0, 0};
            }
        }
    }

    // 2. Prepare Basis
    std::vector<Gaussian> b_bare = state.basis_bare;
    std::vector<Gaussian> b_dress = state.basis_dressed;
    size_t active_idx;
    
    if (is_append) {
        b_bare.push_back(g_bare_new);
        b_dress.push_back(g_dress_new);
        active_idx = K;
    } else {
        b_bare[target_idx] = g_bare_new;
        b_dress[target_idx] = g_dress_new;
        active_idx = target_idx;
    }

    // 3. Compute ONLY the New Diagonal Edges (Overlap & Kinetic Energy)
    for (size_t a = 0; a < n_channels; a++) {
        size_t na = a * K_new;
        
        // Dynamically use the correct basis type
        const auto& bas = channels[a].is_bare ? b_bare : b_dress;

        for (size_t i = 0; i < K_new; i++) {
            for (size_t j = 0; j < K_new; j++) {
                // Skip calculating elements that haven't changed
                if (i != active_idx && j != active_idx) continue;

                GaussianPair gp(bas[i], bas[j]);
                N_out(na + i, na + j) = cld{gp.M, ld{0}};

                ld ke;
                if (channels[a].is_bare) {
                    rvec c0_bare(1); c0_bare[0] = ld{1};
                    KineticParams kp(gp, c0_bare);
                    if (params.relativistic) {
                        ke = ke_relativistic(gp, kp, sys.mu[0]);
                    } else {
                        ke = (phys::hbar_c2 / (ld{2} * sys.mu[0])) * ke_classical(gp, kp);
                    }
                } else {
                    ke = kinetic_energy(gp, sys, params.relativistic);
                    ke += channels[a].pion_mass * gp.M;
                }
                H_out(na + i, na + j) += cld{ke, ld{0}};
            }
        }
    }

    // 4. Compute ONLY the New Off-Diagonal Edges (Pion/Meson Coupling W)
    size_t n0 = 0; // Assuming block 0 is always the base coupling channel
    for (size_t a = 1; a < n_channels; a++) {
        size_t na = a * K_new;
        const Channel& ch = channels[a];

        for (size_t i = 0; i < K_new; i++) {       // dressed index
            for (size_t j = 0; j < K_new; j++) {   // bare index
                if (i != active_idx && j != active_idx) continue;

                cld w = w_matrix_element(b_dress[i], b_bare[j], sys, ch.w_piN, 
                                         params.b_ff, params.S_coupling, 
                                         ch.iso_coeff, ch.spin_type);
                H_out(na + i, n0 + j) += w;
                H_out(n0 + j, na + i) += std::conj(w);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// grow_matrices
// ─────────────────────────────────────────────────────────────────────────────
inline void grow_matrices(SvmState& state,
                           const Gaussian& g_bare_new,
                           const Gaussian& g_dress_new,
                           const JacobiSystem& sys,
                           const std::vector<Channel>& channels,
                           const SvmParams& params)
{
    cmat H_new, N_new;
    incremental_update(state, g_bare_new, g_dress_new, -1, sys, channels, params, H_new, N_new);
    state.H = H_new;
    state.N = N_new;
}

// ─────────────────────────────────────────────────────────────────────────────
// eval_candidate
// ─────────────────────────────────────────────────────────────────────────────
inline ld eval_candidate(const SvmState& state,
                          const Gaussian& g_bare_cand,
                          const Gaussian& g_dress_cand,
                          const JacobiSystem& sys,
                          const std::vector<Channel>& channels,
                          const SvmParams& params)
{
    cmat H, N;
    incremental_update(state, g_bare_cand, g_dress_cand, -1, sys, channels, params, H, N);
    return solve_gevp(H, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// run_svm
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
SvmState run_svm(const JacobiSystem& sys,
                  const std::vector<Channel>& channels,
                  const SvmParams& params,
                  RNG& rng)
{
    SvmState state;
    state.E0 = std::numeric_limits<ld>::max();

    if (params.verbose) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "SVM: K_max=" << params.K_max
                  << "  N_trial=" << params.N_trial
                  << "  b0=" << params.b0 << " fm"
                  << "  s_max=" << params.s_max << " fm^-1"
                  << "  " << (params.relativistic ? "relativistic" : "classical")
                  << " KE";
#ifdef _OPENMP
        std::cout << "  threads=" << omp_get_max_threads();
#endif
        std::cout << "\n" << std::string(60, '-') << "\n";
    }

    // ── Phase 1: Build ────────────────────────────────────────────────────────
    for (int k = 0; k < params.K_max; k++) {

        std::vector<Gaussian> cand_bare(params.N_trial);
        std::vector<Gaussian> cand_dress(params.N_trial);
        for (int t = 0; t < params.N_trial; t++) {
            cand_bare[t]  = random_gaussian_bare(sys, params.b0, params.s_max, rng);
            cand_dress[t] = random_gaussian_dressed(sys, params.b0, params.s_max, rng);
        }

        int    best_idx = -1;
        ld     best_E   = std::numeric_limits<ld>::max();

        #pragma omp parallel
        {
            int local_idx = -1;
            ld  local_E   = std::numeric_limits<ld>::max();

            #pragma omp for schedule(dynamic) nowait
            for (int t = 0; t < params.N_trial; t++) {
                ld E_trial = eval_candidate(state, cand_bare[t], cand_dress[t], sys, channels, params);
                if (std::isfinite(E_trial) && E_trial < local_E) {
                    local_E   = E_trial;
                    local_idx = t;
                }
            }

            #pragma omp critical
            {
                if (local_idx >= 0 && local_E < best_E) {
                    best_E   = local_E;
                    best_idx = local_idx;
                }
            }
        }

        if (best_idx < 0) {
            if (params.verbose)
                std::cout << "  [step " << k << "] all trials failed Cholesky — stopping.\n";
            break;
        }

        const Gaussian& best_bare  = cand_bare [best_idx];
        const Gaussian& best_dress = cand_dress[best_idx];

        ld delta_E = best_E - state.E0;
        grow_matrices(state, best_bare, best_dress, sys, channels, params);
        state.basis_bare.push_back(best_bare);
        state.basis_dressed.push_back(best_dress);
        state.E0 = best_E;
        state.energy_history.push_back(best_E);

        if (params.verbose) {
            std::cout << "\r"
                      << std::setw(5)  << "K = " << (k + 1)
                      << std::setw(10) << "E0 = " << best_E << " MeV"
                      << std::setw(15) << "delta E = " << (k > 0 ? delta_E : ld{0}) << " MeV"
                      << std::flush;
        }

        // ── Phase 2: Refine ───────────────────────────────────────────────────
        if (params.refine_every > 0 && (k + 1) % params.refine_every == 0) {
            if (params.verbose) std::cout << "\n";

            const ld E_floor = ld{-1000}; 

            for (size_t j = 0; j < state.K(); j++) {
                std::vector<Gaussian> rc_bare (params.N_refine_trial);
                std::vector<Gaussian> rc_dress(params.N_refine_trial);
                for (int t = 0; t < params.N_refine_trial; t++) {
                    rc_bare [t] = random_gaussian_bare(sys, params.b0, params.s_max, rng);
                    rc_dress[t] = random_gaussian_dressed(sys, params.b0, params.s_max, rng);
                }

                int  best_rt  = -1;
                ld   best_rE  = state.E0; 
                cmat best_H, best_N;

                #pragma omp parallel
                {
                    int  local_rt = -1;
                    ld   local_rE = state.E0;
                    cmat local_H, local_N;

                    #pragma omp for schedule(dynamic) nowait
                    for (int t = 0; t < params.N_refine_trial; t++) {
                        cmat H_t, N_t;
                        
                        // Use the incremental update in REPLACE mode (target_idx = j)
                        incremental_update(state, rc_bare[t], rc_dress[t], j, sys, channels, params, H_t, N_t);
                        
                        ld E_t = solve_gevp(H_t, N_t);

                        if (std::isfinite(E_t) && E_t > E_floor && E_t < local_rE) {
                            local_rE = E_t;
                            local_rt = t;
                            local_H  = H_t;
                            local_N  = N_t;
                        }
                    }

                    #pragma omp critical
                    {
                        if (local_rt >= 0 && local_rE < best_rE) {
                            best_rE = local_rE;
                            best_rt = local_rt;
                            best_H  = local_H;
                            best_N  = local_N;
                        }
                    }
                }

                if (best_rt >= 0) {
                    state.basis_bare[j]    = rc_bare [best_rt];
                    state.basis_dressed[j] = rc_dress[best_rt];
                    state.H  = best_H;
                    state.N  = best_N;
                    state.E0 = best_rE;
                    if (params.verbose)
                        std::cout << "\r  [Refining]  replaced basis[" << j
                                  << "] -> E0 = " << best_rE << " MeV" << std::flush;
                }
            }

            if (params.verbose) std::cout << "\n";
        }
    }

    if (params.verbose) {
        std::cout << "\n\n" << std::string(60, '-') << "\n";
        std::cout << "Final E0 = " << state.E0 << " MeV"
                  << "  (K=" << state.K() << ")\n";
    }

    return state;
}

} // namespace qm