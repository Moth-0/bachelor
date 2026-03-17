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

// ─────────────────────────────────────────────────────────────────────────────
// Linear Dependence Check
// ─────────────────────────────────────────────────────────────────────────────
inline bool is_linearly_dependent(const Gaussian& trial, const std::vector<Gaussian>& basis, int ignore_idx = -1, ld threshold = 0.999L) {
    GaussianPair gp_tt(trial, trial);
    ld n_tt = gp_tt.M;
    for (int i = 0; i < (int)basis.size(); ++i) {
        if (i == ignore_idx) continue;
        GaussianPair gp_ii(basis[i], basis[i]);
        GaussianPair gp_ti(trial, basis[i]);
        if (std::abs(gp_ti.M) / std::sqrt(n_tt * gp_ii.M) > threshold) return true;
    }
    return false;
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

    // --- Master Annealing Workflow Parameters ---
    bool do_annealing;
    ld   S_build;
    ld   E_target;
    int  relax_sweeps;

    SvmParams()
        : K_max(40), N_trial(50), refine_every(10), N_refine_trial(50),
          b0(ld{1.4L}), s_max(ld{0.12L}), b_ff(ld{1.4L}), S_coupling(ld{15}),
          relativistic(false), verbose(true),
          do_annealing(false), S_build(18.0), E_target(-2.224), relax_sweeps(3)
    {}
};

struct SvmState {
    std::vector<Gaussian> basis_bare;
    std::vector<Gaussian> basis_dressed;

    cmat H;
    cmat N;
    ld   E0;
    ld   S_final; // Tracks the physically tuned coupling 

    std::vector<ld> energy_history;

    size_t K() const { return basis_bare.size(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// incremental_update  —  Generalized O(K) Magic
// ─────────────────────────────────────────────────────────────────────────────
inline void incremental_update(const SvmState& state, const Gaussian& g_bare_new, const Gaussian& g_dress_new,
                               int target_idx, const JacobiSystem& sys, const std::vector<Channel>& channels,
                               const SvmParams& params, cmat& H_out, cmat& N_out)
{
    size_t K = state.K();
    bool is_append = (target_idx < 0);
    size_t K_new = is_append ? K + 1 : K;
    size_t n_channels = channels.size();
    size_t dim_new = n_channels * K_new;

    H_out = zeros<cld>(dim_new, dim_new);
    N_out = zeros<cld>(dim_new, dim_new);

    if (is_append) {
        if (K > 0) {
            for(size_t a = 0; a < n_channels; ++a) {
                size_t oa = a * K; size_t na = a * K_new;
                for(size_t b = 0; b < n_channels; ++b) {
                    size_t ob = b * K; size_t nb = b * K_new;
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
        H_out = state.H; N_out = state.N;
        for (size_t a = 0; a < n_channels; ++a) {
            size_t global_idx = a * K_new + target_idx;
            for (size_t k = 0; k < dim_new; ++k) {
                H_out(global_idx, k) = cld{0, 0}; H_out(k, global_idx) = cld{0, 0};
                N_out(global_idx, k) = cld{0, 0}; N_out(k, global_idx) = cld{0, 0};
            }
        }
    }

    std::vector<Gaussian> b_bare = state.basis_bare;
    std::vector<Gaussian> b_dress = state.basis_dressed;
    size_t active_idx;
    
    if (is_append) {
        b_bare.push_back(g_bare_new); b_dress.push_back(g_dress_new);
        active_idx = K;
    } else {
        b_bare[target_idx] = g_bare_new; b_dress[target_idx] = g_dress_new;
        active_idx = target_idx;
    }

    for (size_t a = 0; a < n_channels; a++) {
        size_t na = a * K_new;
        const auto& bas = channels[a].is_bare ? b_bare : b_dress;
        for (size_t i = 0; i < K_new; i++) {
            for (size_t j = 0; j < K_new; j++) {
                if (i != active_idx && j != active_idx) continue;
                GaussianPair gp(bas[i], bas[j]);
                N_out(na + i, na + j) = cld{gp.M, ld{0}};

                ld ke;
                if (channels[a].is_bare) {
                    rvec c0_bare(1); c0_bare[0] = ld{1};
                    KineticParams kp(gp, c0_bare);
                    ke = params.relativistic ? ke_relativistic(gp, kp, sys.mu[0]) 
                                             : (phys::hbar_c2 / (ld{2} * sys.mu[0])) * ke_classical(gp, kp);
                } else {
                    ke = kinetic_energy(gp, sys, params.relativistic);
                    ke += channels[a].pion_mass * gp.M;
                }
                H_out(na + i, na + j) += cld{ke, ld{0}};
            }
        }
    }

    size_t n0 = 0; 
    for (size_t a = 1; a < n_channels; a++) {
        size_t na = a * K_new;
        const Channel& ch = channels[a];
        for (size_t i = 0; i < K_new; i++) {       
            for (size_t j = 0; j < K_new; j++) {   
                if (i != active_idx && j != active_idx) continue;
                cld w = w_matrix_element(b_dress[i], b_bare[j], sys, ch.w_piN, params.b_ff, params.S_coupling, ch.iso_coeff, ch.spin_type);
                H_out(na + i, n0 + j) += w; H_out(n0 + j, na + i) += std::conj(w);
            }
        }
    }
}

inline void grow_matrices(SvmState& state, const Gaussian& g_bare_new, const Gaussian& g_dress_new,
                           const JacobiSystem& sys, const std::vector<Channel>& channels, const SvmParams& params) {
    cmat H_new, N_new;
    incremental_update(state, g_bare_new, g_dress_new, -1, sys, channels, params, H_new, N_new);
    state.H = H_new; state.N = N_new;
}

inline ld eval_candidate(const SvmState& state, const Gaussian& g_bare_cand, const Gaussian& g_dress_cand,
                          const JacobiSystem& sys, const std::vector<Channel>& channels, const SvmParams& params) {
    cmat H, N;
    incremental_update(state, g_bare_cand, g_dress_cand, -1, sys, channels, params, H, N);
    return solve_gevp(H, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// relax_wavefunction — Reusable logic for Periodic Refinement AND Final Sweeps
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
inline void relax_wavefunction(SvmState& state, const JacobiSystem& sys, const std::vector<Channel>& channels, 
                               const SvmParams& params, RNG& rng, int num_sweeps) 
{
    const ld E_floor = ld{-1000}; 

    for (int sweep = 1; sweep <= num_sweeps; ++sweep) {
        for (size_t j = 0; j < state.K(); j++) {
            
            // 1. Refine BARE Gaussian
            {
                std::vector<Gaussian> rc_bare;
                rc_bare.reserve(params.N_refine_trial);
                int attempts = 0, max_attempts = params.N_refine_trial * 50;

                while (rc_bare.size() < (size_t)params.N_refine_trial && attempts < max_attempts) {
                    Gaussian gb = random_gaussian_bare(sys, params.b0, params.s_max, rng);
                    if (!is_linearly_dependent(gb, state.basis_bare, j, 0.999L)) rc_bare.push_back(gb);
                    attempts++;
                }

                int  best_rt = -1; ld best_rE = state.E0; cmat best_H, best_N;

                #pragma omp parallel
                {
                    int  local_rt = -1; ld local_rE = state.E0; cmat local_H, local_N;
                    #pragma omp for schedule(dynamic) nowait
                    for (int t = 0; t < (int)rc_bare.size(); t++) {
                        cmat H_t, N_t;
                        incremental_update(state, rc_bare[t], state.basis_dressed[j], j, sys, channels, params, H_t, N_t);
                        ld E_t = solve_gevp(H_t, N_t);
                        if (std::isfinite(E_t) && E_t > E_floor && E_t < local_rE) {
                            local_rE = E_t; local_rt = t; local_H = H_t; local_N = N_t;
                        }
                    }
                    #pragma omp critical
                    {
                        if (local_rt >= 0 && local_rE < best_rE) {
                            best_rE = local_rE; best_rt = local_rt; best_H = local_H; best_N = local_N;
                        }
                    }
                }
                if (best_rt >= 0) {
                    state.basis_bare[j] = rc_bare[best_rt]; state.H = best_H; state.N = best_N; state.E0 = best_rE;
                    if (params.verbose) std::cout << "\r  [Refining Bare " << j << "] -> E0 = " << best_rE << " MeV      " << std::flush;
                }
            }

            // 2. Refine DRESSED Gaussian
            {
                std::vector<Gaussian> rc_dress;
                rc_dress.reserve(params.N_refine_trial);
                int attempts = 0, max_attempts = params.N_refine_trial * 50;

                while (rc_dress.size() < (size_t)params.N_refine_trial && attempts < max_attempts) {
                    Gaussian gd = random_gaussian_dressed(sys, params.b0, params.s_max, rng);
                    if (!is_linearly_dependent(gd, state.basis_dressed, j, 0.999L)) rc_dress.push_back(gd);
                    attempts++;
                }

                int  best_rt = -1; ld best_rE = state.E0; cmat best_H, best_N;

                #pragma omp parallel
                {
                    int  local_rt = -1; ld local_rE = state.E0; cmat local_H, local_N;
                    #pragma omp for schedule(dynamic) nowait
                    for (int t = 0; t < (int)rc_dress.size(); t++) {
                        cmat H_t, N_t;
                        incremental_update(state, state.basis_bare[j], rc_dress[t], j, sys, channels, params, H_t, N_t);
                        ld E_t = solve_gevp(H_t, N_t);
                        if (std::isfinite(E_t) && E_t > E_floor && E_t < local_rE) {
                            local_rE = E_t; local_rt = t; local_H = H_t; local_N = N_t;
                        }
                    }
                    #pragma omp critical
                    {
                        if (local_rt >= 0 && local_rE < best_rE) {
                            best_rE = local_rE; best_rt = local_rt; best_H = local_H; best_N = local_N;
                        }
                    }
                }
                if (best_rt >= 0) {
                    state.basis_dressed[j] = rc_dress[best_rt]; state.H = best_H; state.N = best_N; state.E0 = best_rE;
                    if (params.verbose) std::cout << "\r  [Refining Dressed " << j << "] -> E0 = " << best_rE << " MeV   " << std::flush;
                }
            }
        }
        if (params.verbose) std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tune_coupling — Bisection Math bypass to lock S perfectly to E0
// ─────────────────────────────────────────────────────────────────────────────
inline void tune_coupling(SvmState& state, const JacobiSystem& sys, const std::vector<Channel>& channels, SvmParams& active_params) {
    if (active_params.verbose) std::cout << "\n=== Tuning S to Target E0 = " << active_params.E_target << " MeV ===\n";
    
    HamiltonianBuilder hb_base(sys, channels, state.basis_bare, state.basis_dressed, active_params.b_ff, 0.0, active_params.relativistic);
    cmat N = hb_base.build_N();
    cmat H_diag = hb_base.build_H();
    size_t dim = H_diag.size1();

    cmat L = N.cholesky();
    cmat Linv = L.inverse_lower();
    cmat Linv_dag = Linv.adjoint();

    HamiltonianBuilder hb_W(sys, channels, state.basis_bare, state.basis_dressed, active_params.b_ff, 1.0, active_params.relativistic);
    cmat H_S1 = hb_W.build_H();
    cmat W_raw = zeros<cld>(dim, dim);
    for(size_t r = 0; r < dim; ++r) {
        for(size_t c = 0; c < dim; ++c) {
            W_raw(r, c) = H_S1(r, c) - H_diag(r, c);
        }
    }

    // --- OPTIMIZATION 1: Pre-transform matrices to the orthonormal basis ---
    // This saves TWO massive O(N^3) matrix multiplications inside the loop
    cmat Hp_diag = Linv * H_diag * Linv_dag;
    cmat Wp_raw  = Linv * W_raw * Linv_dag;

    // --- OPTIMIZATION 2: Bisection Search ---
    ld S_low = 5.0;
    ld S_high = active_params.S_build + 5.0; 
    ld best_S = active_params.S_build;
    ld tol = 1e-4; // Energy tolerance (0.0001 MeV)
    
    for (int iter = 1; iter <= 50; ++iter) {
        ld S_mid = (S_low + S_high) / 2.0;

        cmat Hp = zeros<cld>(dim, dim);
        for(size_t r = 0; r < dim; ++r) {
            for(size_t c = 0; c < dim; ++c) {
                // Now we just do simple addition
                Hp(r, c) = Hp_diag(r, c) + cld{S_mid, 0.0} * Wp_raw(r, c);
            }
        }

        jacobi_diag(Hp);

        ld E0 = std::numeric_limits<ld>::max();
        for (size_t d = 0; d < dim; d++) {
            ld e = std::real(Hp(d, d));
            if (e < E0) E0 = e;
        }

        if (active_params.verbose) {
            std::cout << "\r  [Tuning] Bisection Step " << iter 
                      << ": S = " << std::fixed << std::setprecision(4) << S_mid 
                      << " MeV  |  E0 = " << std::setprecision(5) << E0 << " MeV    " << std::flush;
        }

        if (std::abs(E0 - active_params.E_target) < tol) {
            best_S = S_mid;
            break;
        }

        // If energy is too high (e.g., -1.0), we need more attraction (higher S)
        if (E0 > active_params.E_target) {
            S_low = S_mid;
        } else {
            S_high = S_mid;
        }
        best_S = S_mid;
    }

    if (active_params.verbose) std::cout << "\n";

    active_params.S_coupling = best_S;
    state.S_final = best_S;

    // Force the un-transformed matrices back to the target physical values
    for(size_t r = 0; r < dim; ++r) {
        for(size_t c = 0; c < dim; ++c) {
            state.H(r, c) = H_diag(r, c) + cld{best_S, 0.0} * W_raw(r, c);
        }
    }
    state.E0 = solve_gevp(state.H, state.N);
    if (active_params.verbose) std::cout << "  -> Target Locked! S = " << best_S << " MeV\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// run_svm  —  The Master Orchestrator (Builds, Tunes, and Relaxes automatically)
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
SvmState run_svm(const JacobiSystem& sys, const std::vector<Channel>& channels, const SvmParams& params, RNG& rng)
{
    SvmParams active_params = params;
    SvmState state;
    state.E0 = std::numeric_limits<ld>::max();

    if (active_params.do_annealing) {
        active_params.S_coupling = active_params.S_build;
        active_params.refine_every = 0; // Disable periodic refinement for speed during build
    }
    state.S_final = active_params.S_coupling;

    if (active_params.verbose) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "SVM: K_max=" << active_params.K_max
                  << "  N_trial=" << active_params.N_trial
                  << "  b0=" << active_params.b0 << " fm"
                  << "  s_max=" << active_params.s_max << " fm^-1\n"
                  << "  Physics: " << (active_params.relativistic ? "relativistic" : "classical") << " KE"
                  << "  |  Annealing: " << (params.do_annealing ? "Enabled" : "Disabled");
#ifdef _OPENMP
        std::cout << "  |  threads=" << omp_get_max_threads();
#endif
        std::cout << "\n" << std::string(60, '-') << "\n";
    }

    // ── Phase 1: Build ────────────────────────────────────────────────────────
    for (int k = 0; k < active_params.K_max; k++) {
        std::vector<Gaussian> cand_bare(active_params.N_trial);
        std::vector<Gaussian> cand_dress(active_params.N_trial);
        for (int t = 0; t < active_params.N_trial; t++) {
            cand_bare[t]  = random_gaussian_bare(sys, active_params.b0, active_params.s_max, rng);
            cand_dress[t] = random_gaussian_dressed(sys, active_params.b0, active_params.s_max, rng);
        }

        int  best_idx = -1; ld best_E = std::numeric_limits<ld>::max();

        #pragma omp parallel
        {
            int local_idx = -1; ld local_E = std::numeric_limits<ld>::max();

            #pragma omp for schedule(dynamic) nowait
            for (int t = 0; t < active_params.N_trial; t++) {
                if (is_linearly_dependent(cand_bare[t], state.basis_bare, -1, 0.9999L) || 
                    is_linearly_dependent(cand_dress[t], state.basis_dressed, -1, 0.9999L)) {
                    continue; 
                }
                ld E_trial = eval_candidate(state, cand_bare[t], cand_dress[t], sys, channels, active_params);
                if (std::isfinite(E_trial) && E_trial < local_E) {
                    local_E = E_trial; local_idx = t;
                }
            }

            #pragma omp critical
            {
                if (local_idx >= 0 && local_E < best_E) {
                    best_E = local_E; best_idx = local_idx;
                }
            }
        }

        if (best_idx < 0) {
            if (active_params.verbose) std::cout << "  [step " << k << "] all trials legitimately failed Cholesky — stopping.\n";
            break;
        }

        const Gaussian& best_bare  = cand_bare [best_idx];
        const Gaussian& best_dress = cand_dress[best_idx];

        ld delta_E = best_E - state.E0;
        grow_matrices(state, best_bare, best_dress, sys, channels, active_params);
        state.basis_bare.push_back(best_bare);
        state.basis_dressed.push_back(best_dress);
        state.E0 = best_E;
        state.energy_history.push_back(best_E);

        if (active_params.verbose) {
            std::cout << "\r" << std::setw(5)  << "K = " << (k + 1) << std::setw(10) << "E0 = " << best_E << " MeV"
                      << std::setw(15) << "delta E = " << (k > 0 ? delta_E : ld{0}) << " MeV" << std::flush;
        }

        // Periodic Refinement (If active during standard runs)
        if (active_params.refine_every > 0 && (k + 1) % active_params.refine_every == 0) {
            if (active_params.verbose) std::cout << "\n";
            relax_wavefunction(state, sys, channels, active_params, rng, 1);
        }
    }
    if (active_params.verbose) std::cout << "\n";

    // ── Phase 2 & 3: Tune and Relax (Annealing Workflow) ──────────────────────
    if (params.do_annealing) {
        tune_coupling(state, sys, channels, active_params);
        if (active_params.verbose) std::cout << "=== Relaxing into True Ground State (" << active_params.relax_sweeps << " Sweeps) ===\n";
        relax_wavefunction(state, sys, channels, active_params, rng, active_params.relax_sweeps);
    }

    if (active_params.verbose) {
        std::cout << "\n\n" << std::string(60, '-') << "\n";
        std::cout << "Final E0 = " << state.E0 << " MeV" << "  (K=" << state.K() << ")\n";
    }

    return state;
}

} // namespace qm