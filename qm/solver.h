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
//
// Changes vs. original (marked NEW / CHANGED below):
//
//   1. SvmParams: new fields for utility selection and multi-step annealing.
//
//   2. Utility selection  (SVM book §4.2.5)
//      The build loop now requires a candidate to lower E0 by at least
//      delta_E_min MeV before it is admitted.  If max_fail_utility
//      consecutive batches all fail the test, delta_E_min is halved
//      automatically — this mirrors the book's recommended strategy and
//      prevents near-redundant states from bloating the basis.
//
//   3. Adaptive E_floor
//      E_floor is no longer a hard -1000 MeV constant.  It is computed
//      from E_target (or, before E_target is known, from the current best
//      energy) as  E_floor = E_target * E_floor_factor.  This rejects
//      unphysical numerical states before they enter the basis.
//
//   4. Multi-step annealing schedule  (new field S_anneal_steps)
//      Instead of building the entire basis at S_build and then doing a
//      single bisection jump to E_target, you can supply a decreasing
//      list of S values.  The basis is grown in equal stages, with a
//      light refinement sweep between stages.  This keeps the Gaussian
//      parameters well-adapted to the coupling at every stage.
//
//   5. Light periodic refinement during the annealing build
//      When do_annealing=true the original code forced refine_every=0.
//      The new code uses refine_during_anneal (default: every 5 steps)
//      with a reduced trial count (N_refine_trial / 2) so the per-step
//      overhead stays modest.
//
//   6. local_refine_accepted  (new helper)
//      After a candidate survives the utility test, a small number of
//      perturbed copies are tested and the best replaces the accepted
//      candidate.  This is the "fine tuning in the vicinity" recommended
//      at the end of SVM book §4.2.5.
// ─────────────────────────────────────────────────────────────────────────────

namespace qm {

// ─────────────────────────────────────────────────────────────────────────────
// jacobi_diag — unchanged from original
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
                    cld hip = H(i, p); cld hiq = H(i, q);
                    H(i, p) = gpp * hip + gqp * hiq;
                    H(i, q) = gpq * hip + gqq * hiq;
                }
                for (size_t j = 0; j < n; j++) {
                    cld hpj = H(p, j); cld hqj = H(q, j);
                    H(p, j) = std::conj(gpp) * hpj + std::conj(gqp) * hqj;
                    H(q, j) = std::conj(gpq) * hpj + std::conj(gqq) * hqj;
                }
                for (size_t i = 0; i < n; i++) {
                    cld vip = V(i, p); cld viq = V(i, q);
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
// Linear Dependence Check — unchanged from original
// ─────────────────────────────────────────────────────────────────────────────
inline bool is_linearly_dependent(const Gaussian& trial,
                                   const std::vector<Gaussian>& basis,
                                   int  ignore_idx = -1,
                                   ld   threshold  = 0.999L)
{
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
// SvmParams  (CHANGED — new fields for utility selection & annealing schedule)
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

    // ── NEW: Utility selection (SVM book §4.2.5) ─────────────────────────────
    //
    // A candidate pair is only admitted if it lowers E0 by more than
    // delta_E_min MeV.  When max_fail_utility consecutive N_trial batches
    // all fail to find such a candidate, delta_E_min is halved and the
    // counter resets.  Set delta_E_min = 0 to revert to pure competitive
    // selection (original behaviour).
    ld  delta_E_min;        // initial minimum energy drop [MeV]  (default 0.05)
    int max_fail_utility;   // halve threshold after this many consecutive fails

    // ── NEW: Multi-step annealing schedule ───────────────────────────────────
    //
    // If non-empty, the basis is built in len(S_anneal_steps) equal stages.
    // Each stage uses the corresponding S value; a light refinement sweep is
    // performed between stages.  The final physical S is found by
    // tune_coupling as usual.
    //
    // Example: S_anneal_steps = {18, 15, 12} with K_max=25
    //   → grow 8 pairs at S=18, refine, grow 8 at S=15, refine, grow 9 at S=12
    //   → tune_coupling, relax_sweeps
    //
    // Leave empty to use the original two-phase approach (build at S_build,
    // then single bisection step).
    std::vector<ld> S_anneal_steps;

    // ── NEW: Light refinement during annealing build ─────────────────────────
    //
    // Original code forced refine_every=0 during annealing to save time.
    // refine_during_anneal (in basis steps) does one sweep with
    // N_refine_trial/2 trials — cheap enough to run often and keeps the
    // Gaussian parameters well-adapted at the current S.
    // Set to 0 to disable (original behaviour).
    int refine_during_anneal;   // default: 5  (refine every 5 accepted steps)

    // ── NEW: Local fine-tuning after acceptance (SVM book §4.2.5, last para) ─
    //
    // After a candidate passes the utility test, local_finetune_trials small
    // perturbations are generated around it.  The best replaces the accepted
    // candidate.  Set to 0 to disable.
    int  local_finetune_trials;  // number of perturbations to try  (default 10)
    ld   local_perturb_scale;    // scale of A/s perturbations      (default 0.15)

    // ── NEW: Adaptive E_floor ────────────────────────────────────────────────
    //
    // Candidates with E_trial < E_target * E_floor_factor are rejected as
    // unphysical regardless of the utility test.  For the deuteron
    // (E_target ≈ -2.2 MeV) with factor=10 → floor at -22 MeV.
    // The floor only activates when do_annealing=true and E_target is set.
    ld E_floor_factor;           // default: 10.0

    SvmParams()
        : K_max(40), N_trial(50), refine_every(10), N_refine_trial(50),
          b0(ld{1.4L}), s_max(ld{0.12L}), b_ff(ld{1.4L}), S_coupling(ld{15}),
          relativistic(false), verbose(true),
          do_annealing(false), S_build(18.0), E_target(-2.224), relax_sweeps(3),
          // NEW defaults
          delta_E_min(ld{0.05L}), max_fail_utility(3),
          S_anneal_steps{},
          refine_during_anneal(5),
          local_finetune_trials(10), local_perturb_scale(ld{0.15L}),
          E_floor_factor(ld{10.0L})
    {}
};

struct SvmState {
    std::vector<Gaussian> basis_bare;
    std::vector<Gaussian> basis_dressed;

    cmat H;
    cmat N;
    ld   E0;
    ld   S_final;

    std::vector<ld> energy_history;

    size_t K() const { return basis_bare.size(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// incremental_update — unchanged from original
// ─────────────────────────────────────────────────────────────────────────────
inline void incremental_update(const SvmState& state,
                                const Gaussian& g_bare_new,
                                const Gaussian& g_dress_new,
                                int             target_idx,
                                const JacobiSystem&         sys,
                                const std::vector<Channel>& channels,
                                const SvmParams&            params,
                                cmat& H_out, cmat& N_out)
{
    size_t K = state.K();
    bool   is_append = (target_idx < 0);
    size_t K_new     = is_append ? K + 1 : K;
    size_t n_channels = channels.size();
    size_t dim_new    = n_channels * K_new;

    H_out = zeros<cld>(dim_new, dim_new);
    N_out = zeros<cld>(dim_new, dim_new);

    if (is_append) {
        if (K > 0) {
            for (size_t a = 0; a < n_channels; ++a) {
                size_t oa = a * K; size_t na = a * K_new;
                for (size_t b = 0; b < n_channels; ++b) {
                    size_t ob = b * K; size_t nb = b * K_new;
                    for (size_t i = 0; i < K; ++i)
                        for (size_t j = 0; j < K; ++j) {
                            H_out(na+i, nb+j) = state.H(oa+i, ob+j);
                            N_out(na+i, nb+j) = state.N(oa+i, ob+j);
                        }
                }
            }
        }
    } else {
        H_out = state.H; N_out = state.N;
        for (size_t a = 0; a < n_channels; ++a) {
            size_t global_idx = a * K_new + target_idx;
            for (size_t k = 0; k < dim_new; ++k) {
                H_out(global_idx, k) = cld{0,0}; H_out(k, global_idx) = cld{0,0};
                N_out(global_idx, k) = cld{0,0}; N_out(k, global_idx) = cld{0,0};
            }
        }
    }

    std::vector<Gaussian> b_bare  = state.basis_bare;
    std::vector<Gaussian> b_dress = state.basis_dressed;
    size_t active_idx;

    if (is_append) {
        b_bare.push_back(g_bare_new); b_dress.push_back(g_dress_new);
        active_idx = K;
    } else {
        b_bare[target_idx]  = g_bare_new;
        b_dress[target_idx] = g_dress_new;
        active_idx = target_idx;
    }

    for (size_t a = 0; a < n_channels; a++) {
        size_t na = a * K_new;
        const auto& bas = channels[a].is_bare ? b_bare : b_dress;
        for (size_t i = 0; i < K_new; i++) {
            for (size_t j = 0; j < K_new; j++) {
                if (i != active_idx && j != active_idx) continue;
                GaussianPair gp(bas[i], bas[j]);
                N_out(na+i, na+j) = cld{gp.M, ld{0}};
                ld ke;
                if (channels[a].is_bare) {
                    rvec c0_bare(1); c0_bare[0] = ld{1};
                    KineticParams kp(gp, c0_bare);
                    ke = params.relativistic
                           ? ke_relativistic(gp, kp, sys.mu[0])
                           : (phys::hbar_c2 / (ld{2} * sys.mu[0])) * ke_classical(gp, kp);
                } else {
                    ke = kinetic_energy(gp, sys, params.relativistic);
                    ke += channels[a].pion_mass * gp.M;
                }
                H_out(na+i, na+j) += cld{ke, ld{0}};
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
                cld w = w_matrix_element(b_dress[i], b_bare[j],
                                         sys, ch.w_piN, params.b_ff,
                                         params.S_coupling, ch.iso_coeff,
                                         ch.spin_type);
                H_out(na+i, n0+j) += w;
                H_out(n0+j, na+i) += std::conj(w);
            }
        }
    }
}

inline void grow_matrices(SvmState& state,
                           const Gaussian& g_bare_new,
                           const Gaussian& g_dress_new,
                           const JacobiSystem&         sys,
                           const std::vector<Channel>& channels,
                           const SvmParams&            params)
{
    cmat H_new, N_new;
    incremental_update(state, g_bare_new, g_dress_new, -1,
                       sys, channels, params, H_new, N_new);
    state.H = H_new; state.N = N_new;
}

inline ld eval_candidate(const SvmState& state,
                          const Gaussian& g_bare_cand,
                          const Gaussian& g_dress_cand,
                          const JacobiSystem&         sys,
                          const std::vector<Channel>& channels,
                          const SvmParams&            params)
{
    cmat H, N;
    incremental_update(state, g_bare_cand, g_dress_cand, -1,
                       sys, channels, params, H, N);
    return solve_gevp(H, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// relax_wavefunction — unchanged from original
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
inline void relax_wavefunction(SvmState&                   state,
                                const JacobiSystem&         sys,
                                const std::vector<Channel>& channels,
                                const SvmParams&            params,
                                RNG& rng, int num_sweeps)
{
    const ld E_floor = ld{-1000};

    for (int sweep = 1; sweep <= num_sweeps; ++sweep) {
        for (size_t j = 0; j < state.K(); j++) {

            // ── Refine BARE ────────────────────────────────────────────────
            {
                std::vector<Gaussian> rc_bare;
                rc_bare.reserve(params.N_refine_trial);
                int attempts = 0, max_attempts = params.N_refine_trial * 50;
                while ((int)rc_bare.size() < params.N_refine_trial &&
                       attempts < max_attempts) {
                    Gaussian gb = random_gaussian_bare(sys, params.b0, params.s_max, rng);
                    if (!is_linearly_dependent(gb, state.basis_bare, j, 0.999L))
                        rc_bare.push_back(gb);
                    attempts++;
                }
                int best_rt = -1; ld best_rE = state.E0; cmat best_H, best_N;

                #pragma omp parallel
                {
                    int local_rt = -1; ld local_rE = state.E0; cmat local_H, local_N;
                    #pragma omp for schedule(dynamic) nowait
                    for (int t = 0; t < (int)rc_bare.size(); t++) {
                        cmat H_t, N_t;
                        incremental_update(state, rc_bare[t], state.basis_dressed[j],
                                           j, sys, channels, params, H_t, N_t);
                        ld E_t = solve_gevp(H_t, N_t);
                        if (std::isfinite(E_t) && E_t > E_floor && E_t < local_rE)
                        { local_rE = E_t; local_rt = t; local_H = H_t; local_N = N_t; }
                    }
                    #pragma omp critical
                    { if (local_rt >= 0 && local_rE < best_rE)
                        { best_rE = local_rE; best_rt = local_rt;
                          best_H = local_H; best_N = local_N; } }
                }
                if (best_rt >= 0) {
                    state.basis_bare[j] = rc_bare[best_rt];
                    state.H = best_H; state.N = best_N; state.E0 = best_rE;
                    if (params.verbose)
                        std::cout << "\r  [Refining Bare " << j << "] -> E0 = "
                                  << best_rE << " MeV      " << std::flush;
                }
            }

            // ── Refine DRESSED ─────────────────────────────────────────────
            {
                std::vector<Gaussian> rc_dress;
                rc_dress.reserve(params.N_refine_trial);
                int attempts = 0, max_attempts = params.N_refine_trial * 50;
                while ((int)rc_dress.size() < params.N_refine_trial &&
                       attempts < max_attempts) {
                    Gaussian gd = random_gaussian_dressed(sys, params.b0, params.s_max, rng);
                    if (!is_linearly_dependent(gd, state.basis_dressed, j, 0.999L))
                        rc_dress.push_back(gd);
                    attempts++;
                }
                int best_rt = -1; ld best_rE = state.E0; cmat best_H, best_N;

                #pragma omp parallel
                {
                    int local_rt = -1; ld local_rE = state.E0; cmat local_H, local_N;
                    #pragma omp for schedule(dynamic) nowait
                    for (int t = 0; t < (int)rc_dress.size(); t++) {
                        cmat H_t, N_t;
                        incremental_update(state, state.basis_bare[j], rc_dress[t],
                                           j, sys, channels, params, H_t, N_t);
                        ld E_t = solve_gevp(H_t, N_t);
                        if (std::isfinite(E_t) && E_t > E_floor && E_t < local_rE)
                        { local_rE = E_t; local_rt = t; local_H = H_t; local_N = N_t; }
                    }
                    #pragma omp critical
                    { if (local_rt >= 0 && local_rE < best_rE)
                        { best_rE = local_rE; best_rt = local_rt;
                          best_H = local_H; best_N = local_N; } }
                }
                if (best_rt >= 0) {
                    state.basis_dressed[j] = rc_dress[best_rt];
                    state.H = best_H; state.N = best_N; state.E0 = best_rE;
                    if (params.verbose)
                        std::cout << "\r  [Refining Dressed " << j << "] -> E0 = "
                                  << best_rE << " MeV   " << std::flush;
                }
            }
        }
        if (params.verbose) std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tune_coupling — unchanged from original
// ─────────────────────────────────────────────────────────────────────────────
inline void tune_coupling(SvmState& state,
                           const JacobiSystem&         sys,
                           const std::vector<Channel>& channels,
                           SvmParams&                  active_params)
{
    if (active_params.verbose)
        std::cout << "\n=== Tuning S to Target E0 = "
                  << active_params.E_target << " MeV ===\n";

    HamiltonianBuilder hb_base(sys, channels,
                                state.basis_bare, state.basis_dressed,
                                active_params.b_ff, 0.0,
                                active_params.relativistic);
    cmat N      = hb_base.build_N();
    cmat H_diag = hb_base.build_H();
    size_t dim  = H_diag.size1();

    cmat L       = N.cholesky();
    cmat Linv    = L.inverse_lower();
    cmat Linv_dag = Linv.adjoint();

    HamiltonianBuilder hb_W(sys, channels,
                             state.basis_bare, state.basis_dressed,
                             active_params.b_ff, 1.0,
                             active_params.relativistic);
    cmat H_S1 = hb_W.build_H();
    cmat W_raw = zeros<cld>(dim, dim);
    for (size_t r = 0; r < dim; ++r)
        for (size_t c = 0; c < dim; ++c)
            W_raw(r, c) = H_S1(r, c) - H_diag(r, c);

    cmat Hp_diag = Linv * H_diag * Linv_dag;
    cmat Wp_raw  = Linv * W_raw  * Linv_dag;

    ld S_low  = ld{5.0L};
    ld S_high = active_params.S_build + ld{5.0L};
    ld best_S = active_params.S_build;
    ld tol    = ld{1e-4L};

    for (int iter = 1; iter <= 50; ++iter) {
        ld S_mid = (S_low + S_high) / ld{2.0L};
        cmat Hp  = zeros<cld>(dim, dim);
        for (size_t r = 0; r < dim; ++r)
            for (size_t c = 0; c < dim; ++c)
                Hp(r, c) = Hp_diag(r, c) + cld{S_mid, 0.0L} * Wp_raw(r, c);

        jacobi_diag(Hp);
        ld E0 = std::numeric_limits<ld>::max();
        for (size_t d = 0; d < dim; d++) {
            ld e = std::real(Hp(d, d));
            if (e < E0) E0 = e;
        }

        if (active_params.verbose)
            std::cout << "\r  [Tuning] Bisection " << iter
                      << ": S = " << std::fixed << std::setprecision(4) << S_mid
                      << " MeV  |  E0 = " << std::setprecision(5) << E0
                      << " MeV    " << std::flush;

        if (std::abs(E0 - active_params.E_target) < tol) { best_S = S_mid; break; }
        if (E0 > active_params.E_target) S_low  = S_mid;
        else                             S_high = S_mid;
        best_S = S_mid;
    }
    if (active_params.verbose) std::cout << "\n";

    active_params.S_coupling = best_S;
    state.S_final = best_S;

    for (size_t r = 0; r < dim; ++r)
        for (size_t c = 0; c < dim; ++c)
            state.H(r, c) = H_diag(r, c) + cld{best_S, 0.0L} * W_raw(r, c);
    state.E0 = solve_gevp(state.H, state.N);
    if (active_params.verbose)
        std::cout << "  -> Target Locked! S = " << best_S << " MeV\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW: local_refine_accepted
//
// After a candidate pair (g_bare, g_dress) is accepted by the utility test,
// this function generates local_finetune_trials small perturbations around it
// and picks the best, following the recommendation at the end of §4.2.5 of
// the SVM book:
//
//   "A better candidate could be found in the neighborhood of the basis
//    element admitted.  Fine tuning in the vicinity of the element would
//    clearly accelerate the energy convergence."
//
// Implementation:
//   - Each perturbed A is A_orig * exp(δ) where δ ~ N(0, scale²) per entry
//     (preserves positive-definiteness for small scale).
//   - Each perturbed s[k] is s_orig[k] + N(0, scale * s_max).
//
// Returns true if a better candidate was found and updates g_bare / g_dress
// and the corresponding H_out / N_out.
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
inline bool local_refine_accepted(Gaussian&        g_bare,
                                   Gaussian&        g_dress,
                                   ld&              E_current,
                                   cmat&            H_current,
                                   cmat&            N_current,
                                   const SvmState&  state,
                                   const JacobiSystem&         sys,
                                   const std::vector<Channel>& channels,
                                   const SvmParams& params,
                                   ld E_floor,
                                   RNG& rng)
{
    if (params.local_finetune_trials <= 0) return false;

    std::normal_distribution<long double> normal(ld{0}, params.local_perturb_scale);

    bool improved = false;

    for (int t = 0; t < params.local_finetune_trials; t++) {

        // ── Perturb bare (1D) ────────────────────────────────────────────────
        Gaussian gb_p = g_bare;
        gb_p.A(0, 0) *= std::exp(normal(rng));     // multiplicative, keeps A > 0
        gb_p.s[0] += normal(rng) * params.s_max;

        // ── Perturb dressed (2D) ─────────────────────────────────────────────
        Gaussian gd_p = g_dress;
        for (size_t i = 0; i < gd_p.dim; i++) {
            for (size_t j = i; j < gd_p.dim; j++) {
                ld delta = std::exp(normal(rng));
                gd_p.A(i, j) *= delta;
                gd_p.A(j, i) *= delta;
            }
            gd_p.s[i] += normal(rng) * params.s_max;
        }

        if (is_linearly_dependent(gb_p, state.basis_bare,  -1, ld{0.9999L}) ||
            is_linearly_dependent(gd_p, state.basis_dressed,-1, ld{0.9999L}))
            continue;

        cmat H_t, N_t;
        incremental_update(state, gb_p, gd_p, -1,
                           sys, channels, params, H_t, N_t);
        ld E_t = solve_gevp(H_t, N_t);

        if (std::isfinite(E_t) && E_t > E_floor && E_t < E_current) {
            E_current  = E_t;
            g_bare     = gb_p;
            g_dress    = gd_p;
            H_current  = H_t;
            N_current  = N_t;
            improved   = true;
        }
    }
    return improved;
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW: build_stage
//
// Grows the basis by exactly n_steps pairs under the given S value,
// applying utility selection, local fine-tuning, and the adaptive E_floor.
//
// Returns the number of pairs actually added (may be less than n_steps if
// the energy curve flattens out entirely).
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
inline int build_stage(SvmState&                   state,
                        const JacobiSystem&         sys,
                        const std::vector<Channel>& channels,
                        SvmParams&                  active_params,
                        int                         n_steps,
                        ld                          E_floor,
                        RNG&                        rng)
{
    ld   delta_E_min    = active_params.delta_E_min;   // local mutable copy
    int  consec_fails   = 0;                           // for utility selection
    int  steps_accepted = 0;

    for (int k = 0; k < n_steps; k++) {

        // ── Generate N_trial candidate pairs ─────────────────────────────────
        std::vector<Gaussian> cand_bare (active_params.N_trial);
        std::vector<Gaussian> cand_dress(active_params.N_trial);
        for (int t = 0; t < active_params.N_trial; t++) {
            cand_bare [t] = random_gaussian_bare   (sys, active_params.b0,
                                                    active_params.s_max, rng);
            cand_dress[t] = random_gaussian_dressed(sys, active_params.b0,
                                                    active_params.s_max, rng);
        }

        // ── Find the best candidate in parallel ───────────────────────────────
        int best_idx = -1;
        ld  best_E   = std::numeric_limits<ld>::max();
        cmat best_H, best_N;

        #pragma omp parallel
        {
            int  local_idx = -1;
            ld   local_E   = std::numeric_limits<ld>::max();
            cmat local_H, local_N;

            #pragma omp for schedule(dynamic) nowait
            for (int t = 0; t < active_params.N_trial; t++) {
                if (is_linearly_dependent(cand_bare [t], state.basis_bare,
                                          -1, ld{0.9999L}) ||
                    is_linearly_dependent(cand_dress[t], state.basis_dressed,
                                          -1, ld{0.9999L}))
                    continue;

                cmat H_t, N_t;
                incremental_update(state, cand_bare[t], cand_dress[t], -1,
                                   sys, channels, active_params, H_t, N_t);
                ld E_t = solve_gevp(H_t, N_t);

                // CHANGED: reject E below adaptive floor
                if (!std::isfinite(E_t) || E_t <= E_floor) continue;

                if (E_t < local_E) {
                    local_E = E_t; local_idx = t;
                    local_H = H_t; local_N = N_t;
                }
            }
            #pragma omp critical
            {
                if (local_idx >= 0 && local_E < best_E) {
                    best_E = local_E; best_idx = local_idx;
                    best_H = local_H; best_N = local_N;
                }
            }
        }

        // ── Cholesky failure or all trials rejected ───────────────────────────
        if (best_idx < 0) {
            if (active_params.verbose)
                std::cout << "  [step " << steps_accepted
                          << "] all trials rejected — stopping stage.\n";
            break;
        }

        // ── NEW: Utility selection (SVM book §4.2.5) ──────────────────────────
        // Only admit if the energy drops by more than delta_E_min.
        ld delta_E = best_E - state.E0;
        if (state.K() > 0 && delta_E > -delta_E_min) {
            // Not enough improvement this batch
            consec_fails++;
            if (consec_fails >= active_params.max_fail_utility) {
                delta_E_min *= ld{0.5L};
                consec_fails = 0;
                if (active_params.verbose)
                    std::cout << "\n  [Utility] Threshold halved → ΔE_min = "
                              << delta_E_min << " MeV\n";
                // If threshold is effectively zero, fall through to competitive
                if (delta_E_min < ld{1e-6L}) goto accept;
            }
            k--;   // don't count this k as a "step" — retry
            // Guard against infinite loop when basis is saturated
            if (delta_E_min < ld{1e-6L}) break;
            continue;
        }
        accept:
        consec_fails = 0;

        // ── NEW: Local fine-tuning around the accepted candidate ───────────────
        Gaussian accepted_bare  = cand_bare [best_idx];
        Gaussian accepted_dress = cand_dress[best_idx];
        local_refine_accepted(accepted_bare, accepted_dress,
                              best_E, best_H, best_N,
                              state, sys, channels, active_params,
                              E_floor, rng);

        // ── Grow basis ────────────────────────────────────────────────────────
        state.H = best_H; state.N = best_N;
        state.basis_bare .push_back(accepted_bare);
        state.basis_dressed.push_back(accepted_dress);
        state.E0 = best_E;
        state.energy_history.push_back(best_E);
        steps_accepted++;

        if (active_params.verbose) {
            std::cout << "\r" << std::setw(5) << "K = " << state.K()
                      << std::setw(10) << "  E0 = " << best_E << " MeV"
                      << std::setw(15) << "  ΔE = "
                      << (state.K() > 1 ? delta_E : ld{0}) << " MeV"
                      << std::flush;
        }

        // ── NEW: Light periodic refinement even during annealing ──────────────
        if (active_params.refine_during_anneal > 0 &&
            steps_accepted % active_params.refine_during_anneal == 0)
        {
            if (active_params.verbose) std::cout << "\n";
            // Use half the normal trial count to keep overhead modest
            SvmParams light = active_params;
            light.N_refine_trial = std::max(5, active_params.N_refine_trial / 2);
            relax_wavefunction(state, sys, channels, light, rng, 1);
        }
    }

    return steps_accepted;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_svm — CHANGED: integrates multi-step annealing and utility selection
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
SvmState run_svm(const JacobiSystem&         sys,
                  const std::vector<Channel>& channels,
                  const SvmParams&            params,
                  RNG&                        rng)
{
    SvmParams active_params = params;
    SvmState  state;
    state.E0     = std::numeric_limits<ld>::max();
    state.S_final = active_params.S_coupling;

    // ── Compute the adaptive E_floor ──────────────────────────────────────────
    // If we know E_target (annealing), floor is E_target * factor.
    // Before any state exists, floor is a safe large-negative fallback.
    //
    // With do_annealing and E_target = -2.224 MeV, factor = 10 gives
    // E_floor = -22.24 MeV — far enough below the deuteron ground state
    // to never reject a physical state, but close enough to reject the
    // spurious -50 / -100 MeV artefacts that appear with large bases.
    auto compute_E_floor = [&]() -> ld {
        if (active_params.do_annealing)
            return active_params.E_target * active_params.E_floor_factor;
        if (state.K() > 0 && std::isfinite(state.E0))
            return state.E0 * active_params.E_floor_factor;
        return ld{-1000};   // fallback before any basis exists
    };

    if (active_params.verbose) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "SVM: K_max=" << active_params.K_max
                  << "  N_trial=" << active_params.N_trial
                  << "  b0=" << active_params.b0 << " fm"
                  << "  s_max=" << active_params.s_max << " fm^-1\n"
                  << "  Physics: "
                  << (active_params.relativistic ? "relativistic" : "classical")
                  << " KE  |  Annealing: "
                  << (params.do_annealing ? "Enabled" : "Disabled")
                  << "\n  Utility: ΔE_min=" << active_params.delta_E_min
                  << " MeV  max_fail=" << active_params.max_fail_utility
                  << "  E_floor_factor=" << active_params.E_floor_factor
                  << "\n  Local fine-tune: " << active_params.local_finetune_trials
                  << " trials  scale=" << active_params.local_perturb_scale;
#ifdef _OPENMP
        std::cout << "  |  threads=" << omp_get_max_threads();
#endif
        std::cout << "\n" << std::string(60, '-') << "\n";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // NEW: Multi-step annealing build
    //
    // If S_anneal_steps is non-empty the basis is grown in stages:
    //
    //   Stage 0 : S = S_anneal_steps[0],  K_per_stage steps
    //   ...
    //   Stage n : S = S_anneal_steps[n],  remaining steps
    //
    // After each stage (except the last) one light refinement sweep is run.
    // This gives the Gaussian parameters time to adapt before the next, lower
    // coupling value is imposed.
    //
    // If S_anneal_steps is empty we fall back to the original single-stage
    // build at S_build (or S_coupling for non-annealing runs).
    // ═══════════════════════════════════════════════════════════════════════════
    if (active_params.do_annealing && !active_params.S_anneal_steps.empty()) {

        const auto& sched = active_params.S_anneal_steps;
        int n_stages      = (int)sched.size();
        int K_per_stage   = active_params.K_max / n_stages;

        if (active_params.verbose) {
            std::cout << "Multi-step annealing: " << n_stages << " stages × ~"
                      << K_per_stage << " steps\n";
            for (int i = 0; i < n_stages; i++)
                std::cout << "  Stage " << i << ": S = " << sched[i] << " MeV\n";
            std::cout << std::string(60, '-') << "\n";
        }

        for (int stage = 0; stage < n_stages; stage++) {
            active_params.S_coupling = sched[stage];
            int target_steps = (stage < n_stages - 1)
                                 ? K_per_stage
                                 : active_params.K_max - (int)state.K();

            if (active_params.verbose)
                std::cout << "\n── Stage " << stage
                          << "  (S = " << sched[stage] << " MeV, target K+" << target_steps
                          << ") ──\n";

            ld E_floor = compute_E_floor();
            build_stage(state, sys, channels, active_params, target_steps,
                        E_floor, rng);

            // Refine between stages (but not after the last one — tune + relax handle that)
            if (stage < n_stages - 1) {
                if (active_params.verbose) {
                    std::cout << "\n  [Inter-stage refinement] ";
                    std::cout << "(K=" << state.K() << ", S=" << sched[stage] << "→"
                              << sched[stage+1] << " MeV)\n";
                }
                relax_wavefunction(state, sys, channels, active_params, rng, 1);
            }
        }

    } else {
        // ── Original single-stage build ────────────────────────────────────
        if (active_params.do_annealing)
            active_params.S_coupling = active_params.S_build;

        ld E_floor = compute_E_floor();
        build_stage(state, sys, channels, active_params,
                    active_params.K_max, E_floor, rng);
    }

    if (active_params.verbose) std::cout << "\n";

    // ── Phase 2 & 3: Tune and Relax (Annealing Workflow) ─────────────────────
    if (params.do_annealing) {
        tune_coupling(state, sys, channels, active_params);
        if (active_params.verbose)
            std::cout << "=== Relaxing into True Ground State ("
                      << active_params.relax_sweeps << " Sweeps) ===\n";
        relax_wavefunction(state, sys, channels, active_params,
                           rng, active_params.relax_sweeps);
    }

    // ── Fallback: standard periodic refinement (non-annealing runs) ──────────
    if (!params.do_annealing && params.refine_every > 0) {
        if (active_params.verbose)
            std::cout << "=== Final Refinement ===\n";
        relax_wavefunction(state, sys, channels, active_params,
                           rng, 1);
    }

    if (active_params.verbose) {
        std::cout << "\n\n" << std::string(60, '-') << "\n";
        std::cout << "Final E0 = " << state.E0 << " MeV"
                  << "  (K=" << state.K() << ")\n";
    }

    return state;
}

} // namespace qm