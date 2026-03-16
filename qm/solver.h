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
// ── Solving H c = E N c ──────────────────────────────────────────────────────
//
//  Step 1.  Cholesky decomposition of the overlap matrix:
//               N = L L†    (L lower-triangular, complex)
//           Fails if N is not positive-definite (basis is linearly dependent).
//
//  Step 2.  Transform to a standard Hermitian eigenvalue problem:
//               H' = L⁻¹ H L⁻†    (H' is Hermitian because H and N are)
//
//  Step 3.  Diagonalise H' by cyclic Jacobi sweeps:
//               H' V = V Λ    (V unitary, Λ = diag of real eigenvalues)
//           Each (p,q) rotation uses the combined phase+Givens unitary:
//               G[p,p] = e^{iφ/2} c,   G[p,q] = −e^{iφ/2} s
//               G[q,p] = e^{−iφ/2} s,  G[q,q] =  e^{−iφ/2} c
//           where H'[p,q] = r e^{iφ}  and  θ = ½ arctan(2r / (H'[pp]−H'[qq]))
//           Converges in ~7 sweeps for n ≤ 20 to machine precision.
//
//  Step 4.  Ground-state energy = smallest eigenvalue E₀ of H'.
//
// ── SVM competitive search (thesis §5.1) ─────────────────────────────────────
//
//  At step k (basis size grows from k−1 to k):
//    Try N_trial random Gaussians.
//    For each candidate, compute only the ONE new row/column of H and N
//    (K−1 new matrix elements, not K²), then solve the (9k × 9k) GEP.
//    Keep the candidate that gives the lowest E₀.
//
//  Every refine_every steps, loop over all existing basis functions and
//  try to replace each one with a better random candidate.
//
// ── Incremental matrix assembly ──────────────────────────────────────────────
//
//  The full matrices H and N grow by one "block-row/col" per SVM step.
//  A block-row means 9 actual rows (1 from the bare block + 8 from the
//  dressed blocks) because all dressed channels share the same Gaussian index.
//
//  grow_matrices() computes only the new elements and appends them.
//
// ── Units ─────────────────────────────────────────────────────────────────────
//  Energies in MeV.  Lengths in fm.  A in fm⁻².  ħc = 197.3269804 MeV·fm.
// ─────────────────────────────────────────────────────────────────────────────

namespace qm {

// ─────────────────────────────────────────────────────────────────────────────
// jacobi_diag  —  cyclic Jacobi diagonalisation of a Hermitian matrix
//
// Modifies H in-place (becomes diagonal with real entries).
// Returns the accumulated eigenvector matrix V (columns = eigenvectors).
// Eigenvalues are the final diagonal entries of H, NOT sorted.
// Sorting is done by solve_gevp which calls this.
//
// The rotation at step (p,q):
//   H'[p,q] = r·e^{iφ}   →   θ = ½·arctan2(2r, H[pp]−H[qq])
//   Combined phase+Givens unitary G_pq with:
//     G[p,p] =  e^{iφ/2}·cos θ,   G[p,q] = −e^{iφ/2}·sin θ
//     G[q,p] =  e^{−iφ/2}·sin θ,  G[q,q] =  e^{−iφ/2}·cos θ
//   Update: H ← G†HG,  V ← VG
//
// Convergence: declared when every |H[p,q]| < tol.
// ─────────────────────────────────────────────────────────────────────────────
inline cmat jacobi_diag(cmat& H, ld tol = ld{1e-12L}, int max_sweeps = 500)
{
    size_t n = H.size1();
    assert(n == H.size2());

    cmat V = eye<cld>(n);   // accumulates the eigenvector columns

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

                // Jacobi rotation angle: θ = ½·arctan2(2r, a_pp − a_qq)
                ld theta = ld{0.5L} * std::atan2(ld{2} * r, a_pp - a_qq);
                ld c     = std::cos(theta);
                ld s     = std::sin(theta);

                // Combined phase+Givens unitary coefficients
                cld ephi_half     = std::exp(cld{ld{0}, phi / ld{2}});   // e^{iφ/2}
                cld ephi_half_neg = std::conj(ephi_half);                  // e^{−iφ/2}

                cld gpp =  ephi_half     * c;   // G[p,p]
                cld gpq = -ephi_half     * s;   // G[p,q]
                cld gqp =  ephi_half_neg * s;   // G[q,p]
                cld gqq =  ephi_half_neg * c;   // G[q,q]

                // ── Right multiply: H ← H·G  (columns p and q change) ───────
                for (size_t i = 0; i < n; i++) {
                    cld hip = H(i, p);
                    cld hiq = H(i, q);
                    H(i, p) = gpp * hip + gqp * hiq;
                    H(i, q) = gpq * hip + gqq * hiq;
                }

                // ── Left multiply: H ← G†·H  (rows p and q change) ──────────
                // G†[p,p]=conj(gpp), G†[p,q]=conj(gqp), etc.
                for (size_t j = 0; j < n; j++) {
                    cld hpj = H(p, j);
                    cld hqj = H(q, j);
                    H(p, j) = std::conj(gpp) * hpj + std::conj(gqp) * hqj;
                    H(q, j) = std::conj(gpq) * hpj + std::conj(gqq) * hqj;
                }

                // ── Accumulate eigenvectors: V ← V·G ────────────────────────
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
// solve_gevp  —  generalised eigenvalue problem  H c = E N c
//
// Returns the ground-state energy E₀ (smallest eigenvalue in MeV).
// Returns NaN if N is not positive-definite (Cholesky fails).
//
// Algorithm:
//   1. N = L L†  (Cholesky)
//   2. L⁻¹ is computed by back-substitution
//   3. H' = L⁻¹ H L⁻†  (Hermitian similarity transformation)
//   4. Jacobi diagonalisation of H' → eigenvalues sorted ascending
// ─────────────────────────────────────────────────────────────────────────────
inline ld solve_gevp(const cmat& H, const cmat& N)
{
    assert(H.size1() == N.size1());
    size_t n = H.size1();

    // Step 1: Cholesky decomposition N = L L†
    //
    // cholesky() returns matrix(0,0) when any diagonal becomes negative.
    // inverse_lower() returns matrix(0,0) when any diagonal of L falls
    // below ZERO_LIMIT (1e-10).  These two thresholds can disagree: a very
    // small but positive Cholesky diagonal (e.g. 1e-21) passes Cholesky but
    // produces L(i,i) ≈ 3e-11 < 1e-10, which makes inverse_lower return 0×0.
    // The missing null-check on Linv caused the assertion A.size2()==B.size1()
    // to fire in Linv * H when Linv was 0×0 and H was 2K×2K.
    //
    // Both failure modes are treated identically: return NaN so the SVM loop
    // rejects this candidate without crashing.
    cmat L = N.cholesky();
    if (L.size1() == 0) {
        return std::numeric_limits<ld>::quiet_NaN();
    }

    // Step 2: L⁻¹  by lower-triangular back-substitution
    cmat Linv = L.inverse_lower();
    if (Linv.size1() == 0) {
        // inverse_lower hit a near-zero diagonal in L — basis near-degenerate
        return std::numeric_limits<ld>::quiet_NaN();
    }

    // Step 3: H' = L⁻¹ H L⁻†   (L⁻† = (L⁻¹)†)
    cmat Linv_dag = Linv.adjoint();
    cmat Hp = Linv * H * Linv_dag;

    // Step 4: Jacobi diagonalisation
    jacobi_diag(Hp);   // Hp is modified in-place; diagonal entries = eigenvalues

    // Extract smallest real eigenvalue
    ld E0 = std::numeric_limits<ld>::max();
    for (size_t i = 0; i < n; i++) {
        ld e = std::real(Hp(i, i));
        if (e < E0) E0 = e;
    }
    return E0;
}

// Return ALL eigenvalues sorted ascending (used for diagnostics)
inline std::vector<ld> solve_gevp_all(const cmat& H, const cmat& N)
{
    size_t n = H.size1();
    cmat L    = N.cholesky();
    if (L.size1() == 0) return {};
    cmat Linv = L.inverse_lower();
    if (Linv.size1() == 0) return {};   // near-zero diagonal in L
    cmat Hp   = Linv * H * Linv.adjoint();
    jacobi_diag(Hp);

    std::vector<ld> evals(n);
    for (size_t i = 0; i < n; i++) evals[i] = std::real(Hp(i, i));
    std::sort(evals.begin(), evals.end());
    return evals;
}


// ─────────────────────────────────────────────────────────────────────────────
// SvmParams  —  configuration for the SVM optimisation loop
// ─────────────────────────────────────────────────────────────────────────────
struct SvmParams {
    int  K_max;          // maximum basis size per channel
    int  N_trial;        // number of random candidates per SVM step
    int  refine_every;   // refine the full basis every this many steps (0 = off)
    int  N_refine_trial; // candidates per basis function during refinement
    ld   b0;             // Gaussian range parameter  (fm,  thesis ~1.4)
    ld   s_max;          // shift bound               (fm⁻¹, ~0.076–0.15)
    ld   b_ff;           // form-factor range         (fm)
    ld   S_coupling;     // pion coupling strength    (MeV)
    bool relativistic;   // true → K^rel,  false → K^cla
    bool verbose;        // print progress each step

    SvmParams()
        : K_max(40), N_trial(50), refine_every(10), N_refine_trial(20),
          b0(ld{1.4L}), s_max(ld{0.12L}), b_ff(ld{1.4L}), S_coupling(ld{15}),
          relativistic(false), verbose(true)
    {}
};


// ─────────────────────────────────────────────────────────────────────────────
// SvmState  —  all state needed to extend the current basis incrementally
//
// The matrices H and N are stored explicitly and grown one "block" at a time.
// One block = 9 rows/cols: row 0 belongs to the bare channel, rows 1..8 to
// the dressed channels (one row per channel, same Gaussian index within each).
// ─────────────────────────────────────────────────────────────────────────────
struct SvmState {
    std::vector<Gaussian> basis_bare;     // current bare Gaussians
    std::vector<Gaussian> basis_dressed;  // current dressed Gaussians

    cmat H;       // current full Hamiltonian  (9K × 9K)
    cmat N;       // current overlap matrix    (9K × 9K)
    ld   E0;      // current ground-state energy (MeV)

    std::vector<ld> energy_history;  // E0 after each accepted step

    size_t K() const { return basis_bare.size(); }  // current basis size
};


// ─────────────────────────────────────────────────────────────────────────────
// grow_matrices  —  extend H and N by one new bare+dressed Gaussian pair
//
// The new basis functions are g_bare (the (K+1)-th bare Gaussian) and
// g_dress (the (K+1)-th dressed Gaussian, shared across all 8 dressed channels).
//
// New matrix size: 9(K+1) × 9(K+1).
// Only the 9 new rows/columns need to be computed:
//   Row 0:     bare  ↔ bare and bare ↔ dressed couplings
//   Rows 1..8: each dressed channel ↔ all previous and new
//
// We build the extended matrix from scratch for clarity.
// For K ≥ ~30, a true incremental append would be faster, but for the
// basis sizes in a bachelor thesis this is fast enough.
// ─────────────────────────────────────────────────────────────────────────────
inline void grow_matrices(SvmState&                    state,
                           const Gaussian&              g_bare_new,
                           const Gaussian&              g_dress_new,
                           const JacobiSystem&          sys,
                           const std::vector<Channel>&  channels,
                           const SvmParams&             params)
{
    // Temporarily extend the basis vectors
    std::vector<Gaussian> new_bare   = state.basis_bare;
    std::vector<Gaussian> new_dressed = state.basis_dressed;
    new_bare.push_back(g_bare_new);
    new_dressed.push_back(g_dress_new);

    // Rebuild full H and N from scratch
    HamiltonianBuilder hb(sys, channels, new_bare, new_dressed,
                          params.b_ff, params.S_coupling, params.relativistic);
    state.H = hb.build_H();
    state.N = hb.build_N();
}


// ─────────────────────────────────────────────────────────────────────────────
// eval_candidate  —  solve the GEP with one additional Gaussian pair
//
// Returns E₀ if the candidate improves the basis, NaN if Cholesky fails.
// Does NOT modify state.
// ─────────────────────────────────────────────────────────────────────────────
inline ld eval_candidate(const SvmState&              state,
                          const Gaussian&              g_bare_cand,
                          const Gaussian&              g_dress_cand,
                          const JacobiSystem&          sys,
                          const std::vector<Channel>&  channels,
                          const SvmParams&             params)
{
    std::vector<Gaussian> trial_bare   = state.basis_bare;
    std::vector<Gaussian> trial_dressed = state.basis_dressed;
    trial_bare.push_back(g_bare_cand);
    trial_dressed.push_back(g_dress_cand);

    HamiltonianBuilder hb(sys, channels, trial_bare, trial_dressed,
                          params.b_ff, params.S_coupling, params.relativistic);
    cmat H = hb.build_H();
    cmat N = hb.build_N();
    return solve_gevp(H, N);
}


// ─────────────────────────────────────────────────────────────────────────────
// run_svm  —  full stochastic variational method loop
//
// Phase 1 — Build  (K = 1 .. K_max):
//   For each new basis size k, try N_trial random Gaussian pairs.
//   Keep the pair that gives the lowest E₀.
//   After accepting, update state.H, state.N, state.E₀.
//
// Phase 2 — Refine  (every refine_every steps):
//   For each existing basis index j, try N_refine_trial replacements.
//   Accept if the replacement reduces E₀.
//
// Returns the final SvmState (basis, matrices, energy).
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
SvmState run_svm(const JacobiSystem&          sys,
                  const std::vector<Channel>&  channels,
                  const SvmParams&             params,
                  RNG&                         rng)
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
        std::cout << "\n";
        std::cout << std::string(60, '-') << "\n";
    }

    // ── Phase 1: Build ────────────────────────────────────────────────────────
    for (int k = 0; k < params.K_max; k++) {

        // Pre-generate all candidates with the shared rng (single-threaded)
        // so results are identical regardless of thread count.
        std::vector<Gaussian> cand_bare(params.N_trial);
        std::vector<Gaussian> cand_dress(params.N_trial);
        for (int t = 0; t < params.N_trial; t++) {
            cand_bare[t]  = random_gaussian_bare   (sys, params.b0, params.s_max, rng);
            cand_dress[t] = random_gaussian_dressed(sys, params.b0, params.s_max, rng);
        }

        // Evaluate all candidates in parallel.
        // Each thread tracks its own local best; we reduce afterwards.
        int    best_idx = -1;
        ld     best_E   = std::numeric_limits<ld>::max();

        #pragma omp parallel
        {
            int local_idx = -1;
            ld  local_E   = std::numeric_limits<ld>::max();

            #pragma omp for schedule(dynamic) nowait
            for (int t = 0; t < params.N_trial; t++) {
                ld E_trial = eval_candidate(state, cand_bare[t], cand_dress[t],
                                            sys, channels, params);
                if (std::isfinite(E_trial) && E_trial < local_E) {
                    local_E   = E_trial;
                    local_idx = t;
                }
            }

            // Reduce: one thread at a time updates the shared best
            #pragma omp critical
            {
                if (local_idx >= 0 && local_E < best_E) {
                    best_E   = local_E;
                    best_idx = local_idx;
                }
            }
        }

        // Accept the best candidate
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

        // ── Phase 2: Refine (periodically) ────────────────────────────────────
        //
        // Single forward pass: cycle once through every basis index j.
        // For each j, try N_refine_trial random replacements and keep the best
        // one that (a) strictly lowers E0 and (b) stays above E_floor.
        //
        // Why a single pass (no while-loop)?
        //   Repeating until no improvement degrades the basis: many functions
        //   become near-identical (especially with s_max=0), the overlap matrix
        //   N loses conditioning, and inverse_lower returns 0×0, which then
        //   propagates a bad eigenvalue that crashes the next matrix multiply.
        //
        // Why E_floor?
        //   isfinite(E) alone admits spurious values like −33 million MeV that
        //   arise from an ill-conditioned but technically-positive-definite N.
        //   With the inverse_lower null-check in solve_gevp these now return
        //   NaN, but the floor adds a second layer of defence.
        if (params.refine_every > 0 && (k + 1) % params.refine_every == 0) {
            if (params.verbose) std::cout << "\n";

            const ld E_floor = ld{-1000};   // reject anything below −1000 MeV

            for (size_t j = 0; j < state.K(); j++) {
                // Pre-generate refine candidates with the shared rng
                std::vector<Gaussian> rc_bare (params.N_refine_trial);
                std::vector<Gaussian> rc_dress(params.N_refine_trial);
                for (int t = 0; t < params.N_refine_trial; t++) {
                    rc_bare [t] = random_gaussian_bare   (sys, params.b0, params.s_max, rng);
                    rc_dress[t] = random_gaussian_dressed(sys, params.b0, params.s_max, rng);
                }

                int  best_rt  = -1;
                ld   best_rE  = state.E0;   // must strictly improve
                cmat best_H, best_N;

                #pragma omp parallel
                {
                    int  local_rt = -1;
                    ld   local_rE = state.E0;
                    cmat local_H, local_N;

                    #pragma omp for schedule(dynamic) nowait
                    for (int t = 0; t < params.N_refine_trial; t++) {
                        SvmState trial_state = state;
                        trial_state.basis_bare[j]    = rc_bare [t];
                        trial_state.basis_dressed[j] = rc_dress[t];

                        HamiltonianBuilder hb(sys, channels,
                                              trial_state.basis_bare,
                                              trial_state.basis_dressed,
                                              params.b_ff, params.S_coupling,
                                              params.relativistic);
                        cmat H_t = hb.build_H();
                        cmat N_t = hb.build_N();
                        ld   E_t = solve_gevp(H_t, N_t);

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