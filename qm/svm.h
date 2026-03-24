#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// svm.h
//
// Stochastic Variational Method (SVM) engine.
// Follows Suzuki & Varga (1998), as described in section 4 of your bachelor.
//
// This file provides:
//   1.  Jacobi diagonalisation of a symmetric real matrix (the eigenvalue step)
//   2.  Generalized eigenvalue solver via Cholesky + standard symmetric eig
//   3.  The SVM iterative basis builder (competitive search + refinement)
//
// The SVM knows nothing about the physics; it only sees:
//   - A function that builds the full (H, N) matrix pair given a basis
//   - A basis element type (template parameter)
//   - The number of basis states to grow to
//
// Usage pattern:
//   1. Define a BasisState struct (A matrix, s vector, channel label, ...)
//   2. Write a MatrixBuilder callable:
//        void build(const std::vector<BasisState>&, qm::rmat& H, qm::rmat& N)
//   3. Call svm::optimize_basis(builder, initial_basis, target_size, n_candidates)
// ─────────────────────────────────────────────────────────────────────────────

#include "matrix.h"
#include <vector>
#include <functional>
#include <limits>
#include <iostream>
#include <cmath>

namespace svm {

using ld   = qm::ld;
using rmat = qm::rmat;
using rvec = qm::rvec;

// ─────────────────────────────────────────────────────────────────────────────
// §1  Jacobi eigenvalue algorithm for symmetric real matrices
//
// Finds all eigenvalues (and optionally eigenvectors) of a symmetric matrix
// by successive Givens rotations that zero off-diagonal elements.
//
// Returns eigenvalues in ascending order.
// If eigvecs != nullptr, fills it with orthonormal eigenvectors as columns.
// ─────────────────────────────────────────────────────────────────────────────

struct EigResult {
    rvec  values;    // eigenvalues, ascending
    rmat  vectors;   // eigenvectors as columns (size n×n)
};

inline EigResult jacobi_eigen(rmat S, bool compute_vectors = true)
{
    size_t n = S.size1();
    assert(n == S.size2());

    rmat V(n, n);
    if (compute_vectors) V.setid();

    const int    MAX_SWEEPS = 100;
    const ld     TOL        = 1e-15L;

    for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
        // Off-diagonal norm
        ld off = 0;
        for (size_t i = 0; i < n; i++)
            for (size_t j = i+1; j < n; j++)
                off += S(i,j) * S(i,j);
        if (off < TOL * TOL) break;

        // One sweep: rotate all (i,j) pairs
        for (size_t p = 0; p < n-1; p++) {
            for (size_t q = p+1; q < n; q++) {
                if (std::fabs(S(p,q)) < TOL) continue;

                // Compute rotation angle
                ld theta = 0.5L * (S(q,q) - S(p,p)) / S(p,q);
                ld t     = (theta >= 0)
                           ? 1.0L / (theta + std::sqrt(1 + theta*theta))
                           : 1.0L / (theta - std::sqrt(1 + theta*theta));
                ld c     = 1.0L / std::sqrt(1 + t*t);
                ld s     = t * c;
                ld tau   = s / (1 + c);

                // Update diagonal
                ld dpp = -t * S(p,q);
                ld dqq = +t * S(p,q);
                S(p,p) += dpp;
                S(q,q) += dqq;
                S(p,q)  = 0;
                S(q,p)  = 0;

                // Update off-diagonals
                for (size_t r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    ld Srp = S(r,p);
                    ld Srq = S(r,q);
                    S(r,p) = Srp - s * (Srq + tau * Srp);
                    S(p,r) = S(r,p);
                    S(r,q) = Srq + s * (Srp - tau * Srq);
                    S(q,r) = S(r,q);
                }

                // Update eigenvectors
                if (compute_vectors) {
                    for (size_t r = 0; r < n; r++) {
                        ld Vrp = V(r,p);
                        ld Vrq = V(r,q);
                        V(r,p) = c * Vrp - s * Vrq;
                        V(r,q) = s * Vrp + c * Vrq;
                    }
                }
            }
        }
    }

    // Extract eigenvalues and sort ascending
    EigResult res;
    res.values.resize(n);
    for (size_t i = 0; i < n; i++) res.values[i] = S(i,i);

    // Sort by eigenvalue (bubble sort — n is small)
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    for (size_t i = 0; i < n; i++)
        for (size_t j = i+1; j < n; j++)
            if (res.values[idx[i]] > res.values[idx[j]])
                std::swap(idx[i], idx[j]);

    rvec sorted_vals(n);
    rmat sorted_vecs(n, n);
    for (size_t i = 0; i < n; i++) {
        sorted_vals[i] = res.values[idx[i]];
        if (compute_vectors)
            for (size_t r = 0; r < n; r++)
                sorted_vecs(r, i) = V(r, idx[i]);
    }
    res.values  = sorted_vals;
    res.vectors = sorted_vecs;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Generalized eigenvalue problem  H c = E N c
//
// Solved via Cholesky decomposition of N:
//   N = L L^T  =>  (L^{-1} H L^{-T})(L^T c) = E (L^T c)
//
// Returns the lowest eigenvalue (ground state energy).
// If eigvec != nullptr, fills the ground-state coefficient vector c.
//
// Returns NaN on failure (e.g. N not positive definite).
// ─────────────────────────────────────────────────────────────────────────────
inline ld solve_gep(const rmat& H, const rmat& N,
                    rvec* eigvec = nullptr)
{
    size_t n = N.size1();

    // Cholesky: N = L L^T
    rmat L = N.cholesky();
    if (L.size1() == 0) {
        // N is not positive definite — basis is ill-conditioned
        return std::numeric_limits<ld>::quiet_NaN();
    }

    // L^{-1}  (lower triangular inverse)
    rmat Linv = L.inverse_lower();
    if (Linv.size1() == 0) {
        return std::numeric_limits<ld>::quiet_NaN();
    }

    // Hbar = L^{-1} H L^{-T}
    rmat LT    = L.transpose();
    rmat LinvT = Linv.transpose();
    rmat Hbar  = Linv * H * LinvT;

    // Symmetrise (guard against floating-point asymmetry)
    for (size_t i = 0; i < n; i++)
        for (size_t j = i+1; j < n; j++) {
            ld avg = 0.5L * (Hbar(i,j) + Hbar(j,i));
            Hbar(i,j) = Hbar(j,i) = avg;
        }

    // Jacobi diagonalisation
    auto eig = jacobi_eigen(Hbar, eigvec != nullptr);

    ld E0 = eig.values[0];

    if (eigvec != nullptr) {
        // Back-substitute: c = L^{-T} y  where y is ground state eigenvector
        rvec y(n);
        for (size_t i = 0; i < n; i++) y[i] = eig.vectors(i, 0);
        // c = L^{-T} y  = (L^{-1})^T y
        *eigvec = LinvT * y;
    }

    return E0;
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  SVM basis optimiser
//
// Template parameters:
//   BasisState : any struct that represents one Gaussian basis state.
//                Must be copyable.
//   MatrixBuilder : callable  void(const std::vector<BasisState>&, rmat&, rmat&)
//                   that fills H and N matrices.
//   CandidateGenerator : callable  BasisState(size_t current_size, size_t trial_index)
//                        that generates a random candidate state.
//
// Algorithm (section 4.2 of your bachelor):
//   - Grow basis one state at a time up to target_size.
//   - For each new slot: generate n_candidates random states, keep the best.
//   - After each addition, do one refinement pass over the entire basis.
//   - Periodically (every refine_period steps) do a full refinement sweep.
// ─────────────────────────────────────────────────────────────────────────────

template<typename BasisState>
struct SVMOptimizer {

    using MatrixBuilder      = std::function<void(const std::vector<BasisState>&, rmat&, rmat&)>;
    using CandidateGenerator = std::function<BasisState(size_t, size_t)>;

    MatrixBuilder      build;       // builds (H, N) from basis
    CandidateGenerator generate;    // generates a random candidate

    int  n_candidates   = 50;       // pool size for competitive search
    int  refine_period  = 5;        // full refinement sweep every N additions
    bool verbose        = true;

    // Safety bounds — energies outside this window are treated as numerical garbage.
    // Set E_min to something well below any physical bound state you expect.
    // For the deuteron, anything below -100 MeV is certainly unphysical.
    ld E_min = -100.0L;   // MeV  — reject candidates below this
    ld E_max =  500.0L;   // MeV  — reject candidates above this (unbound)

    // Condition number threshold for the overlap matrix N.
    // If max(diag(L)) / min(diag(L)) > this, the basis is ill-conditioned.
    ld cond_threshold = 1.0e8L;

    // ── Run the optimisation ────────────────────────────────────────────────
    std::vector<BasisState> optimize(std::vector<BasisState> basis,
                                     size_t target_size)
    {
        if (verbose)
            std::cout << "[SVM] Growing basis to " << target_size << " states\n" << std::flush;

        // Seed with one good candidate if empty
        if (basis.empty()) {
            BasisState best; bool found = false;
            ld best_E = std::numeric_limits<ld>::max();
            for (int t = 0; t < n_candidates; t++) {
                auto candidate = generate(0, t);
                std::vector<BasisState> trial = {candidate};
                rmat H, N;
                build(trial, H, N);
                ld E = safe_solve(H, N);
                if (is_physical(E) && E < best_E) { best_E = E; best = candidate; found = true; }
            }
            if (!found) {
                std::cerr << "[SVM] Could not seed basis — all candidates rejected.\n";
                return basis;
            }
            basis.push_back(best);
            if (verbose)
                std::cout << "[SVM] k=1  E0 = " << best_E << " MeV\n" << std::flush;
        }

        // Grow one state at a time
        while (basis.size() < target_size) {
            size_t k = basis.size();

            // Current energy with existing basis
            rmat H0, N0;
            build(basis, H0, N0);
            ld current_E = safe_solve(H0, N0);

            BasisState best_candidate;
            bool found = false;
            ld best_E = current_E;

            for (int t = 0; t < n_candidates; t++) {
                auto candidate = generate(k, t);
                auto trial = basis;
                trial.push_back(candidate);
                rmat H, N;
                build(trial, H, N);
                ld E = safe_solve(H, N);
                if (is_physical(E) && E < best_E) {
                    best_E = E;
                    best_candidate = candidate;
                    found = true;
                }
            }

            if (!found) {
                // No candidate improved things — keep current basis but warn
                if (verbose)
                    std::cout << "[SVM] k=" << k+1
                              << "  no improvement found, keeping best candidate\n";
                // Force-add the least-bad physical candidate
                for (int t = 0; t < n_candidates && !found; t++) {
                    auto candidate = generate(k, t + n_candidates);
                    auto trial = basis;
                    trial.push_back(candidate);
                    rmat H, N;
                    build(trial, H, N);
                    ld E = safe_solve(H, N);
                    if (is_physical(E)) {
                        best_candidate = candidate;
                        best_E = E;
                        found = true;
                    }
                }
                if (!found) {
                    std::cerr << "[SVM] k=" << k+1 << " all candidates unphysical — stopping.\n";
                    break;
                }
            }

            basis.push_back(best_candidate);
            if (verbose)
                std::cout << "[SVM] k=" << k+1 << "  E0 = " << best_E << " MeV\n" << std::flush;

            // Light refinement after every addition
            basis = refine(basis, current_E);

            // Full sweep periodically
            if ((int)(k+1) % refine_period == 0) {
                if (verbose) std::cout << "[SVM] -- full refinement sweep --\n";
                basis = refine(basis, current_E);
            }
        }

        return basis;
    }

    // ── Public helper: solve GEP with all safety checks ──────────────────────
    ld safe_solve(const rmat& H, const rmat& N) const
    {
        // 1. Cholesky — also gives us a condition estimate from diagonal of L
        rmat L = N.cholesky();
        if (L.size1() == 0) return std::numeric_limits<ld>::quiet_NaN();

        // 2. Condition number estimate: max(L_ii) / min(L_ii)
        ld Lmax = 0, Lmin = std::numeric_limits<ld>::max();
        for (size_t i = 0; i < L.size1(); i++) {
            ld d = std::abs(L(i,i));
            if (d > Lmax) Lmax = d;
            if (d < Lmin) Lmin = d;
        }
        if (Lmin < ZERO_LIMIT || Lmax / Lmin > cond_threshold)
            return std::numeric_limits<ld>::quiet_NaN();

        // 3. Standard solve
        return solve_gep(H, N);
    }

private:
    // Is an energy value within the physical window?
    bool is_physical(ld E) const
    {
        return !std::isnan(E) && !std::isinf(E) && E > E_min && E < E_max;
    }

    // Refinement: try replacing each state with a better one.
    // Only accepts replacements that keep the energy physical and improving.
    std::vector<BasisState> refine(std::vector<BasisState> basis, ld reference_E)
    {
        rmat H, N;
        build(basis, H, N);
        ld current_E = safe_solve(H, N);
        if (!is_physical(current_E)) current_E = reference_E;

        for (size_t slot = 0; slot < basis.size(); slot++) {
            BasisState original = basis[slot];
            ld best_E = current_E;
            BasisState best = original;

            for (int t = 0; t < n_candidates; t++) {
                auto candidate = generate(basis.size(), t + (int)slot * n_candidates);
                basis[slot] = candidate;
                build(basis, H, N);
                ld E = safe_solve(H, N);
                if (is_physical(E) && E < best_E) { best_E = E; best = candidate; }
            }

            basis[slot] = best;
            current_E   = best_E;
        }
        return basis;
    }
};

// Convenience factory
template<typename BasisState>
SVMOptimizer<BasisState>
make_svm(typename SVMOptimizer<BasisState>::MatrixBuilder mb,
         typename SVMOptimizer<BasisState>::CandidateGenerator cg)
{
    SVMOptimizer<BasisState> opt;
    opt.build    = std::move(mb);
    opt.generate = std::move(cg);
    return opt;
}

} // namespace svm