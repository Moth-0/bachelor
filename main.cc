// main.cc  --  Deuteron + explicit pion model using 9-sector SVM.
//
// Hamiltonian structure (complex Hermitian):
//
//   H = [ H_00   W_01   W_02   ...  W_08 ]
//       [ W_01†  H_11    0          0    ]
//       [  ...        H_22               ]
//       [ W_08†   0         ...   H_88   ]
//
// The diagonal blocks H_ss are real symmetric (kinetic energy + meson mass).
// The off-diagonal blocks W_0s are complex because:
//
//   W = (tau.pi)(sigma.r) f(r)
//
// and sigma.r in the spherical basis has components:
//   r[0] = z  (real),  r[1] = x+iy  (complex),  r[2] = x-iy  (complex)
//
// The complex Hermitian GEP  H c = E N c  is solved by the real-doubling trick:
//
//   H_real = [ H.real  -H.imag ]   N_real = [ N  0 ]
//            [ H.imag   H.real ]             [ 0  N ]
//
// which is a real symmetric GEP of size 2n.  Its lowest eigenvalue equals the
// ground-state energy of the complex problem.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <complex>
#include <limits>
#include <cmath>
#include <omp.h>

#include "qm/matrix.h"
#include "qm/eigen.h"
#include "qm/particle.h"
#include "qm/jacobian.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"

using namespace qm;

// ---------------------------------------------------------------------------
// Config key=value store with defaults
// ---------------------------------------------------------------------------
struct Params {
    std::map<std::string, std::string> store;

    Params() {
        store["S"]        = "15.0";
        store["b"]        = "2.0";
        store["n_pn"]     = "15";
        store["n_pnpi"]   = "20";
        store["n_cand"]   = "30";
        store["n_refine"] = "3";
        store["mean_r"]   = "2.0";
        store["mean_R"]   = "1.5";
        store["rel"]      = "0";
        store["macro"]    = "2";
        store["n_opt_S"]  = "15";
        store["n_opt_b"]  = "15";
        store["k_S"]      = "0.01";
        store["k_b"]      = "0.005";
        store["target"]   = "-2.225";
    }

    void load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t\r\n");
                size_t b = s.find_last_not_of (" \t\r\n");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(key); trim(val);
            if (!key.empty()) store[key] = val;
        }
    }

    void set(const std::string& kv) {
        auto eq = kv.find('=');
        if (eq == std::string::npos) return;
        store[kv.substr(0, eq)] = kv.substr(eq + 1);
    }

    long double ld(const std::string& k) const { return std::stold(store.at(k)); }
    int         i (const std::string& k) const { return std::stoi(store.at(k)); }
    size_t      sz(const std::string& k) const { return (size_t)std::stoul(store.at(k)); }

    void print() const {
        std::cerr << "[cfg] Parameters:\n";
        for (auto& kv : store) std::cerr << "  " << kv.first << " = " << kv.second << "\n";
    }
};

// ---------------------------------------------------------------------------
// Real-doubling trick for complex Hermitian GEP  H c = E N c
//
// Given H (complex Hermitian, n x n) and N (real symmetric positive-definite, n x n),
// build the 2n x 2n real symmetric GEP:
//
//   [ H_R  -H_I ] [c_R]     [N  0] [c_R]
//   [ H_I   H_R ] [c_I]  =E [0  N] [c_I]
//
// The n lowest eigenvalues of the 2n system equal the n eigenvalues of the
// complex Hermitian system (each appears twice in the doubled problem).
// We return only the lowest eigenvalue.
// ---------------------------------------------------------------------------
static long double solve_complex_gep(const cmatrix& H, const matrix& N) {
    size_t n = H.size1();
    assert(H.size2() == n && N.size1() == n && N.size2() == n);

    size_t n2 = 2 * n;
    matrix H_real(n2, n2);
    matrix N_real(n2, n2);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            long double hr =  H(i,j).real();
            long double hi =  H(i,j).imag();
            // Top-left block: H_R
            H_real(i,   j)   =  hr;
            // Top-right block: -H_I
            H_real(i,   j+n) = -hi;
            // Bottom-left block: +H_I
            H_real(i+n, j)   =  hi;
            // Bottom-right block: H_R
            H_real(i+n, j+n) =  hr;
            // N is block-diagonal (N, 0; 0, N)
            N_real(i,   j)   = N(i,j);
            N_real(i+n, j+n) = N(i,j);
            // Off-diagonal N blocks remain 0 (default-initialized)
        }
    }

    EigenResult sys = solve_generalized_eigensystem(H_real, N_real);
    if (sys.evals.size()==0) return std::numeric_limits<long double>::quiet_NaN();
    return sys.evals[0];   // lowest eigenvalue (appears twice; take [0])
}

// Convenience wrapper: if the H matrix is known to be purely real, call the
// original real solver directly.  Otherwise use the doubling trick.
static long double solve(const cmatrix& H_c, const matrix& N) {
    // Check whether imaginary parts are all zero
    bool pure_real = true;
    for (size_t i = 0; i < H_c.size1() && pure_real; ++i)
        for (size_t j = 0; j < H_c.size2() && pure_real; ++j)
            if (std::abs(H_c(i,j).imag()) > 1e-14L) pure_real = false;

    if (pure_real) {
        // Extract real part only and use the fast real solver
        matrix H_r(H_c.size1(), H_c.size2());
        for (size_t i = 0; i < H_c.size1(); ++i)
            for (size_t j = 0; j < H_c.size2(); ++j)
                H_r(i,j) = H_c(i,j).real();
        EigenResult sys = solve_generalized_eigensystem(H_r, N);
        if (sys.evals.size()==0) return std::numeric_limits<long double>::quiet_NaN();
        return sys.evals[0];
    }
    return solve_complex_gep(H_c, N);
}

// ---------------------------------------------------------------------------
// 9-sector system
//
// Sector layout:
//   sec 0      : bare PN
//   sec 1,2    : PN+pi0, proton-side Gaussians,  chan A / B
//   sec 3,4    : NN+pi+, proton-side Gaussians,  chan A / B
//   sec 5,6    : PP+pi-, neutron-side Gaussians,  chan A / B
//   sec 7,8    : PN+pi0, neutron-side Gaussians,  chan A / B
//
// "chan A" Gaussians have the pion-nucleon shift in the z direction (real
// spatial structure, connects to the sigma_z / r[0] component of W).
// "chan B" Gaussians have the shift in the x direction (connects to the
// real part of the sigma_pm / r[1,2] complex components).
//
// The W coupling is now computed as W_1 + W_2 for BOTH bare nucleons,
// properly summing over all three spherical components of sigma.r.  The
// full result is complex; the sectors merely provide different spatial
// configurations for the variational basis.
// ---------------------------------------------------------------------------
struct PionSystem {
    static const int N_SEC = 9;

    std::vector<gaus> basis[N_SEC];
    jacobian          jac_phys[5];      // jac_phys[p] for physical channel p=0..4
    long double       meson_mass[5];
    long double       S_pion;
    long double       b_pion;
    bool              relativistic;
    hamiltonian       H;

    // Map sector -> physical channel
    static int  phys(int sec)      { return (sec <= 0) ? 0 : (sec + 1) / 2; }
    // Odd clothed sectors use z-shift (chan A); even use x-shift (chan B)
    static bool is_chan_A(int sec) { return (sec > 0) && (sec % 2 == 1); }

    const jacobian& jac(int sec) const { return jac_phys[phys(sec)]; }

    size_t total_size() const {
        size_t n = 0; for (int s = 0; s < N_SEC; ++s) n += basis[s].size(); return n;
    }
    size_t offset(int sec) const {
        size_t off = 0; for (int s = 0; s < sec; ++s) off += basis[s].size(); return off;
    }

    // -----------------------------------------------------------------------
    // pion_coord: projection vector c such that c^T x = r_pion - r_particle_k
    // for particle index k in the clothed-sector Jacobi coordinates.
    //   k=0: pion relative to first nucleon  (e.g. proton side)
    //   k=1: pion relative to second nucleon (e.g. neutron side)
    // -----------------------------------------------------------------------
    vector pion_coord(int sec, int k) const {
        long double m0   = jac(sec).particles[0].mass;
        long double m1   = jac(sec).particles[1].mass;
        long double mtot = m0 + m1;
        vector c(2);
        if (k == 0) { c[0] = +m1/mtot; c[1] = 1.0L; }
        else        { c[0] = -m0/mtot; c[1] = 1.0L; }
        return c;
    }

    // Gaussian form-factor kernel:  Omega_{ij} = c_i c_j / b^2
    matrix Omega(int sec, const vector& c) const {
        size_t d = jac(sec).dim();
        matrix Om(d, d);
        long double inv_b2 = 1.0L / (b_pion * b_pion);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                Om(i,j) = inv_b2 * c[i] * c[j];
        return Om;
    }

    // -----------------------------------------------------------------------
    // make_couplings: build the NucleonCoupling list for a clothed sector.
    //
    // For EACH of the two bare nucleons (proton=0, neutron=1), apply_vertex
    // is called with the pion type for this sector.  Any forbidden vertex
    // returns an empty VertexTerm list and is skipped.
    //
    // This automatically handles:
    //   - PN+pi0: both proton (iso=+1) and neutron (iso=-1) contribute
    //   - NN+pi+: only the bare proton can emit pi+ (neutron term is empty)
    //   - PP+pi-: only the bare neutron can emit pi- (proton term is empty)
    //
    // The form-factor strength S_pion/b_pion is stored in each coupling.
    // -----------------------------------------------------------------------
    std::vector<NucleonCoupling> make_couplings(int sec) const {
        int p = phys(sec);
        if (p == 0) return {};

        const Pion pion = [p]() -> Pion {
            switch (p) {
                case 1: case 4: return Pion::PiZero();
                case 2:         return Pion::PiPlus();
                case 3:         return Pion::PiMinus();
                default:        return Pion::PiZero();
            }
        }();

        long double strength = S_pion / b_pion;

        Nucleon bare_nucleons[2] = { Nucleon::Proton(), Nucleon::Neutron() };
        std::vector<NucleonCoupling> couplings;

        for (int k = 0; k < 2; ++k) {
            auto terms = apply_vertex(bare_nucleons[k], pion);
            if (terms.empty()) continue;

            NucleonCoupling c;
            c.c_coord = pion_coord(sec, k);
            c.terms   = std::move(terms);
            c.strength = strength;
            couplings.push_back(std::move(c));
        }
        return couplings;
    }

    // -----------------------------------------------------------------------
    // Omega for sector: use the pion coordinate relative to the first
    // non-empty coupling's nucleon.  For sectors where only one nucleon
    // emits, this is unambiguous.  For PN+pi0 where both emit, we use a
    // geometric mean or simply the first coupling.
    // The form factor is the same exp(-r_k^2/b^2) absorbed into Omega.
    // We use the first coupling's c_coord to build Omega (the coupling
    // that dominates spatially, i.e., the one whose Gaussians are centred
    // on that nucleon's side).
    // -----------------------------------------------------------------------
    matrix make_omega(int sec) const {
        auto couplings = make_couplings(sec);
        if (couplings.empty()) return matrix(jac(sec).dim(), jac(sec).dim());
        return Omega(sec, couplings[0].c_coord);
    }

    // -----------------------------------------------------------------------
    // calc_row: compute one new row/column of H_c and N for a trial Gaussian
    // -----------------------------------------------------------------------
    void calc_row(int target_sec, const gaus& g_target,
                  std::vector<cld>& h_row,
                  std::vector<long double>& n_row,
                  cld& h_diag, long double& n_diag) const {
        size_t N = total_size();
        h_row.assign(N, cld(0));
        n_row.assign(N, 0.0L);

        n_diag = overlap(g_target, g_target);
        long double k_diag = 0.0L;
        const jacobian& j_tgt = jac(target_sec);
        for (size_t coord = 0; coord < j_tgt.dim(); ++coord)
            k_diag += relativistic
                ? H.K_rel(g_target, g_target, j_tgt.c(coord), j_tgt.mu(coord))
                : H.K_cla(g_target, g_target, j_tgt.c(coord), j_tgt.mu(coord));
        h_diag = cld(k_diag + meson_mass[phys(target_sec)] * n_diag, 0.0L);

        for (int sec = 0; sec < N_SEC; ++sec) {
            size_t n_s = basis[sec].size();
            if (n_s == 0) continue;
            size_t off = offset(sec);

            for (size_t i = 0; i < n_s; ++i) {
                cld h_val(0); long double n_val = 0.0L;

                if (sec == target_sec) {
                    // Same-sector: real kinetic + overlap
                    n_val = overlap(basis[sec][i], g_target);
                    long double k_val = 0.0L;
                    for (size_t coord = 0; coord < j_tgt.dim(); ++coord)
                        k_val += relativistic
                            ? H.K_rel(basis[sec][i], g_target, j_tgt.c(coord), j_tgt.mu(coord))
                            : H.K_cla(basis[sec][i], g_target, j_tgt.c(coord), j_tgt.mu(coord));
                    h_val = cld(k_val + meson_mass[phys(target_sec)] * n_val, 0.0L);

                } else if (sec == 0 || target_sec == 0) {
                    // Off-diagonal: W coupling between bare (sec=0) and clothed
                    int         clothed_sec = (sec == 0) ? target_sec : sec;
                    const gaus& g_bare  = (sec == 0) ? basis[0][i]  : g_target;
                    const gaus& g_cloth = (sec == 0) ? g_target     : basis[sec][i];
                    auto        couplings = make_couplings(clothed_sec);
                    matrix      Om = make_omega(clothed_sec);
                    
                    cld w = H.W(g_bare, g_cloth, Om, couplings); // Always <bare | W | cloth>
                    
                    // Force h_val to ALWAYS represent <basis_i | H | trial>
                    if (target_sec == 0) {
                        // Trial is bare, basis_i is clothed: w = <trial | W | basis_i>.
                        // We need <basis_i | H | trial> = w*
                        h_val = std::conj(w);
                    } else {
                        // Trial is clothed, basis_i is bare: w = <basis_i | W | trial>.
                        h_val = w;
                    }
                    n_val = 0.0L;
                }
                h_row[off + i] = h_val;
                n_row[off + i] = n_val;
            }
        }
    }

    // -----------------------------------------------------------------------
    // insert_matrix: expand H_c and N by one row/column
    // -----------------------------------------------------------------------
    void insert_matrix(const cmatrix& H_old, const matrix& N_old,
                       int target_sec, const gaus& trial,
                       cmatrix& H_new, matrix& N_new) const {
        size_t N = H_old.size1();
        cmatrix H_temp(N+1, N+1);
        matrix  N_temp(N+1, N+1);

        std::vector<cld>          h_row;
        std::vector<long double>  n_row;
        cld h_diag; long double n_diag;
        calc_row(target_sec, trial, h_row, n_row, h_diag, n_diag);

        size_t ins = offset(target_sec) + basis[target_sec].size();
        auto remap = [&](size_t old_i) { return (old_i >= ins) ? old_i + 1 : old_i; };

        for (size_t i = 0; i < N; ++i) {
            size_t ni = remap(i);
            for (size_t j = 0; j < N; ++j) {
                size_t nj = remap(j);
                H_temp(ni,nj) = H_old(i,j);
                N_temp(ni,nj) = N_old(i,j);
            }
            H_temp(ni, ins) = h_row[i];
            H_temp(ins, ni) = std::conj(h_row[i]);  // Hermitian symmetry
            N_temp(ni, ins) = N_temp(ins, ni) = n_row[i];
        }
        H_temp(ins, ins) = h_diag;
        N_temp(ins, ins) = n_diag;
        H_new = H_temp;
        N_new = N_temp;
    }

    // -----------------------------------------------------------------------
    // replace_matrix: swap row/column target_k in target_sec
    // -----------------------------------------------------------------------
    void replace_matrix(const cmatrix& H_old, const matrix& N_old,
                        int target_sec, size_t target_k, const gaus& trial,
                        cmatrix& H_new, matrix& N_new) const {
        H_new = H_old; N_new = N_old;

        std::vector<cld>         h_row;
        std::vector<long double> n_row;
        cld h_diag; long double n_diag;
        calc_row(target_sec, trial, h_row, n_row, h_diag, n_diag);

        size_t gi = offset(target_sec) + target_k;
        size_t tot = H_old.size1();
        for (size_t i = 0; i < tot; ++i) {
            if (i == gi) continue;
            H_new(i,  gi) = h_row[i];
            H_new(gi, i)  = std::conj(h_row[i]);
            N_new(i,  gi) = N_new(gi, i) = n_row[i];
        }
        H_new(gi, gi) = h_diag;
        N_new(gi, gi) = n_diag;
    }

    // -----------------------------------------------------------------------
    // build_full: construct H_c and N from scratch
    // -----------------------------------------------------------------------
    void build_full(cmatrix& H_tot, matrix& N_tot) const {
        size_t Ntot = total_size();
        H_tot = cmatrix(Ntot, Ntot);
        N_tot = matrix (Ntot, Ntot);

        // Diagonal blocks (real kinetic energy + meson mass)
        for (int sec = 0; sec < N_SEC; ++sec) {
            size_t n_s = basis[sec].size();
            size_t off = offset(sec);
            int    p   = phys(sec);
            const jacobian& j = jac(sec);
            for (size_t i = 0; i < n_s; ++i)
                for (size_t k = 0; k < n_s; ++k) {
                    long double nij = overlap(basis[sec][i], basis[sec][k]);
                    N_tot(off+i, off+k) = nij;
                    long double kij = 0.0L;
                    for (size_t coord = 0; coord < j.dim(); ++coord)
                        kij += relativistic
                            ? H.K_rel(basis[sec][i], basis[sec][k], j.c(coord), j.mu(coord))
                            : H.K_cla(basis[sec][i], basis[sec][k], j.c(coord), j.mu(coord));
                    H_tot(off+i, off+k) = cld(kij + meson_mass[p]*nij, 0.0L);
                }
        }

        // Off-diagonal W blocks: bare (sec=0) <-> clothed (sec=1..8)
        size_t n0 = basis[0].size();
        for (int sec = 1; sec < N_SEC; ++sec) {
            size_t n_s   = basis[sec].size();
            size_t off_s = offset(sec);
            auto   couplings = make_couplings(sec);
            matrix Om        = make_omega(sec);

            for (size_t i = 0; i < n0; ++i)
                for (size_t k = 0; k < n_s; ++k) {
                    cld w = H.W(basis[0][i], basis[sec][k], Om, couplings);
                    H_tot(i,        off_s + k) = w;
                    H_tot(off_s+k,  i)         = std::conj(w);  // Hermitian
                }
        }
    }
};

// ---------------------------------------------------------------------------
// Basis generation helpers
// ---------------------------------------------------------------------------
static gaus make_swave(size_t dim, long double mean_r) {
    gaus g(dim, mean_r, 0.0L);
    g.zero_shifts();
    return g;
}

// P-wave Gaussian: only the pion Jacobi coordinate (last row of s) carries
// a non-zero shift.
//   chan_A -> shift in z-direction  (connects to r[0]=z, real)
//   chan_B -> shift in x-direction  (connects to Re[r[1,2]], and imaginary
//             part comes automatically from the spherical formula in hamiltonian.h)
static gaus make_pwave(size_t dim, long double mean_r, long double mean_R, bool chan_A) {
    gaus g(dim, mean_r, mean_R);
    // Zero all shifts except the last Jacobi coordinate
    for (size_t i = 0; i < dim - 1; ++i)
        for (size_t k = 0; k < 3; ++k)
            g.s(i, k) = 0.0L;
    if (dim >= 2) {
        if (chan_A) {
            // z-shift only (index 2): zero x and y
            g.s(dim-1, 0) = 0.0L;
            g.s(dim-1, 1) = 0.0L;
        } else {
            // x-shift only (index 0): zero y and z
            g.s(dim-1, 1) = 0.0L;
            g.s(dim-1, 2) = 0.0L;
        }
    }
    return g;
}

// ---------------------------------------------------------------------------
// SVM greedy basis selection
// ---------------------------------------------------------------------------
static long double run_selection(PionSystem& sys, const size_t* target_n,
                                 int n_cand, long double mean_r, long double mean_R) {
    for (int s = 0; s < PionSystem::N_SEC; ++s) sys.basis[s].clear();

    size_t total_slots = 0;
    for (int s = 0; s < PionSystem::N_SEC; ++s) total_slots += target_n[s];

    cmatrix H_master; matrix N_master;
    long double E_cur = std::numeric_limits<long double>::quiet_NaN();
    size_t slots_filled = 0;
    int consecutive_fails = 0;
    const int max_fails = 15;

    while (slots_filled < total_slots) {
        // Pick the sector furthest from its target
        int    add_sec  = 0;
        double max_frac = -1.0;
        for (int s = 0; s < PionSystem::N_SEC; ++s) {
            if (sys.basis[s].size() >= target_n[s]) continue;
            double frac = 1.0 - (double)sys.basis[s].size() / target_n[s];
            if (frac > max_frac) { max_frac = frac; add_sec = s; }
        }

        bool   bare  = (add_sec == 0);
        bool   chanA = PionSystem::is_chan_A(add_sec);
        size_t dim   = sys.jac(add_sec).dim();

        long double global_best_E = std::numeric_limits<long double>::infinity();
        gaus        global_best_g;

        std::vector<gaus> candidates(n_cand);
        for (int c = 0; c < n_cand; ++c)
            candidates[c] = bare ? make_swave(dim, mean_r)
                                 : make_pwave(dim, mean_r, mean_R, chanA);

        #pragma omp parallel
        {
            long double local_best_E = std::numeric_limits<long double>::infinity();
            gaus        local_best_g;

            #pragma omp for
            for (int c = 0; c < n_cand; ++c) {
                const gaus& trial = candidates[c];

                // Linear dependence filter
                bool dep = false;
                long double n_tt = overlap(trial, trial);
                for (size_t i = 0; i < sys.basis[add_sec].size(); ++i) {
                    long double n_ii = overlap(sys.basis[add_sec][i], sys.basis[add_sec][i]);
                    long double n_ti = std::abs(overlap(trial, sys.basis[add_sec][i]));
                    if (n_ti / std::sqrt(n_tt * n_ii) > 0.95L) { dep = true; break; }
                }
                if (dep) continue;

                cmatrix H_trial; matrix N_trial;
                sys.insert_matrix(H_master, N_master, add_sec, trial, H_trial, N_trial);
                long double E = solve(H_trial, N_trial);
                if (!std::isnan(E) && E < local_best_E) { local_best_E = E; local_best_g = trial; }
            }

            #pragma omp critical
            {
                if (local_best_E < global_best_E) { global_best_E = local_best_E; global_best_g = local_best_g; }
            }
        }

        if (std::isinf(global_best_E)) {
            if (++consecutive_fails >= max_fails) {
                std::cout << "\n  [!] Basis naturally saturated at size " << slots_filled << "\n";
                break;
            }
            std::cerr << "\n  [Warning] All candidates rejected. Retry " << consecutive_fails << "/" << max_fails << "\n";
            continue;
        }

        consecutive_fails = 0;
        sys.insert_matrix(H_master, N_master, add_sec, global_best_g, H_master, N_master);
        sys.basis[add_sec].push_back(global_best_g);
        slots_filled++;
        E_cur = global_best_E;

        std::cout << "\r  [SVM] Building basis... " << std::setw(3) << slots_filled
                  << "/" << total_slots
                  << " | E = " << std::fixed << std::setprecision(5) << global_best_E
                  << std::flush;
    }
    std::cout << "\n";
    return E_cur;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Params cfg;
    bool file_loaded = false;
    for (int k = 1; k < argc; ++k) {
        std::string arg(argv[k]);
        if (arg.find('=') == std::string::npos) { cfg.load_file(arg); file_loaded = true; }
    }
    if (!file_loaded) cfg.load_file("pion.cfg");
    for (int k = 1; k < argc; ++k) {
        std::string arg(argv[k]);
        if (arg.find('=') != std::string::npos) cfg.set(arg);
    }
    cfg.print();

    long double       S_pion   = cfg.ld("S");
    long double       b_pion   = cfg.ld("b");
    const size_t      n_pn     = cfg.sz("n_pn");
    const size_t      n_pnpi   = cfg.sz("n_pnpi");
    const int         n_cand   = cfg.i("n_cand");
    const int         n_refine = cfg.i("n_refine");
    const long double mean_r   = cfg.ld("mean_r");
    const long double mean_R   = cfg.ld("mean_R");
    const bool        rel      = (cfg.i("rel") != 0);
    const int         n_macro  = cfg.i("macro");
    const int         n_opt_S  = cfg.i("n_opt_S");
    const int         n_opt_b  = cfg.i("n_opt_b");
    const long double k_S      = cfg.ld("k_S");
    long double       k_b      = cfg.ld("k_b");
    const long double E_target = cfg.ld("target");

    std::cout << "===============================================\n";
    std::cout << "[main] Deuteron + explicit pion  (9-sector SVM)\n";
    std::cout << "[main] W = W_1 + W_2, complex Hermitian H\n";
    std::cout << "===============================================\n";

    // ------------------------------------------------------------------
    // Particles
    // ------------------------------------------------------------------
    Nucleon proton  = Nucleon::Proton();
    Nucleon neutron = Nucleon::Neutron();
    Pion    pi0     = Pion::PiZero();
    Pion    piplus  = Pion::PiPlus();
    Pion    piminus = Pion::PiMinus();

    // ------------------------------------------------------------------
    // Print vertex table so we can verify the spin-isospin algebra
    // ------------------------------------------------------------------
    std::cerr << "\n[Vertex table]  (tau.pi)(sigma.r) -- coeff[m] per spherical component\n";
    std::cerr << "  Format: nucleon + pion -> [m=0 (z), m=1 (x+iy), m=2 (x-iy)]\n";
    for (auto& [nuc, pi] : std::vector<std::pair<Nucleon,Pion>>{
            {proton, pi0}, {proton, piplus}, {neutron, piminus}, {neutron, pi0}}) {
        auto terms = apply_vertex(nuc, pi);
        std::cerr << "  " << nuc.name << " + " << pi.name << " :";
        if (terms.empty()) { std::cerr << "  (forbidden)\n"; continue; }
        for (auto& t : terms)
            std::cerr << "  [" << t.coeff[0] << ", " << t.coeff[1]
                      << ", " << t.coeff[2] << "] bra_sz=" << t.bra_sz;
        std::cerr << "\n";
    }

    // ------------------------------------------------------------------
    // Build PionSystem
    // ------------------------------------------------------------------
    PionSystem sys;
    sys.relativistic = rel;
    sys.S_pion       = S_pion;
    sys.b_pion       = b_pion;

    sys.jac_phys[0] = jacobian({proton,  neutron});
    sys.jac_phys[1] = jacobian({proton,  neutron, pi0});
    sys.jac_phys[2] = jacobian({neutron, neutron, piplus});
    sys.jac_phys[3] = jacobian({proton,  proton,  piminus});
    sys.jac_phys[4] = jacobian({proton,  neutron, pi0});    // same particles as p=1

    sys.meson_mass[0] = 0.0L;
    sys.meson_mass[1] = pi0.mass;
    sys.meson_mass[2] = piplus.mass;
    sys.meson_mass[3] = piminus.mass;
    sys.meson_mass[4] = pi0.mass;

    // Target basis sizes
    const size_t target_n[PionSystem::N_SEC] = {
        n_pn,     // sec 0: bare PN
        n_pnpi,   // sec 1: PN+pi0, z-shift, chan A
        n_pnpi,   // sec 2: PN+pi0, x-shift, chan B
        n_pnpi,   // sec 3: NN+pi+, chan A
        n_pnpi,   // sec 4: NN+pi+, chan B
        n_pnpi,   // sec 5: PP+pi-, chan A
        n_pnpi,   // sec 6: PP+pi-, chan B
        n_pnpi,   // sec 7: PN+pi0 neutron side, chan A
        n_pnpi,   // sec 8: PN+pi0 neutron side, chan B
    };

    cmatrix H_cur; matrix N_cur;
    long double E_best = 0.0L;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Phase 0: Generating Initial Basis ===\n";
    run_selection(sys, target_n, n_cand, mean_r, mean_R);

    for (int macro = 1; macro <= n_macro; ++macro) {
        std::cout << "\n====== MACRO CYCLE " << macro << " / " << n_macro << " ======\n";

        // Tune S
        std::cout << "\n=== Phase 0a: Tuning S ===\n";
        long double E_cur = std::numeric_limits<long double>::quiet_NaN();
        long double cur_kS = std::abs(k_S), prev_errS = 0.0L;
        for (int it = 0; it < n_opt_S; ++it) {
            sys.S_pion = S_pion; sys.b_pion = b_pion;
            sys.build_full(H_cur, N_cur);
            E_cur = solve(H_cur, N_cur);
            long double err = E_target - E_cur;
            std::cout << "\r  S it=" << std::setw(2) << it
                      << "  S=" << std::setw(9) << S_pion
                      << "  b=" << std::setw(6) << b_pion
                      << "  E=" << std::setw(10) << E_cur
                      << "  err=" << std::setw(9) << err << " MeV" << std::flush;
            if (std::abs(err) < 0.001L) { std::cout << "\n --- [S converged] ---\n"; break; }
            if (it > 0 && err * prev_errS < 0) cur_kS *= 0.5L;
            long double step = std::abs(cur_kS * err);
            S_pion += (err > 0) ? +step : -step;
            if (S_pion < 0.1L) S_pion = 0.1L;
            prev_errS = err;
        }

        // Tune b
        std::cout << "\n=== Phase 0b: Tuning b ===\n";
        long double prev_err = 0.0L;
        for (int it = 0; it < n_opt_b; ++it) {
            sys.S_pion = S_pion; sys.b_pion = b_pion;
            sys.build_full(H_cur, N_cur);
            E_cur = solve(H_cur, N_cur);
            long double err = E_target - E_cur;
            std::cout << "\r  b it=" << std::setw(2) << it
                      << "  S=" << std::setw(9) << S_pion
                      << "  b=" << std::setw(6) << b_pion
                      << "  E=" << std::setw(10) << E_cur
                      << "  err=" << std::setw(9) << err << " MeV" << std::flush;
            if (std::abs(err) < 0.001L) { std::cout << "\n --- [b converged] ---\n"; break; }
            if (it > 0 && err * prev_err < 0) { k_b *= 0.5L; }
            b_pion += k_b * err;
            prev_err = err;
        }

        // Full basis selection
        if (macro == 1 && (n_opt_S + n_opt_b) > 0) {
            std::cout << "\n=== Phase 1: Full Selection (S=" << S_pion << " b=" << b_pion << ") ===\n";
            E_best = run_selection(sys, target_n, n_cand, mean_r, mean_R);
        } else {
            sys.build_full(H_cur, N_cur);
            E_best = solve(H_cur, N_cur);
            std::cout << "\n[Phase 1 skipped. E=" << E_best << " MeV]\n";
        }

        // Refinement
        std::cout << "\n=== Phase 2: Refining ===\n";
        cmatrix H_master; matrix N_master;
        sys.build_full(H_master, N_master);

        for (int cycle = 1; cycle <= n_refine; ++cycle) {
            long double E_start = E_best;
            size_t state_idx = 0;
            for (int sec = 0; sec < PionSystem::N_SEC; ++sec) {
                bool   bare  = (sec == 0);
                bool   chanA = PionSystem::is_chan_A(sec);
                size_t dim   = sys.jac(sec).dim();

                for (size_t k = 0; k < sys.basis[sec].size(); ++k) {
                    state_idx++;
                    long double global_best_E = E_best;
                    gaus        global_best_g = sys.basis[sec][k];

                    std::vector<gaus> candidates(n_cand);
                    for (int c = 0; c < n_cand; ++c)
                        candidates[c] = bare ? make_swave(dim, mean_r)
                                             : make_pwave(dim, mean_r, mean_R, chanA);

                    #pragma omp parallel
                    {
                        long double local_best_E = global_best_E;
                        gaus        local_best_g  = global_best_g;

                        #pragma omp for
                        for (int c = 0; c < n_cand; ++c) {
                            const gaus& trial = candidates[c];
                            if (is_linearly_dependent(trial, sys.basis[sec], k)) continue;
                            cmatrix H_trial; matrix N_trial;
                            sys.replace_matrix(H_master, N_master, sec, k, trial, H_trial, N_trial);
                            long double E = solve(H_trial, N_trial);
                            if (!std::isnan(E) && E < local_best_E) { local_best_E = E; local_best_g = trial; }
                        }
                        #pragma omp critical
                        {
                            if (local_best_E < global_best_E) { global_best_E = local_best_E; global_best_g = local_best_g; }
                        }
                    }

                    if (global_best_E < E_best) {
                        sys.basis[sec][k] = global_best_g;
                        sys.replace_matrix(H_master, N_master, sec, k, global_best_g, H_master, N_master);
                        E_best = global_best_E;
                    }
                    std::cout << "\r  [Refine] cycle=" << cycle
                              << " state=" << std::setw(3) << state_idx
                              << " E=" << std::fixed << std::setprecision(5) << E_best
                              << std::flush;
                }
            }
            std::cout << "\n";
            if (E_best >= E_start - 0.0001L) break;
        }
    }

    // ------------------------------------------------------------------
    // Final observables
    // ------------------------------------------------------------------
    sys.S_pion = S_pion; sys.b_pion = b_pion;
    cmatrix H_final; matrix N_final;
    sys.build_full(H_final, N_final);

    // Use real-doubled solver to get the ground state
    long double E_gs = solve(H_final, N_final);

    // Norm probabilities: use real part of H (diagonal blocks are real)
    // For sector weights we use the overlap matrix N which is always real.
    // We solve the real symmetric GEP on the real part to get the eigenvector.
    long double prob_bare = 0.0L, prob_pion = 0.0L, r2_exp = 0.0L;
    {
        matrix H_r_final(H_final.size1(), H_final.size2());
        for (size_t i = 0; i < H_final.size1(); ++i)
            for (size_t j = 0; j < H_final.size2(); ++j)
                H_r_final(i,j) = H_final(i,j).real();

        EigenResult res = solve_generalized_eigensystem(H_r_final, N_final);
        if (res.evecs.size1() > 0) {
            const auto& cv = res.evecs;
            // Bare sector probability
            for (size_t i = 0; i < sys.basis[0].size(); ++i)
                for (size_t j = 0; j < sys.basis[0].size(); ++j)
                    prob_bare += cv(i,0) * N_final(i,j) * cv(j,0);
            // Pion sector probability
            for (int sec = 1; sec < PionSystem::N_SEC; ++sec) {
                size_t off = sys.offset(sec);
                for (size_t i = 0; i < sys.basis[sec].size(); ++i)
                    for (size_t j = 0; j < sys.basis[sec].size(); ++j)
                        prob_pion += cv(off+i,0) * N_final(off+i, off+j) * cv(off+j,0);
            }
            // <r^2>
            for (size_t i = 0; i < sys.basis[0].size(); ++i)
                for (size_t j = 0; j < sys.basis[0].size(); ++j)
                    r2_exp += cv(i,0) * sys.H.R2_matrix_element(sys.basis[0][i], sys.basis[0][j]) * cv(j,0);
        }
    }
    long double r_c = std::sqrt(std::abs(r2_exp) / 4.0L);

    std::cout << "\n==========================================\n";
    std::cout << "  Ground-state energy : " << E_gs << " MeV\n";
    std::cout << "  Relativistic        : " << (rel ? "True" : "False") << "\n";
    std::cout << "  S_pion              : " << S_pion << " MeV\n";
    std::cout << "  b_pion              : " << b_pion << " fm\n";
    std::cout << "  --------------------------------------\n";
    std::cout << "  <r^2>_pn            : " << r2_exp << " fm^2\n";
    std::cout << "  Charge radius       : " << r_c    << " fm\n";
    std::cout << "  --------------------------------------\n";
    std::cout << "  Bare PN sector      : " << prob_bare*100.0L << " %\n";
    std::cout << "  Pion cloud sectors  : " << prob_pion*100.0L << " %\n";
    std::cout << "==========================================\n";

    return 0;
}