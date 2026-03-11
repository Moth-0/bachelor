#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
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
        store["n_pnπ"]    = "20";
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
                size_t b = s.find_last_not_of(" \t\r\n");
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
// 9-sector system
//
// Sector layout:
//   sec 0          : bare PN (proton + neutron, no pion)
//   sec 1, 2       : PN + pi0,  proton  emits pi0  (chan A / chan B)
//   sec 3, 4       : NN + pi+,  proton  emits pi+  (chan A / chan B)
//   sec 5, 6       : PP + pi-,  neutron emits pi-  (chan A / chan B)
//   sec 7, 8       : PN + pi0,  neutron emits pi0  (chan A / chan B)
//
// Physical channel index p = phys(sec):
//   p=0  bare PN
//   p=1  PN+pi0  (proton emits,  iso_weight = +1)
//   p=2  NN+pi+  (proton emits,  iso_weight = +sqrt2)
//   p=3  PP+pi-  (neutron emits, iso_weight = +sqrt2)
//   p=4  PN+pi0  (neutron emits, iso_weight = -1)
//
// The neutron-emitting pi0 sector (p=4) has the same particle content as p=1
// but the pion coordinate is measured relative to the neutron (emitting_idx=1).
// Its isospin coefficient is -1, giving destructive interference with p=1 in
// the W matrix and contributing a spatially distinct pion-cloud configuration.
// ---------------------------------------------------------------------------
struct PionSystem {
    static const int N_SEC = 9;

    std::vector<gaus> basis[N_SEC];
    jacobian          jac_phys[5];    // jac_phys[p] for physical channel p = 0..4
    long double       meson_mass[5];
    long double       iso_weight[5];
    long double       S_pion;
    long double       b_pion;
    bool              relativistic;
    hamiltonian       H;

    // Map sector -> physical channel index
    // sec 0     -> 0
    // sec 1,2   -> 1
    // sec 3,4   -> 2
    // sec 5,6   -> 3
    // sec 7,8   -> 4
    static int  phys(int sec)      { return (sec <= 0) ? 0 : (sec + 1) / 2; }

    // Odd-numbered clothed sectors are chan_A (sigma_z / spin-preserving)
    // Even-numbered clothed sectors are chan_B (sigma_perp / spin-flipping)
    static bool is_chan_A(int sec) { return (sec > 0) && (sec % 2 == 1); }

    const jacobian& jac(int sec) const { return jac_phys[phys(sec)]; }

    // Returns true if the pion in sector sec is emitted by the neutron (particle index 1).
    // Used to select the correct pion_coord projection vector.
    static bool neutron_emits(int p) { return (p == 3 || p == 4); }

    // Projection vector c such that c^T x = r_{pion} - r_{emitter}.
    // emitting_idx=0: proton is the emitter (particle 0 in the Jacobi).
    // emitting_idx=1: neutron is the emitter (particle 1 in the Jacobi).
    vector pion_coord(int sec, int emitting_idx) const {
        long double m0   = jac(sec).particles[0].mass;
        long double m1   = jac(sec).particles[1].mass;
        long double mtot = m0 + m1;
        vector c(2);
        if (emitting_idx == 0) { c[0] = +m1 / mtot; c[1] = 1.0; }
        else                   { c[0] = -m0 / mtot; c[1] = 1.0; }
        return c;
    }

    // Gaussian form-factor kernel: Omega_{ij} = c_i c_j / b^2
    matrix Omega(int sec, const vector& c_coord) const {
        size_t d = jac(sec).dim();
        matrix Om(d, d);
        long double inv_b2 = 1.0 / (b_pion * b_pion);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                Om(i, j) = inv_b2 * c_coord[i] * c_coord[j];
        return Om;
    }

    size_t total_size() const {
        size_t n = 0;
        for (int s = 0; s < N_SEC; ++s) n += basis[s].size();
        return n;
    }

    size_t offset(int sec) const {
        size_t off = 0;
        for (int s = 0; s < sec; ++s) off += basis[s].size();
        return off;
    }

    // Build the beta vector for the W operator from the spin and isospin
    // coefficients of the emitting nucleon.
    //   Chan A (is_chan_A): beta[2] = iso_weight * strength * spin.coef_z
    //   Chan B (!is_chan_A): beta[0] = iso_weight * strength * spin.coef_perp
    // The spin coefficients come from Nucleon::spin_dot_r_coefficients() so
    // that all sigma algebra lives in particle.h alongside the tau algebra.
    vector make_beta(int sec) const {
        int  p     = phys(sec);
        bool chanA = is_chan_A(sec);

        // Choose the emitting nucleon to read the spin coupling from.
        // For pp+pi- (p==3) and pn+pi0-neutron-emits (p==4) the neutron emits.
        Nucleon emitter = neutron_emits(p) ? Nucleon::Neutron() : Nucleon::Proton();
        Nucleon::SpinCoupling sc = emitter.spin_dot_r_coefficients();

        long double base = iso_weight[p] * (S_pion / b_pion);
        vector beta(3);
        if (chanA) beta[2] = base * sc.coef_z;
        else       beta[0] = base * sc.coef_perp;
        return beta;
    }

    // Compute a new row/column of H and N for a trial Gaussian in target_sec.
    void calc_row(int target_sec, const gaus& g_target,
                  std::vector<long double>& h_row, std::vector<long double>& n_row,
                  long double& h_diag, long double& n_diag) const {
        size_t N = total_size();
        h_row.assign(N, 0.0);
        n_row.assign(N, 0.0);

        n_diag = overlap(g_target, g_target);
        long double k_diag = 0.0;
        const jacobian& j_tgt = jac(target_sec);
        for (size_t coord = 0; coord < j_tgt.dim(); ++coord) {
            if (relativistic) k_diag += H.K_rel(g_target, g_target, j_tgt.c(coord), j_tgt.mu(coord));
            else              k_diag += H.K_cla(g_target, g_target, j_tgt.c(coord), j_tgt.mu(coord));
        }
        h_diag = k_diag + meson_mass[phys(target_sec)] * n_diag;

        for (int sec = 0; sec < N_SEC; ++sec) {
            size_t n_s = basis[sec].size();
            if (n_s == 0) continue;
            size_t off = offset(sec);

            for (size_t i = 0; i < n_s; ++i) {
                long double h_val = 0.0, n_val = 0.0;
                if (sec == target_sec) {
                    n_val = overlap(basis[sec][i], g_target);
                    long double k_val = 0.0;
                    for (size_t coord = 0; coord < j_tgt.dim(); ++coord) {
                        if (relativistic) k_val += H.K_rel(basis[sec][i], g_target, j_tgt.c(coord), j_tgt.mu(coord));
                        else              k_val += H.K_cla(basis[sec][i], g_target, j_tgt.c(coord), j_tgt.mu(coord));
                    }
                    h_val = k_val + meson_mass[phys(target_sec)] * n_val;
                } else if (sec == 0 || target_sec == 0) {
                    int          clothed_sec = (sec == 0) ? target_sec : sec;
                    const gaus&  g_bare      = (sec == 0) ? basis[0][i] : g_target;
                    const gaus&  g_cloth     = (sec == 0) ? g_target    : basis[sec][i];
                    int          p           = phys(clothed_sec);
                    vector       c_coord     = pion_coord(clothed_sec, neutron_emits(p) ? 1 : 0);
                    matrix       Om          = Omega(clothed_sec, c_coord);
                    vector       beta        = make_beta(clothed_sec);
                    h_val = H.W(g_bare, g_cloth, Om, c_coord, 0.0, beta);
                    n_val = 0.0;
                }
                h_row[off + i] = h_val;
                n_row[off + i] = n_val;
            }
        }
    }

    // Expand H and N by one row/column for a new basis function in target_sec.
    void insert_matrix(const matrix& H_old, const matrix& N_old,
                       int target_sec, const gaus& trial,
                       matrix& H_new, matrix& N_new) const {
        size_t N = H_old.size1();
        matrix H_temp(N + 1, N + 1);
        matrix N_temp(N + 1, N + 1);

        std::vector<long double> h_row, n_row;
        long double h_diag, n_diag;
        calc_row(target_sec, trial, h_row, n_row, h_diag, n_diag);

        size_t ins = offset(target_sec) + basis[target_sec].size();
        auto remap = [&](size_t old_i) { return (old_i >= ins) ? old_i + 1 : old_i; };

        for (size_t i = 0; i < N; ++i) {
            size_t ni = remap(i);
            for (size_t j = 0; j < N; ++j) {
                size_t nj = remap(j);
                H_temp(ni, nj) = H_old(i, j);
                N_temp(ni, nj) = N_old(i, j);
            }
            H_temp(ni, ins) = H_temp(ins, ni) = h_row[i];
            N_temp(ni, ins) = N_temp(ins, ni) = n_row[i];
        }
        H_temp(ins, ins) = h_diag;
        N_temp(ins, ins) = n_diag;
        H_new = H_temp;
        N_new = N_temp;
    }

    // Replace row/column target_k in target_sec with a new basis function.
    void replace_matrix(const matrix& H_old, const matrix& N_old,
                        int target_sec, size_t target_k, const gaus& trial,
                        matrix& H_new, matrix& N_new) const {
        size_t N = H_old.size1();
        matrix H_temp = H_old;
        matrix N_temp = N_old;

        std::vector<long double> h_row, n_row;
        long double h_diag, n_diag;
        calc_row(target_sec, trial, h_row, n_row, h_diag, n_diag);

        size_t gi = offset(target_sec) + target_k;
        for (size_t i = 0; i < N; ++i) {
            if (i == gi) continue;
            H_temp(i, gi) = H_temp(gi, i) = h_row[i];
            N_temp(i, gi) = N_temp(gi, i) = n_row[i];
        }
        H_temp(gi, gi) = h_diag;
        N_temp(gi, gi) = n_diag;
        H_new = H_temp;
        N_new = N_temp;
    }

    // Build full H and N from scratch.
    void build_full(matrix& H_tot, matrix& N_tot) const {
        size_t N = total_size();
        H_tot = matrix(N, N);
        N_tot = matrix(N, N);

        // Diagonal blocks: kinetic energy + meson rest mass
        for (int sec = 0; sec < N_SEC; ++sec) {
            size_t n_s = basis[sec].size();
            size_t off = offset(sec);
            int p = phys(sec);
            const jacobian& j = jac(sec);
            for (size_t i = 0; i < n_s; ++i)
                for (size_t k = 0; k < n_s; ++k) {
                    long double nij = overlap(basis[sec][i], basis[sec][k]);
                    N_tot(off + i, off + k) = nij;
                    long double kij = 0.0;
                    for (size_t coord = 0; coord < j.dim(); ++coord)
                        kij += relativistic
                            ? H.K_rel(basis[sec][i], basis[sec][k], j.c(coord), j.mu(coord))
                            : H.K_cla(basis[sec][i], basis[sec][k], j.c(coord), j.mu(coord));
                    H_tot(off + i, off + k) = kij + meson_mass[p] * nij;
                }
        }

        // Off-diagonal blocks: W coupling between bare (sec=0) and each clothed sector
        size_t n0 = basis[0].size();
        for (int sec = 1; sec < N_SEC; ++sec) {
            size_t n_s   = basis[sec].size();
            size_t off_s = offset(sec);
            int    p     = phys(sec);
            vector c_coord = pion_coord(sec, neutron_emits(p) ? 1 : 0);
            matrix Om      = Omega(sec, c_coord);
            vector beta    = make_beta(sec);
            for (size_t i = 0; i < n0; ++i)
                for (size_t k = 0; k < n_s; ++k) {
                    long double w = H.W(basis[0][i], basis[sec][k], Om, c_coord, 0.0, beta);
                    H_tot(i, off_s + k) = H_tot(off_s + k, i) = w;
                }
        }
    }
};

// ---------------------------------------------------------------------------
// Solver helpers
// ---------------------------------------------------------------------------
static long double solve(const matrix& H_tot, const matrix& N_tot) {
    EigenResult sys = solve_generalized_eigensystem(H_tot, N_tot);
    if (sys.evals.size() == 0) return std::numeric_limits<long double>::quiet_NaN();
    return sys.evals[0];
}

static gaus make_swave(size_t dim, long double mean_r) {
    gaus g(dim, mean_r, 0.0);
    g.zero_shifts();
    return g;
}

// P-wave Gaussian: only the meson Jacobi coordinate (last row of s) carries a shift.
// Chan A -> shift in z-direction (sigma_z channel).
// Chan B -> shift in x-direction (sigma_perp channel, represents x+y via sqrt(2) weight).
static gaus make_pwave(size_t dim, long double mean_r, long double mean_R, bool chan_A) {
    gaus g(dim, mean_r, mean_R);
    for (size_t i = 0; i < dim - 1; ++i)
        for (size_t k = 0; k < 3; ++k)
            g.s(i, k) = 0.0;
    if (dim >= 2) {
        if (chan_A) { g.s(dim - 1, 0) = 0.0; g.s(dim - 1, 1) = 0.0; }  // only z nonzero
        else        { g.s(dim - 1, 1) = 0.0; g.s(dim - 1, 2) = 0.0; }  // only x nonzero
    }
    return g;
}

// ---------------------------------------------------------------------------
// SVM basis selection: greedy, one Gaussian at a time.
// ---------------------------------------------------------------------------
static long double run_selection(PionSystem& sys, const size_t* target_n,
                                 int n_cand, long double mean_r, long double mean_R) {
    for (int s = 0; s < PionSystem::N_SEC; ++s) sys.basis[s].clear();

    size_t total_slots = 0;
    for (int s = 0; s < PionSystem::N_SEC; ++s) total_slots += target_n[s];

    matrix H_master(0, 0), N_master(0, 0);
    long double E_cur = std::numeric_limits<long double>::quiet_NaN();

    size_t slots_filled    = 0;
    int    consecutive_fails = 0;
    const int max_fails    = 15;

    while (slots_filled < total_slots) {
        // Pick the sector furthest from its target
        int    add_sec  = 0;
        double max_frac = -1.0;
        for (int s = 0; s < PionSystem::N_SEC; ++s) {
            if (sys.basis[s].size() >= target_n[s]) continue;
            double frac = 1.0 - (double)sys.basis[s].size() / target_n[s];
            if (frac > max_frac) { max_frac = frac; add_sec = s; }
        }

        bool   bare   = (add_sec == 0);
        bool   chan_A = PionSystem::is_chan_A(add_sec);
        size_t dim    = sys.jac(add_sec).dim();

        long double global_best_E = std::numeric_limits<long double>::infinity();
        gaus        global_best_g;

        std::vector<gaus> candidates;
        for (int c = 0; c < n_cand; ++c)
            candidates.push_back(bare ? make_swave(dim, mean_r)
                                      : make_pwave(dim, mean_r, mean_R, chan_A));

        #pragma omp parallel
        {
            long double local_best_E = std::numeric_limits<long double>::infinity();
            gaus        local_best_g;

            #pragma omp for
            for (int c = 0; c < n_cand; ++c) {
                const gaus& trial = candidates[c];

                // Linear dependence filter
                bool dependent = false;
                long double n_tt = overlap(trial, trial);
                for (size_t i = 0; i < sys.basis[add_sec].size(); ++i) {
                    long double n_ii = overlap(sys.basis[add_sec][i], sys.basis[add_sec][i]);
                    long double n_ti = std::abs(overlap(trial, sys.basis[add_sec][i]));
                    if (n_ti / std::sqrt(n_tt * n_ii) > 0.95L) { dependent = true; break; }
                }
                if (dependent) continue;

                matrix H_trial, N_trial;
                sys.insert_matrix(H_master, N_master, add_sec, trial, H_trial, N_trial);
                long double E = solve(H_trial, N_trial);
                if (!std::isnan(E) && E < local_best_E) {
                    local_best_E = E; local_best_g = trial;
                }
            }

            #pragma omp critical
            {
                if (local_best_E < global_best_E) {
                    global_best_E = local_best_E; global_best_g = local_best_g;
                }
            }
        }

        if (std::isinf(global_best_E)) {
            consecutive_fails++;
            std::cerr << "\n  [Warning] All candidates rejected (too similar). Retry "
                      << consecutive_fails << "/" << max_fails << "\n";
            if (consecutive_fails >= max_fails) {
                std::cout << "\n  [!] Basis naturally saturated at size " << slots_filled
                          << ". Moving to tuning!\n";
                break;
            }
            continue;
        }

        consecutive_fails = 0;
        sys.insert_matrix(H_master, N_master, add_sec, global_best_g, H_master, N_master);
        sys.basis[add_sec].push_back(global_best_g);
        slots_filled++;

        std::cout << "\r  [SVM] Building basis... " << std::setw(3) << slots_filled
                  << "/" << total_slots
                  << " | E = " << std::fixed << std::setprecision(5) << global_best_E
                  << std::flush;
        E_cur = global_best_E;
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
    const size_t      n_pnπ    = cfg.sz("n_pnπ");
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
    std::cout << "===============================================\n";

    // ------------------------------------------------------------------
    // Particles and vertex coefficients
    // ------------------------------------------------------------------
    Nucleon proton  = Nucleon::Proton();
    Nucleon neutron = Nucleon::Neutron();
    Pion    pi0     = Pion::PiZero();
    Pion    piplus  = Pion::PiPlus();
    Pion    piminus = Pion::PiMinus();

    // Isospin vertices (tau.pi coupling)
    VertexResult v0_p = apply_pion_emission(proton,  pi0);     // proton  + pi0  -> coef +1
    VertexResult v_pp = apply_pion_emission(proton,  piplus);  // proton  + pi+  -> coef +sqrt2
    VertexResult v_nm = apply_pion_emission(neutron, piminus); // neutron + pi-  -> coef +sqrt2
    VertexResult v0_n = apply_pion_emission(neutron, pi0);     // neutron + pi0  -> coef -1

    // ------------------------------------------------------------------
    // Build PionSystem
    // ------------------------------------------------------------------
    PionSystem sys;
    sys.relativistic = rel;
    sys.S_pion       = S_pion;
    sys.b_pion       = b_pion;

    // Jacobi systems for each physical channel
    sys.jac_phys[0] = jacobian({proton,  neutron});                   // bare PN
    sys.jac_phys[1] = jacobian({proton,  neutron, pi0});              // PN+pi0 (proton emits)
    sys.jac_phys[2] = jacobian({neutron, neutron, piplus});           // NN+pi+
    sys.jac_phys[3] = jacobian({proton,  proton,  piminus});          // PP+pi-
    sys.jac_phys[4] = jacobian({proton,  neutron, pi0});              // PN+pi0 (neutron emits)

    // Meson rest masses per physical channel
    sys.meson_mass[0] = 0.0;
    sys.meson_mass[1] = pi0.mass;
    sys.meson_mass[2] = piplus.mass;
    sys.meson_mass[3] = piminus.mass;
    sys.meson_mass[4] = pi0.mass;

    // Isospin coupling coefficients from (tau.pi) vertex
    sys.iso_weight[0] = 0.0;
    sys.iso_weight[1] = v0_p.coefficient;   // +1
    sys.iso_weight[2] = v_pp.coefficient;   // +sqrt2
    sys.iso_weight[3] = v_nm.coefficient;   // +sqrt2
    sys.iso_weight[4] = v0_n.coefficient;   // -1  (destructive interference with p=1)

    // ------------------------------------------------------------------
    // Print vertex summary so it is easy to verify the signs
    // ------------------------------------------------------------------
    std::cerr << "\n[Vertices]\n";
    std::cerr << "  p+pi0  -> coef " << v0_p.coefficient << "  (" << v0_p.resulting_nucleon.name << ")\n";
    std::cerr << "  p+pi+  -> coef " << v_pp.coefficient << "  (" << v_pp.resulting_nucleon.name << ")\n";
    std::cerr << "  n+pi-  -> coef " << v_nm.coefficient << "  (" << v_nm.resulting_nucleon.name << ")\n";
    std::cerr << "  n+pi0  -> coef " << v0_n.coefficient << "  (" << v0_n.resulting_nucleon.name << ")\n";

    {
        auto sc_p = proton.spin_dot_r_coefficients();
        auto sc_n = neutron.spin_dot_r_coefficients();
        std::cerr << "\n[Spin couplings]\n";
        std::cerr << "  proton  coef_z=" << sc_p.coef_z << "  coef_perp=" << sc_p.coef_perp << "\n";
        std::cerr << "  neutron coef_z=" << sc_n.coef_z << "  coef_perp=" << sc_n.coef_perp << "\n";
    }

    // ------------------------------------------------------------------
    // SVM target basis sizes: one entry per sector
    // ------------------------------------------------------------------
    const size_t target_n[PionSystem::N_SEC] = {
        n_pn,    // sec 0: bare PN
        n_pnπ,   // sec 1: PN+pi0 chan A (proton emits)
        n_pnπ,   // sec 2: PN+pi0 chan B (proton emits)
        n_pnπ,   // sec 3: NN+pi+ chan A
        n_pnπ,   // sec 4: NN+pi+ chan B
        n_pnπ,   // sec 5: PP+pi- chan A
        n_pnπ,   // sec 6: PP+pi- chan B
        n_pnπ,   // sec 7: PN+pi0 chan A (neutron emits)
        n_pnπ,   // sec 8: PN+pi0 chan B (neutron emits)
    };

    matrix      H_cur, N_cur;
    long double E_best = 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Phase 0.0: Generating Initial Rough Basis ===\n";
    run_selection(sys, target_n, n_cand, mean_r, mean_R);

    for (int macro = 1; macro <= n_macro; ++macro) {
        std::cout << "\n====== MACRO CYCLE " << macro << " / " << n_macro << " ======\n";

        // --- Tune S with frozen basis ---
        std::cout << "\n=== Phase 0a: Tuning S ===\n";
        long double E_cur     = std::numeric_limits<long double>::quiet_NaN();
        long double cur_kS    = std::abs(k_S);
        long double prev_errS = 0.0;
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
            if (std::abs(err) < 0.001) { std::cout << "\n --- [S converged] --- \n"; break; }
            if (it > 0 && err * prev_errS < 0) cur_kS *= 0.5;
            long double step = std::abs(cur_kS * err);
            S_pion += (err > 0) ? +step : -step;
            if (S_pion < 0.1) S_pion = 0.1;
            prev_errS = err;
        }

        // --- Tune b with frozen basis ---
        std::cout << "\n=== Phase 0b: Tuning b ===\n";
        long double prev_err = 0.0;
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
            if (std::abs(err) < 0.001) { std::cout << "\n --- [b converged] --- \n"; break; }
            if (it > 0 && err * prev_err < 0) {
                k_b *= 0.5;
                std::cout << "      [Overshoot: halving k_b to " << k_b << "]\n";
            }
            b_pion += k_b * err;
            prev_err = err;
        }

        // --- Full basis selection ---
        if (macro == 1 && (n_opt_S + n_opt_b) > 0) {
            std::cout << "\n=== Phase 1: Full Selection (S=" << S_pion << " b=" << b_pion << ") ===\n";
            E_best = run_selection(sys, target_n, n_cand, mean_r, mean_R);
        } else {
            sys.build_full(H_cur, N_cur);
            E_best = solve(H_cur, N_cur);
            std::cout << "\n[Phase 1 skipped. E=" << E_best << " MeV]\n";
        }

        // --- Refinement: replace each basis function with a better candidate ---
        std::cout << "\n=== Phase 2: Refining ===\n";
        size_t total_states = sys.total_size();
        matrix H_master, N_master;
        sys.build_full(H_master, N_master);

        for (int cycle = 1; cycle <= n_refine; ++cycle) {
            long double E_start  = E_best;
            size_t      state_idx = 0;

            for (int sec = 0; sec < PionSystem::N_SEC; ++sec) {
                bool   bare   = (sec == 0);
                bool   chan_A = PionSystem::is_chan_A(sec);
                size_t dim    = sys.jac(sec).dim();

                for (size_t k = 0; k < sys.basis[sec].size(); ++k) {
                    state_idx++;
                    long double global_best_E = E_best;
                    gaus        global_best_g = sys.basis[sec][k];

                    std::vector<gaus> candidates;
                    for (int c = 0; c < n_cand; ++c)
                        candidates.push_back(bare ? make_swave(dim, mean_r)
                                                  : make_pwave(dim, mean_r, mean_R, chan_A));

                    #pragma omp parallel
                    {
                        long double local_best_E = global_best_E;
                        gaus        local_best_g = global_best_g;

                        #pragma omp for
                        for (int c = 0; c < n_cand; ++c) {
                            const gaus& trial = candidates[c];
                            if (is_linearly_dependent(trial, sys.basis[sec], k)) continue;
                            matrix H_trial, N_trial;
                            sys.replace_matrix(H_master, N_master, sec, k, trial, H_trial, N_trial);
                            long double E = solve(H_trial, N_trial);
                            if (!std::isnan(E) && E < local_best_E) {
                                local_best_E = E; local_best_g = trial;
                            }
                        }

                        #pragma omp critical
                        {
                            if (local_best_E < global_best_E) {
                                global_best_E = local_best_E; global_best_g = local_best_g;
                            }
                        }
                    }

                    if (global_best_E < E_best) {
                        sys.basis[sec][k] = global_best_g;
                        sys.replace_matrix(H_master, N_master, sec, k, global_best_g, H_master, N_master);
                        E_best = global_best_E;
                    }

                    std::cout << "\r  [Refine] Cycle " << cycle << "/" << n_refine
                              << " | State " << std::setw(3) << state_idx << "/" << total_states
                              << " | E = " << std::fixed << std::setprecision(5) << E_best << std::flush;
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
    matrix H_final, N_final;
    sys.build_full(H_final, N_final);

    EigenResult res = solve_generalized_eigensystem(H_final, N_final);
    long double E_gs = res.evals.size() > 0 ? res.evals[0]
                                            : std::numeric_limits<long double>::quiet_NaN();

    // Norm weights per sector
    long double prob_bare = 0.0, prob_pion = 0.0;
    if (res.evecs.size1() > 0) {
        const auto& c = res.evecs;
        for (size_t i = 0; i < sys.basis[0].size(); ++i)
            for (size_t j = 0; j < sys.basis[0].size(); ++j)
                prob_bare += c(i, 0) * N_final(i, j) * c(j, 0);
        for (int sec = 1; sec < PionSystem::N_SEC; ++sec) {
            size_t off = sys.offset(sec);
            for (size_t i = 0; i < sys.basis[sec].size(); ++i)
                for (size_t j = 0; j < sys.basis[sec].size(); ++j)
                    prob_pion += c(off + i, 0) * N_final(off + i, off + j) * c(off + j, 0);
        }
        prob_bare *= 100.0L;
        prob_pion *= 100.0L;
    }

    // <r^2> and charge radius
    long double r2_exp = 0.0;
    if (res.evecs.size1() > 0) {
        const auto& c = res.evecs;
        for (size_t i = 0; i < sys.basis[0].size(); ++i)
            for (size_t j = 0; j < sys.basis[0].size(); ++j)
                r2_exp += c(i, 0) * sys.H.R2_matrix_element(sys.basis[0][i], sys.basis[0][j]) * c(j, 0);
    }
    long double r_c = std::sqrt(std::abs(r2_exp) / 4.0L);

    std::cout << "\n==========================================\n";
    std::cout << "  Ground-state energy : " << E_gs << " MeV\n";
    std::cout << "  Relativistic        : " << (rel ? "True\n" : "False\n");
    std::cout << "  --------------------------------------\n";
    std::cout << "  <r^2>_pn     : " << r2_exp << " fm^2\n";
    std::cout << "  Charge Rad   : " << r_c    << " fm\n";
    std::cout << "  --------------------------------------\n";
    std::cout << "  Bare Sector (PN)   : " << prob_bare << " %\n";
    std::cout << "  Pion Cloud (PN+pi) : " << prob_pion << " %\n";
    std::cout << "==========================================\n";

    // ============================================================
    // Phase 5: Fast 2D Heatmap Scan (b vs S)
    // ============================================================
    std::cout << "\n=== Phase 5: Generating 2D Heatmap Data ===\n";
    std::ofstream scan_out("heatmap.dat");
    scan_out << "# b_pion (fm) \t S_pion (MeV) \t Energy (MeV)\n";
    
    // Outer loop: b from 0.8 to 2.0
    for (long double b_test = 0.8; b_test <= 2.0; b_test += 0.05) {
        sys.b_pion = b_test;
        
        // Inner loop: S from 10 to 100
        for (long double s_test = 10.0; s_test <= 100.0; s_test += 2.0) {
            sys.S_pion = s_test;
            matrix H_scan, N_scan;
            sys.build_full(H_scan, N_scan); 
            long double e_scan = solve(H_scan, N_scan);
            
            // Cap explosive positive/NaN energies to 0.0 for a cleaner color scale
            if (std::isnan(e_scan) || e_scan > 0.0) e_scan = 0.0;
            
            scan_out << b_test << " \t " << s_test << " \t " << e_scan << "\n";
        }
        scan_out << "\n"; // MANDATORY BLANK LINE for Gnuplot grid formatting!
    }
    scan_out.close();
    std::cout << "  -> Heatmap data saved to 'heatmap.dat'.\n";
    
    // Restore the physical parameters
    sys.b_pion = b_pion;
    sys.S_pion = S_pion;

    return 0;
}