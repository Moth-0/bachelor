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

// Config key=value store with defaults
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

// 7-sector system: sector 0 = bare PN, sectors 1-6 = clothed PN+pi channels.
// Sector layout: 1,2 -> pi0 (chanA/B), 3,4 -> pi+ (chanA/B), 5,6 -> pi- (chanA/B)
struct PionSystem {
    static const int N_SEC = 7;

    std::vector<gaus> basis[N_SEC];
    jacobian          jac_phys[4];    // jac_phys[p] for physical channel p
    long double       meson_mass[4];
    long double       iso_weight[4];
    long double       S_pion;
    long double       b_pion;
    bool              relativistic;
    hamiltonian       H;

    static int  phys(int sec)      { return (sec <= 0) ? 0 : (sec + 1) / 2; }
    static bool is_chan_A(int sec) { return (sec > 0) && (sec % 2 == 1); }
    const jacobian& jac(int sec) const { return jac_phys[phys(sec)]; }

    // Projection vector for pion position relative to emitting nucleon (emitting_idx = 0 or 1)
    vector pion_coord(int sec, int emitting_idx) const {
        long double m0   = jac(sec).particles[0].mass;
        long double m1   = jac(sec).particles[1].mass;
        long double mtot = m0 + m1;
        vector c(2);
        if (emitting_idx == 0) { c[0] = +m1 / mtot; c[1] = 1.0; }
        else                   { c[0] = -m0 / mtot; c[1] = 1.0; }
        return c;
    }

    // Gaussian form factor kernel: Omega_{ij} = c_i c_j / b^2
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

    // Compute a new row/column of H and N for trial Gaussian in target_sec
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
                    int clothed_sec  = (sec == 0) ? target_sec : sec;
                    const gaus& g_bare  = (sec == 0) ? basis[0][i] : g_target;
                    const gaus& g_cloth = (sec == 0) ? g_target    : basis[sec][i];
                    int p = phys(clothed_sec);
                    vector c_coord = pion_coord(clothed_sec, (p == 3) ? 1 : 0);
                    matrix Om = Omega(clothed_sec, c_coord);
                    vector beta(3);
                    long double strength = iso_weight[p] * (S_pion / b_pion);
                    if (is_chan_A(clothed_sec)) beta[2] = strength;
                    else                        beta[0] = strength * std::sqrt(2.0);
                    h_val = H.W(g_bare, g_cloth, Om, c_coord, 0.0, beta);
                    n_val = 0.0;
                }
                h_row[off + i] = h_val;
                n_row[off + i] = n_val;
            }
        }
    }

    // Expand H and N by one row/column for a new basis function in target_sec
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

    // Replace row/column target_k in target_sec with a new basis function
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

    // Build full H and N from scratch
    void build_full(matrix& H_tot, matrix& N_tot) const {
        size_t N = total_size();
        H_tot = matrix(N, N);
        N_tot = matrix(N, N);

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

        size_t n0 = basis[0].size();
        for (int sec = 1; sec < N_SEC; ++sec) {
            size_t n_s  = basis[sec].size();
            size_t off_s = offset(sec);
            int p = phys(sec);
            vector c_coord = pion_coord(sec, (p == 3) ? 1 : 0);
            matrix Om = Omega(sec, c_coord);
            vector beta(3);
            long double strength = iso_weight[p] * (S_pion / b_pion);
            if (is_chan_A(sec)) beta[2] = strength;
            else                beta[0] = strength * std::sqrt(2.0);
            for (size_t i = 0; i < n0; ++i)
                for (size_t k = 0; k < n_s; ++k) {
                    long double w = H.W(basis[0][i], basis[sec][k], Om, c_coord, 0.0, beta);
                    H_tot(i, off_s + k) = H_tot(off_s + k, i) = w;
                }
        }
    }
};

// Solve H c = E N c and return ground-state energy (NaN on failure)
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
// The spin channel sets the non-zero component: chan_A -> z, chan_B -> x.
static gaus make_pwave(size_t dim, long double mean_r, long double mean_R, bool chan_A) {
    gaus g(dim, mean_r, mean_R);
    for (size_t i = 0; i < dim - 1; ++i)
        for (size_t k = 0; k < 3; ++k)
            g.s(i, k) = 0.0;
    if (dim >= 2) {
        if (chan_A) { g.s(dim - 1, 0) = 0.0; g.s(dim - 1, 1) = 0.0; }
        else        { g.s(dim - 1, 1) = 0.0; g.s(dim - 1, 2) = 0.0; }
    }
    return g;
}

// SVM basis selection: greedily add one Gaussian at a time, choosing the candidate
// that minimises the ground-state energy. Linear dependence check in gaussian.h.
static long double run_selection(PionSystem& sys, const size_t* target_n, int n_cand, long double mean_r, long double mean_R) {
    for (int s = 0; s < PionSystem::N_SEC; ++s) sys.basis[s].clear();

    size_t total_slots = 0;
    for (int s = 0; s < PionSystem::N_SEC; ++s) total_slots += target_n[s];

    matrix H_master(0,0), N_master(0,0);
    long double E_cur = std::numeric_limits<long double>::quiet_NaN();

    size_t slots_filled = 0;
    int consecutive_fails = 0;
    int max_fails = 15; // Give up if we fail 15 times in a row

    while (slots_filled < total_slots) {
        int add_sec = 0;
        double max_frac = -1.0;
        for (int s = 0; s < PionSystem::N_SEC; ++s) {
            if (sys.basis[s].size() >= target_n[s]) continue;
            double frac = 1.0 - (double)sys.basis[s].size() / target_n[s];
            if (frac > max_frac) { max_frac = frac; add_sec = s; }
        }

        bool bare = (add_sec == 0);
        bool chan_A = PionSystem::is_chan_A(add_sec);
        size_t dim = sys.jac(add_sec).dim();

        long double global_best_E = std::numeric_limits<long double>::infinity();
        gaus global_best_g;

        std::vector<gaus> candidates;
        for(int c = 0; c < n_cand; ++c) {
            candidates.push_back(bare ? make_swave(dim, mean_r) : make_pwave(dim, mean_r, mean_R, chan_A));
        }

        #pragma omp parallel
        {
            long double local_best_E = std::numeric_limits<long double>::infinity();
            gaus local_best_g;

            #pragma omp for
            for (int c = 0; c < n_cand; ++c) {
                const gaus& trial = candidates[c]; 

                // --- LINEAR DEPENDENCE FILTER ---
                bool dependent = false;
                long double n_tt = overlap(trial, trial);
                for (size_t i = 0; i < sys.basis[add_sec].size(); ++i) {
                    long double n_ii = overlap(sys.basis[add_sec][i], sys.basis[add_sec][i]);
                    long double n_ti = std::abs(overlap(trial, sys.basis[add_sec][i]));
                    if (n_ti / std::sqrt(n_tt * n_ii) > 0.95L) { dependent = true; break; }
                }
                if (dependent) continue; 
                // --------------------------------

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
            std::cerr << "\n  [Warning] All candidates rejected (Too similar). Retry " << consecutive_fails << "/" << max_fails << "\n";
            if (consecutive_fails >= max_fails) {
                std::cout << "\n  [!] Basis naturally saturated at size " << slots_filled << ". Moving to tuning!\n";
                break; // Exit the while loop early, the well is full!
            }
            continue; // Try again without incrementing slots_filled
        }
        
        consecutive_fails = 0; // Reset fails on a successful addition
        sys.insert_matrix(H_master, N_master, add_sec, global_best_g, H_master, N_master);
        sys.basis[add_sec].push_back(global_best_g);
        slots_filled++;

        std::cout << "\r  [SVM] Building basis... " << std::setw(3) << slots_filled 
                  << "/" << total_slots 
                  << " | E = " << std::fixed << std::setprecision(5) << global_best_E << std::flush;
        E_cur = global_best_E;
    }
    
    std::cout << "\n"; 
    return E_cur;
}

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
    const size_t      n_pnπ   = cfg.sz("n_pnπ");
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
    std::cout << "[main] Deuteron + explicit pion  (7-sector SVM)\n";
    std::cout << "===============================================\n";

    Nucleon proton  = Nucleon::Proton();
    Nucleon neutron = Nucleon::Neutron();
    Pion pi0        = Pion::PiZero();
    Pion piplus     = Pion::PiPlus();
    Pion piminus    = Pion::PiMinus();
    VertexResult v0  = apply_pion_emission(proton,  pi0);
    VertexResult v_p = apply_pion_emission(proton,  piplus);
    VertexResult v_m = apply_pion_emission(neutron, piminus);

    PionSystem sys;
    sys.relativistic  = rel;
    sys.S_pion        = S_pion;
    sys.b_pion        = b_pion;
    sys.jac_phys[0]   = jacobian({proton, neutron});
    sys.jac_phys[1]   = jacobian({proton, neutron, pi0});
    sys.jac_phys[2]   = jacobian({neutron, neutron, piplus});
    sys.jac_phys[3]   = jacobian({proton, proton, piminus});
    sys.meson_mass[0] = 0.0;           sys.meson_mass[1] = pi0.mass;
    sys.meson_mass[2] = piplus.mass;   sys.meson_mass[3] = piminus.mass;
    sys.iso_weight[0] = 0.0;           sys.iso_weight[1] = v0.coefficient;
    sys.iso_weight[2] = v_p.coefficient; sys.iso_weight[3] = v_m.coefficient;

    const size_t target_n[PionSystem::N_SEC] = {n_pn, n_pnπ, n_pnπ, n_pnπ, n_pnπ, n_pnπ, n_pnπ};
    matrix H_cur, N_cur;
    long double E_best = 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Phase 0.0: Generating Initial Rough Basis ===\n";
    run_selection(sys, target_n, n_cand, mean_r, mean_R);

    for (int macro = 1; macro <= n_macro; ++macro) {
        std::cout << "\n====== MACRO CYCLE " << macro << " / " << n_macro << " ======\n";

        // --- Tune S with frozen basis ---
        std::cout << "\n=== Phase 0a: Tuning S ===\n";
        long double E_cur = std::numeric_limits<long double>::quiet_NaN();
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
            S_pion += (err > 0) ? -step : +step;
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

        // --- Full basis selection (first macro only) ---
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
            long double E_start = E_best;
            size_t state_idx = 0;

            for (int sec = 0; sec < PionSystem::N_SEC; ++sec) {
                bool   bare   = (sec == 0);
                bool   chan_A = PionSystem::is_chan_A(sec);
                size_t dim    = sys.jac(sec).dim();

                for (size_t k = 0; k < sys.basis[sec].size(); ++k) {
                    state_idx++;
                    long double global_best_E = E_best;
                    gaus global_best_g = sys.basis[sec][k];

                    std::vector<gaus> candidates;
                    for (int c = 0; c < n_cand; ++c)
                        candidates.push_back(bare ? make_swave(dim, mean_r)
                                                  : make_pwave(dim, mean_r, mean_R, chan_A));

                    #pragma omp parallel
                    {
                        long double local_best_E = global_best_E;
                        gaus local_best_g = global_best_g;

                        #pragma omp for
                        for (int c = 0; c < n_cand; ++c) {
                            const gaus& trial = candidates[c];
                            if (is_linearly_dependent(trial, sys.basis[sec], k)) continue;

                            matrix H_trial, N_trial;
                            sys.replace_matrix(H_master, N_master, sec, k, trial, H_trial, N_trial);
                            long double E = solve(H_trial, N_trial);
                            if (!std::isnan(E) && E < local_best_E) { local_best_E = E; local_best_g = trial; }
                        }

                        #pragma omp critical
                        { if (local_best_E < global_best_E) { global_best_E = local_best_E; global_best_g = local_best_g; } }
                    }

                    sys.replace_matrix(H_master, N_master, sec, k, global_best_g, H_master, N_master);
                    sys.basis[sec][k] = global_best_g;
                    E_best = global_best_E;
                    std::cout << "\r  [Refine] Cycle " << cycle << "/" << n_refine
                              << " | State " << std::setw(3) << state_idx << "/" << total_states
                              << " | E = " << std::fixed << std::setprecision(5) << E_best << std::flush;
                }
            }
            long double delta = (E_start - E_best);
            std::cout << "\n  => Cycle " << cycle << " finished.  Delta E = " << delta << " MeV\n";
            if (delta < 0.01) { std::cout << "  [Converged early.]\n"; break; }
        }
    }

    // --- Observables: charge radius and pion cloud fraction ---
    std::cout << "\n=== Phase 3: Observables ===\n";
    sys.build_full(H_cur, N_cur);
    E_best = solve(H_cur, N_cur);
    EigenResult final_sys = solve_generalized_eigensystem(H_cur, N_cur);

    if (final_sys.evals.size() > 0) {
        size_t N_total = final_sys.evecs.size1();
        std::vector<long double> c(N_total);
        for (size_t i = 0; i < N_total; ++i) c[i] = final_sys.evecs(i, 0);

        long double r2_exp = 0.0;
        long double norm   = 0.0;
        std::vector<long double> prob_sec(PionSystem::N_SEC, 0.0);

        for (int si = 0; si < PionSystem::N_SEC; ++si) {
            size_t oi = sys.offset(si);
            for (size_t i = 0; i < sys.basis[si].size(); ++i) {
                size_t gi = oi + i;
                for (int sj = 0; sj < PionSystem::N_SEC; ++sj) {
                    if (si != sj) continue;
                    size_t oj = sys.offset(sj);
                    for (size_t j = 0; j < sys.basis[sj].size(); ++j) {
                        size_t gj = oj + j;
                        long double w   = c[gi] * c[gj];
                        long double r2v = sys.H.R2_matrix_element(sys.basis[si][i], sys.basis[sj][j]);
                        long double nv  = overlap(sys.basis[si][i], sys.basis[sj][j]);
                        r2_exp        += w * r2v;
                        norm          += w * nv;
                        prob_sec[si]  += w * nv;
                    }
                }
            }
        }

        r2_exp /= norm;
        long double prob_bare = (prob_sec[0] / norm) * 100.0;
        long double prob_pion = 0.0;
        for (int s = 1; s < PionSystem::N_SEC; ++s) prob_pion += (prob_sec[s] / norm) * 100.0;

        // Charge radius: r_c^2 = (1/4)<r_pn^2> + r_p^2 + r_n^2
        long double r_p_sq = 0.8414 * 0.8414;
        long double r_n_sq = -0.1161;
        long double r_c    = std::sqrt(0.25 * r2_exp + r_p_sq + r_n_sq);

        std::cout << "\n==========================================\n";
        std::cout << "  Target       : " << E_target  << " MeV\n";
        std::cout << "  Result       : "  << E_best    << " MeV\n";
        std::cout << "  S_pion       : " << S_pion    << " MeV\n";
        std::cout << "  b_pion       : " << b_pion    << " fm\n";
        std::cout << "  Basis sizes  : n_pn=" << sys.basis[0].size() << ", n_pnπ=" << sys.basis[1].size() << " per channel\n";
        std::cout << "  Relativistic : " << (rel ? "True\n" : "False\n");
        std::cout << "  --------------------------------------\n";
        std::cout << "  <r^2>_pn     : " << r2_exp << " fm^2\n";
        std::cout << "  Charge Rad   : " << r_c    << " fm\n";
        std::cout << "  --------------------------------------\n";
        std::cout << "  Bare Sector (PN)   : " << prob_bare << " %\n";
        std::cout << "  Pion Cloud (PN+pi) : " << prob_pion << " %\n";
        //std::cout << "  --------------------------------------\n";
        //std::cout << "  Final basis size   : " << N_total << "\n";
        std::cout << "==========================================\n";
    }

    return 0;
}