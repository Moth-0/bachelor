#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <limits>
#include <cmath>

#include "qm/matrix.h"
#include "qm/eigen.h"
#include "qm/particle.h"
#include "qm/jacobian.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"

// ============================================================
//  main.cc  —  Deuteron with explicit pion exchange
//
//  Usage:
//    ./main                        # reads pion.cfg if present
//    ./main my_run.cfg             # explicit config file
//    ./main S=20 b=1.5             # command-line overrides
//    ./main my_run.cfg S=20 b=1.5  # file + overrides
//
//  Config file format (key = value, # for comments):
//    S        = 15.0    # coupling strength (MeV)
//    b        = 2.0     # vertex range (fm)
//    n0       = 4       # bare sector basis size
//    n1       = 8       # pi0 sector basis size
//    n2       = 8       # pi+ sector basis size
//    n3       = 8       # pi- sector basis size
//    n_cand   = 5       # SVM candidates per slot
//    n_refine = 2       # refining cycles
//    A_min    = 0.001   # min Gaussian width (fm^-2)
//    A_max    = 10.0    # max Gaussian width (fm^-2)
//    rel      = 0       # 0=classical  1=relativistic
// ============================================================

using namespace qm;

// -----------------------------------------------------------
//  Params: key=value store with typed getters and defaults
// -----------------------------------------------------------
struct Params {
    std::map<std::string, std::string> store;

    Params() {
        store["S"]        = "15.0";
        store["b"]        = "2.0";
        store["n0"]       = "4";
        store["n1"]       = "8";
        store["n2"]       = "8";
        store["n3"]       = "8";
        store["n_cand"]   = "5";
        store["n_refine"] = "2";
        store["A_min"]    = "0.001";
        store["A_max"]    = "10.0";
        store["rel"]      = "0";
        store["n_opt_S"]  = "8";    // number of S update steps
        store["n_opt_b"]  = "8";    // number of b update steps (after S is done)
        store["k_S"]      = "0.01"; // S learning rate: S += k_S * (target - E)
        store["k_b"]      = "0.001";// b learning rate: b += k_b * (target - E)
        store["target"]   = "-2.225"; // binding energy target (MeV)
    }

    void load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[cfg] No file '" << path << "', using defaults.\n";
            return;
        }
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
        std::cerr << "[cfg] Loaded '" << path << "'\n";
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
        for (auto& kv : store)
            std::cerr << "  " << kv.first << " = " << kv.second << "\n";
    }
};

// -----------------------------------------------------------
struct PionSystem {
    std::vector<gaus> basis[4];
    jacobian          jac[4];
    long double       meson_mass[4];
    long double       iso_weight[4];
    long double       S_pion;
    long double       b_pion;
    vector            c_spin;
    bool              relativistic;
    hamiltonian       H;

    matrix Omega(size_t sector) const {
        size_t d = jac[sector].dim();
        matrix Om(d, d);
        long double w = 1.0L / (b_pion * b_pion);
        Om(0,0) = w;
        Om(1,1) = w;
        return Om;
    }

    size_t total_size() const {
        size_t n = 0;
        for (int s = 0; s < 4; ++s) n += basis[s].size();
        return n;
    }

    size_t offset(int sector) const {
        size_t off = 0;
        for (int s = 0; s < sector; ++s) off += basis[s].size();
        return off;
    }

    void build(matrix& H_tot, matrix& N_tot) const {
        size_t N = total_size();
        H_tot = matrix(N, N);
        N_tot = matrix(N, N);

        // Diagonal blocks
        for (int sec = 0; sec < 4; ++sec) {
            size_t n_s = basis[sec].size();
            size_t off = offset(sec);
            for (size_t i = 0; i < n_s; ++i) {
                for (size_t j = 0; j < n_s; ++j) {
                    long double nij = overlap(basis[sec][i], basis[sec][j]);
                    N_tot(off+i, off+j) = nij;
                    long double kij = 0.0L;
                    for (size_t coord = 0; coord < jac[sec].dim(); ++coord) {
                        if (relativistic)
                            kij += H.K_rel(basis[sec][i], basis[sec][j],
                                           jac[sec].c(coord), jac[sec].mu(coord));
                        else
                            kij += H.K_cla(basis[sec][i], basis[sec][j],
                                           jac[sec].c(coord), jac[sec].mu(coord));
                    }
                    H_tot(off+i, off+j) = kij + meson_mass[sec] * nij;
                }
            }
        }

        // Off-diagonal W blocks: sector 0 <-> sectors 1,2,3
        for (int sec = 1; sec <= 3; ++sec) {
            size_t n0    = basis[0].size();
            size_t n_s   = basis[sec].size();
            size_t off_s = offset(sec);
            matrix Om    = Omega(sec);

            vector beta(3);
            for (size_t d = 0; d < 3; ++d)
                // Fedorov: A = S_w/b_w (photoproduction eq.38): r_pi carries fm, so
                // dividing by b_pion restores MeV energy units.
                beta[d] = iso_weight[sec] * (S_pion / b_pion) * c_spin[d];

            for (size_t i = 0; i < n0; ++i) {
                for (size_t j = 0; j < n_s; ++j) {
                    long double w = H.W(basis[0][i], basis[sec][j],
                                        Om, jac[sec].meson_index(),
                                        0.0L, beta);
                    H_tot(i,       off_s+j) = w;
                    H_tot(off_s+j, i      ) = w;
                    N_tot(i,       off_s+j) = 0.0L;
                    N_tot(off_s+j, i      ) = 0.0L;
                }
            }
        }
    }
};

static long double solve(const matrix& H_tot, const matrix& N_tot) {
    matrix L = cholesky(N_tot);
    if (L.size1() == 0) return std::numeric_limits<long double>::quiet_NaN();
    matrix L_inv = L.inverse_lower();
    if (L_inv.size1() == 0) return std::numeric_limits<long double>::quiet_NaN();
    vector evals = jacobi_eigenvalues(L_inv * H_tot * L_inv.transpose());
    if (evals.size() == 0) return std::numeric_limits<long double>::quiet_NaN();
    long double E0 = evals[0];
    for (size_t k = 1; k < evals.size(); ++k)
        if (evals[k] < E0) E0 = evals[k];
    return E0;
}

static gaus make_swave(size_t dim, long double A_min, long double A_max) {
    gaus g(dim, A_min, A_max);
    g.zero_shifts();
    return g;
}

static gaus make_pwave(size_t dim, long double A_min, long double A_max) {
    return gaus(dim, A_min, A_max);
}


// -----------------------------------------------------------
//  Run competitive selection from scratch with the current
//  sys.S_pion and sys.b_pion, return the ground-state energy.
//  Clears and rebuilds sys.basis entirely.
// -----------------------------------------------------------
static long double run_selection(
    PionSystem& sys,
    const size_t* target_n,
    int n_cand,
    long double A_min, long double A_max)
{
    for (int s = 0; s < 4; ++s) sys.basis[s].clear();

    size_t total_slots = 0;
    for (int s = 0; s < 4; ++s) total_slots += target_n[s];

    matrix H_cur, N_cur;
    long double E_cur = std::numeric_limits<long double>::quiet_NaN();

    for (size_t slot = 0; slot < total_slots; ++slot) {
        int    add_sec  = 0;
        double max_frac = -1.0;
        for (int s = 0; s < 4; ++s) {
            if (sys.basis[s].size() >= target_n[s]) continue;
            double frac = 1.0 - (double)sys.basis[s].size() / target_n[s];
            if (frac > max_frac) { max_frac = frac; add_sec = s; }
        }

        size_t dim  = sys.jac[add_sec].dim();
        bool   bare = (add_sec == 0);

        long double best_E = std::numeric_limits<long double>::infinity();
        gaus best_g = bare ? make_swave(dim, A_min, A_max)
                           : make_pwave(dim, A_min, A_max);

        for (int c = 0; c < n_cand; ++c) {
            gaus trial = bare ? make_swave(dim, A_min, A_max)
                              : make_pwave(dim, A_min, A_max);
            sys.basis[add_sec].push_back(trial);
            sys.build(H_cur, N_cur);
            long double E = solve(H_cur, N_cur);
            if (!std::isnan(E) && E < best_E) { best_E = E; best_g = trial; }
            sys.basis[add_sec].pop_back();
        }
        sys.basis[add_sec].push_back(best_g);
    }

    sys.build(H_cur, N_cur);
    E_cur = solve(H_cur, N_cur);
    return E_cur;
}

// ============================================================
int main(int argc, char* argv[])
{
    // ----------------------------------------------------------
    // Parse arguments:
    //   arg without '=' -> config file path
    //   arg with    '=' -> key=value override
    // ----------------------------------------------------------
    Params cfg;

    // Two-pass parsing so CLI overrides always win over the config file:
    //   Pass 1: load all config files (left to right)
    //   Pass 2: apply key=value overrides on top
    // This means ./main pion.cfg S=20  AND  ./main S=20  both work correctly.
    bool file_loaded = false;
    for (int k = 1; k < argc; ++k) {
        std::string arg(argv[k]);
        if (arg.find('=') == std::string::npos) {
            cfg.load_file(arg);
            file_loaded = true;
        }
    }
    if (!file_loaded) cfg.load_file("pion.cfg");  // default

    // Now apply CLI key=value overrides (after file, so they win)
    for (int k = 1; k < argc; ++k) {
        std::string arg(argv[k]);
        if (arg.find('=') != std::string::npos)
            cfg.set(arg);
    }

    cfg.print();

    // ----------------------------------------------------------
    // Extract parameters
    // ----------------------------------------------------------
    long double       S_pion  = cfg.ld("S");
    long double       b_pion  = cfg.ld("b");
    const size_t      n0      = cfg.sz("n0");
    const size_t      n1      = cfg.sz("n1");
    const size_t      n2      = cfg.sz("n2");
    const size_t      n3      = cfg.sz("n3");
    const int         n_cand  = cfg.i("n_cand");
    const int         n_refine= cfg.i("n_refine");
    const long double A_min   = cfg.ld("A_min");
    const long double A_max   = cfg.ld("A_max");
    const bool        rel     = (cfg.i("rel") != 0);
    const int         n_opt_S = cfg.i("n_opt_S");
    const int         n_opt_b = cfg.i("n_opt_b");
    const long double k_S     = cfg.ld("k_S");
    const long double k_b     = cfg.ld("k_b");
    const long double E_target= cfg.ld("target");

    // ----------------------------------------------------------
    // 1. Particles and Jacobians
    // ----------------------------------------------------------
    std::cout << "[main] ============================================\n";
    std::cout << "[main] Deuteron + explicit pion (4-sector SVM)\n";
    std::cout << "[main] ============================================\n";

    Nucleon proton  = Nucleon::Proton();
    Nucleon neutron = Nucleon::Neutron();
    Pion    pi0     = Pion::PiZero();
    Pion    piplus  = Pion::PiPlus();
    Pion    piminus = Pion::PiMinus();

    VertexResult v0  = apply_pion_emission(proton,  pi0);
    VertexResult v_p = apply_pion_emission(proton,  piplus);
    VertexResult v_m = apply_pion_emission(neutron, piminus);

    std::cerr << "[main] Isospin weights:\n";
    std::cerr << "  p  + pi0 -> " << v0.resulting_nucleon.name
              << "  w=" << v0.coefficient << "\n";
    std::cerr << "  p  + pi+ -> " << v_p.resulting_nucleon.name
              << "  w=" << v_p.coefficient << "\n";
    std::cerr << "  n  + pi- -> " << v_m.resulting_nucleon.name
              << "  w=" << v_m.coefficient << "\n";

    PionSystem sys;
    sys.relativistic  = rel;
    sys.S_pion        = S_pion;
    sys.b_pion        = b_pion;
    sys.c_spin        = vector(3);
    // c_spin encodes the spin matrix element  <χ_f|σ_d|χ_i>  for direction d.
    // The pion vertex is  W = S·(σ⃗·r_π)·F(r).
    // For a single spin-½ nucleon in a definite spin state, |c_spin|² = 1.
    //
    // Choosing  c_spin = {1/√3, 1/√3, 1/√3}  keeps |c_spin|² = 1 and
    // distributes the coupling equally over x, y, z, so the SVM can use all
    // three spatial directions of the pion shifts — roughly 3× more efficient
    // than the {1,0,0} placeholder that restricts coupling to x only.
    //
    // For the full deuteron spin structure (J=1), the correct c_spin would be
    // derived from angular momentum recoupling.  This isotropic approximation
    // gives the right binding-energy scale and can be refined later.
    {
        long double s3 = 1.0L / std::sqrt(3.0L);
        sys.c_spin[0] = s3;
        sys.c_spin[1] = s3;
        sys.c_spin[2] = s3;
    }

    sys.jac[0] = jacobian({proton,  neutron});
    sys.jac[1] = jacobian({proton,  neutron, pi0});
    sys.jac[2] = jacobian({neutron, neutron, piplus});
    sys.jac[3] = jacobian({proton,  proton,  piminus});

    sys.meson_mass[0] = 0.0L;
    sys.meson_mass[1] = pi0.mass;
    sys.meson_mass[2] = piplus.mass;
    sys.meson_mass[3] = piminus.mass;

    sys.iso_weight[0] = 0.0L;
    sys.iso_weight[1] = v0.coefficient;
    sys.iso_weight[2] = v_p.coefficient;
    sys.iso_weight[3] = v_m.coefficient;

    std::cerr << "[main] Jacobians:\n";
    for (int s = 0; s < 4; ++s) {
        std::cerr << "  sec " << s << ": dim=" << sys.jac[s].dim();
        for (size_t k = 0; k < sys.jac[s].dim(); ++k)
            std::cerr << "  mu[" << k << "]=" << sys.jac[s].mu(k) << " MeV";
        std::cerr << "\n";
    }

    // ----------------------------------------------------------
    // 2. Print active run parameters
    // ----------------------------------------------------------
    std::cout << "[main] S=" << S_pion << " MeV  b=" << b_pion
              << " fm  rel=" << (rel ? "yes" : "no") << "\n";
    std::cout << "[main] Basis: n0=" << n0 << " n1=" << n1
              << " n2=" << n2 << " n3=" << n3
              << "  n_cand=" << n_cand << "  n_refine=" << n_refine
              << "  A=[" << A_min << "," << A_max << "]\n\n";

    // ----------------------------------------------------------
    // 3. Fixed setup (does not change across optimisation)
    // ----------------------------------------------------------
    const size_t target_n[4] = {n0, n1, n2, n3};
    matrix H_cur, N_cur;

    // ----------------------------------------------------------
    // 4. Phase 0a — Tune S with b held fixed
    //
    //    Each iteration:
    //      1. Run competitive selection with current S (b is fixed)
    //      2. err = target - E_cur
    //      3. S += k_S * err
    //
    //    Sign: if E > target (too weakly bound), err < 0.
    //    For the pion system S enters the W vertex as S*(sigma.r),
    //    so |S| controls coupling strength — increase |S| to bind more.
    //    k_S > 0 moves S in the direction that lowers E.
    // ----------------------------------------------------------
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Phase 0a: Tuning S  (" << n_opt_S << " steps, k_S=" << k_S
              << ")  b=" << b_pion << " fm fixed ===\n";
    std::cerr << "=== Phase 0a: Tuning S  k_S=" << k_S
              << "  n=" << n_opt_S << " ===\n";

    long double E_cur = std::numeric_limits<long double>::quiet_NaN();

    for (int it = 0; it < n_opt_S; ++it) {
        sys.S_pion = S_pion;
        sys.b_pion = b_pion;

        E_cur = run_selection(sys, target_n, n_cand, A_min, A_max);
        long double err = E_target - E_cur;

        std::cout << "  S it=" << std::setw(2) << it
                  << "  S=" << std::setw(9) << S_pion
                  << "  b=" << std::setw(6) << b_pion
                  << "  E=" << std::setw(10) << E_cur
                  << "  err=" << std::setw(9) << err << " MeV\n";
        std::cerr << "  S it=" << it << "  S=" << S_pion
                  << "  E=" << E_cur << "  err=" << err << "\n";

        if (std::abs(err) < 0.05L) { std::cout << "  [S converged]\n"; break; }

        S_pion += k_S * err;
    }

    // ----------------------------------------------------------
    // 5. Phase 0b — Tune b with S held fixed at value found above
    //
    //    Same gradient-step logic but now only b moves.
    //    k_b sign: increasing b widens the form factor → weaker
    //    localisation → typically less binding, so k_b > 0 will
    //    decrease b when E is too high (err < 0), sharpening the
    //    vertex and increasing binding.  Adjust sign if you see
    //    b moving the wrong way for your vertex parameterisation.
    // ----------------------------------------------------------
    std::cout << "\n=== Phase 0b: Tuning b  (" << n_opt_b << " steps, k_b=" << k_b
              << ")  S=" << S_pion << " MeV fixed ===\n";
    std::cerr << "=== Phase 0b: Tuning b  k_b=" << k_b
              << "  n=" << n_opt_b << " ===\n";

    for (int it = 0; it < n_opt_b; ++it) {
        sys.S_pion = S_pion;
        sys.b_pion = b_pion;

        E_cur = run_selection(sys, target_n, n_cand, A_min, A_max);
        long double err = E_target - E_cur;

        std::cout << "  b it=" << std::setw(2) << it
                  << "  S=" << std::setw(9) << S_pion
                  << "  b=" << std::setw(6) << b_pion
                  << "  E=" << std::setw(10) << E_cur
                  << "  err=" << std::setw(9) << err << " MeV\n";
        std::cerr << "  b it=" << it << "  b=" << b_pion
                  << "  E=" << E_cur << "  err=" << err << "\n";

        if (std::abs(err) < 0.05L) { std::cout << "  [b converged]\n"; break; }

        b_pion += k_b * err;
    }

    // ----------------------------------------------------------
    // 5. Phase 1 — Final competitive selection with best S, b
    // ----------------------------------------------------------
    std::cout << "\n=== Phase 1: Final competitive selection"
              << "  S=" << S_pion << "  b=" << b_pion << " ===\n";
    std::cerr << "=== Phase 1: Final selection S=" << S_pion
              << "  b=" << b_pion << " ===\n";

    long double E_best = run_selection(sys, target_n, n_cand, A_min, A_max);
    std::cout << "After selection: E=" << std::fixed << std::setprecision(4)
              << E_best << " MeV\n";
    std::cerr << "[main] After selection: E=" << E_best << " MeV\n\n";

    // ----------------------------------------------------------
    // 7. Phase 2 — Refining
    // ----------------------------------------------------------
    std::cout << "\n=== Phase 2: Refining ===\n";
    std::cerr << "\n=== Phase 2: Refining ===\n";

    for (int cycle = 1; cycle <= n_refine; ++cycle) {
        long double E_start = E_best;

        for (int sec = 0; sec < 4; ++sec) {
            bool   bare = (sec == 0);
            size_t dim  = sys.jac[sec].dim();

            for (size_t k = 0; k < sys.basis[sec].size(); ++k) {
                gaus best_g    = sys.basis[sec][k];
                long double best_E = E_best;

                for (int c = 0; c < n_cand; ++c) {
                    sys.basis[sec][k] = bare
                        ? make_swave(dim, A_min, A_max)
                        : make_pwave(dim, A_min, A_max);
                    sys.build(H_cur, N_cur);
                    long double E = solve(H_cur, N_cur);
                    if (!std::isnan(E) && E < best_E) {
                        best_E = E; best_g = sys.basis[sec][k];
                    }
                }
                sys.basis[sec][k] = best_g;
                E_best = best_E;
            }
        }

        long double delta = E_start - E_best;
        std::cout << std::fixed << std::setprecision(4)
                  << "  cycle=" << cycle
                  << "  E=" << std::setw(10) << E_best
                  << " MeV  delta=" << delta << " MeV\n";
        std::cerr << "  cycle=" << cycle
                  << "  E=" << std::setw(10) << E_best
                  << " MeV  delta=" << delta << " MeV\n";
    }

    // ----------------------------------------------------------
    // 6. Result
    // ----------------------------------------------------------
    sys.build(H_cur, N_cur);
    E_best = solve(H_cur, N_cur);

    std::cout << "\n==========================================\n";
    std::cout << "  Target      :  -2.225000 MeV\n";
    std::cout << "  Result      : " << std::setw(10) << E_best << " MeV\n";
    std::cout << "  S_pion      :  " << S_pion  << " MeV\n";
    std::cout << "  b_pion      :  " << b_pion  << " fm\n";
    std::cout << "  relativistic:  " << (rel ? "yes" : "no") << "\n";
    std::cout << "  Basis       :  "
              << sys.basis[0].size() << " + " << sys.basis[1].size()
              << " + " << sys.basis[2].size() << " + "
              << sys.basis[3].size() << "\n";
    std::cout << "==========================================\n";

    return 0;
}