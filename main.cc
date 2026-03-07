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

// ============================================================
//  main.cc  —  Deuteron with explicit pion exchange
//              Two-channel spin split, Thread-Safe SVM, 
//              Matrix Caching, and Self-Consistent Macro-Cycles
// ============================================================

using namespace qm;

struct Params {
    std::map<std::string, std::string> store;

    Params() {
        store["S"]        = "15.0";
        store["b"]        = "2.0";
        store["n_pn"]     = "15";   
        store["n_pnπ"]   = "20";   
        store["n_cand"]   = "30";   
        store["n_refine"] = "3";
        store["mean_r"]   = "2.0";
        store["mean_R"]   = "1.5";
        store["rel"]      = "0";
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
        for (auto& kv : store)
            std::cerr << "  " << kv.first << " = " << kv.second << "\n";
    }
};

struct PionSystem {
    static const int N_SEC = 7;

    std::vector<gaus> basis[N_SEC];
    jacobian          jac_phys[4];   
    long double       meson_mass[4]; 
    long double       iso_weight[4]; 
    long double       S_pion;
    long double       b_pion;
    bool              relativistic;
    hamiltonian       H;

    static int phys(int sec) { return (sec <= 0) ? 0 : (sec + 1) / 2; }
    static bool is_chan_A(int sec) { return (sec > 0) && (sec % 2 == 1); }
    const jacobian& jac(int sec) const { return jac_phys[phys(sec)]; }

    vector pion_coord(int sec, int emitting_idx) const {
        long double m0   = jac(sec).particles[0].mass;
        long double m1   = jac(sec).particles[1].mass;
        long double mtot = m0 + m1;
        vector c(2);
        if (emitting_idx == 0) { c[0] = +m1 / mtot; c[1] = 1.0L; } 
        else                   { c[0] = -m0 / mtot; c[1] = 1.0L; }
        return c;
    }

    matrix Omega(int sec, const vector& c_coord) const {
        size_t d = jac(sec).dim();
        matrix Om(d, d);
        long double inv_b2 = 1.0L / (b_pion * b_pion);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                Om(i,j) = inv_b2 * c_coord[i] * c_coord[j];
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

    void calc_row(int target_sec, const gaus& g_target, std::vector<long double>& h_row, std::vector<long double>& n_row, long double& h_diag, long double& n_diag) const {
        size_t N = total_size();
        h_row.assign(N, 0.0L); n_row.assign(N, 0.0L);

        n_diag = overlap(g_target, g_target);
        long double k_diag = 0.0L;
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
                long double h_val = 0.0L, n_val = 0.0L;
                if (sec == target_sec) {
                    n_val = overlap(basis[sec][i], g_target);
                    long double k_val = 0.0L;
                    for (size_t coord = 0; coord < j_tgt.dim(); ++coord) {
                        if (relativistic) k_val += H.K_rel(basis[sec][i], g_target, j_tgt.c(coord), j_tgt.mu(coord));
                        else              k_val += H.K_cla(basis[sec][i], g_target, j_tgt.c(coord), j_tgt.mu(coord));
                    }
                    h_val = k_val + meson_mass[phys(target_sec)] * n_val;
                } else if (sec == 0 || target_sec == 0) {
                    int clothed_sec = (sec == 0) ? target_sec : sec;
                    const gaus& g_bare = (sec == 0) ? basis[0][i] : g_target;
                    const gaus& g_cloth = (sec == 0) ? g_target : basis[sec][i];
                    int p = phys(clothed_sec);
                    int emitting_idx = (p == 3) ? 1 : 0;
                    vector c_coord = pion_coord(clothed_sec, emitting_idx);
                    matrix Om = Omega(clothed_sec, c_coord);

                    vector beta(3);
                    long double strength = iso_weight[p] * (S_pion / b_pion);
                    if (is_chan_A(clothed_sec)) beta[2] = strength;
                    else beta[0] = strength * std::sqrt(2.0L);

                    h_val = H.W(g_bare, g_cloth, Om, c_coord, 0.0L, beta);
                    n_val = 0.0L;
                }
                h_row[off + i] = h_val;
                n_row[off + i] = n_val;
            }
        }
    }

    void insert_matrix(const matrix& H_old, const matrix& N_old, int target_sec, const gaus& trial, matrix& H_new, matrix& N_new) const {
        size_t N = H_old.size1();
        matrix H_temp(N + 1, N + 1); 
        matrix N_temp(N + 1, N + 1);

        std::vector<long double> h_row, n_row;
        long double h_diag, n_diag;
        calc_row(target_sec, trial, h_row, n_row, h_diag, n_diag);

        size_t insert_idx = offset(target_sec) + basis[target_sec].size();
        auto map_idx = [&](size_t old_i) { return (old_i >= insert_idx) ? old_i + 1 : old_i; };

        for (size_t i = 0; i < N; ++i) {
            size_t new_i = map_idx(i);
            for (size_t j = 0; j < N; ++j) {
                size_t new_j = map_idx(j);
                H_temp(new_i, new_j) = H_old(i, j);
                N_temp(new_i, new_j) = N_old(i, j);
            }
            H_temp(new_i, insert_idx) = h_row[i]; H_temp(insert_idx, new_i) = h_row[i];
            N_temp(new_i, insert_idx) = n_row[i]; N_temp(insert_idx, new_i) = n_row[i];
        }
        H_temp(insert_idx, insert_idx) = h_diag; 
        N_temp(insert_idx, insert_idx) = n_diag;
        
        H_new = H_temp; N_new = N_temp;
    }

    void replace_matrix(const matrix& H_old, const matrix& N_old, int target_sec, size_t target_k, const gaus& trial, matrix& H_new, matrix& N_new) const {
        size_t N = H_old.size1();
        matrix H_temp = H_old; 
        matrix N_temp = N_old;

        std::vector<long double> h_row, n_row;
        long double h_diag, n_diag;
        calc_row(target_sec, trial, h_row, n_row, h_diag, n_diag);

        size_t global_idx = offset(target_sec) + target_k;
        for (size_t i = 0; i < N; ++i) {
            if (i == global_idx) continue;
            H_temp(i, global_idx) = h_row[i]; H_temp(global_idx, i) = h_row[i];
            N_temp(i, global_idx) = n_row[i]; N_temp(global_idx, i) = n_row[i];
        }
        H_temp(global_idx, global_idx) = h_diag; 
        N_temp(global_idx, global_idx) = n_diag;
        
        H_new = H_temp; N_new = N_temp;
    }

    void build_full(matrix& H_tot, matrix& N_tot) const {
        size_t N = total_size();
        H_tot = matrix(N, N); N_tot = matrix(N, N);

        for (int sec = 0; sec < N_SEC; ++sec) {
            size_t n_s = basis[sec].size();
            size_t off = offset(sec);
            int p = phys(sec);
            const jacobian& j = jac(sec);
            for (size_t i = 0; i < n_s; ++i) {
                for (size_t k = 0; k < n_s; ++k) {
                    long double nij = overlap(basis[sec][i], basis[sec][k]);
                    N_tot(off+i, off+k) = nij;
                    long double kij = 0.0L;
                    for (size_t coord = 0; coord < j.dim(); ++coord) {
                        if (relativistic) kij += H.K_rel(basis[sec][i], basis[sec][k], j.c(coord), j.mu(coord));
                        else              kij += H.K_cla(basis[sec][i], basis[sec][k], j.c(coord), j.mu(coord));
                    }
                    H_tot(off+i, off+k) = kij + meson_mass[p] * nij;
                }
            }
        }

        size_t n0_b = basis[0].size();
        for (int sec = 1; sec < N_SEC; ++sec) {
            size_t n_s = basis[sec].size();
            size_t off_s = offset(sec);
            int p = phys(sec);
            int emitting_idx = (p == 3) ? 1 : 0;
            vector c_coord = pion_coord(sec, emitting_idx);
            matrix Om = Omega(sec, c_coord);

            vector beta(3);
            long double strength = iso_weight[p] * (S_pion / b_pion);
            if (is_chan_A(sec)) beta[2] = strength;                    
            else beta[0] = strength * std::sqrt(2.0L); 

            for (size_t i = 0; i < n0_b; ++i) {
                for (size_t k = 0; k < n_s; ++k) {
                    long double w = H.W(basis[0][i], basis[sec][k], Om, c_coord, 0.0L, beta);
                    H_tot(i, off_s+k) = w; H_tot(off_s+k, i) = w;
                }
            }
        }
    }
};

static long double solve(const matrix& H_tot, const matrix& N_tot) {
    EigenResult sys = solve_generalized_eigensystem(H_tot, N_tot);
    if (sys.evals.size() == 0) return std::numeric_limits<long double>::quiet_NaN();
    // Because we sorted them, the ground state is always strictly index 0!
    return sys.evals[0]; 
}

static gaus make_swave(size_t dim, long double mean_r) {
    gaus g(dim, mean_r, 0.0L); g.zero_shifts(); return g;
}

static gaus make_pwave(size_t dim, long double mean_r, long double mean_R, bool chan_A) {
    gaus g(dim, mean_r, mean_R);
    
    // Safety Fix: Ensure core S-wave coordinates have 0 shift!
    for (size_t i = 0; i < dim - 1; ++i) {
        for (size_t k = 0; k < 3; ++k) {
            g.s(i, k) = 0.0L;
        }
    }
    
    if (dim >= 2) {
        if (chan_A) { g.s(dim-1, 0) = 0.0L; g.s(dim-1, 1) = 0.0L; } 
        else        { g.s(dim-1, 1) = 0.0L; g.s(dim-1, 2) = 0.0L; }
    }
    return g;
}

static long double run_selection(PionSystem& sys, const size_t* target_n, int n_cand, long double mean_r, long double mean_R) {
    for (int s = 0; s < PionSystem::N_SEC; ++s) sys.basis[s].clear();

    size_t total_slots = 0;
    for (int s = 0; s < PionSystem::N_SEC; ++s) total_slots += target_n[s];

    matrix H_master(0,0), N_master(0,0);
    long double E_cur = std::numeric_limits<long double>::quiet_NaN();

    for (size_t slot = 0; slot < total_slots; ++slot) {
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
            std::cout << "\n[Warning] Slot " << slot + 1 << " saturated! Skipping addition.\n";
            continue; 
        }
        
        sys.insert_matrix(H_master, N_master, add_sec, global_best_g, H_master, N_master);
        sys.basis[add_sec].push_back(global_best_g);

        std::cout << "\r  [SVM] Building basis... slot " << std::setw(3) << slot + 1 
                  << "/" << total_slots 
                  << " | E = " << std::fixed << std::setprecision(5) << global_best_E << std::flush;
        E_cur = global_best_E;
    }
    
    std::cout << "\n"; 
    return E_cur;
}

int main(int argc, char* argv[])
{
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
    const int         n_opt_S  = cfg.i("n_opt_S");
    const int         n_opt_b  = cfg.i("n_opt_b");
    const long double k_S      = cfg.ld("k_S");
    long double       k_b      = cfg.ld("k_b");
    const long double E_target = cfg.ld("target");

    std::cout << "[main] ============================================\n";
    std::cout << "[main] Deuteron + explicit pion  (7-sector SVM)\n";
    std::cout << "[main] Thread-Safe Multithreading + Matrix Caching\n";
    std::cout << "[main] Self-Consistent Macro-Iterations Enabled\n";
    std::cout << "[main] ============================================\n";

    Nucleon proton = Nucleon::Proton(); Nucleon neutron = Nucleon::Neutron();
    Pion pi0 = Pion::PiZero(); Pion piplus = Pion::PiPlus(); Pion piminus = Pion::PiMinus();
    VertexResult v0 = apply_pion_emission(proton, pi0);
    VertexResult v_p = apply_pion_emission(proton, piplus);
    VertexResult v_m = apply_pion_emission(neutron, piminus);

    PionSystem sys;
    sys.relativistic = rel; sys.S_pion = S_pion; sys.b_pion = b_pion;
    sys.jac_phys[0] = jacobian({proton, neutron});
    sys.jac_phys[1] = jacobian({proton, neutron, pi0});
    sys.jac_phys[2] = jacobian({neutron, neutron, piplus});
    sys.jac_phys[3] = jacobian({proton, proton, piminus});
    sys.meson_mass[0] = 0.0L; sys.meson_mass[1] = pi0.mass; sys.meson_mass[2] = piplus.mass; sys.meson_mass[3] = piminus.mass;
    sys.iso_weight[0] = 0.0L; sys.iso_weight[1] = v0.coefficient; sys.iso_weight[2] = v_p.coefficient; sys.iso_weight[3] = v_m.coefficient;

    const size_t target_n[PionSystem::N_SEC] = {n_pn, n_pnπ, n_pnπ, n_pnπ, n_pnπ, n_pnπ, n_pnπ};
    matrix H_cur, N_cur;
    long double E_best = 0.0;

    std::cout << std::fixed << std::setprecision(4);
    
    // Generate initial rough basis for the very first tuning cycle
    std::cout << "\n=== Phase 0.0: Generating Initial Rough Basis ===\n";
    run_selection(sys, target_n, n_cand, mean_r, mean_R);

    // ============================================================
    // MACRO-ITERATION LOOP: Self-Consistent Tuning & Refinement
    // ============================================================
    int num_macro_cycles = 2; // 2 cycles is generally enough to lock on
    
    for (int macro = 1; macro <= num_macro_cycles; ++macro) {
        std::cout << "\n======================================================\n";
        std::cout << "               MACRO CYCLE " << macro << " OF " << num_macro_cycles << "              \n";
        std::cout << "======================================================\n";

        // --- TUNING PHASES (Uses currently frozen basis) ---
        std::cout << "\n=== Phase 0a: Tuning S (Frozen Basis) ===\n";
        long double E_cur = std::numeric_limits<long double>::quiet_NaN();
        long double current_k_S = std::abs(k_S); // Ensure it's positive for our explicit logic
        long double previous_err_S = 0.0;

        for (int it = 0; it < n_opt_S; ++it) {
            sys.S_pion = S_pion; sys.b_pion = b_pion; // b_pion is now fixed at 1.0!
            sys.build_full(H_cur, N_cur);
            E_cur = solve(H_cur, N_cur);
            long double err = E_target - E_cur;
            
            std::cout << "  S it=" << std::setw(2) << it << "  S=" << std::setw(9) << S_pion
                      << "  b=" << std::setw(6) << b_pion << "  E=" << std::setw(10) << E_cur
                      << "  err=" << std::setw(9) << err << " MeV\n";
                      
            //if (std::abs(err) < 0.05L) { std::cout << "  [S converged]\n"; break; }
            
            // Adaptive step-size: if error flips sign, we overshot. Cut k_S in half.
            if (it > 0 && (err * previous_err_S < 0)) { 
                current_k_S *= 0.5; 
            }
            
            // Calculate step size, but strictly CAP it at a safe 2.0 MeV jump
            long double step = std::abs(current_k_S * err);
            if (step > 2.0L) step = 2.0L; 
            
            if (err > 0) {
                // Energy is too deep. Weaken the potential.
                S_pion -= step;
            } else {
                // Energy is too shallow. Strengthen the potential.
                S_pion += step;
            }
            
            // THE SAFETY CATCH: Never let S cross zero!
            if (S_pion < 0.1L) S_pion = 0.1L; 
            
            previous_err_S = err;
        }

        std::cout << "\n=== Phase 0b: Tuning b (Frozen Basis) ===\n";
        long double previous_err = 0.0;
        for (int it = 0; it < n_opt_b; ++it) {
            sys.S_pion = S_pion; sys.b_pion = b_pion;
            sys.build_full(H_cur, N_cur);
            E_cur = solve(H_cur, N_cur);
            long double err = E_target - E_cur;
            std::cout << "  b it=" << std::setw(2) << it << "  S=" << std::setw(9) << S_pion
                      << "  b=" << std::setw(6) << b_pion << "  E=" << std::setw(10) << E_cur
                      << "  err=" << std::setw(9) << err << " MeV\n";
            //if (std::abs(err) < 0.05L) { std::cout << "  [b converged]\n"; break; }
            if (it > 0 && (err * previous_err < 0)) { k_b *= 0.5; std::cout << "      [Overshoot detected! Halving k_b to " << k_b << "]\n"; }
            b_pion += k_b * err;
            previous_err = err;
        }

        // --- FULL BASIS SELECTION (Only performed on the FIRST macro cycle) ---
        if (macro == 1) {
            std::cout << "\n=== Phase 1: Full Competitive Selection (S=" << S_pion << " b=" << b_pion << ") ===\n";
            E_best = run_selection(sys, target_n, n_cand, mean_r, mean_R);
        } else {
            // For subsequent cycles, we just re-evaluate the already full, refined basis
            sys.build_full(H_cur, N_cur);
            E_best = solve(H_cur, N_cur);
            std::cout << "\n[Skipping Phase 1 selection, basis is already full. Current E=" << E_best << " MeV]\n";
        }
        
        // --- REFINING PHASE ---
        std::cout << "\n=== Phase 2: Refining ===\n";
        size_t total_states = sys.total_size();
        
        matrix H_master, N_master;
        sys.build_full(H_master, N_master);

        for (int cycle = 1; cycle <= n_refine; ++cycle) {
            long double E_start = E_best;
            size_t current_state_idx = 0;

            for (int sec = 0; sec < PionSystem::N_SEC; ++sec) {
                bool bare = (sec == 0);
                bool chan_A = PionSystem::is_chan_A(sec);
                size_t dim = sys.jac(sec).dim();

                for (size_t k = 0; k < sys.basis[sec].size(); ++k) {
                    current_state_idx++;
                    long double global_best_E = E_best;
                    gaus global_best_g = sys.basis[sec][k];

                    std::vector<gaus> candidates;
                    for(int c = 0; c < n_cand; ++c) candidates.push_back(bare ? make_swave(dim, mean_r) : make_pwave(dim, mean_r, mean_R, chan_A));

                    #pragma omp parallel
                    {
                        long double local_best_E = global_best_E;
                        gaus local_best_g = global_best_g;

                        #pragma omp for
                        for (int c = 0; c < n_cand; ++c) {
                            const gaus& trial = candidates[c];
                            matrix H_trial, N_trial;
                            sys.replace_matrix(H_master, N_master, sec, k, trial, H_trial, N_trial);
                            long double E = solve(H_trial, N_trial);
                            if (!std::isnan(E) && E < local_best_E) { local_best_E = E; local_best_g = trial; }
                        }

                        #pragma omp critical
                        {
                            if (local_best_E < global_best_E) { global_best_E = local_best_E; global_best_g = local_best_g; }
                        }
                    }
                    
                    sys.replace_matrix(H_master, N_master, sec, k, global_best_g, H_master, N_master);
                    sys.basis[sec][k] = global_best_g;
                    E_best = global_best_E;

                    std::cout << "\r  [Refine] Cycle " << cycle << "/" << n_refine 
                              << " | State " << std::setw(3) << current_state_idx << "/" << total_states 
                              << " | E = " << std::fixed << std::setprecision(5) << E_best << std::flush;
                }
            }
            std::cout << "\n  => Cycle " << cycle << " finished.  Improved by " << (E_start - E_best) << " MeV\n";
            if ((E_start - E_best) < 1e-4) { std::cout << "  [!] Energy converged. Stopping refinement early.\n"; break; }
        }
    }

    // ============================================================
    // Phase 3: Final Parameter Lock
    // Now that the basis is perfectly refined, do one last tune
    // to map the parameters exactly to the binding energy.
    // ============================================================
    std::cout << "\n=== Phase 3: Final Parameter Lock ===\n";
    long double current_k_S_final = std::abs(k_S);
    long double previous_err_final = 0.0;
    
    for (int it = 0; it < n_opt_S; ++it) {
        sys.S_pion = S_pion; 
        sys.build_full(H_cur, N_cur);
        E_best = solve(H_cur, N_cur);
        long double err = E_target - E_best;
        
        //if (std::abs(err) < 0.01L) break;
        
        if (it > 0 && (err * previous_err_final < 0)) current_k_S_final *= 0.5; 
        
        long double step = std::abs(current_k_S_final * err);
        if (step > 2.0L) step = 2.0L; 
        
        if (err > 0) S_pion -= step;
        else         S_pion += step;
        
        if (S_pion < 0.1L) S_pion = 0.1L;
        previous_err_final = err;
    }

    // Final result output
    sys.build_full(H_cur, N_cur);
    E_best = solve(H_cur, N_cur);

    
    // ============================================================
    // Phase 4: Observables (Charge Radius)
    // ============================================================
    std::cout << "\n=== Phase 4: Calculating Observables ===\n";
    
    // 1. Get the full eigensystem to extract the wavefunctions
    sys.build_full(H_cur, N_cur);
    EigenResult final_sys = solve_generalized_eigensystem(H_cur, N_cur);
    
    if (final_sys.evals.size() > 0) {
        //long double E_final = final_sys.evals[0];
        size_t N_total = final_sys.evecs.size1();
        
        // 2. Extract the ground state eigenvector (Column 0)
        std::vector<long double> c(N_total);
        for (size_t i = 0; i < N_total; ++i) {
            c[i] = final_sys.evecs(i, 0);
        }
        
        // 3. Calculate <r^2> and normalization <N>
        long double r2_exp = 0.0L;
        long double norm   = 0.0L;
        
        // Loop over the matrix blocks
        for (int sec_i = 0; sec_i < PionSystem::N_SEC; ++sec_i) {
            size_t off_i = sys.offset(sec_i);
            
            for (size_t i = 0; i < sys.basis[sec_i].size(); ++i) {
                size_t global_i = off_i + i;
                const gaus& g_i = sys.basis[sec_i][i];
                
                for (int sec_j = 0; sec_j < PionSystem::N_SEC; ++sec_j) {
                    size_t off_j = sys.offset(sec_j);
                    
                    for (size_t j = 0; j < sys.basis[sec_j].size(); ++j) {
                        size_t global_j = off_j + j;
                        const gaus& g_j = sys.basis[sec_j][j];
                        
                        // r^2 operator does not create/destroy pions, 
                        // so it only has non-zero overlaps within the SAME sector!
                        if (sec_i == sec_j) {
                            long double r2_val = sys.H.R2_matrix_element(g_i, g_j);
                            long double n_val  = overlap(g_i, g_j);
                            
                            r2_exp += c[global_i] * c[global_j] * r2_val;
                            norm   += c[global_i] * c[global_j] * n_val;
                        }
                    }
                }
            }
        }
        
        // 4. Enforce perfect normalization to remove any floating-point drift
        r2_exp /= norm; 
        
        // 5. Calculate final Charge Radius
        // r_c = sqrt( 1/4 * <r^2> + r_p^2 + r_n^2 )
        long double r_p_sq = 0.8414L * 0.8414L; // CODATA 2018 proton radius squared
        long double r_n_sq = -0.1161L;          // PDG neutron charge radius squared
        
        long double r_c = std::sqrt(0.25L * r2_exp + r_p_sq + r_n_sq);
        
        std::cout << "\n==========================================\n";
        std::cout << "  Target       : " << E_target << " MeV\n";
        std::cout << "  Result       : " << E_best << " MeV\n";
        std::cout << "  S_pion       : " << S_pion  << " MeV\n";
        std::cout << "  b_pion       : " << b_pion  << " fm\n";
        std::cout << "  Basis sizes  : n_pn=" << n_pn << ", n_pnπ=" << n_pnπ << " per channel\n";
        std::cout << "  Relativistic : " << (rel ? "True\n" : "False\n");
        std::cout << "  --------------------------------------\n";
        std::cout << "  <r^2>_pn    :  " << r2_exp << " fm^2\n";
        std::cout << "  Charge Rad  :  " << r_c << " fm\n";
        std::cout << "==========================================\n";
    }
    return 0;
}