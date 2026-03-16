//
// fastscan.cc  —  Fast (S, b) parameter scan for the pion-deuterium system
//
// Strategy:
//   1. Run a full SVM optimisation at one reference point (S_ref, b_ref)
//      to obtain an optimised Gaussian basis.
//   2. For every (S, b) grid point, rebuild H and N using that FIXED basis
//      with the new coupling parameters, then solve the GEP directly.
//      No SVM trial loop is run for any other point.
//
// This is orders of magnitude faster than running a full SVM per grid point
// (as heatmap.cc does), at the cost of basis quality away from the reference.
// For mapping the landscape and locating the binding region it is excellent;
// for precise energies at specific points use heatmap.cc or main.cc.
//
// The grid evaluation in step 2 is embarrassingly parallel — each (S,b)
// point only needs a matrix build + eigensolve with the shared read-only basis.
//
// Usage:
//   ./fastscan [options]
//
// Options:
//   Reference point (where SVM is run):
//     --S_ref v        Coupling strength for basis optimisation  (default: 20.0)
//     --b_ref v        Form-factor range  for basis optimisation (default: 1.4)
//
//   SVM (only affects the single reference run):
//     --K_max N        Basis size                               (default: 25)
//     --N_trial N      Trial candidates per SVM step            (default: 50)
//     --refine_every N Refinement period (0=off)                (default: 10)
//     --N_refine N     Candidates per refinement step           (default: 20)
//     --b0 v           Gaussian length scale [fm]               (default: 1.4)
//     --s_max v        Shift vector bound [fm^-1]               (default: 0.0)
//     --seed N         RNG seed                                 (default: 42)
//     --relativistic   Use relativistic KE                      (default: classical)
//
//   Grid (evaluated with fixed basis):
//     --S_min v        (default: 5.0)
//     --S_max v        (default: 60.0)
//     --N_S n          (default: 20)
//     --b_min v        (default: 0.5)
//     --b_max v        (default: 4.0)
//     --N_b n          (default: 20)
//
//   Output:
//     --output file    (default: fastscan.dat)
//     --E_clip v       Clip unbound values  (default: 10.0)
//
// Build:
//   g++ -std=c++17 -O2 -fopenmp -o fastscan fastscan.cc
//

#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/solver.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// build_pion_channels  (identical to main.cc / heatmap.cc)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_pion_channels(const JacobiSystem& sys)
{
    constexpr ld m_pi0   = ld{134.977L};
    constexpr ld m_pi_pm = ld{139.570L};
    constexpr ld c1 =  ld{1};
    const     ld c2 =  std::sqrt(ld{2});
    const     ld c3 =  std::sqrt(ld{2});
    constexpr ld c4 = -ld{1};

    rvec w_pp = sys.w_meson_proton();
    rvec w_pn = sys.w_meson_neutron();

    auto make = [&](int idx, ld mp, ld iso, const rvec& w, SpinType st) {
        Channel ch;
        ch.index = idx; ch.is_bare = false; ch.pion_mass = mp;
        ch.iso_coeff = iso; ch.w_piN = w; ch.spin_type = st;
        ch.dim = sys.N - 1;
        return ch;
    };

    std::vector<Channel> ch(9);
    ch[0].index = 0; ch[0].is_bare = true; ch[0].pion_mass = ld{0};
    ch[0].iso_coeff = ld{0}; ch[0].w_piN = rvec(sys.N-1);
    ch[0].spin_type = SpinType::NO_FLIP; ch[0].dim = 1;

    ch[1] = make(1, m_pi0,   +c1, w_pp, SpinType::NO_FLIP);
    ch[2] = make(2, m_pi_pm, +c2, w_pp, SpinType::NO_FLIP);
    ch[3] = make(3, m_pi_pm, +c3, w_pn, SpinType::NO_FLIP);
    ch[4] = make(4, m_pi0,   +c4, w_pn, SpinType::NO_FLIP);
    ch[5] = make(5, m_pi0,   +c1, w_pp, SpinType::SPIN_FLIP);
    ch[6] = make(6, m_pi_pm, +c2, w_pp, SpinType::SPIN_FLIP);
    ch[7] = make(7, m_pi_pm, +c3, w_pn, SpinType::SPIN_FLIP);
    ch[8] = make(8, m_pi0,   +c4, w_pn, SpinType::SPIN_FLIP);
    return ch;
}


// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    // Reference point — SVM is run here
    ld   S_ref  = ld{20.0L};
    ld   b_ref  = ld{1.4L};

    // SVM parameters
    int  K_max        = 25;
    int  N_trial      = 50;
    int  refine_every = 10;
    int  N_refine     = 20;
    ld   b0           = ld{1.4L};
    ld   s_max        = ld{0.1L};
    uint64_t seed     = 42;
    bool relativistic = false;

    // Grid
    ld   S_min = ld{5.0L};
    ld   S_max = ld{60.0L};
    int  N_S   = 20;
    ld   b_min = ld{0.5L};
    ld   b_max = ld{4.0L};
    int  N_b   = 20;

    // Output
    std::string output = "fastscan.dat";
    ld   E_clip = ld{10.0L};
};

Config parse_args(int argc, char* argv[])
{
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << a << "\n"; std::exit(1);
            }
            return argv[++i];
        };
        auto f = [&]{ return static_cast<ld>(std::stold(next())); };
        auto n = [&]{ return std::stoi(next()); };

        if      (a == "--S_ref")        cfg.S_ref        = f();
        else if (a == "--b_ref")        cfg.b_ref        = f();
        else if (a == "--K_max")        cfg.K_max        = n();
        else if (a == "--N_trial")      cfg.N_trial      = n();
        else if (a == "--refine_every") cfg.refine_every = n();
        else if (a == "--N_refine")     cfg.N_refine     = n();
        else if (a == "--b0")           cfg.b0           = f();
        else if (a == "--s_max")        cfg.s_max        = f();
        else if (a == "--seed")         cfg.seed         = static_cast<uint64_t>(std::stoull(next()));
        else if (a == "--relativistic") cfg.relativistic = true;
        else if (a == "--S_min")        cfg.S_min        = f();
        else if (a == "--S_max")        cfg.S_max        = f();
        else if (a == "--N_S")          cfg.N_S          = n();
        else if (a == "--b_min")        cfg.b_min        = f();
        else if (a == "--b_max")        cfg.b_max        = f();
        else if (a == "--N_b")          cfg.N_b          = n();
        else if (a == "--output")       cfg.output       = next();
        else if (a == "--E_clip")       cfg.E_clip       = f();
        else {
            std::cerr << "Unknown option: " << a << "\n"; std::exit(1);
        }
    }
    return cfg;
}

void print_progress(int done, int total, double elapsed_s)
{
    int width  = 40;
    int filled = (total > 0) ? (done * width / total) : 0;
    double eta = (done > 0 && done < total)
                 ? elapsed_s / done * (total - done) : 0.0;
    std::cout << "\r  [";
    for (int i = 0; i < width; i++) std::cout << (i < filled ? '=' : ' ');
    std::cout << "]  " << done << "/" << total
              << "  elapsed=" << std::fixed << std::setprecision(1) << elapsed_s << "s";
    if (done > 0 && done < total)
        std::cout << "  ETA=" << std::setprecision(0) << eta << "s";
    std::cout << "   " << std::flush;
}


// ─────────────────────────────────────────────────────────────────────────────
// eval_fixed_basis  —  solve GEP for a fixed basis at new (S, b)
//
// The basis Gaussians are unchanged.  Only the coupling parameters fed into
// HamiltonianBuilder change, so H is rebuilt but no SVM trial loop runs.
// Cost: O(K^2) matrix elements + O(K^3) eigensolve — very fast.
// ─────────────────────────────────────────────────────────────────────────────
ld eval_fixed_basis(const JacobiSystem&          sys,
                    const std::vector<Channel>&  channels,
                    const std::vector<Gaussian>& basis_bare,
                    const std::vector<Gaussian>& basis_dressed,
                    ld S, ld b,
                    bool relativistic)
{
    HamiltonianBuilder hb(sys, channels,
                          basis_bare, basis_dressed,
                          b, S, relativistic);
    cmat H = hb.build_H();
    cmat N = hb.build_N();
    return solve_gevp(H, N);
}


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    Config cfg = parse_args(argc, argv);

    std::cout << "\n";
    std::cout << "+============================================================+\n";
    std::cout << "|   Fast scan  E0(S,b)  with fixed optimised basis           |\n";
    std::cout << "+============================================================+\n\n";

    // ── Physical setup ────────────────────────────────────────────────────────
    const ld m_proton  = ld{938.272046L};
    const ld m_neutron = ld{939.565379L};
    const ld m_pion    = ld{139.570L};

    JacobiSystem sys({m_proton, m_neutron, m_pion}, {"p", "n", "pi"});
    auto channels = build_pion_channels(sys);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "KE type:   " << (cfg.relativistic ? "relativistic" : "classical") << "\n";
    std::cout << "Reference: S_ref=" << cfg.S_ref << " MeV,  b_ref=" << cfg.b_ref << " fm\n";
    std::cout << "Grid:      S in [" << cfg.S_min << ", " << cfg.S_max << "] x " << cfg.N_S
              << "  b in [" << cfg.b_min << ", " << cfg.b_max << "] x " << cfg.N_b << "\n\n";

    // Pre-warm Gauss-Legendre static cache before any parallel work
    {
        Gaussian g(ld{1.0L}, ld{0.0L});
        GaussianPair gp(g, g);
        rvec c(1); c[0] = ld{1};
        KineticParams kp(gp, c);
        (void) ke_relativistic(gp, kp, sys.mu[0]);
    }

    // ── Step 1: run SVM at the reference point ────────────────────────────────
    std::cout << "Step 1: SVM optimisation at reference point...\n";

    SvmParams svm;
    svm.K_max          = cfg.K_max;
    svm.N_trial        = cfg.N_trial;
    svm.refine_every   = cfg.refine_every;
    svm.N_refine_trial = cfg.N_refine;
    svm.b0             = cfg.b0;
    svm.s_max          = cfg.s_max;
    svm.b_ff           = cfg.b_ref;
    svm.S_coupling     = cfg.S_ref;
    svm.relativistic   = cfg.relativistic;
    svm.verbose        = true;

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed));

    auto t0  = std::chrono::steady_clock::now();
    SvmState ref = run_svm(sys, channels, svm, rng);
    double t_svm = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();

    std::cout << "\nReference done: E0 = " << std::fixed << std::setprecision(4)
              << ref.E0 << " MeV  (K=" << ref.K() << ",  " << t_svm << " s)\n\n";

    // ── Step 2: build (S, b) grids ────────────────────────────────────────────
    std::vector<ld> S_vals(cfg.N_S), b_vals(cfg.N_b);

    if (cfg.N_S == 1) {
        S_vals[0] = cfg.S_min;
    } else {
        ld dS = (cfg.S_max - cfg.S_min) / (cfg.N_S - 1);
        for (int i = 0; i < cfg.N_S; i++) S_vals[i] = cfg.S_min + i * dS;
    }

    if (cfg.N_b == 1) {
        b_vals[0] = cfg.b_min;
    } else {
        // Log spacing for b — physical range spans an order of magnitude
        ld log_min = std::log(cfg.b_min);
        ld log_max = std::log(cfg.b_max);
        ld dlog    = (log_max - log_min) / (cfg.N_b - 1);
        for (int i = 0; i < cfg.N_b; i++)
            b_vals[i] = std::exp(log_min + i * dlog);
    }

    int total = cfg.N_S * cfg.N_b;
    std::vector<std::vector<ld>> results(cfg.N_S, std::vector<ld>(cfg.N_b, ld{0}));

    // ── Step 3: evaluate fixed basis over the grid in parallel ────────────────
    // Each (S,b) point only needs HamiltonianBuilder + solve_gevp.
    // The basis vectors are read-only — no thread safety issues.
    // Nested OMP disabled: no outer parallel context here, but be explicit.
    std::cout << "Step 2: evaluating fixed basis over " << total << " grid points";
#ifdef _OPENMP
    omp_set_max_active_levels(1);
    std::cout << "  (" << omp_get_max_threads() << " threads)";
#endif
    std::cout << "\n";
    print_progress(0, total, 0.0);

    std::atomic<int> done{0};
    auto t1 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int iS = 0; iS < cfg.N_S; iS++) {
        for (int ib = 0; ib < cfg.N_b; ib++) {

            ld E0 = eval_fixed_basis(sys, channels,
                                     ref.basis_bare, ref.basis_dressed,
                                     S_vals[iS], b_vals[ib],
                                     cfg.relativistic);

            if (!std::isfinite(E0) || E0 > cfg.E_clip) E0 = cfg.E_clip;
            results[iS][ib] = E0;

            int n = ++done;
            #pragma omp critical
            {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t1).count();
                print_progress(n, total, elapsed);
            }
        }
    }

    double t_grid = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t1).count();
    std::cout << "\n\nGrid done in " << std::fixed << std::setprecision(1)
              << t_grid << " s  (SVM took " << t_svm << " s)\n\n";

    // ── Write output ──────────────────────────────────────────────────────────
    std::ofstream out(cfg.output);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open '" << cfg.output << "'\n";
        return 1;
    }

    out << "# fastscan: E0(S,b) with fixed basis optimised at S="
        << cfg.S_ref << " b=" << cfg.b_ref << "\n";
    out << "# KE: " << (cfg.relativistic ? "relativistic" : "classical")
        << "  K=" << ref.K() << "  seed=" << cfg.seed << "\n";
    out << "# Columns: S[MeV]  b[fm]  E0[MeV]\n#\n";
    out << std::fixed << std::setprecision(6);

    for (int iS = 0; iS < cfg.N_S; iS++) {
        for (int ib = 0; ib < cfg.N_b; ib++)
            out << S_vals[iS] << "  " << b_vals[ib] << "  " << results[iS][ib] << "\n";
        out << "\n";   // blank line between S rows for gnuplot
    }

    std::cout << "Output written to: " << cfg.output << "\n\n";

    // ── Quick summary: find minimum E0 in the grid ────────────────────────────
    ld   E_min  = cfg.E_clip;
    ld   S_best = S_vals[0];
    ld   b_best = b_vals[0];
    for (int iS = 0; iS < cfg.N_S; iS++)
        for (int ib = 0; ib < cfg.N_b; ib++)
            if (results[iS][ib] < E_min) {
                E_min  = results[iS][ib];
                S_best = S_vals[iS];
                b_best = b_vals[ib];
            }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Grid minimum:  E0 = " << E_min
              << " MeV  at  S=" << S_best << " MeV,  b=" << b_best << " fm\n";
    std::cout << "Reference E0:  " << ref.E0 << " MeV  at  S="
              << cfg.S_ref << " MeV,  b=" << cfg.b_ref << " fm\n";
    std::cout << "Target:        -2.2000 MeV\n\n";

    return 0;
}