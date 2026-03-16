//
// heatmap.cc  —  Parameter scan: deuteron E₀(S, b) for the pion-nucleon system
//
// Reproduces (and extends) the calculation from the bachelor's thesis:
//   "Pion-Nucleon Coupling in the Deuteron" (Mikkel Moth Billing)
//
// System:
//   Three particles:  proton (0),  neutron (1),  pion (2)
//   Jacobi coordinates:
//     x₀ = r_p − r_n            (nucleon–nucleon relative coordinate)
//     x₁ = r_π − r_{pn CM}      (pion relative to nucleon CM)
//
// Nine-channel Hamiltonian  (thesis §3–4):
//   ch[0]       bare pn            (1D, dim=1)
//   ch[1]       pn π⁰  no-flip     (proton emits π⁰,  iso=+1,  σ_z)
//   ch[2]       nn π⁺  no-flip     (proton emits π⁺,  iso=+√2, σ_z)
//   ch[3]       pp π⁻  no-flip     (neutron emits π⁻, iso=+√2, σ_z)
//   ch[4]       np π⁰  no-flip     (neutron emits π⁰, iso=−1,  σ_z)
//   ch[5]       pn π⁰  spin-flip   (proton emits π⁰,  iso=+1,  σ_+)
//   ch[6]       nn π⁺  spin-flip   (proton emits π⁺,  iso=+√2, σ_+)
//   ch[7]       pp π⁻  spin-flip   (neutron emits π⁻, iso=+√2, σ_+)
//   ch[8]       np π⁰  spin-flip   (neutron emits π⁰, iso=−1,  σ_+)
//
// Isospin coefficients from τ·π algebra (thesis eq. 4.6):
//   W_iso |pn⟩ = +1·|pnπ⁰⟩  +  √2·|nnπ⁺⟩  +  √2·|ppπ⁻⟩  −  1·|npπ⁰⟩
//
// Coupling operator for each pion channel (thesis §4.1):
//   W = C_iso · S · (σ·r_πN) · exp(−r_πN²/b²)
//
// This file sweeps over a 2D grid of  (S, b)  values and for each point
// runs a full SVM optimisation to find the ground-state energy E₀.
// The result is written as a gnuplot-compatible space-delimited file with
// columns:    S[MeV]   b[fm]   E₀[MeV]
// with a blank line between each S row, enabling  'plot with image'  directly.
//
// Usage:
//   ./heatmap [options]
//
// Options (all have defaults):
//   --K_max N            Basis size per channel         (default: 15)
//   --N_trial N          SVM trial candidates per step  (default: 30)
//   --relativistic       Use relativistic KE            (default: classical)
//   --S_min v            Min coupling strength [MeV]    (default: 5.0)
//   --S_max v            Max coupling strength [MeV]    (default: 60.0)
//   --N_S n              Grid points in S               (default: 10)
//   --b_min v            Min form-factor range [fm]     (default: 0.5)
//   --b_max v            Max form-factor range [fm]     (default: 4.0)
//   --N_b n              Grid points in b               (default: 10)
//   --b0 v               Gaussian length scale [fm]     (default: 1.4)
//   --s_max v            Shift vector bound [fm⁻¹]      (default: 0.0)
//   --refine_every N     Basis refinement interval      (default: 0 = off)
//   --N_refine_trial N   Candidates per refinement step (default: 10)
//   --seed N             RNG seed                       (default: 42)
//   --output file        Output data file               (default: heatmap.dat)
//   --E_clip v           Clip unbound results to this value (default: +10.0)
//
// Build:
//   g++ -std=c++17 -O2 -o heatmap heatmap.cc
//
// Gnuplot example:
//   set palette rgbformulae 33,13,10
//   set cbrange [-5:0]
//   set xlabel "S (MeV)"
//   set ylabel "b (fm)"
//   plot "heatmap.dat" using 1:2:3 with image title "E_0 (MeV)"
//   set contour base
//   set cntrparam levels discrete -2.2
//   splot "heatmap.dat" using 1:2:3 with pm3d
//

#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/solver.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>
#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// build_pion_channels  —  construct the 9-channel pion-deuterium descriptor
//
// Channel layout (thesis §3–4):
//   ch[0]  bare pn          (dim=1)
//   ch[1]  pnπ⁰  NO_FLIP    proton  emits π⁰,  iso=+1
//   ch[2]  nnπ⁺  NO_FLIP    proton  emits π⁺,  iso=+√2
//   ch[3]  ppπ⁻  NO_FLIP    neutron emits π⁻,  iso=+√2
//   ch[4]  npπ⁰  NO_FLIP    neutron emits π⁰,  iso=−1
//   ch[5]  pnπ⁰  SPIN_FLIP  proton  emits π⁰,  iso=+1
//   ch[6]  nnπ⁺  SPIN_FLIP  proton  emits π⁺,  iso=+√2
//   ch[7]  ppπ⁻  SPIN_FLIP  neutron emits π⁻,  iso=+√2
//   ch[8]  npπ⁰  SPIN_FLIP  neutron emits π⁰,  iso=−1
//
// The w_piN vector for each channel is the Jacobi-space extraction vector
// for the pion-nucleon separation:
//   proton  emits: w_piN = w_rel(0, 2) = r_π − r_p
//   neutron emits: w_piN = w_rel(1, 2) = r_π − r_n
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_pion_channels(const JacobiSystem& sys)
{
    // Physical pion masses (PDG 2024)
    constexpr ld m_pi0      = ld{134.977L};   // MeV  π⁰
    constexpr ld m_pi_pm    = ld{139.570L};   // MeV  π⁺ = π⁻  (equal mass)

    // Isospin coefficients from τ·π algebra (thesis eq. 4.6):
    //   W_iso|pn⟩ = +1|pnπ⁰⟩ + √2|nnπ⁺⟩ + √2|ppπ⁻⟩ − 1|npπ⁰⟩
    constexpr ld c1 = ld{1};                   // pnπ⁰  (proton emits)
    const     ld c2 = std::sqrt(ld{2});        // nnπ⁺  (proton emits, p→n)
    const     ld c3 = std::sqrt(ld{2});        // ppπ⁻  (neutron emits, n→p)
    constexpr ld c4 = ld{-1};                  // npπ⁰  (neutron emits)

    // Jacobi-space pion-nucleon separation vectors (2D, for a 3-body system)
    //   Particle ordering:  0 = proton,  1 = neutron,  2 = pion
    //   sys.w_meson_proton()  = w_rel(0,2) = r_π − r_p
    //   sys.w_meson_neutron() = w_rel(1,2) = r_π − r_n
    rvec w_pp = sys.w_meson_proton();   // π emitted from proton
    rvec w_pn = sys.w_meson_neutron();  // π emitted from neutron

    // ── Helper lambda — fill a dressed channel descriptor ────────────────────
    auto make_dressed = [&](int idx, ld m_pion, ld iso, const rvec& w, SpinType st)
    {
        Channel ch;
        ch.index      = idx;
        ch.is_bare    = false;
        ch.pion_mass  = m_pion;
        ch.iso_coeff  = iso;
        ch.w_piN      = w;
        ch.spin_type  = st;
        ch.dim        = sys.N - 1;  // = 2 for a 3-body system
        return ch;
    };

    std::vector<Channel> channels(9);

    // ── ch[0]: bare pn ────────────────────────────────────────────────────────
    channels[0].index     = 0;
    channels[0].is_bare   = true;
    channels[0].pion_mass = ld{0};
    channels[0].iso_coeff = ld{0};
    channels[0].w_piN     = rvec(sys.N - 1);   // zero vector — unused for bare
    channels[0].spin_type = SpinType::NO_FLIP;  // unused for bare channel
    channels[0].dim       = 1;

    // ── Dressed no-flip channels (σ_z component): ch[1..4] ───────────────────
    channels[1] = make_dressed(1, m_pi0,   +c1, w_pp, SpinType::NO_FLIP);
    channels[2] = make_dressed(2, m_pi_pm, +c2, w_pp, SpinType::NO_FLIP);
    channels[3] = make_dressed(3, m_pi_pm, +c3, w_pn, SpinType::NO_FLIP);
    channels[4] = make_dressed(4, m_pi0,   +c4, w_pn, SpinType::NO_FLIP);

    // ── Dressed spin-flip channels (σ_+ component): ch[5..8] ─────────────────
    channels[5] = make_dressed(5, m_pi0,   +c1, w_pp, SpinType::SPIN_FLIP);
    channels[6] = make_dressed(6, m_pi_pm, +c2, w_pp, SpinType::SPIN_FLIP);
    channels[7] = make_dressed(7, m_pi_pm, +c3, w_pn, SpinType::SPIN_FLIP);
    channels[8] = make_dressed(8, m_pi0,   +c4, w_pn, SpinType::SPIN_FLIP);

    return channels;
}


// ─────────────────────────────────────────────────────────────────────────────
// run_one_point  —  run SVM for a single (S, b) pair and return E₀
//
// A fresh RNG seed is derived from the global seed + a hash of the grid
// index so that different (S, b) points are statistically independent but
// remain exactly reproducible given the same master seed.
// ─────────────────────────────────────────────────────────────────────────────
ld run_one_point(const JacobiSystem&          sys,
                 const std::vector<Channel>&  channels,
                 const SvmParams&             params_template,
                 ld                           S_val,
                 ld                           b_val,
                 uint64_t                     point_seed)
{
    SvmParams params  = params_template;
    params.b_ff       = b_val;
    params.S_coupling = S_val;
    params.verbose    = false;  // silence SVM output during scan

    std::mt19937 rng(static_cast<uint32_t>(point_seed));
    SvmState result = run_svm(sys, channels, params, rng);
    return result.E0;
}


// ─────────────────────────────────────────────────────────────────────────────
// parse_args  —  minimal command-line argument parser
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    // SVM parameters
    int  K_max           = 15;
    int  N_trial         = 30;
    int  refine_every    = 0;
    int  N_refine_trial  = 10;
    ld   b0              = ld{1.4L};
    ld   s_max           = ld{0.0L};
    bool relativistic    = false;

    // Grid parameters
    ld   S_min = ld{5.0L};
    ld   S_max = ld{60.0L};
    int  N_S   = 10;
    ld   b_min = ld{0.5L};
    ld   b_max = ld{4.0L};
    int  N_b   = 10;

    // Output
    std::string output = "heatmap.dat";
    ld   E_clip = ld{10.0L};   // positive "unbound" cap for gnuplot colour scale
    uint64_t seed = 42;
};

Config parse_args(int argc, char* argv[])
{
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << a << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        auto nextf = [&]() { return static_cast<ld>(std::stod(next())); };
        auto nexti = [&]() { return std::stoi(next()); };

        if      (a == "--K_max")          cfg.K_max          = nexti();
        else if (a == "--N_trial")        cfg.N_trial        = nexti();
        else if (a == "--refine_every")   cfg.refine_every   = nexti();
        else if (a == "--N_refine_trial") cfg.N_refine_trial = nexti();
        else if (a == "--b0")             cfg.b0             = nextf();
        else if (a == "--s_max")          cfg.s_max          = nextf();
        else if (a == "--relativistic")   cfg.relativistic   = true;
        else if (a == "--S_min")          cfg.S_min          = nextf();
        else if (a == "--S_max")          cfg.S_max          = nextf();
        else if (a == "--N_S")            cfg.N_S            = nexti();
        else if (a == "--b_min")          cfg.b_min          = nextf();
        else if (a == "--b_max")          cfg.b_max          = nextf();
        else if (a == "--N_b")            cfg.N_b            = nexti();
        else if (a == "--E_clip")         cfg.E_clip         = nextf();
        else if (a == "--seed")           cfg.seed           = static_cast<uint64_t>(std::stoull(next()));
        else if (a == "--output")         cfg.output         = next();
        else {
            std::cerr << "Unknown option: " << a << "\n";
            std::cerr << "Run with --help to see options (or check the header of heatmap.cc).\n";
            std::exit(1);
        }
    }
    return cfg;
}


// ─────────────────────────────────────────────────────────────────────────────
// progress_bar  —  crude ASCII progress indicator
// ─────────────────────────────────────────────────────────────────────────────
void print_progress(int done, int total, double elapsed_s)
{
    int width = 40;
    int filled = (total > 0) ? (done * width / total) : 0;
    double eta = (done > 0) ? elapsed_s / done * (total - done) : 0.0;

    std::cout << "\r  [";
    for (int i = 0; i < width; i++)
        std::cout << (i < filled ? '=' : ' ');
    std::cout << "]  " << done << "/" << total;
    std::cout << "  elapsed=" << std::fixed << std::setprecision(1) << elapsed_s << "s";
    if (done > 0 && done < total)
        std::cout << "  ETA=" << std::setprecision(0) << eta << "s";
    std::cout << "  " << std::flush;
}


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    Config cfg = parse_args(argc, argv);

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Deuteron pion-coupling heatmap  —  E₀(S, b)             ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";

    // ── Physical parameters ───────────────────────────────────────────────────
    const ld m_proton  = ld{938.272046L};   // MeV  (PDG)
    const ld m_neutron = ld{939.565379L};   // MeV
    // For the JacobiSystem we pick a "representative" pion mass to define the
    // reduced masses.  The individual channel pion masses are set inside
    // build_pion_channels().  A reasonable representative is m_π⁺ (or π⁰).
    const ld m_pion    = ld{139.570L};      // MeV  (π⁺ as representative)

    // ── Build the Jacobi system ───────────────────────────────────────────────
    // Ordering convention (matches jacobi.h §3-body aliases):
    //   particle 0 = proton,  1 = neutron,  2 = pion
    JacobiSystem sys({m_proton, m_neutron, m_pion}, {"p", "n", "pi"});

    // ── Build the 9-channel descriptor ───────────────────────────────────────
    auto channels = build_pion_channels(sys);

    // ── Print configuration ───────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Physical system:\n";
    std::cout << "  m_p  = " << m_proton  << " MeV\n";
    std::cout << "  m_n  = " << m_neutron << " MeV\n";
    std::cout << "  m_π  = " << m_pion    << " MeV  (representative, channels use PDG masses)\n";
    std::cout << "  μ₀ (pn)  = " << sys.mu[0] << " MeV  (pn reduced mass)\n";
    std::cout << "  μ₁ (π–NN) = " << sys.mu[1] << " MeV  (π–NN reduced mass)\n\n";

    std::cout << "SVM settings:\n";
    std::cout << "  K_max         = " << cfg.K_max << "\n";
    std::cout << "  N_trial       = " << cfg.N_trial << "\n";
    std::cout << "  b0            = " << cfg.b0 << " fm\n";
    std::cout << "  s_max         = " << cfg.s_max << " fm⁻¹\n";
    std::cout << "  refine_every  = " << cfg.refine_every << "\n";
    std::cout << "  KE type       = " << (cfg.relativistic ? "relativistic" : "classical") << "\n";
    std::cout << "  seed          = " << cfg.seed << "\n\n";

    std::cout << "Grid:\n";
    std::cout << "  S:  [" << cfg.S_min << ", " << cfg.S_max << "] MeV  ×  " << cfg.N_S << " points\n";
    std::cout << "  b:  [" << cfg.b_min << ", " << cfg.b_max << "] fm   ×  " << cfg.N_b << " points\n";
    int total_points = cfg.N_S * cfg.N_b;
    std::cout << "  Total grid points: " << total_points << "\n\n";

    // ── Build the SVM parameter template ─────────────────────────────────────
    // b_ff and S_coupling will be overwritten per grid point.
    SvmParams params;
    params.K_max          = cfg.K_max;
    params.N_trial        = cfg.N_trial;
    params.refine_every   = cfg.refine_every;
    params.N_refine_trial = cfg.N_refine_trial;
    params.b0             = cfg.b0;
    params.s_max          = cfg.s_max;
    params.b_ff           = ld{1.0L};   // placeholder — overwritten per point
    params.S_coupling     = ld{1.0L};   // placeholder — overwritten per point
    params.relativistic   = cfg.relativistic;
    params.verbose        = false;

    // ── Build the (S, b) grids ────────────────────────────────────────────────
    std::vector<ld> S_vals(cfg.N_S), b_vals(cfg.N_b);

    // Logarithmic spacing for b (physical range spans an order of magnitude)
    // Linear spacing for S (coupling strength)
    if (cfg.N_S == 1) {
        S_vals[0] = cfg.S_min;
    } else {
        ld dS = (cfg.S_max - cfg.S_min) / (cfg.N_S - 1);
        for (int i = 0; i < cfg.N_S; i++) S_vals[i] = cfg.S_min + i * dS;
    }

    if (cfg.N_b == 1) {
        b_vals[0] = cfg.b_min;
    } else {
        // Use logarithmic spacing: b_vals[i] = b_min * (b_max/b_min)^(i/(N-1))
        ld log_bmin = std::log(cfg.b_min);
        ld log_bmax = std::log(cfg.b_max);
        ld d_log = (log_bmax - log_bmin) / (cfg.N_b - 1);
        for (int i = 0; i < cfg.N_b; i++)
            b_vals[i] = std::exp(log_bmin + i * d_log);
    }

    // ── Open output file ──────────────────────────────────────────────────────
    std::ofstream out(cfg.output);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open output file '" << cfg.output << "'\n";
        return 1;
    }

    // Header comment block
    out << "# Deuteron pion-coupling heatmap: E0(S, b)\n";
    out << "# System:  p + n + π  (9-channel pion-nucleon Hamiltonian)\n";
    out << "# KE type: " << (cfg.relativistic ? "relativistic" : "classical") << "\n";
    out << "# K_max=" << cfg.K_max << "  N_trial=" << cfg.N_trial
        << "  b0=" << cfg.b0 << "  s_max=" << cfg.s_max
        << "  seed=" << cfg.seed << "\n";
    out << "# E_clip=" << cfg.E_clip << " MeV  (unbound states clamped to this value)\n";
    out << "# Target:  E₀ ≈ −2.2 MeV  (deuteron binding energy)\n";
    out << "# Columns: S[MeV]  b[fm]  E0[MeV]\n";
    out << "#\n";
    out << std::fixed << std::setprecision(6);

    // ── Pre-warm the Gauss-Legendre static cache (MUST be before parallel) ────
    // detail::integrate() in hamiltonian.h uses a lazy static initialisation:
    //   static bool initialised = false;
    //   if (!initialised) { gauss_legendre_nodes(...); initialised = true; }
    // If the first call happens inside the parallel region, all threads race
    // to write the same static vectors simultaneously — undefined behaviour
    // that typically manifests as a hang or silent wrong results.
    // Forcing one call here on the main thread fills the cache before any
    // OpenMP thread can see it uninitialised.
    {
        Gaussian g_warm(ld{1.0L}, ld{0.0L});   // trivial 1D Gaussian
        GaussianPair gp_warm(g_warm, g_warm);
        rvec c1(1); c1[0] = ld{1};
        KineticParams kp_warm(gp_warm, c1);
        (void) ke_relativistic(gp_warm, kp_warm, sys.mu[0]);  // fills the cache
    }

    // ── Allocate results buffer ───────────────────────────────────────────────
    // Each grid point is independent so we compute in parallel, then write
    // sequentially.  results[iS][ib] stores E₀ for that grid point.
    std::vector<std::vector<ld>> results(cfg.N_S, std::vector<ld>(cfg.N_b, ld{0}));

    // ── Run the scan ──────────────────────────────────────────────────────────
    auto t_start = std::chrono::steady_clock::now();
    std::atomic<int> done{0};

    std::cout << "Running scan";
#ifdef _OPENMP
    std::cout << "  (OpenMP threads: " << omp_get_max_threads() << ")";
#endif
    std::cout << "\n";
    print_progress(0, total_points, 0.0);

    // Flatten the 2D grid into a single index so OpenMP can schedule it.
    // dynamic scheduling is important here: computation time varies across
    // the grid (unbound regions converge fast, near-bound regions are slower).
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int iS = 0; iS < cfg.N_S; iS++) {
        for (int ib = 0; ib < cfg.N_b; ib++) {

            // Each grid point gets a unique but reproducible seed.
            // Because each thread owns its own local rng inside run_one_point,
            // there is no shared mutable state — perfectly thread-safe.
            uint64_t point_seed = cfg.seed + static_cast<uint64_t>(iS) * cfg.N_b + ib;

            ld E0 = run_one_point(sys, channels, params,
                                  S_vals[iS], b_vals[ib], point_seed);

            if (!std::isfinite(E0) || E0 > cfg.E_clip)
                E0 = cfg.E_clip;

            results[iS][ib] = E0;

            // Atomic increment + progress update (guarded to avoid interleaved output)
            int n = ++done;
            #pragma omp critical
            {
                auto t_now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                print_progress(n, total_points, elapsed);
            }
        }
    }

    // ── Write results in order (sequential — ofstream is not thread-safe) ─────
    for (int iS = 0; iS < cfg.N_S; iS++) {
        for (int ib = 0; ib < cfg.N_b; ib++) {
            out << S_vals[iS] << "  " << b_vals[ib] << "  " << results[iS][ib] << "\n";
        }
        // Blank line between S rows — required by gnuplot 'with image' / 'pm3d'
        out << "\n";
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_total = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n\nScan complete.\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(1) << elapsed_total << " s\n";
    std::cout << "  Time per point: "
              << std::setprecision(2) << elapsed_total / total_points << " s\n";
    std::cout << "  Output written to: " << cfg.output << "\n\n";

    return 0;
}