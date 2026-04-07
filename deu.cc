/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                  deu.cc - DEUTERON SVM GROUND STATE FINDER                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver orchestrating the complete Stochastic Variational Method (SVM)   ║
║   to find the ground state energy of a deuteron with pion exchange coupling.   ║
║                                                                                ║
║ WORKFLOW OVERVIEW:                                                             ║
║   main()                                                                       ║
║     └─→ run_deuteron_svm(false)   [classic kinetic energy]                     ║
║     └─→ run_deuteron_svm(true)    [relativistic kinetic energy]                ║
║          └─→ Phase 1: Skeleton basis (14 states) → sweep_optimize()            ║
║          └─→ Phase 2: Competitive growth (2 cycles) →                          ║
║               For each channel: test 100 candidates →                          ║
║               add best → sweep_optimize()                                      ║
║                                                                                ║
║ KEY CONCEPTS:                                                                  ║
║                                                                                ║
║   1. SKELETON (Phase 1):                                                       ║
║      • 5 deterministic PN states: geometric widths {0.02, 0.08, 0.3, 1.2, 4.0} ║
║        These ensure spatial scales from short-range to long-range are covered. ║
║      • 9 random pion states: one per channel × spin configuration              ║
║        Seed the optimization for each 3-body coupling mode.                    ║
║      • Polish with Nelder-Mead sweeps: optimize all 14 states' A,s params      ║
║                                                                                ║
║   2. GROWTH (Phase 2):                                                         ║
║      • Competitive: each channel independently tests 100 random candidates     ║
║      • Selects the one lowering total system energy the most                   ║
║      • Locked into basis permanently                                           ║
║      • After all channels: full basis sweep to polish all parameters           ║
║                                                                                ║
║      Why competitive?                                                          ║
║        - Prevents basis bloat (only keeps impactful states)                    ║
║        - Each state must justify its computational cost                        ║
║        - Parallelizable via OpenMP (each candidate tested independently)       ║
║                                                                                ║
║ PARAMETER TUNING:                                                              ║
║                                                                                ║
║   b_range = 1.4 fm:                                                            ║
║     • Search space for Gaussian width parameter                                ║
║     • log-uniform sampling: b_ij = -log(u) × b_range                           ║
║     • Larger → explores wider spatial scales                                   ║
║                                                                                ║
║   b_form = 1.4 fm:                                                             ║
║     • Pion interaction range (form factor)                                     ║
║     • f(r) = exp(-r²/b_form²) coupling strength                                ║
║     • Controls how "soft" the pion exchange is                                 ║
║     • 1.4 fm choosen from compton wavelength of pion                           ║
║                                                                                ║
║   S = 140.0:  **CRITICAL TUNING KNOB**                                         ║
║     • Pion coupling strength parameter                                         ║
║     • Higher S → stronger binding → more negative energy                       ║
║     • Tune to match experimental target E = -2.224 MeV                         ║
║     • Example: S=100 might give -1.8 MeV, S=180 might give -2.5 MeV            ║
║                                                                                ║
║   num_cycles = 2:                                                              ║
║     • How many SVM growth phases to run                                        ║
║     • Each cycle: 10 channels × 100 candidates = 1000 evaluations (parallel)   ║
║     • More cycles → more time but better convergence                           ║
║                                                                                ║
║   num_candidates_per_step = 100:                                               ║
║     • Test this many random states per channel per cycle                       ║
║     • Larger → better chance of finding good state, but slower                 ║
║                                                                                ║
║ PHYSICAL PARAMETERS:                                                           ║
║                                                                                ║
║   m_p = 938.27 MeV, m_n = 939.56 MeV:  nucleon masses                          ║
║   m_pi0 = 134.97 MeV, m_pic = 139.57 MeV:  pion masses                         ║
║   iso_c = sqrt(2):  isospin weighting for charged pions                        ║
║                                                                                ║
║ CHANNEL DESCRIPTION:                                                           ║
║                                                                                ║
║   Channel enum defines 10 distinct physics states:                             ║
║     Channel::PN          - bare proton-neutron (no pion)                       ║
║     Channel::PI_0c_0f    - π⁰ without spin flip                                ║
║     Channel::PI_0c_1f    - π⁰ with particle 1 flipped                          ║
║     Channel::PI_0c_2f    - π⁰ with particle 2 flipped                          ║
║     Channel::PI_pc_0f    - π⁺ without spin flip                                ║
║     Channel::PI_pc_1f    - π⁺ with particle 1 flipped                          ║
║     Channel::PI_pc_2f    - π⁺ with particle 2 flipped                          ║
║     Channel::PI_mc_0f    - π⁻ without spin flip                                ║
║     Channel::PI_mc_1f    - π⁻ with particle 1 flipped                          ║
║     Channel::PI_mc_2f    - π⁻ with particle 2 flipped                          ║
║                                                                                ║
║   Note: spin flips on charged pions give different final states due to         ║
║   Pauli matrix structure. All encoded in SpinChannel enum.                     ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "deuterium.h" 

using namespace qm;

// Evaluate energy: build H,N and solve GEVP
ld evaluate_basis_energy(const std::vector<BasisState>& basis, ld b, ld S, bool relativistic) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    return solve_ground_state_energy(H, N);
}

// Calculate charge radius from ground state
struct GroundStateResult {
    ld energy;
    ld charge_radius;
};

GroundStateResult evaluate_with_radius(const std::vector<BasisState>& basis, ld b, ld S,
                                       bool relativistic, const Jacobian& jac_bare) {
    auto [H, N] = build_matrices(basis, b, S, relativistic);
    auto [E0, eigvec] = solve_ground_state_with_eigenvector(H, N);

    // Build r² matrix
    cmat R2 = build_r2_matrix(basis, jac_bare);

    // Calculate <ψ₀|r²|ψ₀> = Σ_ij c_i* c_j <i|r²|j>
    cld r2_expectation = 0.0;
    for (size_t i = 0; i < basis.size(); ++i) {
        for (size_t j = 0; j < basis.size(); ++j) {
            r2_expectation += std::conj(eigvec[i]) * R2(i, j) * eigvec[j];
        }
    }

    // Calculate charge radius: r_ch = sqrt(<r²>)
    ld r2_val = std::real(r2_expectation);
    ld charge_radius = (r2_val > 0.0) ? std::sqrt(r2_val) : 0.0;

    return {E0, charge_radius};
}

// Optimize basis parameters via Nelder-Mead sweeping with early exit
void sweep_optimize_basis(std::vector<BasisState>& basis, ld b, ld S, bool relativistic, std::vector<ld>& convergence_energies) {
    ld current_E = evaluate_basis_energy(basis, b, S, relativistic);

    ld previous_E = 999999.0;
    int no_improve_count = 0;  // Early stopping: quit if N sweeps show no progress

    // Adaptive parameters based on basis size
    size_t basis_size = basis.size();
    int max_sweeps = (basis_size <= 20) ? 20 : 10;
    ld improvement_threshold = 1e-6;  // Minimum improvement to count as "progress"
    int patience = 3;  // Exit if 3 consecutive sweeps have negligible improvement
    int nm_max_iter = (basis_size <= 20) ? 100 : 60;

    int sweep = 0;

    while (sweep < max_sweeps) {
        previous_E = current_E;

        // Optimize each basis state
        for (size_t k = 0; k < basis.size(); ++k) {
            SpatialWavefunction backup_psi = basis[k].psi;
            rvec p0 = pack_wavefunction(backup_psi);

            auto objective_func = [&](const qm::rvec& p_test) -> qm::ld {
                unpack_wavefunction(basis[k].psi, p_test);

                bool is_physical = true;
                for (size_t i = 0; i < basis[k].psi.A.size1(); ++i) {
                    if (basis[k].psi.A(i, i) <= 0.02) is_physical = false;
                }
                if (basis[k].psi.A.determinant() <= ZERO_LIMIT) is_physical = false;

                for (size_t i = 0; i < basis[k].psi.s.size1(); ++i) {
                    for (size_t col = 0; col < 3; ++col) {
                        if (std::abs(basis[k].psi.s(i, col)) > 5.0) is_physical = false;
                    }
                }

                if (!is_physical) return 999999.0;
                return evaluate_basis_energy(basis, b, S, relativistic);
            };

            rvec p_best = nelder_mead(p0, objective_func, nm_max_iter);
            unpack_wavefunction(basis[k].psi, p_best);
        }

        current_E = evaluate_basis_energy(basis, b, S, relativistic);
        convergence_energies.push_back(current_E);

        // Check for meaningful improvement
        ld improvement = previous_E - current_E;  // positive = improvement
        if (improvement < improvement_threshold) {
            no_improve_count++;
        } else {
            no_improve_count = 0;  // Reset counter on good improvement
        }

        std::cout << "\r" << "Sweep " << sweep << " (Basis=" << basis.size() << "): E=" << current_E
                  << " MeV  (ΔE=" << improvement << ")     " << std::flush;

        // Early exit if stalled
        if (no_improve_count >= patience) {
            std::cout << "\n  → Sweep converged (no improvement for " << patience << " iterations)\n";
            break;
        }

        sweep++;
    }
}

// Run two-phase SVM: skeleton (Phase 1) + competitive growth (Phase 2)
// Returns: pair of (energy, charge_radius)
std::pair<ld, ld> run_deuteron_svm(bool relativistic) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;
    ld m_pi0 = 134.97, m_pic = 139.57;

    ld b_range = 1.4, b_form = 1.4;
    //
    // *** CRITICAL TUNING PARAMETER ***
    // S = coupling strength for pion exchange (in MeV)
    //     Directly controls how strongly pions bind the nucleons

    ld S = 138.0;

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    ld iso_c = std::sqrt(2.0L);

    // Track convergence energies
    std::vector<ld> convergence_energies;
    std::string convergence_file = relativistic ? "conv_rel.data" : "conv_cla.data";

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        {SpatialWavefunction(-1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic}
    };

    // ------- PHASE 1: SKELETON BASIS WITH GEOMETRIC GRID --------
    // Purpose: Build initial 14-state basis with wide spatial coverage
    //
    // Strategy: 5 explicit geometric PN states + 9 random pion channels
    
    std::cout << "--- 1. Planting Geometric PN Grid & Pion Seeds ---\n";
    
    // The Anchor: 5 deterministic PN states
    std::vector<ld> deterministic_widths = {0.02, 0.08, 0.3, 1.2, 4.0};
    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * width;
        rmat s_fixed = zeros<ld>(1, 3);
        SpatialWavefunction psi_fixed(A_fixed, s_fixed, 1);
        basis.push_back({psi_fixed, Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    // The Cloud: 9 random Pion seeds
    for (size_t t = 1; t < channel_templates.size(); ++t) {
        BasisState seed = channel_templates[t];
        Gaussian g;
        g.randomize(seed.jac, b_range);
        seed.psi.set_from_gaussian(g);
        basis.push_back(seed);
    }
    
    std::cout << "Skeleton Size: " << basis.size() << " states.\n";
    sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);
    ld skeleton_E = evaluate_basis_energy(basis, b_form, S, relativistic);
    convergence_energies.push_back(skeleton_E);
    std::cout << "\nSkeleton Energy: " << skeleton_E << " MeV\n\n";

    // ------- PHASE 2: COMPETITIVE SVM GROWTH (INCREMENTAL BASIS EXPANSION) -------
    // Purpose: Grow basis incrementally with best candidates, avoiding bloat
    //
    // Algorithm per cycle:
    //   1. For each of 10 pion channels (π⁰, π⁺, π⁻ × 3 spin flips):
    //      - Generate 100 random candidate states independently (parallel OpenMP)
    //      - Evaluate each: add to basis temporarily → solve GEVP → remove
    //      - Select candidate with LOWEST energy
    //      - Lock it permanently into the basis
    //   2. After all channels: sweep-optimize the expanded basis for polish
    
    int num_cycles = 3;

    std::cout << "--- 2. Competitive SVM Growth ---\n";
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        // Adaptive candidate count: fewer for later cycles (diminishing returns)
        int num_candidates_per_step = (cycle == 0) ? 100 : 60;

        // Loop through all 10 channels, adding 1 state to each
        for (size_t t = 0; t < channel_templates.size(); ++t) {

            BasisState best_candidate = channel_templates[t];
            ld best_E = 999999.0;

            // OpenMP Parallel Candidate Search
            #pragma omp parallel
            {
                BasisState local_best_candidate = channel_templates[t];
                ld local_best_E = 999999.0;
                std::vector<BasisState> local_basis = basis;

                #pragma omp for
                for (int c = 0; c < num_candidates_per_step; ++c) {
                    BasisState test_candidate = channel_templates[t];

                    Gaussian g;
                    g.randomize(test_candidate.jac, b_range);
                    test_candidate.psi.set_from_gaussian(g);

                    local_basis.push_back(test_candidate);
                    ld E = evaluate_basis_energy(local_basis, b_form, S, relativistic);
                    local_basis.pop_back();

                    if (E < local_best_E) {
                        local_best_E = E;
                        local_best_candidate = test_candidate;
                    }
                }

                #pragma omp critical
                {
                    if (local_best_E < best_E) {
                        best_E = local_best_E;
                        best_candidate = local_best_candidate;
                    }
                }
            }

            // Lock in the winner
            basis.push_back(best_candidate);

            ld current_E = evaluate_basis_energy(basis, b_form, S, relativistic);
            convergence_energies.push_back(current_E);

            std::cout << "\rAdded State " << basis.size() << " (Cycle " << cycle+1 << ", Ch " << t << ") -> E = "
                      << std::fixed << std::setprecision(5) << current_E << " MeV    " << std::flush;
        }

        // ONE massive polish per cycle
        std::cout << "\n - Sweeping Cycle " << cycle+1 << " basis -\n";
        sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);
    }

    // Save convergence data
    {
        std::ofstream outfile(convergence_file);
        outfile << "iteration energy\n";
        for (size_t i = 0; i < convergence_energies.size(); ++i) {
            outfile << i << " " << std::fixed << std::setprecision(8) << convergence_energies[i] << "\n";
        }
        outfile.close();
        std::cout << "\nConvergence data saved to: " << convergence_file << "\n";
    }

    // Generate gnuplot script to visualize convergence
    {
        std::string plot_file = relativistic ? "convergence_rel.gp" : "convergence_cla.gp";
        std::string png_file = relativistic ? "convergence_relativistic.png" : "convergence_classic.png";
        std::string title = relativistic ? "Deuteron SVM - Relativistic Kinetic Energy Convergence" : "Deuteron SVM - Classic Kinetic Energy Convergence";

        std::ofstream gp(plot_file);
        gp << "set terminal pngcairo size 1000,600\n";
        gp << "set output '" << png_file << "'\n";
        gp << "set xlabel 'Iteration'\n";
        gp << "set ylabel 'Energy (MeV)'\n";
        gp << "set title '" << title << "'\n";
        gp << "set grid\n";
        gp << "set style data linespoints\n";
        gp << "set pointsize 0.5\n";
        gp << "plot '" << convergence_file << "' using 1:2 skip 1 with linespoints title 'Energy' linecolor rgb 'blue' linewidth 2\n";
        gp.close();
        std::cout << "Gnuplot script saved to: " << plot_file << "\n";
    }

    std::cout << "\n";
    auto result = evaluate_with_radius(basis, b_form, S, relativistic, jac_bare);
    return {result.energy, result.charge_radius};
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n\n";

    // Run with both kinetic energy models
    std::cout << ">>> RUNNING CLASSIC KINETIC ENERGY\n";
    auto [E_classic, R_classic] = run_deuteron_svm(false);

    std::cout << "\n>>> RUNNING RELATIVISTIC KINETIC ENERGY\n";
    auto [E_relativistic, R_relativistic] = run_deuteron_svm(true);

    ld E_diff = E_relativistic - E_classic;
    ld R_diff = R_relativistic - R_classic;

    // Experimental values
    ld E_exp = -2.224;        // MeV (experimental binding energy)
    ld R_exp = 2.128;         // fm (experimental charge radius)

    // --- COMPARISON & INTERPRETATION ---
    std::cout << "========================================\n";
    std::cout << "  FINAL RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(5);

    std::cout << "\n--- GROUND STATE ENERGY ---\n";
    std::cout << "Classic Energy:          " << E_classic << " MeV\n";
    std::cout << "Relativistic Energy:     " << E_relativistic << " MeV\n";
    std::cout << "Difference (rel-cl):     " << E_diff << " MeV\n";
    std::cout << "Experimental Target:     " << E_exp << " MeV\n";
    
    std::cout << "\n--- CHARGE RADIUS ---\n";
    std::cout << "Classic Radius:          " << R_classic << " fm\n";
    std::cout << "Relativistic Radius:     " << R_relativistic << " fm\n";
    std::cout << "Difference (rel-cl):     " << R_diff << " fm\n";
    std::cout << "Experimental Value:      " << R_exp << " fm\n";

    std::cout << "\n========================================\n";
    std::cout << "INTERPRETATION:\n";
    std::cout << "========================================\n";
    std::cout << "Energy:\n";
    std::cout << "  Target: E = " << E_exp << " MeV\n";
    std::cout << "  Relativistic correction: " << (E_diff / E_classic * 100.0) << "%\n";

    std::cout << "\nCharge Radius:\n";
    std::cout << "  Target: r_ch = " << R_exp << " fm\n";
    std::cout << "  Deviation from experiment: " << ((R_relativistic - R_exp) / R_exp * 100.0) << "%\n";
    std::cout << "========================================\n";

    // Generate comparison gnuplot script
    {
        std::ofstream gp("convergence_comparison.gp");
        gp << "set terminal pngcairo size 1400,600\n";
        gp << "set output 'convergence_comparison.png'\n";
        gp << "set multiplot layout 1, 2\n";

        gp << "set xlabel 'Iteration'\n";
        gp << "set ylabel 'Energy (MeV)'\n";
        gp << "set title 'Classic Kinetic Energy Convergence'\n";
        gp << "set grid\n";
        gp << "set style data linespoints\n";
        gp << "set pointsize 0.5\n";
        gp << "plot 'conv_cla.data' using 1:2 skip 1 with linespoints title 'Energy' linecolor rgb 'blue' linewidth 2\n";

        gp << "set title 'Relativistic Kinetic Energy Convergence'\n";
        gp << "plot 'conv_rel.data' using 1:2 skip 1 with linespoints title 'Energy' linecolor rgb 'red' linewidth 2\n";

        gp << "unset multiplot\n";
        gp.close();
        std::cout << "\nPlotting scripts generated. To create plots, run:\n";
        std::cout << "  gnuplot convergence_cla.gp\n";
        std::cout << "  gnuplot convergence_rel.gp\n";
        std::cout << "  gnuplot convergence_comparison.gp\n";
        std::cout << "Or use Makefile: make convergence.png\n";
    }

    return 0;
}