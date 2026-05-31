/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                  deu.cc - DEUTERON SVM GROUND STATE FINDER                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver orchestrating the complete Stochastic Variational Method (SVM)   ║
║   to find the ground state energy of a deuteron with pion exchange coupling.   ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <memory>
#include <omp.h>
#include <string>
#include <chrono>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "deuterium.h"
#include "qm/serialization.h"
#include "qm/csv_writer.h"
#include "SVM.h"

using namespace qm;


// Run two-phase SVM: skeleton (Phase 1) + competitive growth (Phase 2)
// Returns pair of (basis, result) for saving all configurations
std::pair<std::vector<BasisState>, SvmResult> run_deuteron_svm(const std::vector<bool>& relativistic, ld b_range, ld b_form, ld S, const std::vector<ld>& box_strengths) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;
    ld m_pi0 = 134.97, m_pic = 139.57;
    ld iso_c = std::sqrt(2.0L);

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    rvec convergence_energies;

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        
        {SpatialWavefunction(-1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        
        {SpatialWavefunction(-1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        
        {SpatialWavefunction(-1), Channel::PI_mc_0f, NO_FLIP, -iso_c, jac_dressed_c, m_pic}, 
        {SpatialWavefunction(-1), Channel::PI_mc_1f, FLIP_PARTICLE_1, -iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_2f, FLIP_PARTICLE_2, -iso_c, jac_dressed_c, m_pic}
    };

    // ------- PHASE 1: SKELETON BASIS WITH GEOMETRIC GRID --------
    std::cout << "--- 1. Planting Geometric PN Grid & Pion Seeds ---\n";
    
    std::vector<BasisState> bare_basis;
    std::vector<ld> deterministic_widths = {1.0, 3.0, 8.0, 20.0, 100.0};
    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * 1.0 / (width * width);
        rmat s_fixed = zeros<ld>(1, 3);
        bare_basis.push_back({SpatialWavefunction(A_fixed, s_fixed, +1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    // ------- PHASE 2 & 3: COMPETITIVE GROWTH & BOX REGULARIZATION --------
    std::cout << "--- 2. Competitive Growth inside widening HO Box ---\n";
    
    std::vector<BasisState> grand_basis;
    grand_basis.insert(grand_basis.end(), bare_basis.begin(), bare_basis.end());

    for (ld ho_k : box_strengths) {
        std::cout << "\n=== Generating Basis for ho_k = " << ho_k << " ===\n";
        
        std::vector<BasisState> local_basis;
        local_basis.insert(local_basis.end(), bare_basis.begin(), bare_basis.end());
        
        // 1. Add 2 state per channel (18 states total)
        for(int i=0; i<2; i++) competitive_search(local_basis, channel_templates, 10000, b_range, b_form, S, relativistic, ho_k, 3);
        
        // 2. Fast sweep on just these states!
        sweep_optimize_basis(local_basis, b_form, b_range, S, relativistic, convergence_energies, 20, 1e-4, ho_k, 3);

        // 3. Move these highly specialized states into the master pool
        grand_basis.insert(grand_basis.end(), local_basis.begin() + bare_basis.size(), local_basis.end());
    }

    std::cout << "\n=== FINAL GEVP EVALUATION IN FREE SPACE (ho_k = 0.0) ===\n";

    // Do one final, shallow sweep of the grand basis at k=0
    // to let the core, pocket, and tail states slightly adjust to each other.
    sweep_optimize_basis(grand_basis, b_form, b_range, S, relativistic, convergence_energies, 20, 1e-4, 0.0, 3);

    
    SvmResult result = evaluate_observables(grand_basis, b_form, b_range, S, relativistic);

    result.convergence_history = convergence_energies;
    
    // Calculate execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.execution_time = duration.count() / 1000.0;  // convert to seconds

    print_basis_details(grand_basis, result.coefficients);
    std::cout << "\n";

    // Save final basis state for analysis (keep this for individual inspection)
    save_basis_state(grand_basis, result.coefficients, result.energy, result.charge_radius,
                     result.avg_kinetic_energy, "basis_final.txt");
    std::cout << "Saved basis state to basis_final.txt\n";

    return {grand_basis, result};
}

int main(int argc, char* argv[]) {
    // Default values
    ld b_range = 2.24;
    ld b_form = 1.4;
    ld S = 31.29;

    std::string file_name = "convergence.data";
    std::string output_csv = "";
    int max_basis_size = 0;
    std::vector<ld> box_strengths_input = {5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0};  // default 
    bool pn_rel = false, pi_rel = false;  // defaults: both classical

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-b_range") && i + 1 < argc) {
            b_range = std::stold(argv[++i]);
        } else if ((arg == "-b_form") && i + 1 < argc) {
            b_form = std::stold(argv[++i]);
        } else if ((arg == "-S") && i + 1 < argc) {
            S = std::stold(argv[++i]);
        } else if ((arg == "-f" || arg == "--file") && i+1 < argc) {
            file_name = argv[++i];
        } else if ((arg == "--output-csv") && i + 1 < argc) {
            output_csv = argv[++i];
        } else if ((arg == "--max-basis-size") && i + 1 < argc) {
            max_basis_size = std::stoi(argv[++i]);
        } else if ((arg == "-box-strengths" || arg == "--box-strengths") && i + 1 < argc) {
            // Parse comma-separated box strengths: "0.0" or "0.1,0.0" or "0.5,0.1,0.0"
            std::string box_str = argv[++i];
            box_strengths_input.clear();
            size_t start = 0, end = 0;
            while ((end = box_str.find(',', start)) != std::string::npos) {
                box_strengths_input.push_back(std::stold(box_str.substr(start, end - start)));
                start = end + 1;
            }
            box_strengths_input.push_back(std::stold(box_str.substr(start)));
        } else if ((arg == "--pn-rel")) {
            pn_rel = true;
        } else if ((arg == "--pi-rel")) {
            pi_rel = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./deu [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -b_range <value>       Search space for Gaussian width (default: 3.6 fm)\n";
            std::cout << "  -b_form <value>        Pion interaction range (default: 1.5 fm)\n";
            std::cout << "  -S <value>             Pion coupling strength (default: 20.0 MeV)\n";
            std::cout << "  -f, --file <string>    Convergence data file location\n";
            std::cout << "  --output-csv <path>    Write results to CSV file (includes metadata and convergence)\n";
            std::cout << "  --max-basis-size <n>   Limit maximum basis size (0=unlimited, default: 0)\n";
            std::cout << "  -box-strengths <list> Comma-separated HO box strengths (e.g. '0.0' or '0.1,0.0' or '0.5,0.1,0.0')\n";
            std::cout << "  --pn-rel true|false    Use relativistic PN channel (default: false)\n";
            std::cout << "  --pi-rel true|false    Use relativistic pion channel (default: true)\n";
            std::cout << "  -h, --help             Show this help message\n";
            return 0;
        }
    }

    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n";
    
    std::cout << "Parameters:\n";
    std::cout << "  b_range = " << b_range << " fm\n";
    std::cout << "  b_form  = " << b_form << " fm\n";
    std::cout << "  S       = " << S << " MeV\n";
    std::cout << "  PN Treatment: " << (pn_rel ? "Rel" : "Cla") << ", Pion Treatment: " << (pi_rel ? "Rel" : "Cla") << "\n";
    std::cout << "  Box Strengths: ";
    for (size_t i = 0; i < box_strengths_input.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << box_strengths_input[i];
    }
    std::cout << "\n";
    std::cout << "========================================\n\n";

    // Create results directory if output CSV is specified
    std::unique_ptr<qm::CsvWriter> csv_writer;
    if (!output_csv.empty()) {
        csv_writer = std::make_unique<qm::CsvWriter>(output_csv);
        csv_writer->write_metadata("b_range", std::to_string(b_range));
        csv_writer->write_metadata("b_form", std::to_string(b_form));
        csv_writer->write_metadata("S", std::to_string(S));
        
        // Write box strengths as comma-separated list
        std::string box_str;
        for (size_t i = 0; i < box_strengths_input.size(); ++i) {
            if (i > 0) box_str += ",";
            box_str += std::to_string(box_strengths_input[i]);
        }
        csv_writer->write_metadata("box_strengths", box_str);
        
        if (max_basis_size > 0) {
            csv_writer->write_metadata("max_basis_size", std::to_string(max_basis_size));
        }
        csv_writer->write_timestamp();
        csv_writer->write_headers({"iteration", "energy_mev", "kinetic_mev", "radius_fm", "basis_size", "prob_bare", "prob_dressed", "execution_time_s"});
    }

    std::ofstream outfile;
    if (output_csv.empty()) {
        // Only write convergence.data when NOT using CSV output (to avoid I/O contention in parallel sweeps)
        outfile.open(file_name);
    }
    std::vector<SvmResult> all_results;
    std::vector<ConfigurationResult> all_configs;

    // Set labels based on relativistic flags (for display and CSV)
    std::string pn_label = pn_rel ? "Rel" : "Cla";
    std::string pi_label = pi_rel ? "Rel" : "Cla";

    // Build configuration vector from parsed flags
    std::vector<std::pair<std::string, std::vector<bool>>> configurations = {
        {"PN_{" + pn_label + "} Pi_{" + pi_label + "}", {pn_rel, pi_rel}}
    };

    // Run the configurations loop
    for (const auto& config : configurations) {
        std::string label = config.first;
        std::vector<bool> flags = config.second;

        std::cout << "\n>>>>>>>> RUNNING CONFIGURATION: " << label << " <<<<<<<<\n";

        auto [basis, res] = run_deuteron_svm(flags, b_range, b_form, S, box_strengths_input);
        all_results.push_back(res);

        std::cout << "--> FINAL " << label << " | E: " << res.energy << " MeV, R: " << res.charge_radius << " fm\n";

        if (outfile.is_open()) {
            outfile << "\"Iteration\"\t\"" << label << "\"\n";
            for (size_t iter = 0; iter < res.convergence_history.size(); ++iter) {
                outfile << iter << "\t" << std::fixed << std::setprecision(8) << res.convergence_history[iter] << "\n";
            }
            outfile << "\n\n";
        }
        
        // Write convergence history to CSV (always if CSV output requested)
        for (size_t iter = 0; iter < res.convergence_history.size(); ++iter) {
            if (csv_writer) {
                csv_writer->write_row({
                    static_cast<long double>(iter),
                    res.convergence_history[iter],
                    0.0,  // kinetic energy per iteration not tracked
                    0.0,  // radius per iteration not tracked
                    static_cast<long double>(basis.size()),
                    res.prob_bare,
                    res.prob_dressed,
                    0.0   // execution time not tracked per iteration
                });
            }
        }

        // Write final summary row to CSV
        if (csv_writer) {
            csv_writer->write_final_row(res.energy, res.avg_kinetic_energy, 
                                       res.charge_radius, basis.size(),
                                       res.prob_bare, res.prob_dressed, res.execution_time);
        }

        // Collect configuration for saving
        all_configs.push_back({
            label,
            basis,
            res.coefficients,
            res.energy,
            res.charge_radius,
            res.avg_kinetic_energy,
            res.prob_bare,
            res.prob_dressed
        });
    }

    // Print final summary comparison table
    std::cout << "\n======================================================================================================================================\n";
    std::cout << "                                  FINAL RESULTS SUMMARY                                   \n";
    std::cout << "======================================================================================================================================\n";
    std::cout << std::fixed << std::setprecision(5);

    for (size_t i = 0; i < configurations.size(); ++i) {
        std::cout << std::setw(24) << std::left << configurations[i].first
                  << " | E: "       << std::right << all_results[i].energy  << " MeV"
                  << " | R: "       << all_results[i].charge_radius         << " fm"
                  << " | <T>: "     << all_results[i].avg_kinetic_energy    << " MeV"
                  << " | PN: "      << (all_results[i].prob_bare * 100.0)   << " %"
                  << " | PN+pi: "   << (all_results[i].prob_dressed * 100.0)<< " %"
                  << " | Time: "    << std::fixed << std::setprecision(3) << all_results[i].execution_time << " s\n";
    }

    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "Experimental Target      | E: -2.22400 MeV | R: 2.12800 fm\n";
    std::cout << "======================================================================================================================================\n";

    if (outfile.is_open()) {
        outfile.close();
    }

    // Save all configurations to file
    save_all_configurations(all_configs, "all_configurations.txt");
    std::cout << "Saved all configurations to all_configurations.txt\n";

    return 0;
}