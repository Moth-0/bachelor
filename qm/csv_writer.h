/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                    qm/csv_writer.h - CSV RESULT OUTPUT                         ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Writes quantum mechanics SVM results to CSV format with metadata and         ║
║   convergence tracking. Supports parameter reproducibility.                   ║
║                                                                                ║
║ USAGE:                                                                         ║
║   CsvWriter writer("results/energy_sweep_b_form/run_123.csv");                ║
║   writer.write_metadata("b_range", "2.5");                                     ║
║   writer.write_headers({"iteration", "energy_mev", "kinetic_mev", ...});       ║
║   for each convergence point:                                                  ║
║     writer.write_row({0, -8.1, 29.8, ...});                                    ║
║   writer.write_final_row(-8.004, 31.666, 1.451, 39);                          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cmath>

namespace qm {

class CsvWriter {
private:
    std::ofstream file;
    bool metadata_complete = false;
    std::vector<std::string> headers;
    
public:
    CsvWriter(const std::string& filename) {
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV file: " + filename);
        }
    }

    ~CsvWriter() {
        if (file.is_open()) {
            file.close();
        }
    }

    // Write metadata row (appears in CSV as comment)
    void write_metadata(const std::string& key, const std::string& value) {
        if (metadata_complete) {
            throw std::runtime_error("Cannot add metadata after headers written");
        }
        file << "# METADATA: " << key << "=" << value << "\n";
    }

    // Write a timestamp metadata
    void write_timestamp() {
        auto now = std::time(nullptr);
        auto tm = std::localtime(&now);
        char buffer[30];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", tm);
        write_metadata("timestamp", std::string(buffer));
    }

    // Write column headers
    void write_headers(const std::vector<std::string>& column_names) {
        if (metadata_complete) {
            throw std::runtime_error("Headers already written");
        }
        headers = column_names;
        metadata_complete = true;
        
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) file << ",";
        }
        file << "\n";
        file.flush();
    }

    // Write a data row with long doubles
    void write_row(const std::vector<long double>& values) {
        if (!metadata_complete) {
            throw std::runtime_error("Headers not written yet");
        }
        if (values.size() != headers.size()) {
            throw std::runtime_error("Row size mismatch: got " + std::to_string(values.size()) + 
                                   " but expected " + std::to_string(headers.size()));
        }
        
        for (size_t i = 0; i < values.size(); ++i) {
            file << std::fixed << std::setprecision(8) << values[i];
            if (i < values.size() - 1) file << ",";
        }
        file << "\n";
        file.flush();
    }

    // Write final summary row (iteration=-1)
    void write_final_row(long double energy, long double kinetic, 
                        long double radius, long double basis_size,
                        long double prob_bare = 0.0, long double prob_dressed = 0.0) {
        std::vector<long double> final_row = {
            static_cast<long double>(-1),  // iteration = -1 for FINAL marker
            energy,
            kinetic,
            radius,
            basis_size,
            prob_bare,
            prob_dressed
        };
        write_row(final_row);
    }
};

} // namespace qm
