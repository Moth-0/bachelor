import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
# Folder containing the CSV files
DATA_FOLDER = "results/all_configs_20260609_142217" 
OUTPUT_FILE = "latex_tables_output.txt"

# Expected file substrings and their corresponding table labels
# Format: (File substring, Nucleons label, Pion label)
CONFIGS = [
    ("PN_Cla_Pi_Cla", "Classic", "Classic"),
    ("PN_Rel_Pi_Cla", "Relat.", "Classic"),
    ("PN_Cla_Pi_Rel", "Classic", "Relat."),
    ("PN_Rel_Pi_Rel", "Relat.", "Relat.")
]

# ==========================================
# PARSING LOGIC
# ==========================================
def parse_results(folder_path):
    results = {}
    
    # Step 1: Read all the raw data
    for file_sub, nuc_lbl, pi_lbl in CONFIGS:
        search_pattern = os.path.join(folder_path, f"*{file_sub}*.csv")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            print(f"% Warning: No file found containing '{file_sub}'")
            continue
            
        filepath = matching_files[0] 
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        last_line = lines[-1].strip().split(',')
        
        if last_line[0] == "-1.00000000":
            energy_mev = float(last_line[1])
            kinetic_mev = float(last_line[2])
            radius_fm = float(last_line[3])
            prob_dressed = float(last_line[6])
            execution_time_s = float(last_line[7])
            
            p_pi_percent = prob_dressed * 100
            runtime_min = execution_time_s / 60
            
            results[(nuc_lbl, pi_lbl)] = {
                'E': energy_mev,
                'rd': radius_fm,
                'K': kinetic_mev,
                'P_pi': p_pi_percent,
                'time': runtime_min
            }
        else:
            print(f"% Warning: File {filepath} does not end with the -1.0 summary row.")
            
    # Step 2: Calculate Delta E and Deviation using Classic-Classic as baseline
    classic_key = ("Classic", "Classic")
    
    if classic_key in results:
        e_base = results[classic_key]['E']
        
        for key in results:
            delta_e = results[key]['E'] - e_base
            # Avoid division by zero just in case
            deviation = abs(delta_e / e_base) * 100 if e_base != 0 else 0.0
            
            results[key]['dE'] = delta_e
            results[key]['dev'] = deviation
    else:
        print("% Warning: 'Classic-Classic' data missing. Cannot calculate Delta E.")
        for key in results:
            results[key]['dE'] = 0.0
            results[key]['dev'] = 0.0

    # Step 3: Calculate Delta E and Deviation using Classic-Relat as the new baseline
    recal_key = ("Classic", "Relat.")
    
    if recal_key in results:
        e_base_recal = results[recal_key]['E']
        
        for key in results:
            delta_e_recal = results[key]['E'] - e_base_recal
            deviation_recal = abs(delta_e_recal / e_base_recal) * 100 if e_base_recal != 0 else 0.0
            
            results[key]['dE_recal'] = delta_e_recal
            results[key]['dev_recal'] = deviation_recal
    else:
        print("% Warning: 'Classic-Relat' data missing. Cannot calculate re-calibrated Delta E.")
        for key in results:
            results[key]['dE_recal'] = 0.0
            results[key]['dev_recal'] = 0.0
            
    return results

# ==========================================
# LATEX GENERATION
# ==========================================
def generate_latex_tables(results, output_filename):
    with open(output_filename, 'w') as out:
        
        # Table 1: Physical Properties
        out.write("% --- Table 7.1: Physical Properties ---\n")
        out.write("\\begin{tabular}{ll cccc}\n")
        out.write("    \\hline\n")
        out.write("    \\textbf{Nucleons} & \\textbf{Pion} & \\textbf{$E_0$ (MeV)} & \\textbf{$r_d$ (fm)} & \\textbf{$\\langle K \\rangle$ (MeV)} & \\textbf{$P_\\pi$ (\\%)} \\\\\n")
        out.write("    \\hline\n")
        for (nuc, pi), data in results.items():
            out.write(f"    {nuc:7} & {pi:7} & {data['E']:6.3f} & {data['rd']:5.3f} & {data['K']:5.2f} & {data['P_pi']:5.3f} \\\\\n")
        out.write("    \\hline\n")
        out.write("\\end{tabular}\n\n\n")
        
        # Table 2: Performance and Deviations (Baseline: Cla-Cla)
        out.write("% --- Table 7.2: Runtime and Deviation (Baseline: Classic-Classic) ---\n")
        out.write("\\begin{tabular}{ll ccc}\n")
        out.write("    \\hline\n")
        out.write("    \\textbf{Nucleons} & \\textbf{Pion} & \\textbf{$\\Delta E$ (MeV)} & \\textbf{Deviation (\\%)} & \\textbf{Runtime (min)} \\\\\n")
        out.write("    \\hline\n")
        for (nuc, pi), data in results.items():
            out.write(f"    {nuc:7} & {pi:7} & {data['dE']:8.3f} & {data['dev']:5.2f} & {data['time']:6.1f} \\\\\n")
        out.write("    \\hline\n")
        out.write("\\end{tabular}\n\n\n")

        # Table 3: Re-calibrated Deviations (Baseline: Cla-Rel)
        out.write("% --- Table 7.4: Re-calibrated Deviations (Baseline: Classic-Relativistic) ---\n")
        out.write("\\begin{tabular}{ll ccc}\n")
        out.write("    \\hline\n")
        out.write("    \\textbf{Nucleons} & \\textbf{Pion} & \\textbf{$E_0$ (MeV)} & \\textbf{$\\Delta E$ (MeV)} & \\textbf{Deviation (\\%)} \\\\\n")
        out.write("    \\hline\n")
        for (nuc, pi), data in results.items():
            out.write(f"    {nuc:7} & {pi:7} & {data['E']:6.3f} & {data['dE_recal']:8.3f} & {data['dev_recal']:5.2f} \\\\\n")
        out.write("    \\hline\n")
        out.write("\\end{tabular}\n")

    print(f"Success! All tables written to: {output_filename}")

if __name__ == "__main__":
    extracted_data = parse_results(DATA_FOLDER)
    if extracted_data:
        generate_latex_tables(extracted_data, OUTPUT_FILE)