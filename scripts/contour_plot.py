#!/usr/bin/env python3
"""
plot_calibration_contour.py - Generates a SLANTED 2D grid by dynamically tracking the energy contour

Usage:
  python3 scripts/plot_calibration_contour.py \
    --b_range_min 2.0 --b_range_max 4.0 --b_range_steps 10 \
    --S_init_anchor 47.0 --S_window 5.0 --S_steps 8 \
    --b_form 1.2 --jobs 8
"""

import sys
import csv
import os
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Hardcoded experimental targets
ENERGY_TARGET = -2.224
RADIUS_TARGET = 2.128

def run_single_point(args):
    """Executes the deu binary for a single (b_range, S) coordinate"""
    b_range, S, b_form, run_id, results_dir = args
    csv_path = os.path.join(results_dir, f"run_{run_id}.csv")
    
    cmd = [
        "./deu",
        "-b_range", f"{b_range:.4f}",
        "-b_form", f"{b_form:.4f}",
        "-S", f"{S:.4f}",
        "--output-csv", csv_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        with open(csv_path, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
            if len(lines) < 2: return None
            
            reader = csv.DictReader(lines)
            final_row = list(reader)[-1]
            
            energy = float(final_row.get("energy_mev", 0))
            radius = float(final_row.get("radius_fm", 0))
            
        os.remove(csv_path)
        return (b_range, S, energy, radius)
        
    except Exception as e:
        print(f"Error at b_range={b_range}, S={S}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate Dynamic Slanted 2D Contour Plot")
    parser.add_argument("--b_range_min", type=float, required=True)
    parser.add_argument("--b_range_max", type=float, required=True)
    parser.add_argument("--b_range_steps", type=int, default=10)
    
    # Dynamic Grid Parameters
    parser.add_argument("--S_init_anchor", type=float, required=True, help="Starting S center for the first b_range")
    parser.add_argument("--S_window", type=float, default=5.0, help="Plus/Minus S sweep range around the center")
    parser.add_argument("--S_steps", type=int, default=8)
    
    parser.add_argument("--b_form", type=float, default=1.2)
    parser.add_argument("--jobs", type=int, default=8)
    
    args = parser.parse_args()
    
    results_dir = "results/contour_map"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    grid_csv_path = os.path.join(results_dir, "grid_data.csv")
    plot_path = os.path.join(results_dir, "calibration_contour.pdf")
    
    b_ranges = np.linspace(args.b_range_min, args.b_range_max, args.b_range_steps)
    
    print(f"Starting Dynamic Grid Calculation ({args.b_range_steps} slices of {args.S_steps} runs)")
    print(f"Using {args.jobs} CPU cores...\n")
    
    all_results = []
    run_id = 0
    current_s_center = args.S_init_anchor
    
    # ==========================================
    # 1. PROCESS SLICE BY SLICE
    # ==========================================
    for i, b in enumerate(b_ranges):
        print(f"--- Slice {i+1}/{args.b_range_steps}: b_range = {b:.4f} ---")
        print(f"    Searching S window: [{current_s_center - args.S_window:.2f} to {current_s_center + args.S_window:.2f}]")
        
        s_vals = np.linspace(current_s_center - args.S_window, current_s_center + args.S_window, args.S_steps)
        slice_args = [(b, s, args.b_form, run_id + j, results_dir) for j, s in enumerate(s_vals)]
        run_id += len(s_vals)
        
        slice_results = []
        
        # Run this specific slice in parallel
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for res in executor.map(run_single_point, slice_args):
                if res:
                    slice_results.append(res)
                    all_results.append(res)
                    
        # ==========================================
        # 2. DYNAMIC UPDATER (Linear Interpolation)
        # ==========================================
        if slice_results:
            # Sort by S value to ensure proper interpolation
            slice_results.sort(key=lambda x: x[1]) 
            
            crossing_found = False
            for k in range(len(slice_results) - 1):
                s1, e1 = slice_results[k][1], slice_results[k][2]
                s2, e2 = slice_results[k+1][1], slice_results[k+1][2]
                
                # Check if the -2.224 target falls between these two energies
                if (e1 - ENERGY_TARGET) * (e2 - ENERGY_TARGET) <= 0:
                    # Mathematical Intersection!
                    exact_s = s1 + (s2 - s1) * (ENERGY_TARGET - e1) / (e2 - e1)
                    current_s_center = exact_s
                    crossing_found = True
                    print(f"    ✓ Exact crossing found! Next center interpolated to S = {current_s_center:.4f}")
                    break
            
            # Fallback: If the window completely missed the -2.224 crossing
            if not crossing_found:
                best_run = min(slice_results, key=lambda x: abs(x[2] - ENERGY_TARGET))
                current_s_center = best_run[1]
                print(f"    ⚠ Missed crossing. Snapping to closest S = {current_s_center:.4f} (E = {best_run[2]:.4f} MeV)")
        else:
            print("    ⚠ Warning: Slice failed. Keeping same center for next iteration.")
            
        print() # Empty line for readability
                
    # ==========================================
    # 3. SAVE RAW GRID TO CSV
    # ==========================================
    with open(grid_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["b_range", "S", "energy_mev", "radius_fm"])
        writer.writerows(all_results)
    
    # ==========================================
    # 4. PREPARE 1D DATA FOR TRICONTOUR
    # ==========================================
    B_flat = [res[0] for res in all_results]
    S_flat = [res[1] for res in all_results]
    E_flat = [res[2] for res in all_results]
    R_flat = [res[3] for res in all_results]

    # Filter out unbound/gas states (Energy > -0.01) so they don't ruin the plot scale
    valid_indices = [i for i, e in enumerate(E_flat) if e < -0.01]
    B_plot = [B_flat[i] for i in valid_indices]
    S_plot = [S_flat[i] for i in valid_indices]
    E_plot = [E_flat[i] for i in valid_indices]
    R_plot = [R_flat[i] for i in valid_indices]

    if not B_plot:
        print("ERROR: No bound states found. The plot cannot be generated. Check your S_anchor!")
        return

    # ==========================================
    # 5. GENERATE THE CONTOUR PLOT
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    cs_energy = plt.tricontour(B_plot, S_plot, E_plot, levels=[ENERGY_TARGET], colors='blue', linewidths=2.5)
    cs_radius = plt.tricontour(B_plot, S_plot, R_plot, levels=[RADIUS_TARGET], colors='red', linewidths=2.5, linestyles='dashed')
    
    plt.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='Blues', alpha=0.3)
    
    plt.title(f"Calibration Intersection ($b_{{form}}$ = {args.b_form} fm)", fontsize=14, fontweight='bold')
    plt.xlabel("$b_{range}$ (fm)", fontsize=12)
    plt.ylabel("Interaction Strength $S$", fontsize=12)
    
    # Create legend using custom Line2D objects
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2.5, label=f"Energy = {ENERGY_TARGET} MeV"),
        Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label=f"Radius = {RADIUS_TARGET} fm")
    ]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Contour plot generated successfully: {plot_path}")

if __name__ == "__main__":
    sys.exit(main())