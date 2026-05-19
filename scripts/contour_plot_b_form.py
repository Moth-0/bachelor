#!/usr/bin/env python3
"""
contour_plot_b_form.py - Generates a SLANTED 2D grid by dynamically tracking the energy contour

Usage:
  python3 scripts/contour_plot_b_form.py \
    --b_form_min 0.8 --b_form_max 1.6 --b_form_steps 10 \
    --S_init_anchor 47.0 --S_window 5.0 --S_steps 8 \
    --b_range 2.44 --jobs 8
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
    """Executes the deu binary for a single (b_form, S) coordinate"""
    b_form, S, b_range, run_id, results_dir = args
    csv_path = os.path.join(results_dir, f"run_{run_id}.csv")
    
    cmd = [
        "./deu",
        "-b_range", f"{b_range:.4f}",
        "-b_form", f"{b_form:.4f}",
        "-S", f"{S:.4f}",
        "-box-strengths", "10.0,5.0,2.0,1.0",
        "--output-csv", csv_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Don't check exit code - if CSV was written, process it regardless
        
        with open(csv_path, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
            if len(lines) < 2: return None
            
            reader = csv.DictReader(lines)
            final_row = list(reader)[-1]
            
            energy = float(final_row.get("energy_mev", 0))
            radius = float(final_row.get("radius_fm", 0))
            prob_bare = float(final_row.get("prob_bare", 0))
            prob_dressed = float(final_row.get("prob_dressed", 0))
            
        os.remove(csv_path)
        return (b_form, S, energy, radius, prob_bare, prob_dressed)
        
    except Exception as e:
        print(f"Error at b_form={b_form}, S={S}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate Dynamic Slanted 2D Contour Plot (b_form sweep)")
    parser.add_argument("--b_form_min", type=float, required=True)
    parser.add_argument("--b_form_max", type=float, required=True)
    parser.add_argument("--b_form_steps", type=int, default=10)
    
    # Dynamic Grid Parameters
    parser.add_argument("--S_init_anchor", type=float, required=True, help="Starting S center for the first b_form")
    parser.add_argument("--S_window", type=float, default=5.0, help="Plus/Minus S sweep range around the center")
    parser.add_argument("--S_steps", type=int, default=8)
    
    parser.add_argument("--b_range", type=float, default=2.44)
    parser.add_argument("--jobs", type=int, default=8)
    
    args = parser.parse_args()
    
    results_dir = "results/contour_b_form"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    grid_csv_path = os.path.join(results_dir, "grid_data.csv")
    plot_path = os.path.join(results_dir, "calibration_contour.png")
    
    b_forms = np.linspace(args.b_form_min, args.b_form_max, args.b_form_steps)
    
    print(f"Starting Dynamic Grid Calculation ({args.b_form_steps} slices of {args.S_steps} runs)")
    print(f"Using {args.jobs} CPU cores...\n")
    
    all_results = []
    run_id = 0
    current_s_center = args.S_init_anchor
    
    # ==========================================
    # 1. PROCESS SLICE BY SLICE
    # ==========================================
    for i, b in enumerate(b_forms):
        print(f"--- Slice {i+1}/{args.b_form_steps}: b_form = {b:.4f} ---")
        print(f"    Searching S window: [{current_s_center - args.S_window:.2f} to {current_s_center + args.S_window:.2f}]")
        
        s_vals = np.linspace(current_s_center - args.S_window, current_s_center + args.S_window, args.S_steps)
        slice_args = [(b, s, args.b_range, run_id + j, results_dir) for j, s in enumerate(s_vals)]
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
                # Strategy: Find S closest to target energy (-2.224 MeV)
                best_run = min(slice_results, key=lambda x: abs(x[2] - ENERGY_TARGET))  # Find closest to target energy
                current_s_center = best_run[1]
                print(f"    ⚠ Missed target crossing. Shifting to closest energy point at S = {current_s_center:.4f} (E = {best_run[2]:.4f} MeV, target = {ENERGY_TARGET})")
        else:
            print("    ⚠ Warning: Slice failed. Keeping same center for next iteration.")
            
        print() # Empty line for readability
                
    # ==========================================
    # 3. SAVE RAW GRID TO CSV
    # ==========================================
    with open(grid_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["b_form", "S", "energy_mev", "radius_fm", "prob_bare", "prob_dressed"])
        writer.writerows(all_results)
    
    # ==========================================
    # 4. PREPARE 1D DATA FOR TRICONTOUR
    # ==========================================
    B_flat = [res[0] for res in all_results]
    S_flat = [res[1] for res in all_results]
    E_flat = [res[2] for res in all_results]
    R_flat = [res[3] for res in all_results]
    P_bare_flat = [res[4] for res in all_results]
    P_dressed_flat = [res[5] for res in all_results]

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
    # 5. GENERATE THE CONTOUR PLOTS (2 subplots)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy contours with calibration target
    ax = axes[0]
    tcf_energy = ax.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='RdBu_r', alpha=0.8)
    cs_energy = ax.tricontour(B_plot, S_plot, E_plot, levels=[ENERGY_TARGET], colors='green', linewidths=3)
    ax.set_title(f"Ground State Energy\n($b_{{range}}$ = {args.b_range} fm)", fontsize=12, fontweight='bold')
    ax.set_xlabel("$b_{{form}}$ (fm)", fontsize=11)
    ax.set_ylabel("Interaction Strength $S$", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    cbar1 = plt.colorbar(tcf_energy, ax=ax)
    cbar1.set_label("Energy (MeV)", fontsize=10)
    
    # Plot 2: Radius contours with calibration target
    ax = axes[1]
    tcf_radius = ax.tricontourf(B_plot, S_plot, R_plot, levels=20, cmap='YlGn', alpha=0.8)
    cs_radius = ax.tricontour(B_plot, S_plot, R_plot, levels=[RADIUS_TARGET], colors='purple', linewidths=3)
    ax.set_title(f"Charge Radius\n($b_{{range}}$ = {args.b_range} fm)", fontsize=12, fontweight='bold')
    ax.set_xlabel("$b_{{form}}$ (fm)", fontsize=11)
    ax.set_ylabel("Interaction Strength $S$", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    cbar2 = plt.colorbar(tcf_radius, ax=ax)
    cbar2.set_label("Radius (fm)", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Contour plots generated successfully: {plot_path}")
    
    # ==========================================
    # 6. FIND AND PRINT BEST CALIBRATION PARAMETERS
    # ==========================================
    print("\n" + "="*80)
    print("CALIBRATION PARAMETER ANALYSIS")
    print("="*80)
    
    # Find point closest to both targets
    best_distance = float('inf')
    best_params = None
    
    for b, s, e, r, p_bare, p_dressed in all_results:
        # Distance metric: combination of energy and radius deviations
        energy_dev = abs(e - ENERGY_TARGET)
        radius_dev = abs(r - RADIUS_TARGET)
        total_distance = energy_dev + radius_dev
        
        if total_distance < best_distance:
            best_distance = total_distance
            best_params = (b, s, e, r, energy_dev, radius_dev)
    
    if best_params:
        b_best, s_best, e_best, r_best, e_dev, r_dev = best_params
        print(f"\nBest Calibration Point (closest to both targets):")
        print(f"  b_form  = {b_best:.6f} fm")
        print(f"  S       = {s_best:.6f} MeV")
        print(f"  Energy  = {e_best:.6f} MeV (target: {ENERGY_TARGET}, error: {e_dev:.6f})")
        print(f"  Radius  = {r_best:.6f} fm  (target: {RADIUS_TARGET}, error: {r_dev:.6f})")
    
    # Find best energy point
    best_energy_idx = min(range(len(all_results)), key=lambda i: abs(all_results[i][2] - ENERGY_TARGET))
    b_e, s_e, e_e, r_e, _, _ = all_results[best_energy_idx]
    print(f"\nBest Energy Point:")
    print(f"  b_form  = {b_e:.6f} fm")
    print(f"  S       = {s_e:.6f} MeV")
    print(f"  Energy  = {e_e:.6f} MeV (error: {abs(e_e - ENERGY_TARGET):.6f})")
    print(f"  Radius  = {r_e:.6f} fm")
    
    # Find best radius point
    best_radius_idx = min(range(len(all_results)), key=lambda i: abs(all_results[i][3] - RADIUS_TARGET))
    b_r, s_r, e_r, r_r, _, _ = all_results[best_radius_idx]
    print(f"\nBest Radius Point:")
    print(f"  b_form  = {b_r:.6f} fm")
    print(f"  S       = {s_r:.6f} MeV")
    print(f"  Energy  = {e_r:.6f} MeV")
    print(f"  Radius  = {r_r:.6f} fm (error: {abs(r_r - RADIUS_TARGET):.6f})")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    sys.exit(main())
