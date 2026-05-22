#!/usr/bin/env python3
"""
smart_contour_search.py - Adaptive 2D search for b_range and b_form
Finds S for the target energy, then adaptively samples parameter space 
where the charge radius is close to the target.

Usage:
  python3 scripts/smart_contour_search.py \
    --br_min 2.0 --br_max 4.0 \
    --bf_min 1.0 --bf_max 1.5 \
    --coarse_steps 4 --refine_iters 4 \
    --S_guess 45.0 --jobs 8
"""

import os
import sys
import csv
import uuid
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Hardcoded experimental targets
ENERGY_TARGET = -2.224
RADIUS_TARGET = 2.128

def run_deu(b_range, b_form, S, results_dir):
    """Executes deu binary for a specific (b_range, b_form, S) and returns (Energy, Radius)."""
    run_id = uuid.uuid4().hex
    csv_path = os.path.join(results_dir, f"temp_{run_id}.csv")
    
    cmd = [
        "./deu",
        "-b_range", f"{b_range:.4f}",
        "-b_form", f"{b_form:.4f}",
        "-S", f"{S:.4f}",
        "-box-strengths", "10.0,5.0,2.0,1.0",
        "--output-csv", csv_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True)
        with open(csv_path, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
            if len(lines) < 2: 
                return None, None
            
            reader = csv.DictReader(lines)
            final_row = list(reader)[-1]
            
            energy = float(final_row.get("energy_mev", 0))
            radius = float(final_row.get("radius_fm", 0))
            
        os.remove(csv_path)
        return energy, radius
    except Exception:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return None, None

def find_S_and_R(args):
    """Uses the Secant method to find the S that yields ENERGY_TARGET."""
    b_range, b_form, S_guess, results_dir = args
    
    # Initial guesses for Secant method
    S0 = S_guess
    S1 = S_guess + 2.0
    
    E0, _ = run_deu(b_range, b_form, S0, results_dir)
    if E0 is None: return None
    f0 = E0 - ENERGY_TARGET
    
    E1, R1 = run_deu(b_range, b_form, S1, results_dir)
    if E1 is None: return None
    f1 = E1 - ENERGY_TARGET
    
    max_iter = 15
    tol = 0.005 # MeV tolerance
    
    for _ in range(max_iter):
        if abs(f1) < tol:
            return (b_range, b_form, S1, E1, R1)
        if abs(f1 - f0) < 1e-10: # Prevent division by zero
            break
            
        # Secant step
        S2 = S1 - f1 * (S1 - S0) / (f1 - f0)
        
        E2, R2 = run_deu(b_range, b_form, S2, results_dir)
        if E2 is None: break
        
        f2 = E2 - ENERGY_TARGET
        S0, f0 = S1, f1
        S1, f1, E1, R1 = S2, f2, E2, R2
        
    return (b_range, b_form, S1, E1, R1)

def main():
    parser = argparse.ArgumentParser(description="Smart Adaptive Grid Search for Radius Contour")
    parser.add_argument("--br_min", type=float, required=True, help="Min b_range")
    parser.add_argument("--br_max", type=float, required=True, help="Max b_range")
    parser.add_argument("--bf_min", type=float, required=True, help="Min b_form")
    parser.add_argument("--bf_max", type=float, required=True, help="Max b_form")
    
    parser.add_argument("--coarse_steps", type=int, default=4, help="Grid size for initial coarse search (NxN)")
    parser.add_argument("--refine_iters", type=int, default=4, help="Number of adaptive refinement iterations")
    
    parser.add_argument("--S_guess", type=float, default=45.0, help="Initial guess for interaction strength S")
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel deu jobs")
    
    args = parser.parse_args()
    
    results_dir = "results/smart_contour"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    grid_csv_path = os.path.join(results_dir, "smart_grid_data.csv")
    plot_path = os.path.join(results_dir, "adaptive_radius_contour.png")
    
    evaluated_points = []
    pending_points = []
    
    # 1. Initialize coarse grid
    br_vals = np.linspace(args.br_min, args.br_max, args.coarse_steps)
    bf_vals = np.linspace(args.bf_min, args.bf_max, args.coarse_steps)
    for br in br_vals:
        for bf in bf_vals:
            pending_points.append((br, bf))
            
    print(f"Starting Smart Adaptive Search using {args.jobs} cores...")
    
    # 2. Main Active Learning Loop
    for iteration in range(args.refine_iters + 1):
        print(f"\n--- Iteration {iteration}/{args.refine_iters}: Evaluating {len(pending_points)} points ---")
        
        tasks = [(br, bf, args.S_guess, results_dir) for (br, bf) in pending_points]
        pending_points = [] # Clear pending queue
        
        # Run batch in parallel
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(find_S_and_R, tasks))
            
        for res in results:
            if res is not None:
                evaluated_points.append(res)
                
        if iteration == args.refine_iters:
            break
            
        # 3. Adaptive Triangulation to find the contour
        br_arr = np.array([p[0] for p in evaluated_points])
        bf_arr = np.array([p[1] for p in evaluated_points])
        R_arr = np.array([p[4] for p in evaluated_points])
        
        try:
            triang = mtri.Triangulation(br_arr, bf_arr)
        except Exception as e:
            print(f"Triangulation failed (likely too few points): {e}")
            break
            
        new_points = set()
        for edges in triang.edges:
            i, j = edges[0], edges[1]
            R_i, R_j = R_arr[i], R_arr[j]
            
            # If the target radius falls between the radii of these two connected points
            if (R_i - RADIUS_TARGET) * (R_j - RADIUS_TARGET) <= 0:
                # Calculate midpoint of the edge
                br_mid = (br_arr[i] + br_arr[j]) / 2.0
                bf_mid = (bf_arr[i] + bf_arr[j]) / 2.0
                new_points.add((round(br_mid, 4), round(bf_mid, 4)))
                
        # Queue midpoints that haven't been evaluated yet
        existing = {(round(p[0], 4), round(p[1], 4)) for p in evaluated_points}
        for (br, bf) in new_points:
            if (br, bf) not in existing:
                pending_points.append((br, bf))
                
        if not pending_points:
            print("No new boundary points found. Contour is fully resolved or out of bounds.")
            break

    # Save data
    evaluated_points.sort(key=lambda x: (x[0], x[1]))
    with open(grid_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["b_range", "b_form", "S_opt", "energy_mev", "radius_fm"])
        writer.writerows(evaluated_points)

    # 4. Plotting
    br_final = np.array([p[0] for p in evaluated_points])
    bf_final = np.array([p[1] for p in evaluated_points])
    r_final = np.array([p[4] for p in evaluated_points])

    plt.figure(figsize=(8, 6))
    
    # Plot the evaluated points to show the adaptive mesh density
    plt.scatter(br_final, bf_final, c='black', s=10, alpha=0.5, label='Evaluated Points')
    
    # Plot the interpolated charge radius surface
    tcf = plt.tricontourf(br_final, bf_final, r_final, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(tcf, label="Charge Radius (fm)")
    
    # Draw the specific target line
    cs = plt.tricontour(br_final, bf_final, r_final, levels=[RADIUS_TARGET], colors='red', linewidths=3)
    
    # Add a custom legend entry for the contour line
    cs.collections[0].set_label(f'Target R = {RADIUS_TARGET} fm')
    
    plt.title("Adaptive Mesh Search: Charge Radius Contour", fontweight='bold')
    plt.xlabel("$b_{range}$ (fm)")
    plt.ylabel("$b_{form}$ (fm)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved data to {grid_csv_path}")
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    sys.exit(main())