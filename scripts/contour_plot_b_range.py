#!/usr/bin/env python3
"""
contour_plot_b_range.py - Generates a SLANTED 2D grid by dynamically tracking the energy contour

Usage:
  python3 scripts/contour_plot_b_range.py \
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
        return (b_range, S, energy, radius, prob_bare, prob_dressed)
        
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
    parser.add_argument(
        "--S_anchors",
        type=str,
        default=None,
        help="Optional comma-separated list of per-slice S centers (length must equal b_range_steps). If provided, disables dynamic tracking.",
    )
    parser.add_argument("--S_window", type=float, default=5.0, help="Plus/Minus S sweep range around the center")
    parser.add_argument("--S_steps", type=int, default=8)

    parser.add_argument(
        "--refine_iters",
        type=int,
        default=3,
        help="Number of per-slice refinement iterations to hit ENERGY_TARGET after bracketing (runs are parallelized per iteration). Set 0 to disable.",
    )
    parser.add_argument(
        "--refine_tol",
        type=float,
        default=0.01,
        help="Energy tolerance in MeV for refinement convergence (|E-ENERGY_TARGET| < tol).",
    )
    
    parser.add_argument("--b_form", type=float, default=1.2)
    parser.add_argument("--jobs", type=int, default=8)
    
    args = parser.parse_args()

    s_anchors = None
    if args.S_anchors is not None:
        try:
            s_anchors = [float(x.strip()) for x in args.S_anchors.split(",") if x.strip()]
        except ValueError:
            print("ERROR: --S_anchors must be a comma-separated list of floats", file=sys.stderr)
            return 2
        if len(s_anchors) != args.b_range_steps:
            print(
                f"ERROR: --S_anchors length ({len(s_anchors)}) must equal --b_range_steps ({args.b_range_steps})",
                file=sys.stderr,
            )
            return 2
    
    results_dir = "results/contour_b_range"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    grid_csv_path = os.path.join(results_dir, "grid_data.csv")
    plot_path = os.path.join(results_dir, "calibration_contour.png")
    
    b_ranges = np.linspace(args.b_range_min, args.b_range_max, args.b_range_steps)
    b_ranges = np.array([2.0, 2.1, 2.15, 2.20, 2.23, 2.24, 2.25, 2.27, 2.3, 2.4, 2.5, 2.8, 3.0, 3.2, 3.4, 3.6])
    
    print(f"Starting Dynamic Grid Calculation ({args.b_range_steps} slices of {args.S_steps} runs)")
    print(f"Using {args.jobs} CPU cores...\n")
    
    all_results = []
    # Each entry stores a bracket (S_lo,E_lo)-(S_hi,E_hi) that straddles ENERGY_TARGET.
    # We'll refine each bracket toward the root with a few extra runs, executed in parallel batches.
    pending_refine = []  # list[dict]
    run_id = 0
    current_s_center = args.S_init_anchor
    
    # ==========================================
    # 1. PROCESS SLICE BY SLICE
    # ==========================================
    for i, b in enumerate(b_ranges):
        print(f"--- Slice {i+1}/{args.b_range_steps}: b_range = {b:.4f} ---")

        if s_anchors is not None:
            current_s_center = s_anchors[i]
            print(f"    Using provided S anchor: {current_s_center:.4f}")

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
        #    + IMPORTANT FIX: If we find a crossing between two sampled S points,
        #      also run the interpolated S so the radius is evaluated on-energy.
        # ==========================================
        if slice_results:
            slice_results.sort(key=lambda x: x[1])

            if s_anchors is None:
                crossing_found = False
                for k in range(len(slice_results) - 1):
                    s1, e1 = slice_results[k][1], slice_results[k][2]
                    s2, e2 = slice_results[k + 1][1], slice_results[k + 1][2]

                    # Check if the target falls between these two energies
                    if (e1 - ENERGY_TARGET) * (e2 - ENERGY_TARGET) <= 0:
                        if abs(e2 - e1) < 1e-14:
                            exact_s = 0.5 * (s1 + s2)
                        else:
                            exact_s = s1 + (s2 - s1) * (ENERGY_TARGET - e1) / (e2 - e1)

                        # Save the bracket so we can refine toward ENERGY_TARGET using additional runs.
                        s_lo, e_lo, s_hi, e_hi = (s1, e1, s2, e2) if s1 < s2 else (s2, e2, s1, e1)
                        pending_refine.append(
                            {
                                "b": b,
                                "s_lo": float(s_lo),
                                "e_lo": float(e_lo),
                                "s_hi": float(s_hi),
                                "e_hi": float(e_hi),
                            }
                        )
                        print(
                            f"    ✓ Crossing bracketed; queued refinement near S ≈ {exact_s:.4f} (will run in parallel batches)"
                        )

                        current_s_center = exact_s
                        crossing_found = True
                        print(f"    ✓ Next center interpolated to S = {current_s_center:.4f}")
                        break

                if not crossing_found:
                    best_run = min(slice_results, key=lambda x: abs(x[2] - ENERGY_TARGET))
                    current_s_center = best_run[1]
                    print(
                        f"    ⚠ Missed target crossing. Shifting to closest energy point at S = {current_s_center:.4f} (E = {best_run[2]:.4f} MeV, target = {ENERGY_TARGET})"
                    )
        else:
            print("    ⚠ Warning: Slice failed. Keeping same center for next iteration.")
            
        print() # Empty line for readability
                
    # ==========================================
    # 3. SAVE RAW GRID TO CSV
    # ==========================================

    def _secant_or_bisect(s_lo, e_lo, s_hi, e_hi):
        # Secant step; fallback to bisection if slope is tiny or step is out of bracket.
        if abs(e_hi - e_lo) < 1e-14:
            return 0.5 * (s_lo + s_hi)
        s_next = s_lo + (s_hi - s_lo) * (ENERGY_TARGET - e_lo) / (e_hi - e_lo)
        if not (min(s_lo, s_hi) <= s_next <= max(s_lo, s_hi)):
            s_next = 0.5 * (s_lo + s_hi)
        return float(s_next)

    # Refine each bracket toward ENERGY_TARGET with a few parallel iterations.
    if args.refine_iters > 0 and pending_refine:
        active = pending_refine
        print("=" * 80)
        print(
            f"Refining {len(active)} slices toward E = {ENERGY_TARGET} MeV "
            f"(iters={args.refine_iters}, tol={args.refine_tol})"
        )
        print("=" * 80)

        for it in range(args.refine_iters):
            # Build one batch of proposals.
            proposals = []
            for entry in active:
                s_next = _secant_or_bisect(entry["s_lo"], entry["e_lo"], entry["s_hi"], entry["e_hi"])
                entry["s_next"] = s_next
                proposals.append((entry["b"], s_next, args.b_form, None, results_dir))

            # Assign unique run ids.
            batch_args = []
            for (b, s_next, b_form, _rid, resdir) in proposals:
                batch_args.append((b, s_next, b_form, run_id, resdir))
                run_id += 1

            batch_results = []
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                for res in executor.map(run_single_point, batch_args):
                    batch_results.append(res)

            # Update brackets.
            new_active = []
            added = 0
            converged = 0
            for entry, res in zip(active, batch_results):
                if res is None:
                    new_active.append(entry)
                    continue

                b_val, s_val, e_val, r_val, p_bare, p_dressed = res
                all_results.append(res)
                added += 1

                if abs(e_val - ENERGY_TARGET) < args.refine_tol:
                    converged += 1
                    continue

                # Maintain a bracket if possible.
                f_lo = entry["e_lo"] - ENERGY_TARGET
                f_hi = entry["e_hi"] - ENERGY_TARGET
                f_new = e_val - ENERGY_TARGET

                if f_lo * f_new <= 0:
                    entry["s_hi"], entry["e_hi"] = float(s_val), float(e_val)
                elif f_new * f_hi <= 0:
                    entry["s_lo"], entry["e_lo"] = float(s_val), float(e_val)
                else:
                    # If we lost bracketing (non-monotonic), shrink toward best two points.
                    candidates = [
                        (entry["s_lo"], entry["e_lo"]),
                        (entry["s_hi"], entry["e_hi"]),
                        (float(s_val), float(e_val)),
                    ]
                    candidates.sort(key=lambda se: abs(se[1] - ENERGY_TARGET))
                    (s_a, e_a), (s_b, e_b) = candidates[0], candidates[1]
                    if s_a < s_b:
                        entry["s_lo"], entry["e_lo"] = s_a, e_a
                        entry["s_hi"], entry["e_hi"] = s_b, e_b
                    else:
                        entry["s_lo"], entry["e_lo"] = s_b, e_b
                        entry["s_hi"], entry["e_hi"] = s_a, e_a

                new_active.append(entry)

            print(f"Refine iter {it+1}/{args.refine_iters}: added={added}, converged={converged}")

            active = new_active
            if not active:
                break
        print("=" * 80)

    # Sort output for readability (and to keep refined points next to their slices)
    all_results.sort(key=lambda x: (x[0], x[1]))

    with open(grid_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["b_range", "S", "energy_mev", "radius_fm", "prob_bare", "prob_dressed"])
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

    # Build smoothed surfaces for stable target lines.
    # (Radius is energy-sensitive; weight points near the target energy.)
    try:
        from scipy.optimize import curve_fit
    except Exception:
        curve_fit = None

    grid_B = grid_S = grid_E = grid_R = None
    if curve_fit is not None:
        B_arr = np.asarray(B_plot)
        S_arr = np.asarray(S_plot)
        E_arr = np.asarray(E_plot)
        R_arr = np.asarray(R_plot)

        b_lin = np.linspace(B_arr.min(), B_arr.max(), 250)
        s_lin = np.linspace(S_arr.min(), S_arr.max(), 250)
        grid_B, grid_S = np.meshgrid(b_lin, s_lin)

        def poly2d(xy, c0, c1, c2, c3, c4, c5):
            x, y = xy
            return c0 + c1 * x + c2 * y + c3 * x * y + c4 * x**2 + c5 * y**2

        xy = np.column_stack([B_arr, S_arr])
        popt_E, _ = curve_fit(poly2d, xy.T, E_arr, maxfev=20000)

        dE = np.abs(E_arr - ENERGY_TARGET)
        sigma_E = 0.5  # MeV
        w = np.exp(-(dE / sigma_E) ** 2)
        w = np.clip(w, 1e-6, 1.0)
        sigma_R = 1.0 / np.sqrt(w)
        popt_R, _ = curve_fit(poly2d, xy.T, R_arr, sigma=sigma_R, maxfev=30000)

        grid_E = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_E).reshape(grid_B.shape)
        grid_R = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_R).reshape(grid_B.shape)

    # ==========================================
    # 5. GENERATE THE CONTOUR PLOTS (2 subplots)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy contours with calibration target
    ax = axes[0]
    tcf_energy = ax.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='RdBu_r', alpha=0.8)
    if grid_E is not None:
        cs_energy = ax.contour(grid_B, grid_S, grid_E, levels=[ENERGY_TARGET], colors='green', linewidths=3)
    else:
        cs_energy = ax.tricontour(B_plot, S_plot, E_plot, levels=[ENERGY_TARGET], colors='green', linewidths=3)
    ax.set_title(f"Ground State Energy\n($b_{{form}}$ = {args.b_form} fm)", fontsize=12, fontweight='bold')
    ax.set_xlabel("$b_{{range}}$ (fm)", fontsize=11)
    ax.set_ylabel("Interaction Strength $S$", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    cbar1 = plt.colorbar(tcf_energy, ax=ax)
    cbar1.set_label("Energy (MeV)", fontsize=10)
    
    # Plot 2: Radius contours with calibration target
    ax = axes[1]
    tcf_radius = ax.tricontourf(B_plot, S_plot, R_plot, levels=20, cmap='YlGn', alpha=0.8)
    if grid_R is not None:
        cs_radius = ax.contour(grid_B, grid_S, grid_R, levels=[RADIUS_TARGET], colors='purple', linewidths=3)
    else:
        cs_radius = ax.tricontour(B_plot, S_plot, R_plot, levels=[RADIUS_TARGET], colors='purple', linewidths=3)
    ax.set_title(f"Charge Radius\n($b_{{form}}$ = {args.b_form} fm)", fontsize=12, fontweight='bold')
    ax.set_xlabel("$b_{{range}}$ (fm)", fontsize=11)
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
        print(f"  b_range = {b_best:.6f} fm")
        print(f"  S       = {s_best:.6f} MeV")
        print(f"  Energy  = {e_best:.6f} MeV (target: {ENERGY_TARGET}, error: {e_dev:.6f})")
        print(f"  Radius  = {r_best:.6f} fm  (target: {RADIUS_TARGET}, error: {r_dev:.6f})")
    
    # Find best energy point
    best_energy_idx = min(range(len(all_results)), key=lambda i: abs(all_results[i][2] - ENERGY_TARGET))
    b_e, s_e, e_e, r_e, _, _ = all_results[best_energy_idx]
    print(f"\nBest Energy Point:")
    print(f"  b_range = {b_e:.6f} fm")
    print(f"  S       = {s_e:.6f} MeV")
    print(f"  Energy  = {e_e:.6f} MeV (error: {abs(e_e - ENERGY_TARGET):.6f})")
    print(f"  Radius  = {r_e:.6f} fm")
    
    # Find best radius point
    best_radius_idx = min(range(len(all_results)), key=lambda i: abs(all_results[i][3] - RADIUS_TARGET))
    b_r, s_r, e_r, r_r, _, _ = all_results[best_radius_idx]
    print(f"\nBest Radius Point:")
    print(f"  b_range = {b_r:.6f} fm")
    print(f"  S       = {s_r:.6f} MeV")
    print(f"  Energy  = {e_r:.6f} MeV")
    print(f"  Radius  = {r_r:.6f} fm (error: {abs(r_r - RADIUS_TARGET):.6f})")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    sys.exit(main())