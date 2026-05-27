#!/usr/bin/env python3
"""
plot_results.py - Generate publication-quality plots from sweep results

Usage:
  python3 scripts/plot_results.py energy_sweep_b_range
  python3 scripts/plot_results.py energy_sweep_b_form
  python3 scripts/plot_results.py energy_sweep_S
  python3 scripts/plot_results.py energy_sweep_basis_size
  python3 scripts/plot_results.py energy_sweep_calibration
  python3 scripts/plot_results.py smart_contour
"""

import sys
import csv
import os
from pathlib import Path

def plot_smart_contour(csv_file):
    """Plot adaptive mesh search results for charge radius contour"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib or numpy not found. Install with: pip install matplotlib numpy", file=sys.stderr)
        return False

    br_final, bf_final, r_final = [], [], []
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                br_final.append(float(row["b_range"]))
                bf_final.append(float(row["b_form"]))
                r_final.append(float(row["radius_fm"]))
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False

    if not br_final:
        print("No data found in CSV", file=sys.stderr)
        return False

    RADIUS_TARGET = 2.128

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the evaluated points
    ax.scatter(br_final, bf_final, c='black', s=10, alpha=0.5, label='Evaluated Points')

    try:
        # Interpolated surface
        tcf = ax.tricontourf(br_final, bf_final, r_final, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(tcf, ax=ax, label="Charge Radius (fm)")

        # Target contour line
        ax.tricontour(br_final, bf_final, r_final, levels=[RADIUS_TARGET], colors='red', linewidths=3)
    except Exception as e:
        print(f"Triangulation error (you might need more data points): {e}", file=sys.stderr)
        return False

    # Safe legend implementation
    target_line = mlines.Line2D([], [], color='red', linewidth=3, label=f'Target R = {RADIUS_TARGET} fm')
    ax.legend(handles=[target_line], loc="upper right")

    ax.set_title("Adaptive Mesh Search: Charge Radius Contour", fontweight='bold')
    ax.set_xlabel("$b_{range}$ (fm)")
    ax.set_ylabel("$b_{form}$ (fm)")
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    output_file = csv_file.replace('smart_grid_data.csv', 'adaptive_radius_contour.png')
    if output_file == csv_file:
        output_file = csv_file.replace('.csv', '.png')
        
    plt.savefig(output_file, dpi=150)
    print(f"Smart contour plot saved to: {output_file}")
    return True

def plot_contour_b_range(grid_csv_path, b_form):
    """Plot contour map for b_range sweep (from pre-computed grid data)"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import griddata
        from scipy.optimize import curve_fit
    except ImportError:
        print("ERROR: matplotlib, numpy, or scipy not found. Install with: pip install matplotlib numpy scipy", file=sys.stderr)
        return False
    
    ENERGY_TARGET = -2.224
    RADIUS_TARGET = 2.128
    
    B_plot = []
    S_plot = []
    E_plot = []
    R_plot = []
    
    try:
        with open(grid_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    b_range = float(row["b_range"])
                    S = float(row["S"])
                    energy = float(row["energy_mev"])
                    radius = float(row["radius_fm"])
                    
                    B_plot.append(b_range)
                    S_plot.append(S)
                    E_plot.append(energy)
                    R_plot.append(radius)
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading grid CSV: {e}", file=sys.stderr)
        return False
    
    if not B_plot:
        print("No data found in grid CSV", file=sys.stderr)
        return False
    
    # Filter out unbound states
    valid_indices = [i for i, e in enumerate(E_plot) if e < 0]
    B_plot = [B_plot[i] for i in valid_indices]
    S_plot = [S_plot[i] for i in valid_indices]
    E_plot = [E_plot[i] for i in valid_indices]
    R_plot = [R_plot[i] for i in valid_indices]
    
    if not B_plot:
        print("ERROR: No bound states found in grid data", file=sys.stderr)
        return False
    
    # Create interpolated grid for smooth target lines
    B_array = np.array(B_plot)
    S_array = np.array(S_plot)
    E_array = np.array(E_plot)
    R_array = np.array(R_plot)
    
    b_min, b_max = B_array.min(), B_array.max()
    s_min, s_max = S_array.min(), S_array.max()
    grid_b = np.linspace(b_min, b_max, 200)
    grid_s = np.linspace(s_min, s_max, 200)
    grid_B, grid_S = np.meshgrid(grid_b, grid_s)
    
    # Define 2D polynomial fitting function: f(x,y) = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
    def poly2d(xy, c0, c1, c2, c3, c4, c5):
        x, y = xy
        return c0 + c1*x + c2*y + c3*x*y + c4*x**2 + c5*y**2
    
    # Fit polynomials to the data
    xy_data = np.column_stack([B_array, S_array])
    popt_E, _ = curve_fit(poly2d, xy_data.T, E_array, maxfev=1000)

    # Radius is only reliable near the correct (bound-state) energy.
    # Down-weight off-target energies so large-radius points don't distort the R fit.
    dE = np.abs(E_array - ENERGY_TARGET)
    sigma_E = 0.005  # MeV; controls how aggressively we ignore off-target points
    w = np.exp(-(dE / sigma_E) ** 2)
    w = np.clip(w, 1e-6, 1.0)
    sigma_R = 1.0 / np.sqrt(w)
    popt_R, _ = curve_fit(poly2d, xy_data.T, R_array, sigma=sigma_R, maxfev=1000)
    
    # Evaluate fitted polynomials on the mesh grid
    grid_E = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_E).reshape(grid_B.shape)
    grid_R = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_R).reshape(grid_B.shape)

    # Generate single plot with both target lines
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use energy background for reference
    ax.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='Blues', alpha=0.5)
    
    # Overlay both target lines from fitted curves
    cs_energy = ax.contour(grid_B, grid_S, grid_E, levels=[ENERGY_TARGET], colors='blue', linewidths=3)
    cs_radius = ax.contour(grid_B, grid_S, grid_R, levels=[RADIUS_TARGET], colors='red', linewidths=3)
    
    ax.set_title(f"Target Contours: Energy = {ENERGY_TARGET} MeV (blue) & Radius = {RADIUS_TARGET} fm (red)\n($b_{{form}}$ = {b_form} fm)", 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("$b_{{range}}$ (fm)", fontsize=12)
    ax.set_ylabel("Interaction Strength $S$ (MeV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=3, label=f"Energy = {ENERGY_TARGET} MeV"),
        Line2D([0], [0], color='red', linewidth=3, label=f"Radius = {RADIUS_TARGET} fm")
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    output_file = grid_csv_path.replace('grid_data.csv', 'contour_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Contour plot saved to: {output_file}")
    
    # Find intersection of energy and radius target lines
    print("\n" + "="*80)
    print("TARGET LINE INTERSECTION (OPTIMAL PARAMETERS)")
    print("="*80)
    
    try:
        energy_lines = cs_energy.get_paths() if hasattr(cs_energy, 'get_paths') else []
        radius_lines = cs_radius.get_paths() if hasattr(cs_radius, 'get_paths') else []
        
        if len(energy_lines) > 0 and len(radius_lines) > 0:
            energy_verts = energy_lines[0].vertices
            radius_verts = radius_lines[0].vertices
            
            min_dist = float('inf')
            best_idx_e = None
            best_idx_r = None
            
            for i, (b_e, s_e) in enumerate(energy_verts):
                for j, (b_r, s_r) in enumerate(radius_verts):
                    dist = np.sqrt((b_e - b_r)**2 + (s_e - s_r)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx_e = i
                        best_idx_r = j
            
            if best_idx_e is not None and best_idx_r is not None:
                b_energy, s_energy = energy_verts[best_idx_e]
                b_radius, s_radius = radius_verts[best_idx_r]
                
                b_intersect = (b_energy + b_radius) / 2.0
                s_intersect = (s_energy + s_radius) / 2.0
                
                # Evaluate fitted polynomials at intersection (not scattered data)
                xy_intersect = np.array([[b_intersect, s_intersect]])
                e_at_intersect = poly2d(xy_intersect.T, *popt_E)[0]
                r_at_intersect = poly2d(xy_intersect.T, *popt_R)[0]
                
                print(f"\nIntersection of Target Contours:")
                print(f"  b_range = {b_intersect:.6f} fm")
                print(f"  S       = {s_intersect:.6f} MeV")
                print(f"  Energy  = {e_at_intersect:.6f} MeV (target: {ENERGY_TARGET}, error: {abs(e_at_intersect - ENERGY_TARGET):.6f} MeV)")
                print(f"  Radius  = {r_at_intersect:.6f} fm  (target: {RADIUS_TARGET}, error: {abs(r_at_intersect - RADIUS_TARGET):.6f} fm)")
                print(f"  Distance between contours: {min_dist:.6f}")
        else:
            print("Could not extract contour paths.")
    except Exception as e:
        print(f"Error extracting contours: {e}")
    
    print("="*80 + "\n")
    return True

def plot_contour_b_form(grid_csv_path, b_range):
    """Plot contour map for b_form sweep (from pre-computed grid data)"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("ERROR: matplotlib, numpy, or scipy not found. Install with: pip install matplotlib numpy scipy", file=sys.stderr)
        return False
    
    ENERGY_TARGET = -2.224
    RADIUS_TARGET = 2.128
    
    B_plot = []
    S_plot = []
    E_plot = []
    R_plot = []
    
    try:
        with open(grid_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    b_form = float(row["b_form"])
                    S = float(row["S"])
                    energy = float(row["energy_mev"])
                    radius = float(row["radius_fm"])
                    
                    B_plot.append(b_form)
                    S_plot.append(S)
                    E_plot.append(energy)
                    R_plot.append(radius)
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading grid CSV: {e}", file=sys.stderr)
        return False
    
    if not B_plot:
        print("No data found in grid CSV", file=sys.stderr)
        return False
    
    # Filter out unbound states
    valid_indices = [i for i, e in enumerate(E_plot) if e < -0.01]
    B_plot = [B_plot[i] for i in valid_indices]
    S_plot = [S_plot[i] for i in valid_indices]
    E_plot = [E_plot[i] for i in valid_indices]
    R_plot = [R_plot[i] for i in valid_indices]
    
    if not B_plot:
        print("ERROR: No bound states found in grid data", file=sys.stderr)
        return False
    
    # Create interpolated grid for smooth target lines
    B_array = np.array(B_plot)
    S_array = np.array(S_plot)
    E_array = np.array(E_plot)
    R_array = np.array(R_plot)
    
    b_min, b_max = B_array.min(), B_array.max()
    s_min, s_max = S_array.min(), S_array.max()
    grid_b = np.linspace(b_min, b_max, 200)
    grid_s = np.linspace(s_min, s_max, 200)
    grid_B, grid_S = np.meshgrid(grid_b, grid_s)
    
    from scipy.interpolate import RectBivariateSpline
    
    points = np.column_stack([B_array, S_array])
    
    # Fit smooth splines to the data
    from scipy.optimize import curve_fit
    
    # Define 2D polynomial fitting function: f(x,y) = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
    def poly2d(xy, c0, c1, c2, c3, c4, c5):
        x, y = xy
        return c0 + c1*x + c2*y + c3*x*y + c4*x**2 + c5*y**2
    
    # Fit polynomials to the data
    xy_data = np.column_stack([B_array, S_array])
    popt_E, _ = curve_fit(poly2d, xy_data.T, E_array, maxfev=10000)

    dE = np.abs(E_array - ENERGY_TARGET)
    sigma_E = 0.5  # MeV
    w = np.exp(-(dE / sigma_E) ** 2)
    w = np.clip(w, 1e-6, 1.0)
    sigma_R = 1.0 / np.sqrt(w)
    popt_R, _ = curve_fit(poly2d, xy_data.T, R_array, sigma=sigma_R, maxfev=20000)
    
    # Evaluate fitted polynomials on the mesh grid
    grid_E = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_E).reshape(grid_B.shape)
    grid_R = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_R).reshape(grid_B.shape)
    
    # Generate single plot with both target lines
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use energy background for reference
    ax.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='Blues', alpha=0.5)
    
    # Overlay both target lines
    cs_energy = ax.contour(grid_B, grid_S, grid_E, levels=[ENERGY_TARGET], colors='blue', linewidths=3)
    cs_radius = ax.contour(grid_B, grid_S, grid_R, levels=[RADIUS_TARGET], colors='red', linewidths=3)
    
    ax.set_title(f"Target Contours: Energy = {ENERGY_TARGET} MeV (blue) & Radius = {RADIUS_TARGET} fm (red)\n($b_{{range}}$ = {b_range} fm)", 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("$b_{{form}}$ (fm)", fontsize=12)
    ax.set_ylabel("Interaction Strength $S$ (MeV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=3, label=f"Energy = {ENERGY_TARGET} MeV"),
        Line2D([0], [0], color='red', linewidth=3, label=f"Radius = {RADIUS_TARGET} fm")
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    output_file = grid_csv_path.replace('grid_data.csv', 'contour_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Contour plot saved to: {output_file}")
    
    # Find intersection of energy and radius target lines
    print("\n" + "="*80)
    print("TARGET LINE INTERSECTION (OPTIMAL PARAMETERS)")
    print("="*80)
    
    try:
        energy_lines = cs_energy.get_paths() if hasattr(cs_energy, 'get_paths') else []
        radius_lines = cs_radius.get_paths() if hasattr(cs_radius, 'get_paths') else []
        
        if len(energy_lines) > 0 and len(radius_lines) > 0:
            energy_verts = energy_lines[0].vertices
            radius_verts = radius_lines[0].vertices
            
            min_dist = float('inf')
            best_idx_e = None
            best_idx_r = None
            
            for i, (b_e, s_e) in enumerate(energy_verts):
                for j, (b_r, s_r) in enumerate(radius_verts):
                    dist = np.sqrt((b_e - b_r)**2 + (s_e - s_r)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx_e = i
                        best_idx_r = j
            
            if best_idx_e is not None and best_idx_r is not None:
                b_energy, s_energy = energy_verts[best_idx_e]
                b_radius, s_radius = radius_verts[best_idx_r]
                
                b_intersect = (b_energy + b_radius) / 2.0
                s_intersect = (s_energy + s_radius) / 2.0
                
                # Evaluate fitted polynomials at intersection (not scattered data)
                xy_intersect = np.array([[b_intersect, s_intersect]])
                e_at_intersect = poly2d(xy_intersect.T, *popt_E)[0]
                r_at_intersect = poly2d(xy_intersect.T, *popt_R)[0]
                
                print(f"\nIntersection of Target Contours:")
                print(f"  b_form  = {b_intersect:.6f} fm")
                print(f"  S       = {s_intersect:.6f} MeV")
                print(f"  Energy  = {e_at_intersect:.6f} MeV (target: {ENERGY_TARGET}, error: {abs(e_at_intersect - ENERGY_TARGET):.6f} MeV)")
                print(f"  Radius  = {r_at_intersect:.6f} fm  (target: {RADIUS_TARGET}, error: {abs(r_at_intersect - RADIUS_TARGET):.6f} fm)")
                print(f"  Distance between contours: {min_dist:.6f}")
        else:
            print("Could not extract contour paths.")
    except Exception as e:
        print(f"Error extracting contours: {e}")
    
    print("="*80 + "\n")
    return True

def plot_contour_map(grid_csv_path):
    """Plot contour map from pre-computed grid data (no need to re-run deu)"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("ERROR: matplotlib, numpy, or scipy not found. Install with: pip install matplotlib numpy scipy", file=sys.stderr)
        return False
    
    # Hardcoded experimental targets
    ENERGY_TARGET = -2.224
    RADIUS_TARGET = 2.128
    
    B_plot = []
    S_plot = []
    E_plot = []
    R_plot = []
    all_points = []  # Store all data points for analysis
    
    try:
        with open(grid_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    b_range = float(row["b_range"])
                    S = float(row["S"])
                    energy = float(row["energy_mev"])
                    radius = float(row["radius_fm"])
                    
                    B_plot.append(b_range)
                    S_plot.append(S)
                    E_plot.append(energy)
                    R_plot.append(radius)
                    all_points.append((b_range, S, energy, radius))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading grid CSV: {e}", file=sys.stderr)
        return False
    
    if not B_plot:
        print("No data found in grid CSV", file=sys.stderr)
        return False
    
    # Filter out unbound/gas states (Energy > -0.01)
    valid_indices = [i for i, e in enumerate(E_plot) if e < -0.01]
    B_plot = [B_plot[i] for i in valid_indices]
    S_plot = [S_plot[i] for i in valid_indices]
    E_plot = [E_plot[i] for i in valid_indices]
    R_plot = [R_plot[i] for i in valid_indices]
    all_points_valid = [all_points[i] for i in valid_indices]
    
    if not B_plot:
        print("ERROR: No bound states found in grid data", file=sys.stderr)
        return False
    
    # Create interpolated grid for smooth target lines only
    B_array = np.array(B_plot)
    S_array = np.array(S_plot)
    E_array = np.array(E_plot)
    R_array = np.array(R_plot)
    
    b_min, b_max = B_array.min(), B_array.max()
    s_min, s_max = S_array.min(), S_array.max()
    grid_b = np.linspace(b_min, b_max, 200)
    grid_s = np.linspace(s_min, s_max, 200)
    grid_B, grid_S = np.meshgrid(grid_b, grid_s)
    
    from scipy.optimize import curve_fit
    
    # Define 2D polynomial fitting function: f(x,y) = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
    def poly2d(xy, c0, c1, c2, c3, c4, c5):
        x, y = xy
        return c0 + c1*x + c2*y + c3*x*y + c4*x**2 + c5*y**2
    
    points = np.column_stack([B_array, S_array])
    
    # Fit polynomials to the data
    xy_data = np.column_stack([B_array, S_array])
    popt_E, _ = curve_fit(poly2d, xy_data.T, E_array, maxfev=10000)

    dE = np.abs(E_array - ENERGY_TARGET)
    sigma_E = 0.5  # MeV
    w = np.exp(-(dE / sigma_E) ** 2)
    w = np.clip(w, 1e-6, 1.0)
    sigma_R = 1.0 / np.sqrt(w)
    popt_R, _ = curve_fit(poly2d, xy_data.T, R_array, sigma=sigma_R, maxfev=20000)
    
    # Evaluate fitted polynomials on the mesh grid
    grid_E = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_E).reshape(grid_B.shape)
    grid_R = poly2d(np.column_stack([grid_B.ravel(), grid_S.ravel()]).T, *popt_R).reshape(grid_B.shape)
    
    # Generate single plot showing both target lines
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use energy background for reference
    ax.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='Blues', alpha=0.5)
    
    # Overlay both target lines
    cs_energy = ax.contour(grid_B, grid_S, grid_E, levels=[ENERGY_TARGET], colors='blue', linewidths=3)
    cs_radius = ax.contour(grid_B, grid_S, grid_R, levels=[RADIUS_TARGET], colors='red', linewidths=3)
    
    ax.set_title(f"Target Contours: Energy = {ENERGY_TARGET} MeV (blue) & Radius = {RADIUS_TARGET} fm (red)", 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("$b_{{range}}$ (fm)", fontsize=12)
    ax.set_ylabel("Interaction Strength $S$ (MeV)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=3, label=f"Energy = {ENERGY_TARGET} MeV"),
        Line2D([0], [0], color='red', linewidth=3, label=f"Radius = {RADIUS_TARGET} fm")
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    output_file = grid_csv_path.replace('grid_data.csv', 'contour_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Contour plot saved to: {output_file}")
    
    # Find intersection of energy and radius target lines
    print("\n" + "="*80)
    print("TARGET LINE INTERSECTION (OPTIMAL PARAMETERS)")
    print("="*80)
    
    # Extract contour line coordinates from the contour objects
    try:
        # Get paths from the contour line collections
        energy_lines = cs_energy.get_paths() if hasattr(cs_energy, 'get_paths') else []
        radius_lines = cs_radius.get_paths() if hasattr(cs_radius, 'get_paths') else []
        
        if len(energy_lines) > 0 and len(radius_lines) > 0:
            # Get the first contour line for each
            energy_verts = energy_lines[0].vertices
            radius_verts = radius_lines[0].vertices
            
            # Find closest approach between the two curves
            min_dist = float('inf')
            best_idx_e = None
            best_idx_r = None
            
            for i, (b_e, s_e) in enumerate(energy_verts):
                for j, (b_r, s_r) in enumerate(radius_verts):
                    dist = np.sqrt((b_e - b_r)**2 + (s_e - s_r)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx_e = i
                        best_idx_r = j
            
            if best_idx_e is not None and best_idx_r is not None:
                b_energy, s_energy = energy_verts[best_idx_e]
                b_radius, s_radius = radius_verts[best_idx_r]
                
                # Interpolate to find the closest intersection point
                b_intersect = (b_energy + b_radius) / 2.0
                s_intersect = (s_energy + s_radius) / 2.0
                
                # Evaluate fitted polynomials at the intersection point
                xy_intersect = np.array([[b_intersect, s_intersect]])
                e_at_intersect = poly2d(xy_intersect.T, *popt_E)[0]
                r_at_intersect = poly2d(xy_intersect.T, *popt_R)[0]
                
                print(f"\nIntersection of Target Contours:")
                print(f"  b_range = {b_intersect:.6f} fm")
                print(f"  S       = {s_intersect:.6f} MeV")
                print(f"  Energy  = {e_at_intersect:.6f} MeV (target: {ENERGY_TARGET}, error: {abs(e_at_intersect - ENERGY_TARGET):.6f} MeV)")
                print(f"  Radius  = {r_at_intersect:.6f} fm  (target: {RADIUS_TARGET}, error: {abs(r_at_intersect - RADIUS_TARGET):.6f} fm)")
                print(f"  Distance between contours: {min_dist:.6f}")
        else:
            print("Could not extract contour paths.")
    except Exception as e:
        print(f"Error extracting contours: {e}")
    
    print("="*80 + "\n")
    
    return True

def plot_energy_vs_b_form(csv_file):
    """Plot energy vs b_form"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        return False
    
    data = {"b_form": [], "energy": [], "radius": []}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["b_form"].append(float(row["b_form"]))
                data["energy"].append(float(row["energy_mev"]))
                try:
                    data["radius"].append(float(row["radius_fm"]))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False
    
    if not data["b_form"]:
        print("No data found in CSV", file=sys.stderr)
        return False
    
    # Sort by parameter
    sorted_data = sorted(zip(data["b_form"], data["energy"], data["radius"]))
    b_forms, energies, radii = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Energy vs b_form
    ax1.plot(b_forms, energies, 'o-', linewidth=2, markersize=8, label='Computed')
    ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
    ax1.set_xlabel('b_form (fm)', fontsize=12)
    ax1.set_ylabel('Energy (MeV)', fontsize=12)
    ax1.set_title('Deuteron Ground State Energy vs b_form', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Radius vs b_form
    if radii and any(r > 0 for r in radii):
        ax2.plot(b_forms, radii, 's-', linewidth=2, markersize=8, color='green', label='Computed')
        ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
        ax2.set_xlabel('b_form (fm)', fontsize=12)
        ax2.set_ylabel('Charge Radius (fm)', fontsize=12)
        ax2.set_title('Charge Radius vs b_form', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = csv_file.replace('aggregated.csv', 'plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    return True

def plot_energy_vs_b_range(csv_file):
    """Plot energy vs b_range"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        return False
    
    data = {"b_range": [], "energy": [], "radius": []}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["b_range"].append(float(row["b_range"]))
                data["energy"].append(float(row["energy_mev"]))
                try:
                    data["radius"].append(float(row["radius_fm"]))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False
    
    if not data["b_range"]:
        print("No data found in CSV", file=sys.stderr)
        return False
    
    # Sort by parameter
    sorted_data = sorted(zip(data["b_range"], data["energy"], data["radius"]))
    b_ranges, energies, radii = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Energy vs b_range
    ax1.plot(b_ranges, energies, 'o-', linewidth=2, markersize=8, label='Computed')
    ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
    ax1.set_xlabel('b_range (fm)', fontsize=12)
    ax1.set_ylabel('Energy (MeV)', fontsize=12)
    ax1.set_title('Deuteron Ground State Energy vs b_range', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Radius vs b_range
    if radii and any(r > 0 for r in radii):
        ax2.plot(b_ranges, radii, 's-', linewidth=2, markersize=8, color='green', label='Computed')
        ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
        ax2.set_xlabel('b_range (fm)', fontsize=12)
        ax2.set_ylabel('Charge Radius (fm)', fontsize=12)
        ax2.set_title('Charge Radius vs b_range', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = csv_file.replace('aggregated.csv', 'plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    return True

def plot_energy_vs_S(csv_file):
    """Plot energy vs S (coupling strength)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        return False
    
    data = {"S": [], "energy": [], "radius": []}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["S"].append(float(row["S"]))
                data["energy"].append(float(row["energy_mev"]))
                try:
                    data["radius"].append(float(row["radius_fm"]))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False
    
    if not data["S"]:
        print("No data found in CSV", file=sys.stderr)
        return False
    
    # Sort by parameter
    sorted_data = sorted(zip(data["S"], data["energy"], data["radius"]))
    S_vals, energies, radii = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Energy vs S
    ax1.plot(S_vals, energies, 'o-', linewidth=2, markersize=8, label='Computed')
    ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
    ax1.set_xlabel('S - Pion Coupling Strength (MeV)', fontsize=12)
    ax1.set_ylabel('Energy (MeV)', fontsize=12)
    ax1.set_title('Deuteron Ground State Energy vs Coupling Strength', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Radius vs S
    if radii and any(r > 0 for r in radii):
        ax2.plot(S_vals, radii, 's-', linewidth=2, markersize=8, color='green', label='Computed')
        ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
        ax2.set_xlabel('S - Pion Coupling Strength (MeV)', fontsize=12)
        ax2.set_ylabel('Charge Radius (fm)', fontsize=12)
        ax2.set_title('Charge Radius vs Coupling Strength', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = csv_file.replace('aggregated.csv', 'plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    return True

def plot_calibration_convergence(csv_file):
    """Plot calibration convergence: energy, radius, and parameters vs iteration"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        return False
    
    data = {"iteration": [], "b_range": [], "S": [], "energy": [], "radius": []}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data["iteration"].append(int(float(row["iteration"])))  # Convert float string first
                    data["b_range"].append(float(row["b_range"]))
                    data["S"].append(float(row["S"]))
                    data["energy"].append(float(row["energy_mev"]))
                    data["radius"].append(float(row["radius_fm"]))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False
    
    if not data["iteration"]:
        print("No calibration data found in CSV", file=sys.stderr)
        return False
    
    # Sort by iteration
    sorted_data = sorted(zip(data["iteration"], data["b_range"], data["S"], data["energy"], data["radius"]))
    iterations, b_ranges, S_vals, energies, radii = zip(*sorted_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Energy convergence
    ax1 = axes[0]
    ax1.plot(b_ranges, energies, 'o-', linewidth=2, markersize=8, color='steelblue', label='Computed')
    ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
    ax1.set_xlabel('b_range (fm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Energy (MeV)', fontsize=11, fontweight='bold')
    ax1.set_title('Energy Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Radius convergence
    ax2 = axes[1]
    ax2.plot(b_ranges, radii, 's-', linewidth=2, markersize=8, color='green', label='Computed')
    ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
    ax2.set_xlabel('b_range (fm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Charge Radius (fm)', fontsize=11, fontweight='bold')
    ax2.set_title('Radius Convergence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Parameter evolution (S vs b_range)
    ax3 = axes[2]
    ax3.plot(b_ranges, S_vals, '^-', linewidth=2, markersize=8, color='orange', label='S')
    ax3.set_xlabel('b_range (fm)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('S - Coupling Strength (MeV)', fontsize=11, fontweight='bold')
    ax3.set_title('S Parameter Evolution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = csv_file.replace('aggregated.csv', 'calibration_convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Calibration convergence plot saved to: {output_file}")
    return True

def plot_basis_size_convergence(csv_file):
    """Plot energy and radius convergence vs box strengths count"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        return False
    
    data = {"step": [], "num_boxes": [], "energy": [], "basis_size": [], "radius": [], "execution_time": []}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row.get("step", 0))
                    box_strengths = row.get("box_strengths", "")
                    # Count number of box strengths (elements separated by commas)
                    num_boxes = len([x for x in box_strengths.split(',') if x.strip()])
                    
                    data["step"].append(step)
                    data["num_boxes"].append(num_boxes)
                    data["energy"].append(float(row["energy_mev"]))
                    data["basis_size"].append(float(row["basis_size"]))
                    try:
                        data["radius"].append(float(row["radius_fm"]))
                    except (KeyError, ValueError):
                        data["radius"].append(None)
                    try:
                        data["execution_time"].append(float(row.get("execution_time_s", 0)))
                    except (KeyError, ValueError):
                        data["execution_time"].append(0)
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False
    
    if not data["step"]:
        print("No basis_size convergence data found in CSV", file=sys.stderr)
        return False
    
    # Sort by step
    sorted_data = sorted(zip(data["step"], data["num_boxes"], data["energy"], data["basis_size"], data["radius"], data["execution_time"]))
    steps, num_boxes, energies, basis_sizes, radii, exec_times = zip(*sorted_data)
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    
    # Target values and tolerance
    ENERGY_TARGET = -2.224
    RADIUS_TARGET = 2.128
    TOLERANCE_PERCENT = 0.001  # ±0.1%
    
    # Plot 1: Energy convergence with ±0.1% band
    ax1 = axes[0]
    energy_tolerance = abs(ENERGY_TARGET * TOLERANCE_PERCENT)
    ax1.plot(basis_sizes, energies, 'o-', linewidth=2.5, markersize=10, label='Computed', color='steelblue')
    ax1.axhline(y=ENERGY_TARGET, color='r', linestyle='--', linewidth=2, label=f'Target ({ENERGY_TARGET} MeV)')
    ax1.fill_between(basis_sizes, ENERGY_TARGET - energy_tolerance, ENERGY_TARGET + energy_tolerance, 
                     alpha=0.2, color='red', label='±0.1% Target Band')
    ax1.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (MeV)', fontsize=12, fontweight='bold')
    ax1.set_title('Basis Size Convergence: Ground State Energy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(basis_sizes)
    
    # Plot 2: Radius convergence with ±0.1% band
    if radii and any(r is not None and r > 0 for r in radii):
        ax2 = axes[1]
        radii_clean = [r if r is not None else 0 for r in radii]
        radius_tolerance = RADIUS_TARGET * TOLERANCE_PERCENT
        ax2.plot(basis_sizes, radii_clean, 's-', linewidth=2.5, markersize=10, color='green', label='Computed')
        ax2.axhline(y=RADIUS_TARGET, color='r', linestyle='--', linewidth=2, label=f'Target ({RADIUS_TARGET} fm)')
        ax2.fill_between(basis_sizes, RADIUS_TARGET - radius_tolerance, RADIUS_TARGET + radius_tolerance,
                        alpha=0.2, color='red', label='±0.1% Target Band')
        ax2.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Charge Radius (fm)', fontsize=12, fontweight='bold')
        ax2.set_title('Basis Size Convergence: Charge Radius', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend(fontsize=11, loc='best')
        ax2.set_xticks(basis_sizes)
    
    # Plot 3: Execution time convergence
    if exec_times and any(t > 0 for t in exec_times):
        ax3 = axes[2]
        ax3.plot(basis_sizes, exec_times, '^-', linewidth=2.5, markersize=10, color='darkorange', label='Execution Time')
        ax3.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_title('Basis Size Convergence: Execution Time', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.legend(fontsize=11, loc='best')
        ax3.set_xticks(basis_sizes)
    
    plt.tight_layout()
    output_file = csv_file.replace('aggregated.csv', 'basis_convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Basis size convergence plot saved to: {output_file}")
    return True

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <scan_type> [fixed_param]", file=sys.stderr)
        print(f"  scan_type: energy_sweep_b_range | energy_sweep_b_form | energy_sweep_S | energy_sweep_basis_size | energy_sweep_calibration | contour_map | contour_b_range | contour_b_form | smart_contour", file=sys.stderr)
        print(f"  fixed_param: for contour_b_range, pass b_form value; for contour_b_form, pass b_range value", file=sys.stderr)
        return 1
    
    scan_type = sys.argv[1]
    
    # Special case for contour plots (use grid_data.csv instead of aggregated.csv)
    if scan_type == "contour_map":
        csv_file = "results/contour_map/grid_data.csv"
        if not os.path.exists(csv_file):
            print(f"ERROR: Grid data file not found: {csv_file}", file=sys.stderr)
            return 1
        success = plot_contour_map(csv_file)
        return 0 if success else 1
        
    if scan_type == "smart_contour":
        csv_file = "results/smart_contour/smart_grid_data.csv"
        if not os.path.exists(csv_file):
            print(f"ERROR: Grid data file not found: {csv_file}", file=sys.stderr)
            return 1
        success = plot_smart_contour(csv_file)
        return 0 if success else 1
    
    if scan_type == "contour_b_range":
        if len(sys.argv) < 3:
            print(f"ERROR: contour_b_range requires b_form parameter", file=sys.stderr)
            print(f"Usage: {sys.argv[0]} contour_b_range <b_form>", file=sys.stderr)
            return 1
        csv_file = "results/contour_b_range/grid_data.csv"
        if not os.path.exists(csv_file):
            print(f"ERROR: Grid data file not found: {csv_file}", file=sys.stderr)
            return 1
        try:
            b_form = float(sys.argv[2])
        except ValueError:
            print(f"ERROR: b_form must be a number, got: {sys.argv[2]}", file=sys.stderr)
            return 1
        success = plot_contour_b_range(csv_file, b_form)
        return 0 if success else 1
    
    if scan_type == "contour_b_form":
        if len(sys.argv) < 3:
            print(f"ERROR: contour_b_form requires b_range parameter", file=sys.stderr)
            print(f"Usage: {sys.argv[0]} contour_b_form <b_range>", file=sys.stderr)
            return 1
        csv_file = "results/contour_b_form/grid_data.csv"
        if not os.path.exists(csv_file):
            print(f"ERROR: Grid data file not found: {csv_file}", file=sys.stderr)
            return 1
        try:
            b_range = float(sys.argv[2])
        except ValueError:
            print(f"ERROR: b_range must be a number, got: {sys.argv[2]}", file=sys.stderr)
            return 1
        success = plot_contour_b_form(csv_file, b_range)
        return 0 if success else 1
    
    csv_file = f"results/{scan_type}/aggregated.csv"
    
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}", file=sys.stderr)
        return 1
    
    if scan_type == "energy_sweep_b_range":
        success = plot_energy_vs_b_range(csv_file)
    elif scan_type == "energy_sweep_b_form":
        success = plot_energy_vs_b_form(csv_file)
    elif scan_type == "energy_sweep_S":
        success = plot_energy_vs_S(csv_file)
    elif scan_type == "energy_sweep_basis_size":
        success = plot_basis_size_convergence(csv_file)
    elif scan_type == "energy_sweep_calibration":
        success = plot_calibration_convergence(csv_file)
    else:
        print(f"ERROR: Unknown scan type: {scan_type}", file=sys.stderr)
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())