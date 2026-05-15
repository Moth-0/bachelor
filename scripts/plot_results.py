#!/usr/bin/env python3
"""
plot_results.py - Generate publication-quality plots from sweep results

Usage:
  python3 scripts/plot_results.py energy_sweep_b_range
  python3 scripts/plot_results.py energy_sweep_b_form
  python3 scripts/plot_results.py energy_sweep_S
  python3 scripts/plot_results.py energy_sweep_basis_size
  python3 scripts/plot_results.py energy_sweep_calibration
"""

import sys
import csv
import os
from pathlib import Path

def plot_contour_map(grid_csv_path):
    """Plot contour map from pre-computed grid data (no need to re-run deu)"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib or numpy not found. Install with: pip install matplotlib numpy", file=sys.stderr)
        return False
    
    # Hardcoded experimental targets
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
    
    # Filter out unbound/gas states (Energy > -0.01)
    valid_indices = [i for i, e in enumerate(E_plot) if e < -0.01]
    B_plot = [B_plot[i] for i in valid_indices]
    S_plot = [S_plot[i] for i in valid_indices]
    E_plot = [E_plot[i] for i in valid_indices]
    R_plot = [R_plot[i] for i in valid_indices]
    
    if not B_plot:
        print("ERROR: No bound states found in grid data", file=sys.stderr)
        return False
    
    # Generate contour plot
    plt.figure(figsize=(8, 6))
    
    cs_energy = plt.tricontour(B_plot, S_plot, E_plot, levels=[ENERGY_TARGET], colors='blue', linewidths=2.5)
    cs_radius = plt.tricontour(B_plot, S_plot, R_plot, levels=[RADIUS_TARGET], colors='red', linewidths=2.5, linestyles='dashed')
    
    plt.tricontourf(B_plot, S_plot, E_plot, levels=20, cmap='Blues', alpha=0.3)
    
    plt.title(f"Parameter Sweep contour map", fontsize=14, fontweight='bold')
    plt.xlabel("$b_{{N}}$ (fm)", fontsize=12)
    plt.ylabel("$S$ (MeV)", fontsize=12)
    
    # Create legend using custom Line2D objects
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2.5, label=f"Energy = {ENERGY_TARGET} MeV"),
        Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label=f"Radius = {RADIUS_TARGET} fm")
    ]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    output_file = grid_csv_path.replace('grid_data.csv', 'contour_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Contour plot saved to: {output_file}")
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
    
    data = {"step": [], "num_boxes": [], "energy": [], "basis_size": [], "radius": []}
    
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
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return False
    
    if not data["step"]:
        print("No basis_size convergence data found in CSV", file=sys.stderr)
        return False
    
    # Sort by step
    sorted_data = sorted(zip(data["step"], data["num_boxes"], data["energy"], data["basis_size"], data["radius"]))
    steps, num_boxes, energies, basis_sizes, radii = zip(*sorted_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    
    # Plot 1: Energy convergence
    ax1 = axes[0]
    ax1.plot(basis_sizes, energies, 'o-', linewidth=2.5, markersize=10, label='Computed', color='steelblue')
    ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
    ax1.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (MeV)', fontsize=12, fontweight='bold')
    ax1.set_title('Basis Size Convergence: Ground State Energy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(basis_sizes)
    
    # Plot 2: Radius convergence
    if radii and any(r is not None and r > 0 for r in radii):
        ax2 = axes[1]
        radii_clean = [r if r is not None else 0 for r in radii]
        ax2.plot(basis_sizes, radii_clean, 's-', linewidth=2.5, markersize=10, color='green', label='Computed')
        ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
        ax2.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Charge Radius (fm)', fontsize=12, fontweight='bold')
        ax2.set_title('Basis Size Convergence: Charge Radius', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend(fontsize=11, loc='best')
        ax2.set_xticks(basis_sizes)
    
    plt.tight_layout()
    output_file = csv_file.replace('aggregated.csv', 'basis_convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Basis size convergence plot saved to: {output_file}")
    return True

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <scan_type>", file=sys.stderr)
        print(f"  scan_type: energy_sweep_b_range | energy_sweep_b_form | energy_sweep_S | energy_sweep_basis_size | energy_sweep_calibration | contour_map", file=sys.stderr)
        return 1
    
    scan_type = sys.argv[1]
    
    # Special case for contour_map (uses grid_data.csv instead of aggregated.csv)
    if scan_type == "contour_map":
        csv_file = "results/contour_map/grid_data.csv"
        if not os.path.exists(csv_file):
            print(f"ERROR: Grid data file not found: {csv_file}", file=sys.stderr)
            return 1
        success = plot_contour_map(csv_file)
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
