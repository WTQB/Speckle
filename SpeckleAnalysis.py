"""
Speckle Analysis - Fluorescent security tag speckling under smartphone flash

Analyzes 4D parameter sweeps (height, yaw, pitch, roll) from the automated
testing rig to characterize reflective speckling across tag orientations.

Usage:
    python SpeckleAnalysis.py                          # Analyze all JSON files in directory
    python SpeckleAnalysis.py file1.json               # Analyze specific file
    python SpeckleAnalysis.py file1.json file2.json    # Compare datasets
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import sys

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'figures'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': False,
    'font.size': 10,
})

# ============================================================
# DATA LOADING
# ============================================================

def extract_tag_id(json_path):
    """Extract tag ID from filename pattern: {TAG_ID}_{date}_{time}.json"""
    stem = json_path.stem
    parts = stem.split('_')
    if len(parts) >= 3:
        return parts[0]
    return stem


def classify_outcome(position):
    """Classify position outcome into three tiers."""
    sr = position.get('server_response', {})
    failed_checks = sr.get('failed_checks', [])

    if 'missing_uid' in failed_checks:
        return 'unreadable'
    elif sr.get('status') == 'Not Registered':
        return 'readable'
    else:
        return 'server_error'


def build_data_arrays(positions):
    """Build numpy arrays from positions list."""
    n = len(positions)
    heights = np.zeros(n, dtype=int)
    yaws = np.zeros(n, dtype=float)
    pitches = np.zeros(n, dtype=float)
    rolls = np.zeros(n, dtype=float)
    is_readable = np.zeros(n, dtype=bool)
    speckle_area = np.full(n, np.nan)
    processing_time = np.zeros(n, dtype=float)
    capture_time = np.zeros(n, dtype=float)
    outcomes = []

    for i, pos in enumerate(positions):
        heights[i] = pos['height_mm']
        yaws[i] = pos['yaw_degrees']
        pitches[i] = pos['pitch_degrees']
        rolls[i] = pos['roll_degrees']
        capture_time[i] = pos.get('capture_time_seconds', 0)
        processing_time[i] = pos.get('server_response', {}).get('processing_time_ms', 0)

        outcome = classify_outcome(pos)
        outcomes.append(outcome)
        is_readable[i] = (outcome == 'readable')

        if 'speckle_area_percent' in pos:
            speckle_area[i] = pos['speckle_area_percent']

    return {
        'heights': heights,
        'yaws': yaws,
        'pitches': pitches,
        'rolls': rolls,
        'is_readable': is_readable,
        'outcomes': np.array(outcomes),
        'speckle_area': speckle_area,
        'processing_time': processing_time,
        'capture_time': capture_time,
    }


def get_axis_values(sweep_config):
    """Extract unique axis values from sweep config."""
    axes = {}
    for axis_name in ['height', 'yaw', 'pitch', 'roll']:
        cfg = sweep_config[axis_name]
        vals = np.arange(cfg['min'], cfg['max'] + cfg['step'], cfg['step'])
        axes[axis_name] = vals.astype(int) if axis_name == 'height' else vals
    return axes


def load_sweep(json_path):
    """Load a sweep JSON file and return structured data."""
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        raw = json.load(f)

    tag_id = extract_tag_id(json_path)
    positions = raw['positions']
    sweep_config = raw['sweep']['config']
    sweep_results = raw['sweep']
    data = build_data_arrays(positions)
    axis_values = get_axis_values(sweep_config)

    return {
        'tag_id': tag_id,
        'data': data,
        'axis_values': axis_values,
        'sweep_config': sweep_config,
        'sweep_results': sweep_results,
        'json_path': json_path,
    }


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_readability_rate(data, param_name, param_values):
    """Compute readability rate grouped by a parameter."""
    params = data[param_name]
    rates = []
    for val in param_values:
        mask = np.isclose(params, val)
        valid = data['outcomes'][mask] != 'server_error'
        readable = data['is_readable'][mask] & valid
        rate = np.sum(readable) / np.sum(valid) * 100 if np.sum(valid) > 0 else 0
        rates.append(rate)
    return np.array(rates)


def compute_speckle_by_param(data, param_name, param_values):
    """Compute mean speckle grouped by a parameter (readable entries only)."""
    params = data[param_name]
    means = []
    stds = []
    for val in param_values:
        mask = np.isclose(params, val) & data['is_readable']
        speckles = data['speckle_area'][mask]
        valid = speckles[~np.isnan(speckles)]
        if len(valid) > 0:
            means.append(np.mean(valid))
            stds.append(np.std(valid))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def compute_2d_grid(data, row_param, row_values, col_param, col_values,
                    fixed_params=None, metric='readability'):
    """Compute a 2D grid of a metric for heatmap display."""
    grid = np.full((len(row_values), len(col_values)), np.nan)
    rows = data[row_param]
    cols = data[col_param]

    for i, rv in enumerate(row_values):
        for j, cv in enumerate(col_values):
            mask = np.isclose(rows, rv) & np.isclose(cols, cv)

            if fixed_params:
                for fp_name, fp_val in fixed_params.items():
                    mask &= np.isclose(data[fp_name], fp_val)

            if metric == 'readability':
                valid = data['outcomes'][mask] != 'server_error'
                readable = data['is_readable'][mask] & valid
                grid[i, j] = np.sum(readable) / np.sum(valid) * 100 if np.sum(valid) > 0 else np.nan
            elif metric == 'speckle_mean':
                readable_mask = mask & data['is_readable']
                speckles = data['speckle_area'][readable_mask]
                valid = speckles[~np.isnan(speckles)]
                grid[i, j] = np.mean(valid) if len(valid) > 0 else np.nan
            elif metric == 'speckle_max':
                readable_mask = mask & data['is_readable']
                speckles = data['speckle_area'][readable_mask]
                valid = speckles[~np.isnan(speckles)]
                grid[i, j] = np.max(valid) if len(valid) > 0 else np.nan

    return grid


# ============================================================
# PLOT 1: READABILITY HEATMAP GRID
# ============================================================

def plot_readability_heatmap_grid(data, axis_values, tag_id):
    """4x4 grid of pitch vs roll heatmaps, faceted by height x yaw."""
    height_vals = axis_values['height']
    yaw_vals = axis_values['yaw']
    pitch_vals = axis_values['pitch']
    roll_vals = axis_values['roll']

    fig, axes = plt.subplots(len(height_vals), len(yaw_vals),
                              figsize=(14, 14), squeeze=False)
    fig.suptitle(f'QR Readability Across Parameter Space\nTag: {tag_id}',
                 fontsize=16, fontweight='bold', y=0.98)

    cmap = ListedColormap(['#d32f2f', '#66bb6a'])
    norm = BoundaryNorm([0, 50, 100], cmap.N)

    for i, h in enumerate(height_vals):
        for j, y in enumerate(yaw_vals):
            ax = axes[i, j]
            grid = compute_2d_grid(data, 'pitches', pitch_vals, 'rolls', roll_vals,
                                   fixed_params={'heights': h, 'yaws': y},
                                   metric='readability')

            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal',
                          origin='lower', interpolation='nearest')

            ax.set_xticks(range(len(roll_vals)))
            ax.set_xticklabels([f'{int(v)}' for v in roll_vals], fontsize=7)
            ax.set_yticks(range(len(pitch_vals)))
            ax.set_yticklabels([f'{int(v)}' for v in pitch_vals], fontsize=7)

            if i == len(height_vals) - 1:
                ax.set_xlabel('Roll (deg)', fontsize=8)
            if j == 0:
                ax.set_ylabel('Pitch (deg)', fontsize=8)

            ax.set_title(f'H={h}mm, Yaw={y}\u00b0', fontsize=9, pad=4)

            # Annotate cells
            for ri in range(len(pitch_vals)):
                for ci in range(len(roll_vals)):
                    val = grid[ri, ci]
                    if not np.isnan(val):
                        label = 'R' if val > 50 else 'U'
                        color = 'white' if val < 50 else 'black'
                        ax.text(ci, ri, label, ha='center', va='center',
                               fontsize=6, fontweight='bold', color=color)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#66bb6a', label='Readable'),
                       Patch(facecolor='#d32f2f', label='Unreadable')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    save_path = OUTPUT_DIR / f'{tag_id}_readability_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 2: SPECKLE AREA HEATMAP GRID
# ============================================================

def plot_speckle_heatmap_grid(data, axis_values, tag_id):
    """Speckle area heatmaps for readable heights only."""
    readable_heights = [h for h in axis_values['height']
                        if np.any(data['is_readable'] & np.isclose(data['heights'], h))]
    yaw_vals = axis_values['yaw']
    pitch_vals = axis_values['pitch']
    roll_vals = axis_values['roll']

    if not readable_heights:
        print("  Skipped speckle grid: no readable entries")
        return

    fig, axes = plt.subplots(len(readable_heights), len(yaw_vals),
                              figsize=(14, 4 * len(readable_heights)), squeeze=False)
    fig.suptitle(f'Speckle Area (%) - Readable Positions Only\nTag: {tag_id}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Find global max for consistent colorscale
    readable_speckle = data['speckle_area'][data['is_readable']]
    valid_speckle = readable_speckle[~np.isnan(readable_speckle)]
    vmax = np.max(valid_speckle) if len(valid_speckle) > 0 else 0.1

    for i, h in enumerate(readable_heights):
        for j, y in enumerate(yaw_vals):
            ax = axes[i, j]
            grid = compute_2d_grid(data, 'pitches', pitch_vals, 'rolls', roll_vals,
                                   fixed_params={'heights': h, 'yaws': y},
                                   metric='speckle_mean')

            ax_im = ax.imshow(grid, cmap='YlOrRd', aspect='equal',
                             origin='lower', interpolation='nearest',
                             vmin=0, vmax=vmax)

            ax.set_xticks(range(len(roll_vals)))
            ax.set_xticklabels([f'{int(v)}' for v in roll_vals], fontsize=7)
            ax.set_yticks(range(len(pitch_vals)))
            ax.set_yticklabels([f'{int(v)}' for v in pitch_vals], fontsize=7)

            if i == len(readable_heights) - 1:
                ax.set_xlabel('Roll (deg)', fontsize=8)
            if j == 0:
                ax.set_ylabel('Pitch (deg)', fontsize=8)

            ax.set_title(f'H={h}mm, Yaw={y}\u00b0', fontsize=9, pad=4)

            # Annotate cells with values
            for ri in range(len(pitch_vals)):
                for ci in range(len(roll_vals)):
                    val = grid[ri, ci]
                    if not np.isnan(val):
                        color = 'white' if val > vmax * 0.6 else 'black'
                        ax.text(ci, ri, f'{val:.3f}', ha='center', va='center',
                               fontsize=5, color=color)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(ax_im, cax=cbar_ax)
    cbar.set_label('Speckle Area (%)', fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.08, wspace=0.3, hspace=0.35)
    save_path = OUTPUT_DIR / f'{tag_id}_speckle_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 3: MARGINAL EFFECTS
# ============================================================

def plot_marginal_effects(data, axis_values, tag_id):
    """Marginal effect of each parameter on readability and speckle."""
    param_map = {
        'Height (mm)': ('heights', axis_values['height']),
        'Yaw (deg)': ('yaws', axis_values['yaw']),
        'Pitch (deg)': ('pitches', axis_values['pitch']),
        'Roll (deg)': ('rolls', axis_values['roll']),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Marginal Parameter Effects\nTag: {tag_id}',
                 fontsize=16, fontweight='bold')

    for ax, (label, (param_name, param_vals)) in zip(axes.flat, param_map.items()):
        # Readability rate
        rates = compute_readability_rate(data, param_name, param_vals)
        x = np.arange(len(param_vals))

        color_read = '#1976d2'
        color_speckle = '#d32f2f'

        ax.bar(x - 0.15, rates, 0.3, color=color_read, alpha=0.7, label='Readability (%)')
        ax.set_ylabel('Readability (%)', color=color_read, fontsize=10)
        ax.set_ylim(0, 110)
        ax.tick_params(axis='y', labelcolor=color_read)

        # Speckle on secondary axis
        ax2 = ax.twinx()
        means, stds = compute_speckle_by_param(data, param_name, param_vals)
        valid_mask = ~np.isnan(means)
        if np.any(valid_mask):
            ax2.errorbar(x[valid_mask], means[valid_mask], yerr=stds[valid_mask],
                        color=color_speckle, marker='o', linewidth=2,
                        markersize=6, capsize=3, label='Mean Speckle (%)')
            ax2.set_ylabel('Speckle Area (%)', color=color_speckle, fontsize=10)
            ax2.tick_params(axis='y', labelcolor=color_speckle)
            ax2.set_ylim(bottom=0)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' if v == int(v) else f'{v}' for v in param_vals])
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')

    # Combined legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_read, alpha=0.7, label='Readability (%)'),
        Line2D([0], [0], color=color_speckle, marker='o', linewidth=2, label='Mean Speckle (%)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    save_path = OUTPUT_DIR / f'{tag_id}_marginal_effects.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 4: ANGULAR SENSITIVITY
# ============================================================

def plot_angular_sensitivity(data, axis_values, tag_id):
    """Angular sensitivity: pitch-roll plane contour + radial plot."""
    pitch_vals = axis_values['pitch']
    roll_vals = axis_values['roll']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Angular Sensitivity Analysis\nTag: {tag_id}',
                 fontsize=16, fontweight='bold')

    # Left: Pitch-roll plane contour (averaged over yaw, at most speckled height)
    readable_heights = [h for h in axis_values['height']
                        if np.any(data['is_readable'] & np.isclose(data['heights'], h))]

    if readable_heights:
        # Use the height with most speckle for contour
        best_h = None
        best_mean = 0
        for h in readable_heights:
            mask = data['is_readable'] & np.isclose(data['heights'], h)
            sp = data['speckle_area'][mask]
            sp = sp[~np.isnan(sp)]
            if len(sp) > 0 and np.mean(sp) > best_mean:
                best_mean = np.mean(sp)
                best_h = h

        if best_h is not None:
            # Average over yaw for this height
            grid = compute_2d_grid(data, 'pitches', pitch_vals, 'rolls', roll_vals,
                                   fixed_params={'heights': best_h},
                                   metric='speckle_mean')

            P, R = np.meshgrid(roll_vals, pitch_vals)
            contour = ax1.contourf(P, R, grid, levels=15, cmap='YlOrRd')
            ax1.contour(P, R, grid, levels=15, colors='gray', linewidths=0.3, alpha=0.5)
            plt.colorbar(contour, ax=ax1, label='Speckle Area (%)')
            ax1.set_xlabel('Roll (deg)', fontsize=11)
            ax1.set_ylabel('Pitch (deg)', fontsize=11)
            ax1.set_title(f'Speckle at H={best_h}mm\n(averaged over yaw)', fontsize=12)
            ax1.set_aspect('equal')
            ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax1.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    # Right: Mean speckle vs angular deviation, per readable height
    colors = plt.cm.tab10(np.linspace(0, 1, len(readable_heights)))
    for h, color in zip(readable_heights, colors):
        mask = data['is_readable'] & np.isclose(data['heights'], h)
        p = data['pitches'][mask]
        r = data['rolls'][mask]
        sp = data['speckle_area'][mask]

        # Compute angular distance from normal
        ang_dist = np.sqrt(p**2 + r**2)
        unique_dists = np.sort(np.unique(np.round(ang_dist, 1)))

        mean_speckles = []
        for d in unique_dists:
            d_mask = np.isclose(np.round(ang_dist, 1), d)
            vals = sp[d_mask]
            vals = vals[~np.isnan(vals)]
            mean_speckles.append(np.mean(vals) if len(vals) > 0 else np.nan)

        ax2.plot(unique_dists, mean_speckles, 'o-', color=color,
                label=f'H={h}mm', linewidth=2, markersize=5)

    ax2.set_xlabel('Angular Deviation from Normal (deg)', fontsize=11)
    ax2.set_ylabel('Mean Speckle Area (%)', fontsize=11)
    ax2.set_title('Speckle vs Angular Deviation', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = OUTPUT_DIR / f'{tag_id}_angular_sensitivity.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 5: SUMMARY STATISTICS
# ============================================================

def plot_summary_statistics(data, axis_values, sweep_config, sweep_results, tag_id):
    """Text-based summary figure."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    # Compute stats
    total = len(data['outcomes'])
    n_readable = np.sum(data['is_readable'])
    n_unreadable = np.sum(data['outcomes'] == 'unreadable')
    n_error = np.sum(data['outcomes'] == 'server_error')

    readable_speckle = data['speckle_area'][data['is_readable']]
    valid_speckle = readable_speckle[~np.isnan(readable_speckle)]

    # Per-height readability
    height_lines = []
    for h in axis_values['height']:
        mask = np.isclose(data['heights'], h)
        valid = data['outcomes'][mask] != 'server_error'
        n_read = np.sum(data['is_readable'][mask])
        n_total = np.sum(valid)
        rate = n_read / n_total * 100 if n_total > 0 else 0
        height_lines.append(f"    {h}mm: {n_read}/{n_total} ({rate:.0f}%)")

    # Best/worst angular positions
    if len(valid_speckle) > 0:
        readable_mask = data['is_readable'] & ~np.isnan(data['speckle_area'])
        sp = data['speckle_area'][readable_mask]
        pitches = data['pitches'][readable_mask]
        rolls = data['rolls'][readable_mask]
        heights = data['heights'][readable_mask]

        worst_idx = np.argmax(sp)
        best_idx = np.argmin(sp)

        speckle_text = (
            f"  Mean:       {np.mean(valid_speckle):.4f}%\n"
            f"  Std:        {np.std(valid_speckle):.4f}%\n"
            f"  Median:     {np.median(valid_speckle):.4f}%\n"
            f"  Max:        {np.max(valid_speckle):.4f}%\n"
            f"  Non-zero:   {np.sum(valid_speckle > 0)}/{len(valid_speckle)}\n"
            f"\n"
            f"  Worst: {sp[worst_idx]:.4f}% at H={heights[worst_idx]:.0f}mm, "
            f"P={pitches[worst_idx]:.0f}, R={rolls[worst_idx]:.0f}\n"
            f"  Best:  {sp[best_idx]:.4f}% at H={heights[best_idx]:.0f}mm, "
            f"P={pitches[best_idx]:.0f}, R={rolls[best_idx]:.0f}"
        )
    else:
        speckle_text = "  No speckle data available"

    duration = sweep_results.get('duration_seconds', 0)
    duration_min = duration / 60

    text = (
        f"SPECKLE ANALYSIS SUMMARY\n"
        f"{'='*50}\n\n"
        f"Tag ID:        {tag_id}\n"
        f"Sweep Duration: {duration_min:.1f} minutes\n"
        f"Total Positions: {total}\n\n"
        f"READABILITY\n"
        f"{'-'*50}\n"
        f"  Readable:     {n_readable}/{total} ({n_readable/total*100:.1f}%)\n"
        f"  Unreadable:   {n_unreadable}/{total} ({n_unreadable/total*100:.1f}%)\n"
        f"  Server Error: {n_error}\n\n"
        f"  Per Height:\n"
        + '\n'.join(height_lines) +
        f"\n\n"
        f"SPECKLE AREA (readable positions)\n"
        f"{'-'*50}\n"
        + speckle_text +
        f"\n\n"
        f"SWEEP CONFIGURATION\n"
        f"{'-'*50}\n"
        f"  Heights:  {list(axis_values['height'])} mm\n"
        f"  Yaw:      {list(axis_values['yaw'])} deg\n"
        f"  Pitch:    {list(axis_values['pitch'])} deg\n"
        f"  Roll:     {list(axis_values['roll'])} deg\n"
        f"  Settle:   {sweep_config.get('settle_time_seconds', '?')}s\n"
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=11, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5f5',
                     edgecolor='#333333', linewidth=1.5))

    fig.suptitle(f'Summary Report - Tag: {tag_id}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = OUTPUT_DIR / f'{tag_id}_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 6: TIMING DIAGNOSTICS
# ============================================================

def plot_timing_diagnostics(data, axis_values, tag_id):
    """Processing time and capture time diagnostics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Timing Diagnostics\nTag: {tag_id}',
                 fontsize=16, fontweight='bold')

    # Top: Histogram of processing_time_ms for readable entries
    readable_proc = data['processing_time'][data['is_readable']]
    readable_proc = readable_proc[readable_proc > 0]

    if len(readable_proc) > 0:
        ax1.hist(readable_proc, bins=30, color='#1976d2', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(readable_proc), color='red', linewidth=2,
                    linestyle='--', label=f'Mean: {np.mean(readable_proc):.1f}ms')
        ax1.axvline(np.median(readable_proc), color='orange', linewidth=2,
                    linestyle='--', label=f'Median: {np.median(readable_proc):.1f}ms')
        ax1.legend(fontsize=10)

    ax1.set_xlabel('Processing Time (ms)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Server Processing Time (readable entries)', fontsize=12)

    # Bottom: Capture time vs index, colored by height
    indices = np.arange(len(data['capture_time']))
    colors_map = {70: '#d32f2f', 90: '#ff9800', 110: '#4caf50', 130: '#1976d2'}

    for h in axis_values['height']:
        mask = np.isclose(data['heights'], h)
        ax2.scatter(indices[mask], data['capture_time'][mask],
                   s=3, color=colors_map.get(h, 'gray'), alpha=0.6, label=f'H={h}mm')

    ax2.set_xlabel('Position Index', fontsize=11)
    ax2.set_ylabel('Capture Time (s)', fontsize=11)
    ax2.set_title('Capture Time per Position', fontsize=12)
    ax2.legend(fontsize=9, markerscale=3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = OUTPUT_DIR / f'{tag_id}_timing.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 7: MULTI-DATASET COMPARISON
# ============================================================

def plot_comparison(datasets):
    """Compare multiple datasets side by side."""
    n = len(datasets)
    tag_ids = [ds['tag_id'] for ds in datasets]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Dataset Comparison: {" vs ".join(tag_ids)}',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    # Panel 1: Readability per height
    all_heights = datasets[0]['axis_values']['height']
    x = np.arange(len(all_heights))
    width = 0.8 / n

    for k, ds in enumerate(datasets):
        rates = compute_readability_rate(ds['data'], 'heights', all_heights)
        ax1.bar(x + k * width - 0.4 + width / 2, rates, width,
                color=colors[k], label=ds['tag_id'], edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(h)}mm' for h in all_heights])
    ax1.set_ylabel('Readability (%)')
    ax1.set_title('Readability by Height')
    ax1.legend()
    ax1.set_ylim(0, 110)

    # Panel 2: Speckle distribution box plots
    speckle_data = []
    labels = []
    for ds in datasets:
        sp = ds['data']['speckle_area'][ds['data']['is_readable']]
        sp = sp[~np.isnan(sp)]
        speckle_data.append(sp)
        labels.append(ds['tag_id'])

    bp = ax2.boxplot(speckle_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Speckle Area (%)')
    ax2.set_title('Speckle Distribution')

    # Panel 3: Speckle vs pitch per tag
    for k, ds in enumerate(datasets):
        pitch_vals = ds['axis_values']['pitch']
        means, stds = compute_speckle_by_param(ds['data'], 'pitches', pitch_vals)
        valid = ~np.isnan(means)
        if np.any(valid):
            ax3.errorbar(pitch_vals[valid], means[valid], yerr=stds[valid],
                        color=colors[k], marker='o', linewidth=2,
                        capsize=3, label=ds['tag_id'])

    ax3.set_xlabel('Pitch (deg)')
    ax3.set_ylabel('Mean Speckle Area (%)')
    ax3.set_title('Speckle vs Pitch')
    ax3.legend()
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    tag_str = '_vs_'.join(tag_ids)
    save_path = OUTPUT_DIR / f'comparison_{tag_str}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_summary(ds):
    """Print text summary to console."""
    data = ds['data']
    tag_id = ds['tag_id']
    axis_values = ds['axis_values']

    total = len(data['outcomes'])
    n_readable = np.sum(data['is_readable'])
    n_unreadable = np.sum(data['outcomes'] == 'unreadable')
    n_error = np.sum(data['outcomes'] == 'server_error')

    print(f"\n{'='*60}")
    print(f"  Tag: {tag_id}")
    print(f"{'='*60}")
    print(f"  Positions: {total}")
    print(f"  Readable:  {n_readable} ({n_readable/total*100:.1f}%)")
    print(f"  Unreadable: {n_unreadable} ({n_unreadable/total*100:.1f}%)")
    print(f"  Errors:    {n_error}")

    # Per-height breakdown
    print(f"\n  Per Height:")
    for h in axis_values['height']:
        mask = np.isclose(data['heights'], h)
        valid = data['outcomes'][mask] != 'server_error'
        n_read = np.sum(data['is_readable'][mask])
        n_total = np.sum(valid)
        rate = n_read / n_total * 100 if n_total > 0 else 0
        print(f"    {h}mm: {n_read}/{n_total} ({rate:.0f}%)")

    # Speckle stats
    readable_sp = data['speckle_area'][data['is_readable']]
    valid_sp = readable_sp[~np.isnan(readable_sp)]
    if len(valid_sp) > 0:
        print(f"\n  Speckle Area (readable):")
        print(f"    Mean:   {np.mean(valid_sp):.4f}%")
        print(f"    Std:    {np.std(valid_sp):.4f}%")
        print(f"    Max:    {np.max(valid_sp):.4f}%")
        print(f"    >0:     {np.sum(valid_sp > 0)}/{len(valid_sp)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) > 1:
        json_paths = [Path(p) for p in sys.argv[1:]]
    else:
        json_paths = sorted(SCRIPT_DIR.glob('*_*_*.json'))

    if not json_paths:
        print("No sweep JSON files found.")
        print("Place JSON files in the script directory or pass them as arguments.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load all datasets
    datasets = []
    for jp in json_paths:
        print(f"Loading: {jp.name}")
        ds = load_sweep(jp)
        datasets.append(ds)

    # Single-dataset analysis
    for ds in datasets:
        tag_id = ds['tag_id']
        data = ds['data']
        axis_values = ds['axis_values']

        print_summary(ds)

        print(f"\n  Generating plots...")
        plot_readability_heatmap_grid(data, axis_values, tag_id)
        plot_speckle_heatmap_grid(data, axis_values, tag_id)
        plot_marginal_effects(data, axis_values, tag_id)
        plot_angular_sensitivity(data, axis_values, tag_id)
        plot_summary_statistics(data, axis_values, ds['sweep_config'],
                                ds['sweep_results'], tag_id)
        plot_timing_diagnostics(data, axis_values, tag_id)

    # Multi-dataset comparison
    if len(datasets) > 1:
        print(f"\n{'='*60}")
        print("Generating comparison plots...")
        print(f"{'='*60}")
        plot_comparison(datasets)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")
