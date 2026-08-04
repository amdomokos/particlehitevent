"""Figures for the Phase 7 perturbation-robustness study.

A separate module from ``Models/common/plotting.py``, which is training-run
specific (loss curves, parity). Same conventions as that module: the ``Agg``
backend is selected at import so this is safe on a headless pod, every figure
is built with ``fig, ax = plt.subplots``, laid out with ``tight_layout``, saved
at dpi 120, and closed.

Design choices worth stating, because they are load-bearing rather than taste:

  - LOG-LOG axes. Both the sigma grid and the resulting MSE span three-plus
    decades, and second-order theory predicts a power law (dMSE ~ sigma^2). On
    linear axes every point below sigma=0.1 collapses onto the origin and the
    only visible feature is the destructive end. Log-log makes the exponent
    readable as a gradient, which is exactly what the proposal asks to compare.
  - Color identifies the CHECKPOINT and is fixed by model name, not by plot
    order — a run that omits a model must not repaint the survivors. Slots are
    taken in order from a categorical palette validated for colorblind
    separation on all pairs (worst CVD dE 9.2, normal-vision dE 24.0).
  - Every series carries a distinct marker as well as a hue, and the curve
    plot direct-labels each line at its right end. That is the required relief
    for the aqua slot, which sits below 3:1 contrast on a light surface, and it
    also survives a grayscale print of the figure.
  - One y-axis, never two. MSE and dMSE are different quantities on different
    scales, so they get different figures rather than a twin axis.
"""
import math
import os

import matplotlib
matplotlib.use('Agg')          # headless: set before pyplot is imported
import matplotlib.pyplot as plt   # noqa: E402

# Categorical slots 1-3 of the validated reference palette, light mode. Three
# is the documented all-pairs-safe cap; a fourth model would need a facet or an
# "other" fold rather than a new hue.
_SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
_SERIES_MARKERS = ('o', 's', '^')

_SURFACE = '#fcfcfb'
_INK = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_MUTED = '#898781'
_GRID = '#e1e0d9'
_AXIS = '#c3c2b7'

JOINT_SUBSET = 'A+B+C'
PANEL_SUBSETS = ('A', 'B', 'C', JOINT_SUBSET)

CURVES_PNG = 'robustness_curves.png'
BY_MATRIX_PNG = 'robustness_by_matrix.png'
DELTA_PNG = 'robustness_delta.png'


def _model_order(summary):
    """Stable model ordering: first appearance in the (already sorted) summary.

    Color is bound to this ordering once and reused by every figure, so the
    same checkpoint is the same hue across all three.
    """
    seen = []
    for row in summary:
        if row['model'] not in seen:
            seen.append(row['model'])
    return seen


def _style(models):
    """-> {model: (color, marker)}. Cycling is deliberately not implemented:
    past three series the palette stops being colorblind-separable, so a
    fourth model should fail loudly here rather than ship an unreadable
    figure."""
    if len(models) > len(_SERIES_COLORS):
        raise ValueError(
            f"{len(models)} models but only {len(_SERIES_COLORS)} validated "
            f"categorical slots; facet the figure or fold models rather than "
            f"generating a new hue")
    return {m: (_SERIES_COLORS[i], _SERIES_MARKERS[i])
            for i, m in enumerate(models)}


def _series(summary, model, subset, key='mse_mean'):
    """-> (sigmas, means, lo, hi) for one model+subset, ascending in sigma.

    The band is the OBSERVED min-max across seeds, not mean +/- std. On a log
    axis a symmetric std band is the wrong geometry: perturbation-instance
    spread routinely approaches or exceeds the mean at moderate sigma, and
    ``mean - std`` then goes non-positive, which either has to be clamped to
    an arbitrary floor — dragging a shared log axis down a dozen decades and
    flattening every real feature — or dropped. Min-max is always positive,
    needs no fiction, and states the full range that was actually measured.
    The sample std is still reported in robustness_summary.csv.
    """
    rows = sorted((r for r in summary
                   if r['model'] == model and r['subset'] == subset),
                  key=lambda r: r['sigma'])
    return ([r['sigma'] for r in rows],
            [r[key] for r in rows],
            [r.get('mse_min', r[key]) for r in rows],
            [r.get('mse_max', r[key]) for r in rows])


def _dress(ax, xlabel, ylabel, title=None):
    """The recessive-chrome pass every axes gets."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, color=_INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=9)
    if title:
        ax.set_title(title, color=_INK, fontsize=10, loc='left')
    ax.grid(True, which='major', color=_GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which='minor', color=_GRID, linewidth=0.4, alpha=0.6,
            zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_AXIS)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=_SURFACE)
    plt.close(fig)
    return path


def plot_curves(summary, baselines, output_dir, subset=JOINT_SUBSET):
    """MSE vs sigma for one subset, one line per checkpoint.

    The headline figure. Shaded band is the observed min-max over the
    independent seeds at that sigma — the visual answer to "is the degradation
    smooth, or does it just look noisy because we drew once?". Each sigma=0
    baseline is drawn as a horizontal dashed line in its own color, so the
    distance between a curve and its own floor is readable directly rather
    than by subtracting two numbers off the axis.
    """
    models = _model_order(summary)
    style = _style(models)
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    plotted = 0
    for model in models:
        sigmas, means, lo, hi = _series(summary, model, subset)
        if not sigmas:
            continue
        color, marker = style[model]
        ax.fill_between(sigmas, lo, hi, color=color, alpha=0.18, linewidth=0,
                        zorder=2)
        ax.plot(sigmas, means, color=color, marker=marker, markersize=5,
                linewidth=2, label=model, zorder=3,
                markeredgecolor=_SURFACE, markeredgewidth=0.8)
        base = baselines.get(model, {}).get('mse')
        if base:
            ax.axhline(base, color=color, linestyle=':', linewidth=1.2,
                       alpha=0.7, zorder=1)
        # Direct label at the right end — the relief the palette's contrast
        # WARN requires, and it survives a grayscale print.
        ax.annotate(model, xy=(sigmas[-1], means[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    color=color, fontsize=8, va='center', ha='left',
                    fontweight='bold')
        plotted += 1

    _dress(ax, r'relative perturbation $\sigma$',
           'test MSE (mean over seeds)',
           f'Weight-perturbation robustness — {subset} perturbed')
    if plotted >= 2:
        leg = ax.legend(frameon=False, fontsize=8, loc='upper left',
                        labelcolor=_INK_SECONDARY)
        leg.set_zorder(4)
    ax.margins(x=0.16)
    fig.text(0.01, 0.005,
             "shaded band: observed min-max over seeds   "
             "...... each model's sigma=0 baseline",
             color=_MUTED, fontsize=7)
    return _save(fig, os.path.join(output_dir, CURVES_PNG))


def plot_by_matrix(summary, baselines, output_dir):
    """2x2 panels, one per perturbed matrix subset, shared axes.

    Shared limits are the point: the per-matrix sensitivity ordering is meant
    to be read by comparing panels, and independently scaled panels would make
    an insensitive matrix look exactly like a sensitive one.
    """
    models = _model_order(summary)
    style = _style(models)
    present = [s for s in PANEL_SUBSETS
               if any(r['subset'] == s for r in summary)]
    if not present:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=_SURFACE,
                             sharex=True, sharey=True)
    flat = axes.flatten()
    for ax in flat:
        ax.set_facecolor(_SURFACE)

    for i, subset in enumerate(PANEL_SUBSETS):
        ax = flat[i]
        if subset not in present:
            ax.set_visible(False)
            continue
        for model in models:
            sigmas, means, lo, hi = _series(summary, model, subset)
            if not sigmas:
                continue
            color, marker = style[model]
            ax.fill_between(sigmas, lo, hi, color=color, alpha=0.18,
                            linewidth=0, zorder=2)
            ax.plot(sigmas, means, color=color, marker=marker, markersize=4.5,
                    linewidth=2, label=model, zorder=3,
                    markeredgecolor=_SURFACE, markeredgewidth=0.8)
            base = baselines.get(model, {}).get('mse')
            if base:
                ax.axhline(base, color=color, linestyle=':', linewidth=1.1,
                           alpha=0.7, zorder=1)
        _dress(ax, r'relative perturbation $\sigma$', 'test MSE',
               f'{subset} perturbed')

    handles, labels = flat[0].get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(labels),
                   loc='lower center', labelcolor=_INK_SECONDARY,
                   bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Perturbation sensitivity by state-space matrix',
                 color=_INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(output_dir, BY_MATRIX_PNG), dpi=120,
                facecolor=_SURFACE)
    plt.close(fig)
    return os.path.join(output_dir, BY_MATRIX_PNG)


def plot_delta_with_fits(summary, slopes, output_dir, subset=JOINT_SUBSET):
    """dMSE vs sigma with the fitted power laws overlaid.

    This is the figure that makes the proposal's question visual: parallel
    lines mean the models share a degradation exponent and differ only in
    scale; converging or crossing lines mean one really does degrade more
    slowly. Only the fitted (pre-saturation) points are drawn as fit lines, so
    the saturated tail cannot make a flat slope look like robustness.
    """
    models = _model_order(summary)
    style = _style(models)
    fits = {r['model']: r for r in slopes if r['subset'] == subset}

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    plotted = 0
    for model in models:
        rows = sorted((r for r in summary if r['model'] == model
                       and r['subset'] == subset), key=lambda r: r['sigma'])
        pts = [(r['sigma'], r['delta_mse_mean']) for r in rows
               if r['delta_mse_mean'] is not None and r['delta_mse_mean'] > 0]
        if not pts:
            continue
        color, marker = style[model]
        fit = fits.get(model)
        # The slope goes in the legend label, not in an annotation at the end
        # of each fit segment: every model saturates at roughly the same
        # sigma, so those segments all end at the same place and the labels
        # collide into an unreadable stack.
        label = model
        if fit and fit.get('slope') is not None:
            label = f"{model}  (slope {fit['slope']:.2f})"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                marker=marker, markersize=5, linewidth=2, zorder=3,
                markeredgecolor=_SURFACE, markeredgewidth=0.8,
                label=label)
        if fit and fit.get('slope') is not None:
            lo, hi = fit['fit_sigma_lo'], fit['fit_sigma_hi']
            ys = [10 ** (fit['intercept'] + fit['slope'] * math.log10(x))
                  for x in (lo, hi)]
            ax.plot([lo, hi], ys, color=color, linestyle='--', linewidth=1.3,
                    alpha=0.85, zorder=2)
        # Direct label at the data's right end — clear of the fit segments,
        # and the required relief for the low-contrast palette slot.
        ax.annotate(model, xy=pts[-1], xytext=(6, 0),
                    textcoords='offset points', color=color, fontsize=8,
                    va='center', ha='left', fontweight='bold')
        plotted += 1

    _dress(ax, r'relative perturbation $\sigma$',
           r'$\Delta$MSE vs. that model' + "'s own baseline",
           f'Degradation slope — {subset} perturbed')
    if plotted >= 2:
        ax.legend(frameon=False, fontsize=8, loc='upper left',
                  labelcolor=_INK_SECONDARY)
    ax.margins(x=0.22)
    fig.text(0.01, 0.005,
             'dashed: least-squares fit of log10(dMSE) on log10(sigma) over '
             'the pre-saturation region',
             color=_MUTED, fontsize=7)
    return _save(fig, os.path.join(output_dir, DELTA_PNG))


def build_all(summary, slopes, baselines, output_dir):
    """-> list of written figure paths. Subsets absent from the run are
    skipped rather than drawn empty."""
    written = []
    have_joint = any(r['subset'] == JOINT_SUBSET for r in summary)
    if have_joint:
        written.append(plot_curves(summary, baselines, output_dir))
        written.append(plot_delta_with_fits(summary, slopes, output_dir))
    elif summary:
        # A run restricted to single matrices still deserves a headline curve;
        # use whichever subset it does have.
        subset = summary[0]['subset']
        written.append(plot_curves(summary, baselines, output_dir, subset))
        written.append(plot_delta_with_fits(summary, slopes, output_dir,
                                            subset))
    by_matrix = plot_by_matrix(summary, baselines, output_dir)
    if by_matrix:
        written.append(by_matrix)
    return [p for p in written if p]
