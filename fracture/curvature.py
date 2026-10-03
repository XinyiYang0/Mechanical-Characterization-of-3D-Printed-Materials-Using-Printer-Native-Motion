"""Legacy image-to-curvature analysis from the supplied break.ipynb.

This polynomial/spline method is retained for traceability. It is distinct from
the physics-informed reconstruction described in the final manuscript.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import splprep, splev


def fit_centerline(y, x, clamp_slope=False):
    """Sixth-degree least squares on a scaled coordinate to avoid ill-conditioning.

    Removing the linear term in u=(y-y_min)/(y_max-y_min) imposes dx/dy=0
    at y_min. This is the same polynomial family as the source constraint.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    span = np.ptp(y)
    if len(np.unique(y)) < 7 or span <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("At least seven finite centerline rows with positive span are required")
    u = (y - y.min()) / span
    powers = [0, 2, 3, 4, 5, 6] if clamp_slope else list(range(7))
    coefficients, _, _, _ = np.linalg.lstsq(np.column_stack([u**p for p in powers]), x, rcond=None)
    full = np.zeros(7)
    full[powers] = coefficients
    return Polynomial(full)(Polynomial([-y.min()/span, 1/span]))


def legacy_curvature_adjustment(curvature):
    """Source notebook's heuristic; it is not an equilibrium constraint.

    Starting at sample 150, replace a rise exceeding 1% by a 1% decrease.
    Smaller increases are allowed, so this does not guarantee monotonicity.
    """
    adjusted = np.array(curvature, copy=True)
    for i in range(150, len(adjusted)):
        if adjusted[i] > adjusted[i-1] * 1.01:
            adjusted[i] = adjusted[i-1] * 0.99
    return adjusted


def analyze_image(image_path, output_dir, *, beam_length_m, flip_vertical=True,
                  hsv_lower=(10, 60, 30), hsv_upper=(35, 255, 255),
                  spline_smoothing=1e-4, include_legacy_adjustment=False):
    """Export centerline, curvature, parameters, and overlay from one intact frame.

    beam_length_m must describe the complete tracked segment. The source pipeline
    uses arc length for scale, a constrained pixel-space fit for calibration, an
    unconstrained metric fit, then a quintic parametric spline. It assumes x(y)
    is single-valued and that the first rows after flipping correspond to root.
    """
    import matplotlib.pyplot as plt

    if not np.isfinite(beam_length_m) or beam_length_m <= 0:
        raise ValueError("beam_length_m must be positive")
    if not np.isfinite(spline_smoothing) or spline_smoothing < 0:
        raise ValueError("spline_smoothing must be nonnegative")
    bounds = np.asarray([hsv_lower, hsv_upper])
    if bounds.shape != (2, 3) or np.any(bounds < 0) or np.any(bounds > [179, 255, 255]) or np.any(bounds[0] > bounds[1]):
        raise ValueError("Invalid OpenCV HSV bounds")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if flip_vertical:
        image = cv2.flip(image, 0)
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), np.array(hsv_lower, np.uint8), np.array(hsv_upper, np.uint8))
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 7:
        raise ValueError("Too few beam rows detected; inspect orientation and HSV thresholds")
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    center_x = []
    for y in rows:
        columns = np.flatnonzero(mask[y])
        # Preserve the source's special treatment of the first 11 rows.
        x = np.argmax(dist[y]) if y > rows.min() + 10 else (columns.min() + columns.max()) // 2
        center_x.append(x)
    center_x = np.asarray(center_x, dtype=float)
    pixel_polynomial = fit_centerline(rows, center_x, clamp_slope=True)
    derivative = pixel_polynomial.deriv()
    arc_px, _ = quad(lambda y: np.sqrt(1 + derivative(y)**2), float(rows.min()), float(rows.max()))
    scale = beam_length_m / arc_px
    x_m, y_m = (center_x - center_x.min()) * scale, (rows - rows.min()) * scale
    metric_polynomial = fit_centerline(y_m, x_m)
    y_uniform = np.linspace(y_m.min(), y_m.max(), 1000)
    tck, _ = splprep([metric_polynomial(y_uniform), y_uniform], s=spline_smoothing, k=5)
    u = np.linspace(0, 1, 3000)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    speed = np.hypot(dx, dy)
    if np.any(speed <= 0):
        raise ValueError("Degenerate spline derivative")
    curvature = np.abs(dx * ddy - dy * ddx) / speed**3
    arc_m = cumulative_trapezoid(speed, u, initial=0)
    profile = pd.DataFrame(dict(arc_length_m=arc_m, curvature_per_m=curvature))
    if include_legacy_adjustment:
        profile['legacy_adjusted_curvature_per_m'] = legacy_curvature_adjustment(curvature)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_dir / 'curvature.csv', index=False)
    pd.DataFrame(dict(x_m=x_m, y_m=y_m)).to_csv(output_dir / 'centerline.csv', index=False)
    metadata = dict(input_image=Path(image_path).name, method='legacy_polynomial_spline', beam_length_m=beam_length_m,
        metres_per_pixel=scale, flip_vertical=flip_vertical, hsv_lower=list(hsv_lower), hsv_upper=list(hsv_upper),
        spline_smoothing=spline_smoothing, include_legacy_adjustment=include_legacy_adjustment,
        max_raw_curvature_per_m=float(curvature.max()))
    (output_dir / 'settings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].plot(center_x, rows, '.', ms=1, label='Detected centerline')
    axes[0].plot(x / scale + center_x.min(), y / scale + rows.min(), label='Fitted spline')
    axes[0].legend(); axes[0].set_title('Inspect segmentation and root orientation')
    axes[1].plot(arc_m, curvature, label='Raw spline curvature')
    if include_legacy_adjustment:
        axes[1].plot(arc_m, profile['legacy_adjusted_curvature_per_m'], label='Legacy heuristic')
    axes[1].set(xlabel='Arc length (m)', ylabel='Curvature (1/m)')
    axes[1].legend(); fig.tight_layout()
    fig.savefig(output_dir / 'diagnostics.png', dpi=160)
    plt.close(fig)
    return profile
