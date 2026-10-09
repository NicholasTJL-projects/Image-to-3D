"""Judge whether a posed capture is good enough for Gaussian Splatting, and say what to add.

Called after Structure-from-Motion so the user learns early whether the result will hold up
from new viewpoints, and what to capture next if it won't.
"""

from __future__ import annotations

import numpy as np

from .scene import Scene


def assess_coverage(scene: Scene, photos_submitted: int | None = None) -> dict:
    """Return coverage statistics, a verdict (``good`` / ``fair`` / ``poor``) and advice."""
    n = len(scene)
    submitted = photos_submitted or n
    pts = scene.points_xyz
    out: dict = {"cameras_posed": n, "photos_submitted": submitted, "points": int(len(pts))}
    if n < 2:
        out.update(verdict="poor", azimuth_coverage_deg=0.0, elevation_range_deg=0.0,
                   advice=["Fewer than two photos could be posed. Shoot an orbit video or 30+ overlapping photos."])
        return out

    centers = np.stack([c.center for c in scene.cameras])
    target = np.median(pts, axis=0) if len(pts) >= 10 else centers.mean(axis=0)
    up = -np.mean([c.R.T @ np.array([0.0, 1.0, 0.0]) for c in scene.cameras], axis=0)
    up = up / (np.linalg.norm(up) + 1e-9)

    # camera directions relative to the subject, split into azimuth (around `up`) and elevation
    d = centers - target
    dist = np.linalg.norm(d, axis=1) + 1e-9
    elev = np.degrees(np.arcsin(np.clip(d @ up / dist, -1, 1)))
    a = d - np.outer(d @ up, up)
    ref = a[0] / (np.linalg.norm(a[0]) + 1e-9)
    side = np.cross(up, ref)
    az = np.degrees(np.arctan2(a @ side, a @ ref))  # -180..180

    # azimuth coverage = total angle covered when each camera claims +/- 20 degrees around it
    az_sorted = np.sort(np.mod(az, 360.0))
    gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360.0]]))
    uncovered = float(np.clip(gaps - 40.0, 0, None).sum())
    az_cov = float(max(0.0, 360.0 - uncovered))
    elev_range = float(elev.max() - elev.min())
    out.update(azimuth_coverage_deg=round(az_cov, 1), elevation_range_deg=round(elev_range, 1),
               largest_gap_deg=round(float(gaps.max()), 1))

    advice: list[str] = []
    dropped = submitted - n
    if dropped > 0:
        advice.append(f"{dropped} of {submitted} photos could not be matched to the others, usually because they "
                      "share too little with their neighbours. Move the camera in smaller steps.")
    if az_cov < 150:
        advice.append("The cameras cover less than half the circle around the object; the far side will be "
                      "missing. Continue the loop all the way round.")
    elif az_cov < 300:
        advice.append("Part of the circle around the object is not covered; views from that side will break up. "
                      "Add photos from the uncovered side.")
    if elev_range < 15:
        advice.append("All photos are from the same height. Add a second loop from higher or lower to recover the "
                      "top and the sides.")
    if len(pts) < 1500:
        advice.append("Few matched points were found. Put the object on a textured surface (newspaper, a patterned "
                      "cloth) and avoid plain backgrounds.")
    if n < 20:
        advice.append(f"Only {n} usable views. Aim for 40 to 80, or shoot a 20-second orbit video.")

    score = 0
    score += 2 if az_cov >= 300 else 1 if az_cov >= 150 else 0
    score += 1 if elev_range >= 15 else 0
    score += 1 if len(pts) >= 1500 else 0
    score += 1 if n >= 20 else 0
    out["verdict"] = "good" if score >= 4 and not advice else "fair" if score >= 2 else "poor"
    if az_cov < 150:  # less than half the circle: the far side does not exist, whatever else is right
        out["verdict"] = "poor"
    if not advice:
        advice.append("Good coverage. Longer training and a GPU rasteriser will sharpen the result further.")
    out["advice"] = advice
    return out
