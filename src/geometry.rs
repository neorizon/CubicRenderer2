//! Small CPU-side mesh helpers shared by both scenes: the padded bounding
//! quad a cubic's coverage is rasterized into, plus circle/line meshes for
//! drawing control points and the control polygon.

use crate::cubic::{self, ImplicitCubic, Pt};
use crate::gl::CubicVertex;

/// How many points the curve is sampled at to build `coverage_mesh`'s hull.
/// 16 already captures nearly all the benefit over the raw 4-control-point
/// hull (tested against a deliberately pathological near-cusp Loop curve:
/// going from 16 to 64 samples changed the result by under 1%), so there's
/// no real quality reason to go higher; it's cheap regardless at this scale.
const COVERAGE_SAMPLES: usize = 16;

/// The curve always lies within the convex hull of its 4 control points (the
/// Bezier convex-hull property), so a mesh over the hull -- padded outward a
/// few pixels for the ~1px AA falloff -- is guaranteed to contain the whole
/// curve, with zero tessellation error. K, L, M are affine everywhere, so
/// evaluating them at the hull's corners and interpolating is exact.
///
/// A hull of just those 4 points is often needlessly loose, though: K^3 - L*M
/// is a full algebraic cubic curve, and a stray branch of it (away from the
/// actual Bezier arc -- e.g. the same t-parametrized cubic evaluated outside
/// [0,1], which is exactly what happens on either side of a
/// `cubic::chop_at_loop_intersection` chop point) can pass through a hull
/// built from just the 4 control points, since those are frequently far from
/// the curve itself. So this hulls the control points *and* points sampled
/// along the curve together: wherever the samples are tighter than the
/// control-point hull, they dominate and exclude more of any stray branch;
/// the control points are still there as a backstop, so the hull can never
/// be tighter than the curve itself actually allows. Earlier versions of
/// this hulled only the samples, which is tighter still but reintroduces
/// tessellation error -- confirmed: it let the true curve poke outside the
/// padded hull (and get clipped, a visible gap in the stroke) for a
/// high-curvature piece where 16 samples wasn't dense enough. Keeping the
/// control points in the hull set costs nothing (the algorithm just drops
/// them if a sample already dominates) and restores the zero-tessellation-
/// error guarantee unconditionally.
///
/// Even so, `pad` should stay close to the ~1px AA falloff it exists for: a
/// stray branch passing close enough to the true curve gets readmitted once
/// padded outward, no matter how tight the unpadded hull was. That's most
/// visible right at a Loop chop node (unavoidable -- the branches touch
/// there by definition) and, for a Loop curve whose self-intersection angle
/// is shallow enough, can extend over a long stretch of the visible arc.
/// That's an inherent conflict between AA margin and branch exclusion at
/// sub-pixel distances, not something this function can geometry its way
/// out of.
pub fn coverage_mesh(points: &[Pt; 4], implicit: &ImplicitCubic, pad: f32) -> Vec<CubicVertex> {
    let mut hull_points: Vec<Pt> =
        (0..=COVERAGE_SAMPLES).map(|i| cubic::eval(points, i as f32 / COVERAGE_SAMPLES as f32)).collect();
    hull_points.extend_from_slice(points);
    let hull = convex_hull(hull_points);
    let n = hull.len().max(1) as f32;
    let cx = hull.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy = hull.iter().map(|p| p[1]).sum::<f32>() / n;

    hull.into_iter()
        .map(|p| {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let len = (dx * dx + dy * dy).sqrt().max(1e-3);
            let pos = [p[0] + dx / len * pad, p[1] + dy / len * pad];
            CubicVertex { pos, klm: implicit.eval_klm(pos) }
        })
        .collect()
}

/// Andrew's monotone chain: a standard O(n log n) convex hull, returned
/// counter-clockwise. Fine for the small point counts here (no need for a
/// crate dependency).
fn convex_hull(mut pts: Vec<Pt>) -> Vec<Pt> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: Pt, a: Pt, b: Pt| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

    let mut lower: Vec<Pt> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Pt> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// A closed triangle-fan disc: `[center, rim..., rim[0]]`, ready for
/// `gl::TRIANGLE_FAN`.
pub fn circle_fan(center: Pt, radius: f32, segments: usize) -> Vec<Pt> {
    let mut verts = Vec::with_capacity(segments + 2);
    verts.push(center);
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        verts.push([center[0] + radius * a.cos(), center[1] + radius * a.sin()]);
    }
    verts
}

/// Scales and centers a set of control points (e.g. one of `cubic::preset_curves()`)
/// so it fills `fill_fraction` of a `size`-sized region placed at `origin`,
/// preserving aspect ratio. Used to place presets into the interactive
/// viewport or into a gallery grid cell.
pub fn fit_points(preset: &[Pt; 4], origin: [f32; 2], size: [f32; 2], fill_fraction: f32) -> [Pt; 4] {
    let min_x = preset.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let max_x = preset.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
    let min_y = preset.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_y = preset.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let (pw, ph) = ((max_x - min_x).max(1e-3), (max_y - min_y).max(1e-3));

    let scale = (size[0] * fill_fraction / pw).min(size[1] * fill_fraction / ph);
    let center = [origin[0] + size[0] * 0.5, origin[1] + size[1] * 0.5];
    let preset_center = [min_x + pw * 0.5, min_y + ph * 0.5];

    preset.map(|p| {
        [center[0] + (p[0] - preset_center[0]) * scale, center[1] + (p[1] - preset_center[1]) * scale]
    })
}

/// A thin quad along segment a-b, ready for `gl::TRIANGLE_FAN`.
pub fn thick_line(a: Pt, b: Pt, half_width: f32) -> [Pt; 4] {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let nx = -dy / len * half_width;
    let ny = dx / len * half_width;
    [[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_in_convex_polygon(pt: Pt, poly: &[Pt]) -> bool {
        let n = poly.len();
        let mut sign = 0i32;
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            let cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]);
            // A little slack: the true curve sample can legitimately land
            // right on a hull edge (e.g. at a sample point itself).
            let s = if cross > 1e-3 {
                1
            } else if cross < -1e-3 {
                -1
            } else {
                0
            };
            if s != 0 {
                if sign == 0 {
                    sign = s;
                } else if s != sign {
                    return false;
                }
            }
        }
        true
    }

    /// The whole point of hulling the control points alongside the curve
    /// samples: no matter how few samples there are, or how sharply the
    /// curve bends between them, the true curve can never poke outside the
    /// padded mesh. Regression test for the bug where a samples-only hull
    /// let a high-curvature piece's stroke render with a visible gap.
    #[test]
    fn coverage_mesh_always_contains_the_true_curve() {
        const PAD: f32 = 1.5;
        for (name, _, p) in cubic::preset_curves() {
            for piece in cubic::chop_at_loop_intersection(&p) {
                if !piece.implicit.is_renderable() {
                    continue;
                }
                let mesh = coverage_mesh(&piece.points, &piece.implicit, PAD);
                let poly: Vec<Pt> = mesh.iter().map(|v| v.pos).collect();
                if poly.len() < 3 {
                    continue;
                }
                for i in 0..=400 {
                    let t = i as f32 / 400.0;
                    let pt = cubic::eval(&piece.points, t);
                    assert!(
                        point_in_convex_polygon(pt, &poly),
                        "{name}: curve point at t={t} ({pt:?}) escaped its own coverage mesh"
                    );
                }
            }
        }
    }
}
