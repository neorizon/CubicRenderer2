//! Small CPU-side mesh helpers shared by both scenes: the padded bounding
//! quad a cubic's coverage is rasterized into, plus circle/line meshes for
//! drawing control points and the control polygon.

use crate::cubic::{ImplicitCubic, Pt};
use crate::gl::CubicVertex;

/// The curve always lies within the convex hull of its 4 control points (the
/// Bezier convex-hull property), so a mesh over just those points -- padded
/// outward a few pixels for the ~1px AA falloff -- is guaranteed to contain
/// the whole curve. K, L, M are affine everywhere, so evaluating them at
/// these corners and interpolating is exact: no tessellation of the curve
/// itself is needed.
///
/// This deliberately hugs the points' own hull rather than their axis-aligned
/// bounding box: k^3 - l*m is a full algebraic cubic curve, and a stray
/// branch of it (away from the actual Bezier arc) can pass through a bbox
/// corner that the tighter hull excludes.
pub fn coverage_quad(points: &[Pt; 4], implicit: &ImplicitCubic, pad: f32) -> [CubicVertex; 4] {
    let hull = angular_hull_order(points);
    let cx = hull.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = hull.iter().map(|p| p[1]).sum::<f32>() / 4.0;

    hull.map(|p| {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        let len = (dx * dx + dy * dy).sqrt().max(1e-3);
        let pos = [p[0] + dx / len * pad, p[1] + dy / len * pad];
        CubicVertex { pos, klm: implicit.eval_klm(pos) }
    })
}

/// Orders 4 points by angle around their centroid, giving a simple
/// (non-self-intersecting) polygon -- the convex hull when the points are in
/// convex position, or a slightly concave quad otherwise. Either way it's a
/// tighter cover than the bounding box.
fn angular_hull_order(points: &[Pt; 4]) -> [Pt; 4] {
    let cx = points.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = points.iter().map(|p| p[1]).sum::<f32>() / 4.0;
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|&a, &b| {
        let angle_of = |i: usize| (points[i][1] - cy).atan2(points[i][0] - cx);
        angle_of(a).total_cmp(&angle_of(b))
    });
    idx.map(|i| points[i])
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
