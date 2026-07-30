//! Loop-Blinn implicitization of a cubic Bezier: classify the curve from its
//! control points, then derive three linear functionals K, L, M (each affine
//! in screen-space x,y) such that the curve satisfies K^3 - L*M = 0.
//!
//! This is a port of the classification/derivation math in Skia's
//! `src/gpu/GrPathUtils.cpp` (`classify_cubic`, `calc_cubic_inflection_func`,
//! `set_serp_klm`, `set_loop_klm`, `set_cusp_klm`, `set_quadratic_klm`,
//! `calc_cubic_klm`), following Loop & Blinn, "Resolution Independent Curve
//! Rendering using Programmable Graphics Hardware" (2005).

pub type Pt = [f32; 2];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubicKind {
    Serpentine,
    Loop,
    Cusp,
    Quadratic,
    Line,
    Point,
}

impl CubicKind {
    pub fn name(self) -> &'static str {
        match self {
            CubicKind::Serpentine => "Serpentine",
            CubicKind::Loop => "Loop",
            CubicKind::Cusp => "Cusp",
            CubicKind::Quadratic => "Quadratic",
            CubicKind::Line => "Line",
            CubicKind::Point => "Point",
        }
    }

    /// A stable accent color per classification, shared by the interactive
    /// HUD and the gallery legend.
    pub fn accent_color(self) -> [f32; 3] {
        match self {
            CubicKind::Serpentine => [0.30, 0.70, 1.00],
            CubicKind::Loop => [1.00, 0.45, 0.45],
            CubicKind::Cusp => [1.00, 0.80, 0.20],
            CubicKind::Quadratic => [0.55, 1.00, 0.55],
            CubicKind::Line => [0.75, 0.55, 1.00],
            CubicKind::Point => [0.80, 0.80, 0.80],
        }
    }
}

/// The implicit form of a cubic: three affine functionals K, L, M (each
/// stored as [a, b, c] meaning f(x,y) = a*x + b*y + c) such that a point on
/// the curve satisfies K(x,y)^3 - L(x,y)*M(x,y) == 0.
#[derive(Clone, Copy, Debug)]
pub struct ImplicitCubic {
    pub kind: CubicKind,
    pub k: [f32; 3],
    pub l: [f32; 3],
    pub m: [f32; 3],
    /// Inflection-polynomial coefficients d0,d1,d2 (post scale-normalized),
    /// exposed for the HUD.
    pub d: [f32; 3],
    pub discriminant: f32,
}

impl ImplicitCubic {
    pub fn is_renderable(&self) -> bool {
        !matches!(self.kind, CubicKind::Line | CubicKind::Point)
    }

    fn eval_functional(f: [f32; 3], p: Pt) -> f32 {
        f[0] * p[0] + f[1] * p[1] + f[2]
    }

    pub fn eval_klm(&self, p: Pt) -> [f32; 3] {
        [
            Self::eval_functional(self.k, p),
            Self::eval_functional(self.l, p),
            Self::eval_functional(self.m, p),
        ]
    }
}

const NEARLY_ZERO: f32 = 1.0 / 4096.0; // matches SK_ScalarNearlyZero

fn dot_cross(p0: Pt, p1: Pt, p2: Pt) -> f32 {
    // p0 . (p1 x p2), treating points as homogeneous (x, y, 1).
    let x_comp = p0[0] * (p1[1] - p2[1]);
    let y_comp = p0[1] * (p2[0] - p1[0]);
    let w_comp = p1[0] * p2[1] - p1[1] * p2[0];
    x_comp + y_comp + w_comp
}

/// Coefficients d0,d1,d2 of the inflection polynomial I(s,t) whose roots are
/// the curve's inflection points, normalized so max(|a1|,|a2|,|a3|) == 1.
fn inflection_coeffs(p: &[Pt; 4]) -> [f32; 3] {
    let a1 = dot_cross(p[0], p[3], p[2]);
    let a2 = dot_cross(p[1], p[0], p[3]);
    let a3 = dot_cross(p[2], p[1], p[0]);

    let max_abs = a1.abs().max(a2.abs()).max(a3.abs());
    let scale = if max_abs > 0.0 { 1.0 / max_abs } else { 1.0 };
    let (a1, a2, a3) = (a1 * scale, a2 * scale, a3 * scale);

    let d2 = 3.0 * a3;
    let d1 = d2 - a2;
    let d0 = d1 - a2 + a1;
    [d0, d1, d2]
}

fn classify(p: &[Pt; 4], d: [f32; 3]) -> CubicKind {
    if p[0] == p[1] && p[0] == p[2] && p[0] == p[3] {
        return CubicKind::Point;
    }
    let discr = d[0] * d[0] * (3.0 * d[1] * d[1] - 4.0 * d[0] * d[2]);
    if discr > NEARLY_ZERO {
        CubicKind::Serpentine
    } else if discr < -NEARLY_ZERO {
        CubicKind::Loop
    } else if d[0].abs() < NEARLY_ZERO && d[1].abs() < NEARLY_ZERO {
        // Skia's literal source compares against exact 0.0 here, which is
        // fine there since it classifies points straight from a fixed path;
        // we re-scale/translate presets per grid cell, which reintroduces
        // float noise into an otherwise-exact cancellation, so this needs
        // an epsilon instead.
        if d[2].abs() < NEARLY_ZERO { CubicKind::Line } else { CubicKind::Quadratic }
    } else {
        CubicKind::Cusp
    }
}

/// Per-classification control values of K, L, M at parameter t = 0, 1/3,
/// 2/3, 1 (i.e. the canonical Bezier-basis weights that make K(t)^3 -
/// L(t)*M(t) vanish identically in t).
type ControlKlm = ([f32; 4], [f32; 4], [f32; 4]);

fn set_serpentine_klm(d: [f32; 3]) -> ControlKlm {
    let root = (9.0 * d[1] * d[1] - 12.0 * d[0] * d[2]).max(0.0).sqrt();
    let ls = 3.0 * d[1] - root;
    let lt = 6.0 * d[0];
    let ms = 3.0 * d[1] + root;
    let mt = 6.0 * d[0];

    let mut k = [
        ls * ms,
        (3.0 * ls * ms - ls * mt - lt * ms) / 3.0,
        (lt * (mt - 2.0 * ms) + ls * (3.0 * ms - 2.0 * mt)) / 3.0,
        (lt - ls) * (mt - ms),
    ];
    let lt_ls = lt - ls;
    let mut l = [ls * ls * ls, -ls * ls * lt_ls, lt_ls * lt_ls * ls, -lt_ls * lt_ls * lt_ls];
    let mt_ms = mt - ms;
    let m = [ms * ms * ms, -ms * ms * mt_ms, mt_ms * mt_ms * ms, -mt_ms * mt_ms * mt_ms];

    // Negative distances must land inside the curve; flip orientation if not.
    if d[0] > 0.0 {
        for v in k.iter_mut() {
            *v = -*v;
        }
        for v in l.iter_mut() {
            *v = -*v;
        }
    }
    (k, l, m)
}

fn set_loop_klm(d: [f32; 3]) -> ControlKlm {
    let root = (4.0 * d[0] * d[2] - 3.0 * d[1] * d[1]).max(0.0).sqrt();
    let ls = d[1] - root;
    let lt = 2.0 * d[0];
    let ms = d[1] + root;
    let mt = 2.0 * d[0];

    let mut k = [
        ls * ms,
        (3.0 * ls * ms - ls * mt - lt * ms) / 3.0,
        (lt * (mt - 2.0 * ms) + ls * (3.0 * ms - 2.0 * mt)) / 3.0,
        (lt - ls) * (mt - ms),
    ];
    let mut l = [
        ls * ls * ms,
        (ls * (ls * (mt - 3.0 * ms) + 2.0 * lt * ms)) / -3.0,
        ((lt - ls) * (ls * (2.0 * mt - 3.0 * ms) + lt * ms)) / 3.0,
        -(lt - ls) * (lt - ls) * (mt - ms),
    ];
    let m = [
        ls * ms * ms,
        (ms * (ls * (2.0 * mt - 3.0 * ms) + lt * ms)) / -3.0,
        ((mt - ms) * (ls * (mt - 3.0 * ms) + 2.0 * lt * ms)) / 3.0,
        -(lt - ls) * (mt - ms) * (mt - ms),
    ];

    if (d[0] < 0.0 && k[1] > 0.0) || (d[0] > 0.0 && k[1] < 0.0) {
        for v in k.iter_mut() {
            *v = -*v;
        }
        for v in l.iter_mut() {
            *v = -*v;
        }
    }
    (k, l, m)
}

fn set_cusp_klm(d: [f32; 3]) -> ControlKlm {
    let ls = d[2];
    let lt = 3.0 * d[1];

    let k = [ls, ls - lt / 3.0, ls - 2.0 * lt / 3.0, ls - lt];
    let ls_lt = ls - lt;
    let l = [ls * ls * ls, ls * ls * ls_lt, ls_lt * ls_lt * ls, ls_lt * ls_lt * ls_lt];
    let m = [1.0, 1.0, 1.0, 1.0];
    (k, l, m)
}

fn set_quadratic_klm(d: [f32; 3]) -> ControlKlm {
    let mut k = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    let l = [0.0, 0.0, 1.0 / 3.0, 1.0];
    let m = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    let mut l = l;
    if d[2] > 0.0 {
        for v in k.iter_mut() {
            *v = -*v;
        }
        for v in l.iter_mut() {
            *v = -*v;
        }
    }
    (k, l, m)
}

fn invert3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = if det.abs() > 1e-12 { 1.0 / det } else { 0.0 };
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

fn matvec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Fits an affine functional F(x,y) = a*x + b*y + c through its values at
/// p0, p1, p2 (control_values[0..3]); the 4th control value at p3 is implied
/// by construction and not used as a fitting constraint.
fn fit_affine_functionals(
    p: &[Pt; 4],
    control_k: [f32; 4],
    control_l: [f32; 4],
    control_m: [f32; 4],
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let basis = [[p[0][0], p[0][1], 1.0], [p[1][0], p[1][1], 1.0], [p[2][0], p[2][1], 1.0]];
    let inv = invert3(basis);
    let k = matvec3(inv, [control_k[0], control_k[1], control_k[2]]);
    let l = matvec3(inv, [control_l[0], control_l[1], control_l[2]]);
    let m = matvec3(inv, [control_m[0], control_m[1], control_m[2]]);
    (k, l, m)
}

/// Classify a cubic Bezier and derive its Loop-Blinn implicit form.
pub fn compute_implicit(p: &[Pt; 4]) -> ImplicitCubic {
    let d = inflection_coeffs(p);
    let kind = classify(p, d);
    let discriminant = d[0] * d[0] * (3.0 * d[1] * d[1] - 4.0 * d[0] * d[2]);

    let (k, l, m) = match kind {
        CubicKind::Serpentine => {
            let (ck, cl, cm) = set_serpentine_klm(d);
            fit_affine_functionals(p, ck, cl, cm)
        }
        CubicKind::Cusp if d[0] != 0.0 => {
            // Numerically a cusp but not the exact d0==0 case that
            // set_cusp_klm assumes; the serpentine formula still applies.
            let (ck, cl, cm) = set_serpentine_klm(d);
            fit_affine_functionals(p, ck, cl, cm)
        }
        CubicKind::Cusp => {
            let (ck, cl, cm) = set_cusp_klm(d);
            fit_affine_functionals(p, ck, cl, cm)
        }
        CubicKind::Loop => {
            let (ck, cl, cm) = set_loop_klm(d);
            fit_affine_functionals(p, ck, cl, cm)
        }
        CubicKind::Quadratic => {
            let (ck, cl, cm) = set_quadratic_klm(d);
            fit_affine_functionals(p, ck, cl, cm)
        }
        // Degenerate: no meaningful implicit curve. Callers must check
        // `is_renderable()` before drawing the fill.
        CubicKind::Line | CubicKind::Point => ([0.0; 3], [0.0; 3], [0.0; 3]),
    };

    ImplicitCubic { kind, k, l, m, d, discriminant }
}

/// Reference control points hitting each of the six classifications, in a
/// local ~600x300 pixel space. Shared by the gallery scene and the tests
/// below (numerically verified, not hand-guessed: see the derivation notes
/// in the project history for how these were found).
pub fn preset_curves() -> [(CubicKind, [Pt; 4]); 6] {
    [
        (CubicKind::Serpentine, [[40.0, 260.0], [460.0, 40.0], [160.0, 40.0], [460.0, 260.0]]),
        (CubicKind::Loop, [[40.0, 260.0], [460.0, 40.0], [-80.0, 40.0], [460.0, 260.0]]),
        (CubicKind::Cusp, [[40.0, 260.0], [460.0, 40.0], [40.0, 40.0], [460.0, 260.0]]),
        (CubicKind::Quadratic, [[0.0, 300.0], [200.0, 100.0], [400.0, 100.0], [600.0, 300.0]]),
        (CubicKind::Line, [[40.0, 260.0], [180.0, 200.0], [320.0, 140.0], [460.0, 80.0]]),
        (CubicKind::Point, [[200.0, 200.0]; 4]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_classify_as_labeled() {
        for (expected, p) in preset_curves() {
            let c = compute_implicit(&p);
            assert_eq!(c.kind, expected, "discriminant = {}, d = {:?}", c.discriminant, c.d);
        }
    }

    #[test]
    fn klm_vanishes_along_the_curve() {
        // For every renderable classification, sample the actual parametric
        // curve and check the implicit form is ~0 there (correctness of the
        // fitted functionals, not just the classification).
        for (kind, p) in preset_curves() {
            let c = compute_implicit(&p);
            if !c.is_renderable() {
                continue;
            }
            // Absolute, not relative: k, l, m reach the thousands for these
            // ~600px presets, so k^3 and l*m are computed as a difference of
            // two ~1e6 f32 terms near t=1. A few units of cancellation noise
            // there is expected f32 precision, not a derivation bug -- it's
            // many orders of magnitude below the gradient scale (~1e4) that
            // the AA shader actually divides by.
            let mut max_abs_err: f32 = 0.0;
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let pt = cubic_point(&p, t);
                let klm = c.eval_klm(pt);
                let f = klm[0].powi(3) - klm[1] * klm[2];
                max_abs_err = max_abs_err.max(f.abs());
            }
            assert!(max_abs_err < 10.0, "{kind:?} max abs error {max_abs_err}");
        }
    }

    fn cubic_point(p: &[Pt; 4], t: f32) -> Pt {
        let mt = 1.0 - t;
        let a = mt * mt * mt;
        let b = 3.0 * mt * mt * t;
        let c = 3.0 * mt * t * t;
        let d = t * t * t;
        [
            a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0],
            a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1],
        ]
    }
}
