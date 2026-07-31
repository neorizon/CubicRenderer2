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

/// How close `d[0]` must be to zero for `compute_implicit` to trust
/// `set_cusp_klm`'s exact-cusp derivation instead of falling back to the
/// (also-valid-at-d0=0, but always-correct) serpentine formula. Deliberately
/// much tighter than `NEARLY_ZERO`: see the guard in `compute_implicit`.
const CUSP_KLM_EPSILON: f32 = 1e-5;

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

/// Signed area x2 of the (x,y,1) basis formed by 3 of the 4 control points,
/// i.e. the determinant `invert3` would compute for that basis -- used to
/// pick the best-conditioned triple without actually inverting each one.
fn basis_det(p: &[Pt; 4], idx: [usize; 3]) -> f32 {
    let [i, j, k] = idx;
    let (a, b, c) = (p[i], p[j], p[k]);
    a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
}

/// Fits an affine functional F(x,y) = a*x + b*y + c through its values at 3
/// of the 4 control points (the 4th is implied by construction and not
/// needed as a fitting constraint: K(P_i) = control_value[i] holds exactly
/// for *all four* control points, since a Bezier curve composed with an
/// affine map is itself the Bezier curve of the mapped control points -- so
/// any non-degenerate 3-point subset gives the same, exact answer).
///
/// Which 3 is chosen dynamically: hardcoding p0,p1,p2 makes this singular
/// whenever those three happen to be collinear, which includes the
/// completely ordinary case of an exact cusp constructed the standard way
/// (P0 == P1, so the curve leaves its start point with zero tangent -- see
/// the "Cusp (at start)"/"(arch)"/"(tent)" presets). A singular basis makes
/// `invert3` fall back to an all-zero inverse, which makes K, L, M all
/// identically zero -- not just near the curve, everywhere -- so
/// `k^3 - l*m` is 0 at every pixel and the whole coverage mesh fills in
/// solid instead of tracing a curve. Picking whichever triple has the
/// largest |determinant| sidesteps this with no accuracy cost.
fn fit_affine_functionals(
    p: &[Pt; 4],
    control_k: [f32; 4],
    control_l: [f32; 4],
    control_m: [f32; 4],
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    const TRIPLES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let [i, j, k] = TRIPLES
        .into_iter()
        .max_by(|&a, &b| basis_det(p, a).abs().total_cmp(&basis_det(p, b).abs()))
        .unwrap();

    let basis = [[p[i][0], p[i][1], 1.0], [p[j][0], p[j][1], 1.0], [p[k][0], p[k][1], 1.0]];
    let inv = invert3(basis);
    let kf = matvec3(inv, [control_k[i], control_k[j], control_k[k]]);
    let lf = matvec3(inv, [control_l[i], control_l[j], control_l[k]]);
    let mf = matvec3(inv, [control_m[i], control_m[j], control_m[k]]);
    (kf, lf, mf)
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
        CubicKind::Cusp if d[0].abs() > CUSP_KLM_EPSILON => {
            // Numerically a cusp but not (close enough to) the d0==0 case
            // that set_cusp_klm assumes; the serpentine formula still
            // applies. Epsilon rather than exact `!= 0.0`, because callers
            // rescale/retranslate presets (fit_points for the gallery/
            // interactive viewport), and that reintroduces float noise into
            // what would otherwise be an exact cancellation. A strict
            // `!= 0.0` guard would make the set_cusp_klm branch below
            // unreachable in practice -- any preset engineered to hit
            // d0==0.0 exactly loses that exactness the moment it's rescaled
            // to a window size that wasn't the one it was tuned against.
            //
            // This is deliberately its own (much tighter) constant, not
            // classify()'s NEARLY_ZERO (1/4096): set_cusp_klm's derivation
            // assumes d0 is *exactly* zero, and its error grows linearly in
            // d0, not as a step function -- reusing NEARLY_ZERO here let a
            // curve with d0 legitimately near, but not at, the classification
            // boundary take the cusp formula anyway, producing a klm fit
            // that doesn't vanish on the true curve (residual ~0.0036 right
            // at that boundary, measured against the curve that surfaced
            // this: a visibly thick, gradient-y hairline instead of a crisp
            // line, since the shader's f=k^3-lm never gets close to 0 on the
            // actual curve). CUSP_KLM_EPSILON only needs to clear the float
            // noise a rescale introduces (~1e-7, measured), so 1e-5 leaves a
            // huge margin against both that noise and any visible fit error.
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

fn lerp(a: Pt, b: Pt, t: f32) -> Pt {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Evaluates the cubic Bezier at parameter `t` via its Bernstein basis.
pub fn eval(p: &[Pt; 4], t: f32) -> Pt {
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

/// De Casteljau subdivision at a single parameter t, matching `SkChopCubicAt`'s
/// one-t overload: returns the [0,t] and [t,1] halves, sharing the split point.
fn chop_cubic_single(p: &[Pt; 4], t: f32) -> ([Pt; 4], [Pt; 4]) {
    let p01 = lerp(p[0], p[1], t);
    let p12 = lerp(p[1], p[2], t);
    let p23 = lerp(p[2], p[3], t);
    let p012 = lerp(p01, p12, t);
    let p123 = lerp(p12, p23, t);
    let p0123 = lerp(p012, p123, t);
    ([p[0], p01, p012, p0123], [p0123, p123, p23, p[3]])
}

/// De Casteljau subdivision at up to 2 ascending parameter values, matching
/// `SkChopCubicAt`: each additional t is re-parametrized relative to the
/// remaining tail before it's applied.
fn chop_cubic_at(p: &[Pt; 4], ts: &[f32]) -> Vec<[Pt; 4]> {
    match ts {
        [] => vec![*p],
        &[t0] => {
            let (left, right) = chop_cubic_single(p, t0);
            vec![left, right]
        }
        &[t0, t1] => {
            let (left, tail) = chop_cubic_single(p, t0);
            let t1_in_tail = (t1 - t0) / (1.0 - t0).max(1e-6);
            let (mid, right) = chop_cubic_single(&tail, t1_in_tail);
            vec![left, mid, right]
        }
        _ => unreachable!("at most 2 chop points"),
    }
}

/// One sub-curve of a (possibly chopped) cubic, with its own control points
/// and an implicit form that's already sign-corrected for this piece.
#[derive(Clone, Copy, Debug)]
pub struct CubicPiece {
    pub points: [Pt; 4],
    pub implicit: ImplicitCubic,
}

/// Port of `GrPathUtils::chopCubicAtLoopIntersection`. A Loop-classified
/// cubic self-intersects at two parameter values; this splits the curve
/// there so the closed loop and its two open tails each get their own
/// piece. That matters for rendering: at the exact self-intersection point,
/// both f = k^3 - l*m and its gradient are genuinely zero (a real
/// singularity of the implicit curve, not a tessellation artifact), so a
/// single mesh spanning both sides of the node can pick up a soft smear
/// there in filled mode. Splitting keeps every rendered mesh's interior
/// free of the singularity.
///
/// Non-loop curves are returned as a single, unsplit piece.
pub fn chop_at_loop_intersection(p: &[Pt; 4]) -> Vec<CubicPiece> {
    let d = inflection_coeffs(p);
    let kind = classify(p, d);

    if kind != CubicKind::Loop {
        return vec![CubicPiece { points: *p, implicit: compute_implicit(p) }];
    }

    let root = (4.0 * d[0] * d[2] - 3.0 * d[1] * d[1]).max(0.0).sqrt();
    let ls = (d[1] - root) / (2.0 * d[0]);
    let ms = (d[1] + root) / (2.0 * d[0]);
    let (small_s, large_s) = if ls <= ms { (ls, ms) } else { (ms, ls) };

    let mut chop_ts = Vec::with_capacity(2);
    if small_s > 0.0 && small_s < 1.0 {
        chop_ts.push(small_s);
    }
    if large_s > 0.0 && large_s < 1.0 {
        chop_ts.push(large_s);
    }

    // Which piece(s) are "the loop" (the closed middle section between the
    // two self-intersection parameters) versus a tail -- the loop formula's
    // K,L need negating on the loop side relative to the tails.
    //
    // Skia's GrPathUtils::chopCubicAtLoopIntersection hardcodes this flip to
    // a constant -1 (klm_rev = {1,-1,1}), because Skia only ever consumes it
    // through a stencil/cover pipeline that combines this klm-tested sliver
    // with separate plain-filled interior triangles -- the *sign* only has
    // to be internally consistent for that stencil accounting, never
    // correct in isolation. We draw each chopped piece as its own
    // standalone alpha-blended mesh (no stencil compositing), so the loop
    // piece's sign must independently match its own true enclosed interior.
    // Verified numerically (see `loop_piece_fill_matches_true_interior`
    // below) that the correct standalone sign is `d[0].signum()`, not a
    // fixed -1: Skia's constant happens to agree only when d[0] < 0, and is
    // inverted (paints the loop's exterior instead of its interior) when
    // d[0] > 0.
    let rev = d[0].signum();
    let klm_rev: Vec<f32> = match chop_ts.len() {
        2 => vec![1.0, rev, 1.0],
        1 => {
            if small_s < 0.0 { vec![rev, 1.0] } else { vec![1.0, rev] }
        }
        _ => {
            if small_s < 0.0 && large_s > 1.0 { vec![rev] } else { vec![1.0] }
        }
    };

    let (ck, cl, cm) = set_loop_klm(d);
    let (k, l, m) = fit_affine_functionals(p, ck, cl, cm);
    let discriminant = d[0] * d[0] * (3.0 * d[1] * d[1] - 4.0 * d[0] * d[2]);

    chop_cubic_at(p, &chop_ts)
        .into_iter()
        .zip(klm_rev)
        .map(|(points, rev)| {
            let signed = |f: [f32; 3]| [f[0] * rev, f[1] * rev, f[2] * rev];
            let implicit = ImplicitCubic {
                kind: CubicKind::Loop,
                k: signed(k),
                l: signed(l),
                m,
                d,
                discriminant,
            };
            CubicPiece { points, implicit }
        })
        .collect()
}

/// Reference control points hitting each of the six classifications, plus
/// four curves that are all classified `Loop` but exercise every distinct
/// branch of `chop_at_loop_intersection`'s chop-count/sign logic (numerically
/// verified, not hand-guessed: see the derivation notes in the project
/// history for how these were found). A label is carried alongside each
/// preset because several share a `CubicKind` (all five Loop entries) and
/// need distinguishing names for the gallery/legend.
///
/// The four Loop variants, by how many of the curve's two self-intersection
/// parameters (in the Bezier's own `t` terms) land inside the visible
/// `[0, 1]` range:
///   - "Loop (2 chops)": both land in range -- the classic, visibly
///     self-crossing loop.
///   - "Loop (tail -> loop)": only the *later* parameter is in range, so the
///     visible arc starts already inside the loop region and exits it
///     partway through. This is the case that produced a visible stray-hook
///     artifact when `geometry::coverage_mesh`'s AA pad was too generous
///     (see the pad size note there) -- included here so a regression shows
///     up in the gallery, not just in one hand-dragged interactive session.
///   - "Loop (loop -> tail)": the mirror image -- only the *earlier*
///     parameter is in range.
///   - "Loop (fully inside)": neither parameter is in range, but they
///     straddle it (one < 0, one > 1) -- the entire visible arc sits inside
///     the loop region.
///   - "Loop (no visible loop)": neither parameter is in range, and both are
///     on the same side -- the visible arc never comes near the
///     self-intersection at all, so it looks like an ordinary curve despite
///     classifying as Loop.
///
/// The remaining entries add variety within each classification (multiple
/// Serpentine/Loop/Cusp shapes, not just one) plus one more targeted case:
/// "Cusp (exact d0=0)" is engineered so `d[0]` lands on bitwise-exact `0.0`
/// (an integer-ratio construction: raw a1,a2,a3 = 1,2,1 before
/// renormalization), so `compute_implicit`'s dedicated `set_cusp_klm` formula
/// -- as opposed to the numeric-cusp fallback that reuses the serpentine
/// formula -- actually gets exercised. See the `d[0].abs() > CUSP_KLM_EPSILON`
/// guard note in `compute_implicit` for why this needs to be *close to* zero
/// rather than exactly zero to survive `fit_points` rescaling into an actual
/// window size.
pub fn preset_curves() -> [(&'static str, CubicKind, [Pt; 4]); 23] {
    [
        ("Serpentine", CubicKind::Serpentine, [[40.0, 260.0], [460.0, 40.0], [160.0, 40.0], [460.0, 260.0]]),
        ("Loop (2 chops)", CubicKind::Loop, [[40.0, 260.0], [460.0, 40.0], [-80.0, 40.0], [460.0, 260.0]]),
        ("Cusp", CubicKind::Cusp, [[40.0, 260.0], [460.0, 40.0], [40.0, 40.0], [460.0, 260.0]]),
        ("Quadratic", CubicKind::Quadratic, [[0.0, 300.0], [200.0, 100.0], [400.0, 100.0], [600.0, 300.0]]),
        ("Line", CubicKind::Line, [[40.0, 260.0], [180.0, 200.0], [320.0, 140.0], [460.0, 80.0]]),
        ("Point", CubicKind::Point, [[200.0, 200.0]; 4]),
        (
            "Loop (tail -> loop)",
            CubicKind::Loop,
            [[240.0, 590.0], [370.0, 416.0], [614.0, 246.0], [806.0, 565.0]],
        ),
        (
            "Loop (loop -> tail)",
            CubicKind::Loop,
            [[806.0, 565.0], [614.0, 246.0], [370.0, 416.0], [240.0, 590.0]],
        ),
        (
            "Loop (fully inside)",
            CubicKind::Loop,
            [[240.0, 590.0], [370.0, 416.0], [700.0, 246.0], [806.0, 565.0]],
        ),
        (
            "Loop (no visible loop)",
            CubicKind::Loop,
            [[240.0, 590.0], [370.0, 416.0], [590.0, 246.0], [806.0, 565.0]],
        ),
        // --- Serpentine: two distinct real inflection points visible in (0,1) ---
        ("Serpentine (bowtie)", CubicKind::Serpentine, [[0.1, 0.9], [0.7, 0.1], [0.3, 0.1], [0.9, 0.9]]),
        ("Serpentine (arch)", CubicKind::Serpentine, [[0.1, 0.2], [0.5, 0.9], [0.5, 0.9], [0.9, 0.2]]),
        ("Serpentine (lean)", CubicKind::Serpentine, [[0.1, 0.1], [0.9, 0.9], [0.9, 0.9], [0.1, 0.9]]),
        // --- Loop: curve self-intersects; both t-values in (0,1) -> 3 pieces ---
        ("Loop (medium)", CubicKind::Loop, [[0.3, 0.9], [0.9, 0.1], [0.1, 0.1], [0.7, 0.9]]),
        ("Loop (wide)", CubicKind::Loop, [[0.2, 0.9], [0.85, 0.1], [0.15, 0.1], [0.8, 0.9]]),
        ("Loop (narrow)", CubicKind::Loop, [[0.3, 0.95], [0.95, 0.1], [0.05, 0.1], [0.7, 0.95]]),
        // --- Cusp: single inflection point collapsed to a cusp ---
        ("Cusp (at start)", CubicKind::Cusp, [[0.1, 0.5], [0.1, 0.5], [0.7, 0.1], [0.9, 0.5]]),
        ("Cusp (at end)", CubicKind::Cusp, [[0.1, 0.5], [0.5, 0.1], [0.9, 0.5], [0.9, 0.5]]),
        ("Cusp (arch)", CubicKind::Cusp, [[0.5, 0.1], [0.5, 0.1], [0.9, 0.9], [0.1, 0.9]]),
        ("Cusp (tent)", CubicKind::Cusp, [[0.5, 0.9], [0.5, 0.9], [0.9, 0.1], [0.1, 0.1]]),
        // --- Line: degenerate -- all four control points collinear ---
        ("Line (horizontal)", CubicKind::Line, [[0.1, 0.5], [0.4, 0.5], [0.6, 0.5], [0.9, 0.5]]),
        // --- Gentle S-curve (classifies Cusp: inflection between convex/concave) ---
        ("S-curve (gentle)", CubicKind::Cusp, [[0.1, 0.4], [0.7, 0.1], [0.3, 0.9], [0.9, 0.6]]),
        // --- Exact cusp: d0 == 0.0 bitwise, exercising set_cusp_klm directly ---
        (
            "Cusp (exact d0=0)",
            CubicKind::Cusp,
            [[100.0, 100.0], [300.0, 500.0], [300.0, 300.0], [300.0, 100.0]],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_classify_as_labeled() {
        for (name, expected, p) in preset_curves() {
            let c = compute_implicit(&p);
            assert_eq!(c.kind, expected, "{name}: discriminant = {}, d = {:?}", c.discriminant, c.d);
        }
    }

    #[test]
    fn klm_vanishes_along_the_curve() {
        // For every renderable classification, sample the actual parametric
        // curve and check the implicit form is ~0 there (correctness of the
        // fitted functionals, not just the classification).
        for (name, kind, p) in preset_curves() {
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
            assert!(max_abs_err < 10.0, "{name} ({kind:?}) max abs error {max_abs_err}");
        }
    }

    /// `klm_vanishes_along_the_curve` above wouldn't catch a degenerate
    /// K=L=M=[0,0,0] fit: k^3-l*m is identically 0 everywhere in that case,
    /// which trivially passes an "is it ~0 on the curve" check. This checks
    /// the functionals aren't *identically* zero instead, by confirming
    /// func is meaningfully nonzero at a point clearly outside the curve's
    /// own bounding box (a genuine affine cubic implicit curve is a 1D
    /// zero set, not the whole plane). Regression test for the bug where
    /// fit_affine_functionals hardcoded control points 0,1,2 as its basis:
    /// singular whenever those three are collinear, which includes the
    /// ordinary P0==P1 construction for an exact cusp (e.g. "Cusp (at
    /// start)"/"(arch)"/"(tent)"), and made the whole coverage mesh render
    /// as a solid fill instead of tracing a curve.
    #[test]
    fn klm_is_not_identically_zero() {
        for (name, kind, p) in preset_curves() {
            let c = compute_implicit(&p);
            if !c.is_renderable() {
                continue;
            }
            let (min_x, max_x, min_y, max_y) = p.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY),
                |(min_x, max_x, min_y, max_y), pt| {
                    (min_x.min(pt[0]), max_x.max(pt[0]), min_y.min(pt[1]), max_y.max(pt[1]))
                },
            );
            let far = [max_x + (max_x - min_x).max(1.0) * 10.0, max_y + (max_y - min_y).max(1.0) * 10.0];
            let klm = c.eval_klm(far);
            let f = klm[0].powi(3) - klm[1] * klm[2];
            assert!(
                f.abs() > 1e-6,
                "{name} ({kind:?}): k^3-l*m is ~0 even far off-curve at {far:?} \
                 (k={:?} l={:?} m={:?}) -- functionals are likely identically zero",
                c.k,
                c.l,
                c.m
            );
        }
    }

    /// All five presets classified `Loop`, by name -- covers every branch of
    /// `chop_at_loop_intersection`'s chop-count/sign logic, not just the
    /// classic both-crossings-visible case.
    fn loop_presets() -> Vec<(&'static str, [Pt; 4])> {
        preset_curves()
            .into_iter()
            .filter(|(_, kind, _)| *kind == CubicKind::Loop)
            .map(|(name, _, p)| (name, p))
            .collect()
    }

    #[test]
    fn loop_chop_pieces_are_contiguous() {
        for (name, p) in loop_presets() {
            let pieces = chop_at_loop_intersection(&p);
            assert_eq!(pieces[0].points[0], p[0], "{name}: first piece should start at p0");
            assert_eq!(pieces.last().unwrap().points[3], p[3], "{name}: last piece should end at p3");
            for pair in pieces.windows(2) {
                assert_eq!(
                    pair[0].points[3], pair[1].points[0],
                    "{name}: chopped pieces must share an endpoint"
                );
            }
        }
    }

    /// Locks in the exact chop-count each Loop preset is meant to exercise,
    /// so every branch of `chop_at_loop_intersection`'s `match chop_ts.len()`
    /// is actually covered by a test rather than just the 2-chop case.
    #[test]
    fn loop_chop_piece_counts_cover_every_case() {
        let expected: &[(&str, usize)] = &[
            ("Loop (2 chops)", 3),
            ("Loop (tail -> loop)", 2),
            ("Loop (loop -> tail)", 2),
            ("Loop (fully inside)", 1),
            ("Loop (no visible loop)", 1),
            ("Loop (medium)", 3),
            ("Loop (wide)", 3),
            ("Loop (narrow)", 3),
        ];
        for (name, p) in loop_presets() {
            let want = expected.iter().find(|(n, _)| *n == name).map(|(_, n)| *n).unwrap_or_else(|| {
                panic!("{name}: no expected piece count registered for this Loop preset")
            });
            let pieces = chop_at_loop_intersection(&p);
            assert_eq!(pieces.len(), want, "{name}: expected {want} piece(s), got {}", pieces.len());
        }
    }

    #[test]
    fn loop_chop_klm_vanishes_on_each_piece() {
        for (name, p) in loop_presets() {
            for piece in chop_at_loop_intersection(&p) {
                let mut max_abs_err: f32 = 0.0;
                for i in 0..=20 {
                    let t = i as f32 / 20.0;
                    let pt = cubic_point(&piece.points, t);
                    let klm = piece.implicit.eval_klm(pt);
                    let f = klm[0].powi(3) - klm[1] * klm[2];
                    max_abs_err = max_abs_err.max(f.abs());
                }
                assert!(
                    max_abs_err < 10.0,
                    "{name}: piece {:?} max abs error {max_abs_err}",
                    piece.points
                );
            }
        }
    }

    /// The loop preset's middle chopped piece starts and ends at (nearly)
    /// the same point -- the self-intersection node -- so it traces a
    /// closed shape on its own. This checks that piece's klm sign against
    /// that shape's *true* geometric interior (via point-in-polygon on the
    /// sampled arc), not just that klm vanishes on the boundary. This is
    /// the check that would have caught the standalone-fill sign bug fixed
    /// alongside this test: Skia's fixed klm_rev = {1,-1,1} only agrees
    /// with the true interior when d[0] < 0, and paints the *exterior*
    /// instead when d[0] > 0 (the Loop preset's case).
    #[test]
    fn loop_piece_fill_matches_true_interior() {
        let (_, _, p) = preset_curves()[1]; // "Loop (2 chops)"
        let pieces = chop_at_loop_intersection(&p);
        assert_eq!(pieces.len(), 3, "expected two chops (tail, loop, tail)");
        let loop_piece = &pieces[1];

        let poly: Vec<Pt> = (0..=300).map(|i| cubic_point(&loop_piece.points, i as f32 / 300.0)).collect();

        let (min_x, max_x, min_y, max_y) = loop_piece.points.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY),
            |(min_x, max_x, min_y, max_y), p| {
                (min_x.min(p[0]), max_x.max(p[0]), min_y.min(p[1]), max_y.max(p[1]))
            },
        );

        let n = 30;
        let mut agree = 0;
        let mut total = 0;
        for gy in 0..n {
            for gx in 0..n {
                let x = min_x + (max_x - min_x) * gx as f32 / (n - 1) as f32;
                let y = min_y + (max_y - min_y) * gy as f32 / (n - 1) as f32;
                let pt = [x, y];
                let truth = point_in_polygon(pt, &poly);
                let klm = loop_piece.implicit.eval_klm(pt);
                let inside = klm[0].powi(3) - klm[1] * klm[2] < 0.0;
                total += 1;
                if truth == inside {
                    agree += 1;
                }
            }
        }
        // Boundary grid points can legitimately land on either side of the
        // curve by float noise; a large majority match is the real signal.
        assert!(
            agree as f32 / total as f32 > 0.95,
            "loop piece klm sign disagrees with true interior on {}/{} grid points",
            total - agree,
            total
        );
    }

    fn point_in_polygon(pt: Pt, poly: &[Pt]) -> bool {
        let (x, y) = (pt[0], pt[1]);
        let mut inside = false;
        let mut j = poly.len() - 1;
        for i in 0..poly.len() {
            let (xi, yi) = (poly[i][0], poly[i][1]);
            let (xj, yj) = (poly[j][0], poly[j][1]);
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    #[test]
    fn non_loop_curves_are_returned_unchopped() {
        for (name, kind, p) in preset_curves() {
            if kind == CubicKind::Loop {
                continue;
            }
            let pieces = chop_at_loop_intersection(&p);
            assert_eq!(pieces.len(), 1, "{name} ({kind:?}) should not be chopped");
            assert_eq!(pieces[0].points, p);
        }
    }

    /// "Cusp (exact d0=0)" was engineered so `compute_implicit`'s
    /// `d[0].abs() > CUSP_KLM_EPSILON` guard takes the *other* branch --
    /// `set_cusp_klm` -- rather than the numeric-cusp fallback that every
    /// other Cusp preset exercises. Without this, that formula has zero
    /// coverage anywhere in the suite.
    #[test]
    fn exact_cusp_preset_exercises_set_cusp_klm() {
        let (_, kind, p) = preset_curves()
            .into_iter()
            .find(|(name, _, _)| *name == "Cusp (exact d0=0)")
            .expect("preset not found");
        assert_eq!(kind, CubicKind::Cusp);
        let d = inflection_coeffs(&p);
        assert!(d[0].abs() <= CUSP_KLM_EPSILON, "expected d0 within epsilon of 0, got {}", d[0]);
    }

    /// The flip side of the test above: a Cusp-classified curve whose d0 is
    /// clearly outside CUSP_KLM_EPSILON (but still comfortably inside
    /// classify()'s much looser NEARLY_ZERO) must take the serpentine-formula
    /// branch, not set_cusp_klm -- otherwise the implicit curve doesn't
    /// vanish along the true curve (this is exactly the bug that shipped:
    /// reusing NEARLY_ZERO here let it happen). Regression test using a
    /// point known to sit at 1x the old, wrong threshold.
    #[test]
    fn near_cusp_within_classify_epsilon_but_outside_cusp_epsilon_uses_serpentine_formula() {
        let p: [Pt; 4] = [[250.0, 635.0], [306.0, 383.0], [454.62, 258.0], [969.0, 635.0]];
        let d = inflection_coeffs(&p);
        assert!(d[0].abs() > CUSP_KLM_EPSILON, "test fixture drifted: d0 = {}", d[0]);
        assert!(d[0].abs() < NEARLY_ZERO, "test fixture drifted: d0 = {}", d[0]);
        let c = compute_implicit(&p);
        assert_eq!(c.kind, CubicKind::Cusp);
        let mut max_abs_err: f32 = 0.0;
        for i in 0..=200 {
            let t = i as f32 / 200.0;
            let pt = cubic_point(&p, t);
            let klm = c.eval_klm(pt);
            let f = klm[0].powi(3) - klm[1] * klm[2];
            max_abs_err = max_abs_err.max(f.abs());
        }
        // Correct (serpentine formula): ~1e-9. Wrong (cusp formula, the bug
        // this regression-tests): ~0.0024. 0.001 sits comfortably between.
        assert!(max_abs_err < 0.001, "klm didn't vanish on the true curve: max abs error {max_abs_err}");
    }

    fn cubic_point(p: &[Pt; 4], t: f32) -> Pt {
        super::eval(p, t)
    }
}
