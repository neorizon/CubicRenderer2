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

    // Canonicalize the traversal direction before doing any of the loop
    // math, because none of that math is reversal-invariant on its own.
    //
    // d0 is exactly negated by t -> 1-t, and `set_loop_klm` keys its
    // orientation normalization off `d[0]`'s sign *combined* with the sign
    // of k[1] -- which does not flip in lockstep. The upshot is that the
    // raw implicit f comes out negated under reversal for most curves but
    // identical for others, so no sign derived from d0 can make the fill
    // direction-independent. Measured across the Loop presets: reversal
    // negates f for six of them and leaves it alone for two.
    //
    // Rather than chase that with a correction factor, run the whole
    // derivation on one fixed orientation (d0 > 0) and map the pieces back
    // to the caller's direction. Reversal invariance then holds by
    // construction instead of by tuning. See
    // `loop_fill_is_invariant_under_reversal`.
    if d[0] < 0.0 {
        let mut pieces = chop_at_loop_intersection(&[p[3], p[2], p[1], p[0]]);
        pieces.reverse();
        for piece in pieces.iter_mut() {
            let q = piece.points;
            piece.points = [q[3], q[2], q[1], q[0]];
        }
        return pieces;
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

    // Crossing a double point always swaps which side of the arc the
    // negative region lies on -- near a crunode f = k^3 - l*m is a saddle,
    // so {f < 0} occupies two *opposite* quadrants of the crossing. The
    // loop piece's K,L therefore always need negating relative to the
    // tails, by a fixed -1.
    //
    // The case split below is Skia's
    // GrPathUtils::chopCubicAtLoopIntersection verbatim ({1,-1,1} and its
    // shorter forms): that *relative* pattern is direction-invariant and
    // correct, and it is the part worth copying.
    //
    // The overall sign is where we deliberately part company. Skia leaves
    // it entirely to whatever `set_loop_klm` happened to produce, with no
    // global correction. Measured by running Skia's exact constants through
    // this port's tests, that inverts the fill (paints the loop's exterior)
    // on every d0 > 0 loop -- `loop_piece_fill_matches_true_interior` fails
    // 900/900 samples -- and is not reversal-invariant. Skia gets away with
    // it because its only caller is gm/beziereffects.cpp, a visual GM that
    // draws each chopped piece as a standalone quad and never asserts which
    // side came out filled. We render for real, so the loop piece must
    // paint its own enclosed interior; with the direction canonicalized to
    // d0 > 0 above, that pins the loop piece at +1 and the tails at -1.
    let (loop_sign, tail_sign) = (1.0f32, -1.0f32);
    let klm_rev: Vec<f32> = match chop_ts.len() {
        2 => vec![tail_sign, loop_sign, tail_sign],
        1 => {
            if small_s < 0.0 {
                vec![loop_sign, tail_sign]
            } else {
                vec![tail_sign, loop_sign]
            }
        }
        _ => {
            if small_s < 0.0 && large_s > 1.0 { vec![loop_sign] } else { vec![tail_sign] }
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

/// Radius, in *screen* pixels, of the region around an acnode that the
/// coverage shader discards. The acnode is a single point mathematically,
/// but the shader's AA distance estimate `f / |grad f|` degenerates around
/// it (both f and grad f vanish there, and their ratio stays below 1 for
/// several pixels), which is what paints it as a visible blob rather than a
/// lone pixel. This is sized to cover that blob -- see
/// `stray_shaded_pixels_all_lie_within_the_acnode_halo`.
pub const ACNODE_HALO_PX: f32 = 6.0;

/// Fraction of the acnode's clearance from the arc that its halo may
/// occupy, so the discarded disc never reaches the curve itself.
const ACNODE_HALO_ARC_MARGIN_PX: f32 = 1.0;

/// The isolated real singular point ("acnode") of a serpentine's implicit
/// cubic: a point where `k^3 - l*m` and its gradient both vanish, but which
/// is *not* on the Bezier arc.
///
/// Every rational cubic has exactly one double point, and which kind it is
/// *is* the Loop-Blinn classification: a Loop's is a crunode (the visible
/// self-intersection, which `chop_at_loop_intersection` splits at), a Cusp's
/// sits right on the curve, and a Serpentine's is an acnode -- real, but
/// isolated and detached from the arc. The acnode still satisfies the
/// implicit test the coverage shader applies, so it lights up as a stray dot
/// floating inside the coverage mesh, drifting towards the arc as the curve
/// approaches a cusp and disappearing into the node once it becomes a Loop.
#[derive(Clone, Copy, Debug)]
pub struct Acnode {
    pub point: Pt,
    /// Distance from `point` to the nearest point of the Bezier arc, in
    /// world units.
    pub dist_to_arc: f32,
}

impl Acnode {
    /// Radius of the disc the shader discards, in screen pixels, or 0 when
    /// no halo should be applied.
    ///
    /// It is all-or-nothing on purpose. As the curve approaches a cusp the
    /// acnode migrates onto the arc, and once there is not enough clearance
    /// for the full disc the right answer is to leave the region alone: a
    /// partial disc would punch a hole through the middle of the blob and
    /// could strand its far rim as a *new* detached crescent, and by that
    /// point the dot has fused with the cusp tip and no longer reads as a
    /// stray mark anyway.
    pub fn halo_radius_px(&self, zoom: f32) -> f32 {
        if self.dist_to_arc * zoom > ACNODE_HALO_PX + ACNODE_HALO_ARC_MARGIN_PX {
            ACNODE_HALO_PX
        } else {
            0.0
        }
    }
}

/// Locates the acnode of `p`'s implicit cubic, if it has one.
///
/// `chop_at_loop_intersection` finds a Loop's double point as the two real
/// roots `(d1 +/- sqrt(4*d0*d2 - 3*d1^2)) / (2*d0)` of the same quadratic.
/// For a serpentine that radicand is negative, so the double point is
/// reached at a *complex conjugate* pair of parameters instead. Evaluating
/// the Bezier there is still well defined, and because the pair is
/// conjugate the two images are conjugates of each other -- and they are by
/// definition the same double point, so that point is real and the
/// imaginary parts cancel exactly. Hence: evaluate the Bernstein basis in
/// complex arithmetic and keep the real part.
pub fn acnode(p: &[Pt; 4]) -> Option<Acnode> {
    let d = inflection_coeffs(p);
    if !matches!(classify(p, d), CubicKind::Serpentine | CubicKind::Cusp) {
        return None;
    }
    // d0 == 0 is the exact-cusp case: the double point degenerates onto the
    // curve and the parameter below runs off to infinity.
    if d[0].abs() <= NEARLY_ZERO {
        return None;
    }
    let radicand = 3.0 * d[1] * d[1] - 4.0 * d[0] * d[2];
    if radicand <= 0.0 {
        return None; // real roots -> crunode, not an acnode
    }

    let inv = 1.0 / (2.0 * d[0]);
    let (tr, ti) = (d[1] * inv, radicand.sqrt() * inv);

    let mul = |a: (f32, f32), b: (f32, f32)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let t = (tr, ti);
    let mt = (1.0 - tr, -ti);
    let t2 = mul(t, t);
    let mt2 = mul(mt, mt);
    // Real parts of the four Bernstein weights; the control points are real,
    // so only these are needed.
    let w = [
        mul(mt2, mt).0,
        3.0 * mul(mt2, t).0,
        3.0 * mul(mt, t2).0,
        mul(t2, t).0,
    ];
    let point = [
        w[0] * p[0][0] + w[1] * p[1][0] + w[2] * p[2][0] + w[3] * p[3][0],
        w[0] * p[0][1] + w[1] * p[1][1] + w[2] * p[2][1] + w[3] * p[3][1],
    ];
    if !point[0].is_finite() || !point[1].is_finite() {
        return None;
    }

    const ARC_SAMPLES: usize = 256;
    let dist_to_arc = (0..=ARC_SAMPLES)
        .map(|i| {
            let e = eval(p, i as f32 / ARC_SAMPLES as f32);
            ((e[0] - point[0]).powi(2) + (e[1] - point[1]).powi(2)).sqrt()
        })
        .fold(f32::INFINITY, f32::min);

    Some(Acnode { point, dist_to_arc })
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

    /// The `func`/`gradMag` pair `shaders/cubic.frag` computes per fragment,
    /// mirrored on the CPU so the coverage a pixel *would* get is testable
    /// without a GL context. K, L, M are affine, so their gradients are just
    /// their own x/y coefficients (this is exactly what `dFdx`/`dFdy` of the
    /// interpolated klm recover on the GPU).
    fn shader_coverage(c: &ImplicitCubic, q: Pt) -> f32 {
        let [k, l, m] = c.eval_klm(q);
        let func = k * k * k - l * m;
        let gx = 3.0 * k * k * c.k[0] - l * c.m[0] - m * c.l[0];
        let gy = 3.0 * k * k * c.k[1] - l * c.m[1] - m * c.l[1];
        let grad_mag = (gx * gx + gy * gy).sqrt();
        // Hairline AA: the edge type the artifact was reported against.
        1.0 - func.abs() / grad_mag.max(1e-12)
    }

    fn point_in_convex_polygon(pt: Pt, poly: &[Pt]) -> bool {
        let mut sign = 0i32;
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            let cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]);
            let s = if cross > 0.0 {
                1
            } else if cross < 0.0 {
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

    /// A serpentine's implicit cubic has an acnode: a real, isolated
    /// solution of k^3 - l*m = 0 that lies *off* the Bezier arc. It is a
    /// genuine singular point, so both the implicit function and its
    /// gradient vanish there.
    #[test]
    fn acnode_is_a_singular_point_of_the_implicit_cubic() {
        let mut checked = 0;
        for (name, _, p) in preset_curves() {
            let Some(ac) = acnode(&p) else { continue };
            let c = compute_implicit(&p);
            let [k, l, m] = c.eval_klm(ac.point);
            let f = k * k * k - l * m;
            let gx = 3.0 * k * k * c.k[0] - l * c.m[0] - m * c.l[0];
            let gy = 3.0 * k * k * c.k[1] - l * c.m[1] - m * c.l[1];
            // Scale-free: compare against the gradient magnitude a whole
            // pixel away from the curve, i.e. the shader's own unit.
            let scale = c.k.iter().chain(c.l.iter()).chain(c.m.iter()).fold(0.0f32, |a, v| a.max(v.abs()));
            assert!(
                f.abs() / scale.powi(3) < 1e-6,
                "{name}: f = {f} at the acnode {:?} (should vanish)",
                ac.point
            );
            assert!(
                (gx * gx + gy * gy).sqrt() / (scale * scale) < 1e-6,
                "{name}: grad = ({gx}, {gy}) at the acnode (should vanish)"
            );
            checked += 1;
        }
        assert!(checked >= 4, "expected several serpentine presets to have an acnode, got {checked}");
    }

    #[test]
    fn only_serpentines_have_an_acnode() {
        for (name, _, p) in preset_curves() {
            let kind = compute_implicit(&p).kind;
            if matches!(kind, CubicKind::Loop | CubicKind::Line | CubicKind::Point) {
                assert!(acnode(&p).is_none(), "{name} ({kind:?}) should have no acnode");
            }
        }
    }

    /// The regression test for the floating blue dot.
    ///
    /// The defect is not "a pixel far from the arc" -- the implicit curve
    /// legitimately continues past the arc's endpoints inside the padded
    /// hull. It is specifically a *detached* mark: a shaded blob with a gap
    /// of empty pixels between it and the stroke. So this rasterizes the
    /// coverage mesh at one sample per pixel, applies the acnode halo the
    /// shader will apply, and flood-fills the shaded pixels starting from
    /// the arc. Anything left unreached is a floating artifact.
    #[test]
    fn no_shaded_pixels_are_detached_from_the_curve() {
        let a: [Pt; 4] = [[40.0, 260.0], [460.0, 40.0], [160.0, 40.0], [460.0, 260.0]];
        let b: [Pt; 4] = [[40.0, 260.0], [460.0, 40.0], [-80.0, 40.0], [460.0, 260.0]];

        for step in 0..=20 {
            let u = step as f32 / 20.0;
            let mut p = a;
            p[2] = [a[2][0] + (b[2][0] - a[2][0]) * u, a[2][1] + (b[2][1] - a[2][1]) * u];

            for piece in chop_at_loop_intersection(&p) {
                if !piece.implicit.is_renderable() {
                    continue;
                }
                let mesh = crate::geometry::coverage_mesh(&piece.points, &piece.implicit, 1.5);
                let poly: Vec<Pt> = mesh.iter().map(|v| v.pos).collect();
                if poly.len() < 3 {
                    continue;
                }
                let ac = acnode(&piece.points);
                let halo = ac.map_or(0.0, |ac| ac.halo_radius_px(1.0)); // zoom 1: world px == screen px

                let (mut mnx, mut mny, mut mxx, mut mxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for q in poly.iter() {
                    mnx = mnx.min(q[0]);
                    mny = mny.min(q[1]);
                    mxx = mxx.max(q[0]);
                    mxy = mxy.max(q[1]);
                }
                let (w, h) = ((mxx - mnx).ceil() as usize + 2, (mxy - mny).ceil() as usize + 2);
                let at = |i: usize, j: usize| [mnx + i as f32, mny + j as f32];

                let mut shaded = vec![false; w * h];
                for j in 0..h {
                    for i in 0..w {
                        let q = at(i, j);
                        if !point_in_convex_polygon(q, &poly) {
                            continue;
                        }
                        if shader_coverage(&piece.implicit, q) <= 0.0 {
                            continue;
                        }
                        if let Some(ac) = ac {
                            let d = ((q[0] - ac.point[0]).powi(2) + (q[1] - ac.point[1]).powi(2)).sqrt();
                            if halo > 0.0 && d <= halo {
                                continue; // the shader discards this fragment
                            }
                        }
                        shaded[j * w + i] = true;
                    }
                }

                // Seed the flood fill from the shaded pixels the arc passes
                // through (3x3 around each sample, so a sample landing between
                // pixel centres still seeds), then spread 8-connected.
                let mut seen = vec![false; w * h];
                let mut stack: Vec<(usize, usize)> = Vec::new();
                for s in 0..=4000 {
                    let e = eval(&piece.points, s as f32 / 4000.0);
                    let (ci, cj) = ((e[0] - mnx).round() as i32, (e[1] - mny).round() as i32);
                    for dj in -1i32..=1 {
                        for di in -1i32..=1 {
                            let (i, j) = (ci + di, cj + dj);
                            if i < 0 || j < 0 || i >= w as i32 || j >= h as i32 {
                                continue;
                            }
                            let (i, j) = (i as usize, j as usize);
                            if shaded[j * w + i] && !seen[j * w + i] {
                                seen[j * w + i] = true;
                                stack.push((i, j));
                            }
                        }
                    }
                }
                while let Some((i, j)) = stack.pop() {
                    for dj in -1i32..=1 {
                        for di in -1i32..=1 {
                            let (ni, nj) = (i as i32 + di, j as i32 + dj);
                            if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                                continue;
                            }
                            let (ni, nj) = (ni as usize, nj as usize);
                            if shaded[nj * w + ni] && !seen[nj * w + ni] {
                                seen[nj * w + ni] = true;
                                stack.push((ni, nj));
                            }
                        }
                    }
                }

                let orphans: Vec<Pt> = (0..w * h)
                    .filter(|&n| shaded[n] && !seen[n])
                    .map(|n| at(n % w, n / w))
                    .collect();
                assert!(
                    orphans.is_empty(),
                    "u={u:.2} ({:?}): {} shaded pixel(s) detached from the curve, e.g. {:?} \
                     (acnode = {:?}, halo = {halo}px)",
                    piece.implicit.kind,
                    orphans.len(),
                    &orphans[..orphans.len().min(4)],
                    ac.map(|a| (a.point, a.dist_to_arc)),
                );
            }
        }
    }

    /// Discarding the halo must never eat into the curve itself: with the
    /// halo active, no pixel within half a pixel of the true arc is inside
    /// it.
    #[test]
    fn the_acnode_halo_never_covers_the_curve() {
        for (name, _, p) in preset_curves() {
            let Some(ac) = acnode(&p) else { continue };
            for zoom in [0.25f32, 1.0, 4.0] {
                let radius = ac.halo_radius_px(zoom);
                if radius <= 0.0 {
                    continue;
                }
                for i in 0..=2000 {
                    let q = eval(&p, i as f32 / 2000.0);
                    let d = ((q[0] - ac.point[0]).powi(2) + (q[1] - ac.point[1]).powi(2)).sqrt() * zoom;
                    assert!(
                        d > radius,
                        "{name} @zoom {zoom}: curve point {q:?} is {d}px from the acnode, \
                         inside the {radius}px halo"
                    );
                }
            }
        }
    }

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

    /// Reversing a cubic's control points traces the identical curve, so it
    /// must produce the identical filled region. It did not: `klm_rev` used
    /// to derive the loop-vs-tail flip from `d[0].signum()`, and d0 changes
    /// sign under t -> 1-t, so the flip was applied for one traversal
    /// direction and skipped for the other. The visible symptom was a
    /// near-cusp loop rendering as two triangles meeting at the double
    /// point, one filled on each side of the arc, that collapsed into a
    /// single region when the endpoints were dragged in the opposite order.
    #[test]
    fn loop_fill_is_invariant_under_reversal() {
        // Shaded region of a whole curve: the union over its chopped pieces
        // of {f < 0} clipped to that piece's coverage mesh, i.e. exactly
        // what the renderer paints.
        fn shaded_set(p: &[Pt; 4]) -> Vec<(Vec<Pt>, ImplicitCubic)> {
            chop_at_loop_intersection(p)
                .into_iter()
                .filter(|pc| pc.implicit.is_renderable())
                .map(|pc| {
                    let mesh = crate::geometry::coverage_mesh(&pc.points, &pc.implicit, 1.5);
                    (mesh.iter().map(|v| v.pos).collect::<Vec<Pt>>(), pc.implicit)
                })
                .collect()
        }
        fn is_shaded(set: &[(Vec<Pt>, ImplicitCubic)], q: Pt) -> bool {
            set.iter().any(|(poly, imp)| {
                poly.len() >= 3 && point_in_convex_polygon(q, poly) && {
                    let klm = imp.eval_klm(q);
                    klm[0].powi(3) - klm[1] * klm[2] < 0.0
                }
            })
        }

        let mut cases: Vec<(String, [Pt; 4])> = preset_curves()
            .into_iter()
            .filter(|(_, kind, _)| *kind == CubicKind::Loop)
            .map(|(name, _, p)| (name.to_string(), p))
            .collect();
        // The near-cusp loop from the reported screenshot: its double point
        // falls at t = 0.417 while its partner root is at t = 2.197, so the
        // arc crosses the node without ever closing the loop.
        cases.push((
            "reported near-cusp loop".to_string(),
            [[963.0, 620.0], [720.0, 213.0], [410.0, 431.0], [245.0, 652.0]],
        ));

        for (name, p) in cases {
            let rev = [p[3], p[2], p[1], p[0]];
            let (fwd_set, rev_set) = (shaded_set(&p), shaded_set(&rev));

            let (mut mnx, mut mny, mut mxx, mut mxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for q in p.iter() {
                mnx = mnx.min(q[0]);
                mny = mny.min(q[1]);
                mxx = mxx.max(q[0]);
                mxy = mxy.max(q[1]);
            }

            let n = 80;
            let (mut differ, mut total) = (0, 0);
            for gy in 0..n {
                for gx in 0..n {
                    let q = [
                        mnx + (mxx - mnx) * gx as f32 / (n - 1) as f32,
                        mny + (mxy - mny) * gy as f32 / (n - 1) as f32,
                    ];
                    total += 1;
                    if is_shaded(&fwd_set, q) != is_shaded(&rev_set, q) {
                        differ += 1;
                    }
                }
            }
            // Samples landing on the boundary can fall either way on float
            // noise; a whole mis-signed piece is orders of magnitude larger.
            assert!(
                differ * 100 <= total,
                "{name}: fill differs on {differ}/{total} samples when the \
                 control points are reversed"
            );
        }
    }

    /// The reported symptom, stated directly: the filled region must stay on
    /// the same side of the arc across a chop. It did not for a near-cusp
    /// loop -- the two pieces met at the double point with one filled above
    /// the curve and the other below, reading as two triangles joined at a
    /// point.
    #[test]
    fn loop_fill_stays_on_one_side_across_chops() {
        let mut cases: Vec<(String, [Pt; 4])> = preset_curves()
            .into_iter()
            .filter(|(_, kind, _)| *kind == CubicKind::Loop)
            .map(|(name, _, p)| (name.to_string(), p))
            .collect();
        cases.push((
            "reported near-cusp loop".to_string(),
            [[963.0, 620.0], [720.0, 213.0], [410.0, 431.0], [245.0, 652.0]],
        ));

        for (name, p) in cases {
            let pieces = chop_at_loop_intersection(&p);
            let mut sides: Vec<(usize, char)> = Vec::new();
            for (i, piece) in pieces.iter().enumerate() {
                for j in 1..=9 {
                    let t = j as f32 / 10.0;
                    let e = eval(&piece.points, t);
                    let e2 = eval(&piece.points, t + 1e-3);
                    let (tx, ty) = (e2[0] - e[0], e2[1] - e[1]);
                    let n = (tx * tx + ty * ty).sqrt();
                    if n < 1e-6 {
                        continue;
                    }
                    // Probe symmetrically about the arc, along its normal.
                    let (nx, ny) = (-ty / n * 2.0, tx / n * 2.0);
                    let f = |q: Pt| {
                        let klm = piece.implicit.eval_klm(q);
                        klm[0].powi(3) - klm[1] * klm[2]
                    };
                    let left = f([e[0] - nx, e[1] - ny]) < 0.0;
                    let right = f([e[0] + nx, e[1] + ny]) < 0.0;
                    // Only samples that cleanly straddle the boundary say
                    // anything about orientation.
                    if left != right {
                        sides.push((i, if left { 'L' } else { 'R' }));
                    }
                }
            }
            let first = match sides.first() {
                Some(&(_, c)) => c,
                None => continue,
            };
            assert!(
                sides.iter().all(|&(_, c)| c == first),
                "{name}: filled side flips between pieces: {sides:?}"
            );
        }
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




