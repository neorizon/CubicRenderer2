#version 300 es
precision highp float;

// Port of Skia's GrGLCubicEffect (src/gpu/effects/GrBezierEffect.cpp).
// The Loop-Blinn implicit test for a non-rational cubic is
//   f(k,l,m) = k^3 - l*m,   f < 0 inside, f > 0 outside.
//
// AA coverage turns f into an approximate signed distance in pixels via a
// first-order Taylor expansion: d = f / |grad f|, where grad f is obtained
// by the chain rule from the screen-space derivatives of k, l, m themselves
// (dFdx/dFdy of the *inputs*, not of the composite f) -- exactly as Skia's
// shader does it. dFdx/dFdy are core in GLSL ES 3.00.
in vec3 v_klm;
in vec2 v_screen;
out vec4 fragColor;

uniform vec4 u_color;
// 0 = fill, antialiased | 1 = hairline stroke, antialiased | 2 = fill, no AA
uniform int u_edgeType;

// Screen-space centre and radius of the acnode halo; radius 0 disables it.
//
// A serpentine's implicit cubic k^3 - l*m = 0 has an *acnode*: an isolated
// real solution that is not on the Bezier arc at all (it is the image of a
// complex-conjugate parameter pair). It satisfies the test below and falls
// inside the coverage mesh's convex hull, so it renders as a floating dot.
// The CPU solves for it in closed form; see `cubic::Acnode`.
uniform vec2 u_acnode;
uniform float u_acnodeRadius;

void main() {
    if (u_acnodeRadius > 0.0 && distance(v_screen, u_acnode) < u_acnodeRadius) {
        discard;
    }

    float k = v_klm.x;
    float l = v_klm.y;
    float m = v_klm.z;
    float func = k * k * k - l * m;

    float coverage;
    if (u_edgeType == 2) {
        coverage = func < 0.0 ? 1.0 : 0.0;
    } else {
        vec3 dklmdx = dFdx(v_klm);
        vec3 dklmdy = dFdy(v_klm);
        // d/dx(k^3 - l*m) = 3k^2 dk/dx - l dm/dx - m dl/dx, and likewise for y.
        float dfdx = 3.0 * k * k * dklmdx.x - l * dklmdx.z - m * dklmdx.y;
        float dfdy = 3.0 * k * k * dklmdy.x - l * dklmdy.z - m * dklmdy.y;
        float gradMag = length(vec2(dfdx, dfdy));

        float f = (u_edgeType == 1) ? abs(func) : func;
        // gradMag is the true, mathematically exact |grad f|, which is
        // ~1px worth of change in f -- not an arbitrary quantity that needs
        // clamping to some fixed scale. It legitimately gets very small
        // (verified: ~1e-8, not float noise) for some curves, because the
        // affine k, l, m end up with small linear coefficients for that
        // curve's specific geometry -- nothing to do with being close to a
        // cusp or Loop node specifically. A clamp of 1e-6 was previously
        // used here to guard the single true zero (an exact cusp/node
        // singularity, one pixel), but 1e-6 is itself large enough to
        // misfire on those legitimately-small-but-correct gradients,
        // inflating the ~1px AA falloff into a many-pixel-wide gradient
        // band across the whole curve. 1e-12 only guards the literal 0/0
        // case; it's far below any gradient magnitude a real curve produces.
        float dist = f / max(gradMag, 1e-12);

        if (u_edgeType == 1) {
            coverage = max(1.0 - dist, 0.0);
        } else {
            coverage = clamp(1.0 - dist, 0.0, 1.0);
        }
    }

    if (coverage <= 0.0) {
        discard;
    }
    fragColor = vec4(u_color.rgb, u_color.a * coverage);
}
