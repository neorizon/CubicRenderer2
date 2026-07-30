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
out vec4 fragColor;

uniform vec4 u_color;
// 0 = fill, antialiased | 1 = hairline stroke, antialiased | 2 = fill, no AA
uniform int u_edgeType;

void main() {
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
        float dist = f / max(gradMag, 1e-6);

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
