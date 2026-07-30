#version 300 es
precision highp float;

// Pixel-space position (y-down, origin top-left) and the per-vertex value of
// the K, L, M functionals at that position. Since K, L, M are affine in
// (x,y), linear interpolation across the triangle reproduces their exact
// value at every covered pixel -- no tessellation of the curve is needed.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_klm;

uniform vec2 u_viewportSize;

out vec3 v_klm;

void main() {
    vec2 ndc = (a_pos / u_viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_klm = a_klm;
}
