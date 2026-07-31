#version 300 es
precision highp float;

// Pixel-space position (y-down, origin top-left) and the per-vertex value of
// the K, L, M functionals at that position. Since K, L, M are affine in
// (x,y), linear interpolation across the triangle reproduces their exact
// value at every covered pixel -- no tessellation of the curve is needed.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_klm;

uniform vec2 u_viewportSize;
uniform vec2 u_pan;
uniform float u_zoom;

out vec3 v_klm;
// Screen-space position (y-down, matching the app's convention). Passed as a
// varying rather than read from gl_FragCoord because gl_FragCoord is y-up.
out vec2 v_screen;

void main() {
    vec2 screen = (a_pos + u_pan) * u_zoom;
    vec2 ndc = (screen / u_viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_klm = a_klm;
    v_screen = screen;
}
