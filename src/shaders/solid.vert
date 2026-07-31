#version 300 es
precision highp float;

layout(location = 0) in vec2 a_pos;

uniform vec2 u_viewportSize;
uniform vec2 u_pan;
uniform float u_zoom;

void main() {
    vec2 screen = (a_pos + u_pan) * u_zoom;
    vec2 ndc = (screen / u_viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
