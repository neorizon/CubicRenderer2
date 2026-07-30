#version 300 es
precision highp float;

layout(location = 0) in vec2 a_pos;

uniform vec2 u_viewportSize;

void main() {
    vec2 ndc = (a_pos / u_viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
