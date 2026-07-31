//! Thin glow wrappers: shader compilation and the two draw pipelines used by
//! both scenes -- `CubicPipeline` for Loop-Blinn coverage fills, and
//! `SolidPipeline` for flat-colored control points / polygons / grid chrome.

use glow::HasContext;

use crate::camera::Camera;

unsafe fn compile_shader(gl: &glow::Context, kind: u32, src: &str) -> glow::NativeShader {
    unsafe {
        let shader = gl.create_shader(kind).expect("create_shader failed");
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            panic!("shader compile error:\n{log}\n--- source ---\n{src}");
        }
        shader
    }
}

unsafe fn link_program(gl: &glow::Context, vs_src: &str, fs_src: &str) -> glow::NativeProgram {
    unsafe {
        let vs = compile_shader(gl, glow::VERTEX_SHADER, vs_src);
        let fs = compile_shader(gl, glow::FRAGMENT_SHADER, fs_src);
        let program = gl.create_program().expect("create_program failed");
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("program link error:\n{}", gl.get_program_info_log(program));
        }
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        program
    }
}

/// Screen-space fill/stroke coverage for a single cubic, matching Skia's
/// `GrCubicEffect` edge types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeType {
    FillAA = 0,
    HairlineAA = 1,
    FillNoAA = 2,
}

impl EdgeType {
    pub fn name(self) -> &'static str {
        match self {
            EdgeType::FillAA => "fill (AA)",
            EdgeType::HairlineAA => "hairline (AA)",
            EdgeType::FillNoAA => "fill (no AA)",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            EdgeType::FillAA => EdgeType::HairlineAA,
            EdgeType::HairlineAA => EdgeType::FillNoAA,
            EdgeType::FillNoAA => EdgeType::FillAA,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CubicVertex {
    pub pos: [f32; 2],
    pub klm: [f32; 3],
}

pub struct CubicPipeline {
    program: glow::NativeProgram,
    u_viewport: glow::UniformLocation,
    u_pan: glow::UniformLocation,
    u_zoom: glow::UniformLocation,
    u_color: glow::UniformLocation,
    u_edge_type: glow::UniformLocation,
    u_acnode: glow::UniformLocation,
    u_acnode_radius: glow::UniformLocation,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ebo: glow::Buffer,
}

impl CubicPipeline {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link_program(
                gl,
                include_str!("shaders/cubic.vert"),
                include_str!("shaders/cubic.frag"),
            );
            let u_viewport = gl.get_uniform_location(program, "u_viewportSize").unwrap();
            let u_pan = gl.get_uniform_location(program, "u_pan").unwrap();
            let u_zoom = gl.get_uniform_location(program, "u_zoom").unwrap();
            let u_color = gl.get_uniform_location(program, "u_color").unwrap();
            let u_edge_type = gl.get_uniform_location(program, "u_edgeType").unwrap();
            let u_acnode = gl.get_uniform_location(program, "u_acnode").unwrap();
            let u_acnode_radius = gl.get_uniform_location(program, "u_acnodeRadius").unwrap();

            let vao = gl.create_vertex_array().unwrap();
            let vbo = gl.create_buffer().unwrap();
            let ebo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let stride = std::mem::size_of::<CubicVertex>() as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, stride, 8);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.bind_vertex_array(None);

            CubicPipeline {
                program,
                u_viewport,
                u_pan,
                u_zoom,
                u_color,
                u_edge_type,
                u_acnode,
                u_acnode_radius,
                vao,
                vbo,
                ebo,
            }
        }
    }

    /// Draws a fan of `verts` (already triangulated as a convex polygon,
    /// vertex 0 shared by every triangle) covering the region the caller
    /// wants shaded; each vertex carries its own exact K,L,M value.
    /// `camera` is applied on the GPU: `verts` stay in world space (KLM is
    /// evaluated there), only the final screen position moves.
    /// `acnode`, when present, marks a stray isolated solution of the implicit
    /// equation that the shader must not paint; see `cubic::Acnode`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_fan(
        &self,
        gl: &glow::Context,
        viewport: [f32; 2],
        camera: &Camera,
        verts: &[CubicVertex],
        color: [f32; 4],
        edge_type: EdgeType,
        acnode: Option<crate::cubic::Acnode>,
    ) {
        if verts.len() < 3 {
            return;
        }
        let mut indices: Vec<u16> = Vec::with_capacity((verts.len() - 2) * 3);
        for i in 1..verts.len() as u16 - 1 {
            indices.extend_from_slice(&[0, i, i + 1]);
        }
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_2_f32(Some(&self.u_viewport), viewport[0], viewport[1]);
            gl.uniform_2_f32(Some(&self.u_pan), camera.pan[0], camera.pan[1]);
            gl.uniform_1_f32(Some(&self.u_zoom), camera.zoom);
            gl.uniform_4_f32(Some(&self.u_color), color[0], color[1], color[2], color[3]);
            gl.uniform_1_i32(Some(&self.u_edge_type), edge_type as i32);
            // Same transform the vertex shader applies, so v_screen and
            // u_acnode land in the same space.
            let (ac_pos, ac_radius) = match acnode {
                Some(ac) => (
                    [
                        (ac.point[0] + camera.pan[0]) * camera.zoom,
                        (ac.point[1] + camera.pan[1]) * camera.zoom,
                    ],
                    ac.halo_radius_px(camera.zoom),
                ),
                None => ([0.0, 0.0], 0.0),
            };
            gl.uniform_2_f32(Some(&self.u_acnode), ac_pos[0], ac_pos[1]);
            gl.uniform_1_f32(Some(&self.u_acnode_radius), ac_radius);

            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_of(verts), glow::STREAM_DRAW);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, bytes_of(&indices), glow::STREAM_DRAW);

            gl.draw_elements(glow::TRIANGLES, indices.len() as i32, glow::UNSIGNED_SHORT, 0);
            gl.bind_vertex_array(None);
        }
    }
}

pub struct SolidPipeline {
    program: glow::NativeProgram,
    u_viewport: glow::UniformLocation,
    u_pan: glow::UniformLocation,
    u_zoom: glow::UniformLocation,
    u_color: glow::UniformLocation,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
}

impl SolidPipeline {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link_program(
                gl,
                include_str!("shaders/solid.vert"),
                include_str!("shaders/solid.frag"),
            );
            let u_viewport = gl.get_uniform_location(program, "u_viewportSize").unwrap();
            let u_pan = gl.get_uniform_location(program, "u_pan").unwrap();
            let u_zoom = gl.get_uniform_location(program, "u_zoom").unwrap();
            let u_color = gl.get_uniform_location(program, "u_color").unwrap();

            let vao = gl.create_vertex_array().unwrap();
            let vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
            gl.bind_vertex_array(None);

            SolidPipeline { program, u_viewport, u_pan, u_zoom, u_color, vao, vbo }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        gl: &glow::Context,
        viewport: [f32; 2],
        camera: &Camera,
        mode: u32,
        points: &[[f32; 2]],
        color: [f32; 4],
    ) {
        if points.is_empty() {
            return;
        }
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_2_f32(Some(&self.u_viewport), viewport[0], viewport[1]);
            gl.uniform_2_f32(Some(&self.u_pan), camera.pan[0], camera.pan[1]);
            gl.uniform_1_f32(Some(&self.u_zoom), camera.zoom);
            gl.uniform_4_f32(Some(&self.u_color), color[0], color[1], color[2], color[3]);

            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_of(points), glow::STREAM_DRAW);
            gl.draw_arrays(mode, 0, points.len() as i32);
            gl.bind_vertex_array(None);
        }
    }
}

fn bytes_of<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    }
}
