//! The gallery: every Loop-Blinn classification (plus every distinct
//! Loop-chop case) laid out in a grid. Clicking a cell jumps into Interactive
//! mode loaded with that exact curve -- see `App::handle_left_click` in
//! `main.rs`.

use crate::camera::Camera;
use crate::cubic::{self, Pt};
use crate::geometry;
use crate::gl::{CubicPipeline, EdgeType, SolidPipeline};

const COLS: usize = 5;
const CELL_FILL_FRACTION: f32 = 0.55;
const BORDER_INSET: f32 = 10.0;

pub struct GalleryScene {
    pub camera: Camera,
}

impl GalleryScene {
    pub fn new() -> Self {
        GalleryScene { camera: Camera::default() }
    }

    fn rows() -> usize {
        let n = cubic::preset_curves().len();
        n.div_ceil(COLS)
    }

    fn cell_size(viewport: [f32; 2]) -> [f32; 2] {
        [viewport[0] / COLS as f32, viewport[1] / Self::rows() as f32]
    }

    /// Which preset (if any) `cursor` falls over, for click-to-Interactive.
    /// `cursor` is a screen position; converted through the camera before
    /// the grid math, so panning/zooming the gallery doesn't break clicks.
    pub fn cell_at(&self, cursor: Pt, viewport: [f32; 2]) -> Option<usize> {
        let world = self.camera.to_world(cursor);
        let cell_size = Self::cell_size(viewport);
        if world[0] < 0.0 || world[1] < 0.0 {
            return None;
        }
        let col = (world[0] / cell_size[0]) as usize;
        let row = (world[1] / cell_size[1]) as usize;
        if col >= COLS || row >= Self::rows() {
            return None;
        }
        let idx = row * COLS + col;
        (idx < cubic::preset_curves().len()).then_some(idx)
    }

    pub fn print_legend(&self) {
        println!("Gallery — one cell per Loop-Blinn cubic classification (click a cell to edit it):");
        for (name, kind, _) in cubic::preset_curves() {
            let [r, g, b] = kind.accent_color();
            println!("  {:<24} accent rgb=({:.2}, {:.2}, {:.2})", name, r, g, b);
        }
    }

    pub fn render(
        &self,
        gl: &glow::Context,
        cubic_pipe: &CubicPipeline,
        solid_pipe: &SolidPipeline,
        viewport: [f32; 2],
    ) {
        let cell_size = Self::cell_size(viewport);

        for (idx, (_, kind, preset)) in cubic::preset_curves().into_iter().enumerate() {
            let col = (idx % COLS) as f32;
            let row = (idx / COLS) as f32;
            let origin = [col * cell_size[0], row * cell_size[1]];

            let [r, g, b] = kind.accent_color();
            draw_cell_border(gl, solid_pipe, viewport, &self.camera, origin, cell_size, [r, g, b, 0.5]);

            let points = geometry::fit_points(&preset, origin, cell_size, CELL_FILL_FRACTION);

            // See interactive.rs: Loop curves are split at their
            // self-intersection so no mesh straddles the singular node.
            for piece in cubic::chop_at_loop_intersection(&points) {
                if !piece.implicit.is_renderable() {
                    continue;
                }
                let quad = geometry::coverage_mesh(&piece.points, &piece.implicit, 1.5);
                let acnode = cubic::acnode(&piece.points);
                cubic_pipe.draw_fan(gl, viewport, &self.camera, &quad, [r, g, b, 0.15], EdgeType::FillAA, acnode);
                cubic_pipe.draw_fan(gl, viewport, &self.camera, &quad, [r, g, b, 0.95], EdgeType::HairlineAA, acnode);
            }

            for edge in points.windows(2) {
                let line = geometry::thick_line(edge[0], edge[1], 0.75);
                solid_pipe.draw(gl, viewport, &self.camera, glow::TRIANGLE_FAN, &line, [0.55, 0.55, 0.62, 0.85]);
            }
            for p in points.iter() {
                let disc = geometry::circle_fan(*p, 4.5, 16);
                solid_pipe.draw(gl, viewport, &self.camera, glow::TRIANGLE_FAN, &disc, [0.9, 0.9, 0.92, 1.0]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_cell_border(
    gl: &glow::Context,
    solid_pipe: &SolidPipeline,
    viewport: [f32; 2],
    camera: &Camera,
    origin: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
) {
    let (x0, y0) = (origin[0] + BORDER_INSET, origin[1] + BORDER_INSET);
    let (x1, y1) = (origin[0] + size[0] - BORDER_INSET, origin[1] + size[1] - BORDER_INSET);
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let line = geometry::thick_line(a, b, 1.0);
        solid_pipe.draw(gl, viewport, camera, glow::TRIANGLE_FAN, &line, color);
    }
}
