//! The gallery: every Loop-Blinn classification, laid out in a grid, none of
//! it interactive -- a static reference for what each curve type looks like.

use crate::cubic;
use crate::geometry;
use crate::gl::{CubicPipeline, EdgeType, SolidPipeline};

const COLS: usize = 3;
const ROWS: usize = 2;
const CELL_FILL_FRACTION: f32 = 0.55;
const BORDER_INSET: f32 = 10.0;

pub struct GalleryScene;

impl GalleryScene {
    pub fn new() -> Self {
        GalleryScene
    }

    pub fn print_legend(&self) {
        println!("Gallery — one cell per Loop-Blinn cubic classification:");
        for (kind, _) in cubic::preset_curves() {
            let [r, g, b] = kind.accent_color();
            println!(
                "  {:<10} accent rgb=({:.2}, {:.2}, {:.2})",
                kind.name(),
                r,
                g,
                b
            );
        }
    }

    pub fn render(
        &self,
        gl: &glow::Context,
        cubic_pipe: &CubicPipeline,
        solid_pipe: &SolidPipeline,
        viewport: [f32; 2],
    ) {
        let cell_size = [viewport[0] / COLS as f32, viewport[1] / ROWS as f32];

        for (idx, (kind, preset)) in cubic::preset_curves().into_iter().enumerate() {
            let col = (idx % COLS) as f32;
            let row = (idx / COLS) as f32;
            let origin = [col * cell_size[0], row * cell_size[1]];

            let [r, g, b] = kind.accent_color();
            draw_cell_border(gl, solid_pipe, viewport, origin, cell_size, [r, g, b, 0.5]);

            let points = geometry::fit_points(&preset, origin, cell_size, CELL_FILL_FRACTION);
            let implicit = cubic::compute_implicit(&points);

            if implicit.is_renderable() {
                let quad = geometry::coverage_quad(&points, &implicit, 4.0);
                cubic_pipe.draw_fan(gl, viewport, &quad, [r, g, b, 0.15], EdgeType::FillAA);
                cubic_pipe.draw_fan(gl, viewport, &quad, [r, g, b, 0.95], EdgeType::HairlineAA);
            }

            for edge in points.windows(2) {
                let line = geometry::thick_line(edge[0], edge[1], 0.75);
                solid_pipe.draw(gl, viewport, glow::TRIANGLE_FAN, &line, [0.55, 0.55, 0.62, 0.85]);
            }
            for p in points.iter() {
                let disc = geometry::circle_fan(*p, 4.5, 16);
                solid_pipe.draw(gl, viewport, glow::TRIANGLE_FAN, &disc, [0.9, 0.9, 0.92, 1.0]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_cell_border(
    gl: &glow::Context,
    solid_pipe: &SolidPipeline,
    viewport: [f32; 2],
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
        solid_pipe.draw(gl, viewport, glow::TRIANGLE_FAN, &line, color);
    }
}
