//! The playground: drag the 4 control points of a single cubic and watch its
//! Loop-Blinn classification and AA coverage update live.

use crate::cubic::{self, Pt};
use crate::geometry;
use crate::gl::{CubicPipeline, EdgeType, SolidPipeline};

const HANDLE_RADIUS: f32 = 8.0;
const HIT_RADIUS: f32 = 16.0;
const HANDLE_COLORS: [[f32; 4]; 4] =
    [[0.30, 1.00, 0.45, 1.0], [0.35, 0.80, 1.00, 1.0], [0.35, 0.80, 1.00, 1.0], [1.00, 0.35, 0.35, 1.0]];

pub struct InteractiveScene {
    pub points: [Pt; 4],
    pub edge_type: EdgeType,
    dragging: Option<usize>,
}

impl InteractiveScene {
    pub fn new(viewport: [f32; 2]) -> Self {
        InteractiveScene { points: default_points(viewport), edge_type: EdgeType::HairlineAA, dragging: None }
    }

    pub fn reset(&mut self, viewport: [f32; 2]) {
        self.points = default_points(viewport);
    }

    pub fn cycle_edge_type(&mut self) {
        self.edge_type = self.edge_type.cycle();
    }

    pub fn on_mouse_down(&mut self, cursor: Pt) {
        self.dragging = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dist(*p, cursor)))
            .filter(|(_, d)| *d <= HIT_RADIUS)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
    }

    pub fn on_mouse_move(&mut self, cursor: Pt) {
        if let Some(i) = self.dragging {
            self.points[i] = cursor;
        }
    }

    pub fn on_mouse_up(&mut self) {
        self.dragging = None;
    }

    pub fn status_line(&self) -> String {
        let c = cubic::compute_implicit(&self.points);
        format!(
            "Interactive — {}  (d0={:.2} d1={:.2} d2={:.2} discr={:.2})  edge={}  \
             — drag points, E=edge type, R=reset, Tab=Gallery",
            c.kind.name(),
            c.d[0],
            c.d[1],
            c.d[2],
            c.discriminant,
            self.edge_type.name()
        )
    }

    pub fn render(
        &self,
        gl: &glow::Context,
        cubic_pipe: &CubicPipeline,
        solid_pipe: &SolidPipeline,
        viewport: [f32; 2],
    ) {
        let implicit = cubic::compute_implicit(&self.points);

        if implicit.is_renderable() {
            let quad = geometry::coverage_quad(&self.points, &implicit, 6.0);
            let [r, g, b] = implicit.kind.accent_color();
            // A faint fill wash shows the implicit region even when the
            // primary edge type is a thin hairline.
            if self.edge_type != EdgeType::FillAA {
                cubic_pipe.draw_fan(gl, viewport, &quad, [r, g, b, 0.15], EdgeType::FillAA);
            }
            cubic_pipe.draw_fan(gl, viewport, &quad, [r, g, b, 0.95], self.edge_type);
        }

        for edge in self.points.windows(2) {
            let line = geometry::thick_line(edge[0], edge[1], 1.0);
            solid_pipe.draw(gl, viewport, glow::TRIANGLE_FAN, &line, [0.55, 0.55, 0.62, 0.9]);
        }

        for (i, p) in self.points.iter().enumerate() {
            let disc = geometry::circle_fan(*p, HANDLE_RADIUS, 24);
            solid_pipe.draw(gl, viewport, glow::TRIANGLE_FAN, &disc, HANDLE_COLORS[i]);
        }
    }
}

fn dist(a: Pt, b: Pt) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// The Serpentine preset, scaled and centered into whatever viewport we're
/// given (with a margin so control points never start flush against an
/// edge).
fn default_points(viewport: [f32; 2]) -> [Pt; 4] {
    let (_, preset) = cubic::preset_curves()[0];
    geometry::fit_points(&preset, [0.0, 0.0], viewport, 0.6)
}
