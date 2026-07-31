//! A simple 2D screen-space camera: uniform zoom plus pan, applied entirely
//! on the GPU (the vertex shaders take world-space positions and this
//! camera's `u_pan`/`u_zoom` uniforms). CPU-side geometry -- Bezier math,
//! KLM functionals, hit testing -- never has to know about the camera; it
//! all stays in one consistent "world" coordinate space that's identical to
//! plain screen-pixel space whenever the camera is at its default (zoom=1,
//! pan=[0,0]), exactly reproducing pre-camera behavior.

pub type Pt = [f32; 2];

const MIN_ZOOM: f32 = 0.05;
const MAX_ZOOM: f32 = 50.0;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub zoom: f32,
    pub pan: Pt,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { zoom: 1.0, pan: [0.0, 0.0] }
    }
}

impl Camera {
    /// screen -> world, the inverse of what the vertex shaders compute
    /// (`screen = (world + pan) * zoom`). Used to convert cursor positions
    /// before hit-testing control points or gallery cells.
    pub fn to_world(self, screen: Pt) -> Pt {
        [screen[0] / self.zoom - self.pan[0], screen[1] / self.zoom - self.pan[1]]
    }

    /// Multiplies zoom by `factor` (clamped), keeping the world point
    /// currently under `screen_cursor` fixed on screen -- i.e. zoom centers
    /// on the cursor rather than the viewport origin.
    pub fn zoom_at(&mut self, screen_cursor: Pt, factor: f32) {
        let world = self.to_world(screen_cursor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan = [screen_cursor[0] / self.zoom - world[0], screen_cursor[1] / self.zoom - world[1]];
    }

    /// Shifts the view by a screen-space drag delta (e.g. right-mouse-drag).
    pub fn pan_by(&mut self, screen_delta: Pt) {
        self.pan[0] += screen_delta[0] / self.zoom;
        self.pan[1] += screen_delta[1] / self.zoom;
    }

    pub fn reset(&mut self) {
        *self = Camera::default();
    }
}
