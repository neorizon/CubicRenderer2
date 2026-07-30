mod cubic;
mod gallery;
mod geometry;
mod gl;
mod interactive;

use std::error::Error;
use std::ffi::CString;
use std::num::NonZeroU32;

use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use glutin::config::{Config, ConfigTemplateBuilder, GetGlConfig};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentContext, PossiblyCurrentContext, Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};

use gallery::GalleryScene;
use interactive::InteractiveScene;

const WINDOW_TITLE: &str = "Loop-Blinn Cubic Renderer";

fn main() -> Result<(), Box<dyn Error>> {
    println!("{WINDOW_TITLE}");
    println!("  Tab      switch Interactive <-> Gallery");
    println!("  drag     move a control point (Interactive)");
    println!("  E        cycle edge type: fill AA / hairline AA / fill no-AA (Interactive)");
    println!("  R        reset the curve (Interactive)");
    println!("  Esc      quit");
    println!();

    let event_loop = EventLoop::new()?;
    let template = ConfigTemplateBuilder::new().with_alpha_size(8).with_multisampling(4);
    let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes()));

    let mut app = App {
        template,
        display_state: DisplayState::Pending(Box::new(display_builder)),
        gl_context: None,
        gl: None,
        surface_state: None,
        scene: None,
        cursor: [0.0, 0.0],
        exit_state: Ok(()),
    };
    event_loop.run_app(&mut app)?;
    app.exit_state
}

fn window_attributes() -> WindowAttributes {
    Window::default_attributes()
        .with_title(WINDOW_TITLE)
        .with_inner_size(winit::dpi::PhysicalSize::new(1200, 800))
}

fn gl_config_picker(configs: Box<dyn Iterator<Item = Config> + '_>) -> Config {
    configs
        .reduce(|best, candidate| if candidate.num_samples() > best.num_samples() { candidate } else { best })
        .expect("no GL configs available")
}

/// Requests, in order: default desktop GL, then GLES, then a plain GL 3.3
/// core context. The shaders themselves are GLSL ES 3.00 throughout, which
/// desktop drivers compile natively.
fn create_gl_context(window: &Window, gl_config: &Config) -> NotCurrentContext {
    let raw_window_handle = window.window_handle().ok().map(|wh| wh.as_raw());
    let default_attrs = ContextAttributesBuilder::new().build(raw_window_handle);
    let gles_attrs =
        ContextAttributesBuilder::new().with_context_api(ContextApi::Gles(None)).build(raw_window_handle);
    let gl33_attrs = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(raw_window_handle);

    let display = gl_config.display();
    unsafe {
        display.create_context(gl_config, &default_attrs).unwrap_or_else(|_| {
            display
                .create_context(gl_config, &gles_attrs)
                .unwrap_or_else(|_| display.create_context(gl_config, &gl33_attrs).expect("no usable GL context"))
        })
    }
}

enum DisplayState {
    Pending(Box<DisplayBuilder>),
    Ready,
}

enum Scene {
    Interactive(InteractiveScene),
    Gallery(GalleryScene),
}

struct Renderer {
    cubic: gl::CubicPipeline,
    solid: gl::SolidPipeline,
}

struct SurfaceState {
    gl_surface: Surface<WindowSurface>,
    window: Window,
}

struct App {
    template: ConfigTemplateBuilder,
    display_state: DisplayState,
    gl_context: Option<PossiblyCurrentContext>,
    gl: Option<glow::Context>,
    surface_state: Option<SurfaceState>,
    scene: Option<(Scene, Renderer)>,
    cursor: [f32; 2],
    exit_state: Result<(), Box<dyn Error>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        let (window, gl_config) = match &self.display_state {
            DisplayState::Pending(builder) => {
                let (window, gl_config) =
                    match builder.clone().build(event_loop, self.template.clone(), gl_config_picker) {
                        Ok((window, gl_config)) => (window.unwrap(), gl_config),
                        Err(err) => {
                            self.exit_state = Err(err);
                            event_loop.exit();
                            return;
                        }
                    };
                self.display_state = DisplayState::Ready;
                self.gl_context = Some(create_gl_context(&window, &gl_config).treat_as_possibly_current());
                (window, gl_config)
            }
            DisplayState::Ready => {
                let gl_config = self.gl_context.as_ref().unwrap().config();
                match glutin_winit::finalize_window(event_loop, window_attributes(), &gl_config) {
                    Ok(window) => (window, gl_config),
                    Err(err) => {
                        self.exit_state = Err(err.into());
                        event_loop.exit();
                        return;
                    }
                }
            }
        };

        let attrs = window
            .build_surface_attributes(glutin::surface::SurfaceAttributesBuilder::new())
            .expect("failed to build surface attributes");
        let gl_surface = unsafe { gl_config.display().create_window_surface(&gl_config, &attrs).unwrap() };

        let gl_context = self.gl_context.as_ref().unwrap();
        gl_context.make_current(&gl_surface).unwrap();
        if let Err(err) =
            gl_surface.set_swap_interval(gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
        {
            eprintln!("warning: vsync not available: {err:?}");
        }

        if self.gl.is_none() {
            let gl = unsafe {
                glow::Context::from_loader_function(|name| {
                    let cname = CString::new(name).unwrap();
                    gl_config.display().get_proc_address(cname.as_c_str()).cast()
                })
            };
            self.gl = Some(gl);
        }

        if self.scene.is_none() {
            let gl = self.gl.as_ref().unwrap();
            let renderer = Renderer { cubic: gl::CubicPipeline::new(gl), solid: gl::SolidPipeline::new(gl) };
            let gallery = GalleryScene::new();
            gallery.print_legend();
            println!();
            self.scene = Some((Scene::Gallery(gallery), renderer));
        }

        self.surface_state = Some(SurfaceState { gl_surface, window });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width != 0 && size.height != 0 => {
                if let (Some(surface), Some(ctx)) = (self.surface_state.as_ref(), self.gl_context.as_ref()) {
                    surface.gl_surface.resize(
                        ctx,
                        NonZeroU32::new(size.width).unwrap(),
                        NonZeroU32::new(size.height).unwrap(),
                    );
                }
                if let Some(gl) = self.gl.as_ref() {
                    unsafe {
                        use glow::HasContext;
                        gl.viewport(0, 0, size.width as i32, size.height as i32);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = [position.x as f32, position.y as f32];
                if let Some((Scene::Interactive(scene), _)) = self.scene.as_mut() {
                    scene.on_mouse_move(self.cursor);
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                if let Some((Scene::Interactive(scene), _)) = self.scene.as_mut() {
                    match state {
                        ElementState::Pressed => scene.on_mouse_down(self.cursor),
                        ElementState::Released => scene.on_mouse_up(),
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, logical_key, .. },
                ..
            } => self.handle_key(logical_key, event_loop),
            WindowEvent::RedrawRequested => self.draw(),
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(surface) = self.surface_state.as_ref() {
            surface.window.request_redraw();
        }
    }
}

impl App {
    fn viewport_size(&self) -> [f32; 2] {
        self.surface_state
            .as_ref()
            .map(|s| {
                let size = s.window.inner_size();
                [size.width as f32, size.height as f32]
            })
            .unwrap_or([1200.0, 800.0])
    }

    fn handle_key(&mut self, key: Key, event_loop: &ActiveEventLoop) {
        match key {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::Tab) => self.toggle_scene(),
            Key::Character(s) => match s.as_str() {
                "e" | "E" => {
                    if let Some((Scene::Interactive(scene), _)) = self.scene.as_mut() {
                        scene.cycle_edge_type();
                    }
                }
                "r" | "R" => {
                    let viewport = self.viewport_size();
                    if let Some((Scene::Interactive(scene), _)) = self.scene.as_mut() {
                        scene.reset(viewport);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn toggle_scene(&mut self) {
        let viewport = self.viewport_size();
        if let Some((scene, _)) = self.scene.as_mut() {
            *scene = match scene {
                Scene::Interactive(_) => Scene::Gallery(GalleryScene::new()),
                Scene::Gallery(_) => Scene::Interactive(InteractiveScene::new(viewport)),
            };
        }
    }

    fn draw(&mut self) {
        let (Some(surface), Some(ctx), Some(gl), Some((scene, renderer))) =
            (self.surface_state.as_ref(), self.gl_context.as_ref(), self.gl.as_ref(), self.scene.as_ref())
        else {
            return;
        };

        unsafe {
            use glow::HasContext;
            gl.clear_color(0.10, 0.11, 0.14, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        }

        let viewport = self.viewport_size();
        match scene {
            Scene::Interactive(s) => {
                s.render(gl, &renderer.cubic, &renderer.solid, viewport);
                surface.window.set_title(&format!("{WINDOW_TITLE} — {}", s.status_line()));
            }
            Scene::Gallery(g) => {
                g.render(gl, &renderer.cubic, &renderer.solid, viewport);
                surface.window.set_title(&format!("{WINDOW_TITLE} — Gallery (Tab=Interactive)"));
            }
        }

        surface.gl_surface.swap_buffers(ctx).unwrap();
    }
}
