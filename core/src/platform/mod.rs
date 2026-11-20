//! The platform layer's API-neutral contract: the window, the event queue
//! and the native surface the render backends consume. The same rule the
//! render contract carries: the shared module never contains platform-API
//! source — the per-platform implementations live in their own library
//! files, and the contract's types carry only the data the app logic and
//! the render backends need.
//!
//! The milestone 9 rule the render contract's headless rule extends: the
//! platform layer compiles everywhere (the trait and the types are pure
//! Rust), the per-platform backends gate behind their own features, and
//! the render backends consume the native surface's raw handles through
//! the `NativeSurface` enum's platform-tagged variants.

/// The window's creation description. The platform backend validates the
/// dimensions (positive) and carries the title the window manager shows.
#[derive(Clone, Debug)]
pub struct WindowDesc {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Whether the window manager may resize the window (the event
    /// queue's Resize events carry the actual size either way).
    pub resizable: bool,
}

impl Default for WindowDesc {
    fn default() -> Self {
        WindowDesc {
            title: "OpenGraceII".to_string(),
            width: 1280,
            height: 720,
            resizable: true,
        }
    }
}

/// The mouse button a press/release event names. The wheel is not a
/// button: the platform backends report it as the `MouseWheel` event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    X1,
    X2,
}

/// The layout-agnostic key a press/release event names. The letters are
/// the uppercase form (the platform backends normalise the layout's own
/// case), the digits carry their value, and the modifiers are separate
/// keys the app logic tracks itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Escape,
    Enter,
    Space,
    Tab,
    Backspace,
    Delete,
    Insert,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Letter(char),
    Digit(u8),
    Shift,
    Control,
    Alt,
    /// The key the backend's map does not name (the layout's other
    /// keysyms — the app logic reads the ones it plans for).
    Unknown,
}

impl Key {
    /// The key's stable name (the docs' own wording).
    pub fn name(&self) -> String {
        match self {
            Key::Letter(c) => format!("Letter({})", c),
            Key::Digit(d) => format!("Digit({})", d),
            other => format!("{:?}", other),
        }
    }
}

/// The window's event queue: the platform backends translate the native
/// event stream into these. The app logic polls the queue each frame and
/// consumes the events in order — the queue is never reordered.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The window manager or the user asked to close the window (the
    /// WM_DELETE_WINDOW protocol on X11, the WM_CLOSE on Win32, the
    /// windowShouldClose on Cocoa).
    Close,
    /// The window manager resized the window (the actual size).
    Resize { width: u32, height: u32 },
    FocusGained,
    FocusLost,
    KeyDown { key: Key },
    KeyUp { key: Key },
    MouseMove { x: i32, y: i32 },
    MouseDown { button: MouseButton, x: i32, y: i32 },
    MouseUp { button: MouseButton, x: i32, y: i32 },
    /// The mouse wheel stepped. Positive is up (away from the user).
    MouseWheel { delta: i32 },
}

/// The native surface the render backends consume: the raw platform
/// handles the window carries, tagged by the platform the backend needs.
/// The render backends match the variant they serve and reject the rest
/// (the headless one carries no surface at all).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NativeSurface {
    /// The X11 window: the display connection and the window's XID.
    X11 { display: *mut std::os::raw::c_void, window: u64 },
    /// No surface: the headless render path (the off-screen target the
    /// milestones 2-8 validated).
    Headless,
}

unsafe impl Send for NativeSurface {}

/// The platform layer's API-neutral contract. The same shape the render
/// contract carries: one trait, deterministic translations, and the
/// protocol rules the per-platform implementations follow.
pub trait PlatformBackend {
    /// Human-readable backend description, for example "x11 (X11 1.8)".
    fn describe(&self) -> String;
    /// Drains the native event queue and translates it into the
    /// contract's events. Non-blocking: when the native queue is empty,
    /// returns an empty vector (the app's frame loop keeps running).
    fn poll_events(&mut self) -> Vec<Event>;
    /// Window's current size in pixels.
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    /// Requests the window manager to size the window. The platform's
    /// own confirmation arrives as a Resize event.
    fn resize(&mut self, width: u32, height: u32);
    /// The native surface the render backends consume.
    fn native_surface(&self) -> NativeSurface;
    /// Releases the platform resources. Exactly one successful call per
    /// backend lifetime; calls after shutdown are protocol violations.
    fn shutdown(&mut self);
    /// Number of protocol violations observed so far.
    fn violation_count(&self) -> u64;
}

#[cfg(all(unix, feature = "x11"))]
pub mod x11;
