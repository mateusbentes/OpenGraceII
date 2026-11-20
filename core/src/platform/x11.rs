//! The X11 platform backend: the engine's own Xlib loader, no third-party
//! binding (the same rule the render backends' loaders carry). The module
//! dlopen()s the platform's libX11.so.6 at runtime and resolves the entry
//! points the windowing path needs.
//!
//! The event loop uses Xlib's own event queue: XPending + XNextEvent on
//! the display connection, translated into the contract's events. The
//! window close follows the WM_DELETE_WINDOW protocol the window manager
//! sends as a ClientMessage. The keymap normalises the layout's keysyms
//! onto the contract's Key names.

use std::os::raw::{c_char, c_int, c_uint, c_ulong, c_void};
use super::{
    Event, Key, MouseButton, NativeSurface, PlatformBackend,
};

// ---------------------------------------------------------------------------
// The Xlib types (the published C layouts)
// ---------------------------------------------------------------------------

pub type Display = c_void;
pub type Window = c_ulong;
pub type XID = c_ulong;
pub type Bool = c_int;
pub type Status = c_int;
pub type XPointer = *mut c_char;

// The event types (X.h's own values).
pub const X_KEY_PRESS: c_int = 2;
pub const X_KEY_RELEASE: c_int = 3;
pub const X_BUTTON_PRESS: c_int = 4;
pub const X_BUTTON_RELEASE: c_int = 5;
pub const X_MOTION_NOTIFY: c_int = 6;
pub const X_CONFIGURE_NOTIFY: c_int = 22;
pub const X_CLIENT_MESSAGE: c_int = 33;
pub const X_FOCUS_IN: c_int = 9;
pub const X_FOCUS_OUT: c_int = 10;
pub const X_DESTROY_NOTIFY: c_int = 17;

// The event masks (X.h's own values).
pub const X_KEY_PRESS_MASK: c_ulong = 0x1;
pub const X_KEY_RELEASE_MASK: c_ulong = 0x2;
pub const X_BUTTON_PRESS_MASK: c_ulong = 0x4;
pub const X_BUTTON_RELEASE_MASK: c_ulong = 0x8;
pub const X_POINTER_MOTION_MASK: c_ulong = 0x40;
pub const X_STRUCTURE_NOTIFY_MASK: c_ulong = 0x20000;
pub const X_FOCUS_CHANGE_MASK: c_ulong = 0x400000;

// The event masks the window selects: the keyboard, the mouse, the
// structure (the resize's ConfigureNotify) and the focus.
pub const X_EVENT_MASK: c_ulong = X_KEY_PRESS_MASK
    | X_KEY_RELEASE_MASK
    | X_BUTTON_PRESS_MASK
    | X_BUTTON_RELEASE_MASK
    | X_POINTER_MOTION_MASK
    | X_STRUCTURE_NOTIFY_MASK
    | X_FOCUS_CHANGE_MASK;

// The input focus (X.h's own values).
pub const X_INPUT_OUTPUT: c_int = 1;

// The window class (X.h's own values).
pub const X_INPUT_OUTPUT_CLASS: c_int = 1;

// The client message's format (X.h's own values).
pub const X_WM_DELETE_WINDOW: c_ulong = 0;

// The keysym group masks (X.h's own values).
pub const X_SHIFT_MASK: c_uint = 0x1;
pub const X_CONTROL_MASK: c_uint = 0x4;
pub const X_MOD1_MASK: c_uint = 0x8;
pub const X_MOD4_MASK: c_uint = 0x40;

// The keysym base values (X.h's own values). The keysym's own values
// range 0xff00-0xffff for the function keys, 0x20-0x7e for the printable
// ASCII, and 0x30-0x39 for the digits.
pub const XKS_BACKSPACE: c_ulong = 0xff08;
pub const XKS_TAB: c_ulong = 0xff09;
pub const XKS_RETURN: c_ulong = 0xff0d;
pub const XKS_ESCAPE: c_ulong = 0xff1b;
// The space's keysym is the printable ASCII's own value (0x20).
pub const XKS_SPACE: c_ulong = 0x0020;
pub const XKS_DELETE: c_ulong = 0xffff;
pub const XKS_INSERT: c_ulong = 0xff63;
pub const XKS_HOME: c_ulong = 0xff50;
pub const XKS_END: c_ulong = 0xff57;
pub const XKS_PAGE_UP: c_ulong = 0xff55;
pub const XKS_PAGE_DOWN: c_ulong = 0xff56;
pub const XKS_LEFT: c_ulong = 0xff51;
pub const XKS_UP: c_ulong = 0xff52;
pub const XKS_RIGHT: c_ulong = 0xff53;
pub const XKS_DOWN: c_ulong = 0xff54;
pub const XKS_SHIFT_L: c_ulong = 0xffe1;
pub const XKS_SHIFT_R: c_ulong = 0xffe2;
pub const XKS_CONTROL_L: c_ulong = 0xffe3;
pub const XKS_CONTROL_R: c_ulong = 0xffe4;
pub const XKS_ALT_L: c_ulong = 0xffe9;
pub const XKS_ALT_R: c_ulong = 0xffe7;
pub const XKS_F1: c_ulong = 0xffbe;
pub const XKS_F12: c_ulong = 0xffc9;

// The button values (X.h's own values).
pub const X_BUTTON1: c_uint = 1;
pub const X_BUTTON2: c_uint = 2;
pub const X_BUTTON3: c_uint = 3;
pub const X_BUTTON4: c_uint = 4;
pub const X_BUTTON5: c_uint = 5;
pub const X_BUTTON6: c_uint = 6;
pub const X_BUTTON7: c_uint = 7;
pub const X_BUTTON8: c_uint = 8;
pub const X_BUTTON9: c_uint = 9;

// The XAnyEvent's layout: the type (c_int), the serial (c_ulong), the
// send_event (Bool), the display pointer, the window (Window) — the
// header's first five fields every event type carries. The total XEvent
// is 192 bytes on the header's layout.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct XAnyEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XKeyEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    root: Window,
    subwindow: Window,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    keycode: c_uint,
    same_screen: Bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XButtonEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    root: Window,
    subwindow: Window,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    button: c_uint,
    same_screen: Bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XMotionEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    root: Window,
    subwindow: Window,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    is_hint: c_char,
    same_screen: Bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XFocusChangeEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    mode: c_int,
    detail: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XConfigureEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,
    border_width: c_int,
    above: Window,
    override_redirect: Bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XClientMessageEvent {
    event_type: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    message_type: XID,
    format: c_int,
    data: [c_ulong; 5],
}

/// The union's own layout: the largest member (the XKeyEvent's 32 bytes
/// of payload plus the 24-byte header the XAnyEvent carries) bounds the
/// event. The layout check's C ground truth (the header's sizeof(XEvent))
/// pinned the union at 192 bytes on the platform's layout.
#[repr(C)]
union XEvent {
    event_type: c_int,
    key: XKeyEvent,
    button: XButtonEvent,
    motion: XMotionEvent,
    focus: XFocusChangeEvent,
    configure: XConfigureEvent,
    client_message: XClientMessageEvent,
    pad: [c_char; 192],
}

#[repr(C)]
#[allow(dead_code)]
struct XSetWindowAttributes {
    background_pixmap: XID,
    background_pixel: c_ulong,
    border_pixmap: XID,
    border_pixel: c_ulong,
    bit_gravity: c_int,
    win_gravity: c_int,
    backing_store: c_int,
    backing_planes: c_ulong,
    backing_pixel: c_ulong,
    save_under: Bool,
    event_mask: c_ulong,
    do_not_propagate_mask: c_ulong,
    override_redirect: Bool,
    colormap: XID,
    cursor: XID,
}

// ---------------------------------------------------------------------------
// The Xlib entry points (the engine's own loader's surface)
// ---------------------------------------------------------------------------

type FnOpenDisplay = unsafe extern "C" fn(*const c_char) -> *mut Display;
type FnCloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type FnDefaultScreen = unsafe extern "C" fn(*mut Display) -> c_int;
type FnRootWindow = unsafe extern "C" fn(*mut Display, c_int) -> Window;
type FnCreateSimpleWindow = unsafe extern "C" fn(
    *mut Display,
    Window,
    c_int,
    c_int,
    c_uint,
    c_uint,
    c_uint,
    c_ulong,
    c_ulong,
) -> Window;
type FnSelectInput = unsafe extern "C" fn(*mut Display, Window, c_ulong) -> c_int;
type FnMapWindow = unsafe extern "C" fn(*mut Display, Window) -> c_int;
type FnDestroyWindow = unsafe extern "C" fn(*mut Display, Window) -> c_int;
type FnStoreName = unsafe extern "C" fn(*mut Display, Window, *const c_char) -> c_int;
type FnPending = unsafe extern "C" fn(*mut Display) -> c_int;
type FnNextEvent = unsafe extern "C" fn(*mut Display, *mut XEvent) -> c_int;
type FnFlush = unsafe extern "C" fn(*mut Display) -> c_int;
type FnSync = unsafe extern "C" fn(*mut Display) -> c_int;
type FnInternAtom = unsafe extern "C" fn(
    *mut Display,
    *const c_char,
    Bool,
) -> XID;
type FnSetWMProtocols = unsafe extern "C" fn(
    *mut Display,
    Window,
    // The protocols is a POINTER to the atoms' array (the Xlib's own
    // signature: the value dereference the earlier call's segfault
    // carried).
    *const XID,
    c_int,
) -> Status;
type FnResizeWindow = unsafe extern "C" fn(
    *mut Display,
    Window,
    c_uint,
    c_uint,
) -> c_int;
type FnLookupKeysym = unsafe extern "C" fn(*mut XKeyEvent, c_int) -> c_ulong;
type FnMoveResizeWindow = unsafe extern "C" fn(
    *mut Display,
    Window,
    c_int,
    c_int,
    c_uint,
    c_uint,
) -> c_int;

/// The Xlib loader: the platform's libX11.so.6 dlopen()ed at runtime.
struct X11Loader {
    library: *mut c_void,
    open_display: Option<FnOpenDisplay>,
    close_display: Option<FnCloseDisplay>,
    default_screen: Option<FnDefaultScreen>,
    root_window: Option<FnRootWindow>,
    create_simple_window: Option<FnCreateSimpleWindow>,
    select_input: Option<FnSelectInput>,
    map_window: Option<FnMapWindow>,
    destroy_window: Option<FnDestroyWindow>,
    store_name: Option<FnStoreName>,
    pending: Option<FnPending>,
    next_event: Option<FnNextEvent>,
    flush: Option<FnFlush>,
    sync: Option<FnSync>,
    intern_atom: Option<FnInternAtom>,
    set_wm_protocols: Option<FnSetWMProtocols>,
    resize_window: Option<FnResizeWindow>,
    lookup_keysym: Option<FnLookupKeysym>,
    move_resize_window: Option<FnMoveResizeWindow>,
}

#[cfg(windows)]
unsafe fn load_library() -> Result<*mut c_void, String> {
    Err("X11Backend: the platform's libX11 does not exist on Windows".into())
}

#[cfg(not(windows))]
unsafe fn load_library() -> Result<*mut c_void, String> {
    let names = [
        b"libX11.so.6\0".as_ref(),
        b"libX11.so\0".as_ref(),
    ];
    for name in names {
        let handle = libc::dlopen(name.as_ptr() as *const _, libc::RTLD_NOW);
        if !handle.is_null() {
            return Ok(handle);
        }
    }
    Err("X11Backend: the platform's libX11.so.6 could not be loaded".into())
}

impl X11Loader {
    fn new() -> Result<Self, String> {
        let library = unsafe { load_library() }?;
        let mut missing = String::new();
        let mut loader = X11Loader {
            library,
            open_display: None,
            close_display: None,
            default_screen: None,
            root_window: None,
            create_simple_window: None,
            select_input: None,
            map_window: None,
            destroy_window: None,
            store_name: None,
            pending: None,
            next_event: None,
            flush: None,
            sync: None,
            intern_atom: None,
            set_wm_protocols: None,
            resize_window: None,
            lookup_keysym: None,
            move_resize_window: None,
        };
        macro_rules! sym {
            ($field:ident, $name:literal, $ty:ty) => {
                let symbol = unsafe {
                    libc::dlsym(
                        library,
                        concat!($name, "\0").as_ptr() as *const _,
                    )
                };
                if symbol.is_null() {
                    missing.push_str(concat!($name, " "));
                } else {
                    loader.$field = Some(unsafe {
                        std::mem::transmute::<*mut c_void, $ty>(symbol)
                    });
                }
            };
        }
        sym!(open_display, "XOpenDisplay", FnOpenDisplay);
        sym!(close_display, "XCloseDisplay", FnCloseDisplay);
        sym!(default_screen, "XDefaultScreen", FnDefaultScreen);
        sym!(root_window, "XRootWindow", FnRootWindow);
        sym!(create_simple_window, "XCreateSimpleWindow", FnCreateSimpleWindow);
        sym!(select_input, "XSelectInput", FnSelectInput);
        sym!(map_window, "XMapWindow", FnMapWindow);
        sym!(destroy_window, "XDestroyWindow", FnDestroyWindow);
        sym!(store_name, "XStoreName", FnStoreName);
        sym!(pending, "XPending", FnPending);
        sym!(next_event, "XNextEvent", FnNextEvent);
        sym!(flush, "XFlush", FnFlush);
        sym!(sync, "XSync", FnSync);
        sym!(intern_atom, "XInternAtom", FnInternAtom);
        sym!(set_wm_protocols, "XSetWMProtocols", FnSetWMProtocols);
        sym!(resize_window, "XResizeWindow", FnResizeWindow);
        sym!(lookup_keysym, "XLookupKeysym", FnLookupKeysym);
        sym!(move_resize_window, "XMoveResizeWindow", FnMoveResizeWindow);
        if !missing.is_empty() {
            unsafe { libc::dlclose(library) };
            return Err(format!(
                "X11Backend: the platform's libX11.so.6 does not export {}",
                missing.trim_end()
            ));
        }
        Ok(loader)
    }
}

/// The keysym's contract Key: the layout's normalisation the keymap
/// carries. The printable ASCII keysyms (0x20-0x7e) map to their own
/// value; the function keys map to the contract's named keys.
fn key_for_keysym(keysym: c_ulong) -> Key {
    match keysym {
        XKS_ESCAPE => Key::Escape,
        XKS_RETURN => Key::Enter,
        XKS_SPACE => Key::Space,
        XKS_TAB => Key::Tab,
        XKS_BACKSPACE => Key::Backspace,
        XKS_DELETE => Key::Delete,
        XKS_INSERT => Key::Insert,
        XKS_LEFT => Key::Left,
        XKS_RIGHT => Key::Right,
        XKS_UP => Key::Up,
        XKS_DOWN => Key::Down,
        XKS_PAGE_UP => Key::PageUp,
        XKS_PAGE_DOWN => Key::PageDown,
        XKS_HOME => Key::Home,
        XKS_END => Key::End,
        XKS_SHIFT_L | XKS_SHIFT_R => Key::Shift,
        XKS_CONTROL_L | XKS_CONTROL_R => Key::Control,
        XKS_ALT_L | XKS_ALT_R => Key::Alt,
        // The function keys' keysym range 0xffbe (F1) through 0xffc9
        // (F12): the index the range carries, in order.
        k if (0xffbe..=0xffc9).contains(&k) => {
            const F_KEYS: [Key; 12] = [
                Key::F1, Key::F2, Key::F3, Key::F4, Key::F5, Key::F6,
                Key::F7, Key::F8, Key::F9, Key::F10, Key::F11, Key::F12,
            ];
            F_KEYS[(k - 0xffbe) as usize]
        }
        _ => {
            let code = keysym as u32;
            if (0x30..=0x39).contains(&code) {
                Key::Digit((code - 0x30) as u8)
            } else if (0x41..=0x5a).contains(&code) {
                Key::Letter(code as u8 as char)
            } else if (0x61..=0x7a).contains(&code) {
                Key::Letter((code - 0x20) as u8 as char)
            } else {
                Key::Unknown
            }
        }
    }
}

/// The X11 platform backend: the engine's own Xlib loader, the window
/// the create path establishes, and the event loop's translations.
pub struct X11Backend {
    loader: X11Loader,
    display: *mut Display,
    window: Window,
    delete_atom: XID,
    width: u32,
    height: u32,
    shutdown: bool,
    violations: u64,
    /// The shutdown path owns the close: the display's XCloseDisplay and
    /// the library's dlclose.
    closed: bool,
}

unsafe impl Send for X11Backend {}

impl X11Backend {
    /// Creates the backend: the display connection, the window and the
    /// WM_DELETE_WINDOW protocol registration. Throws on the loader's
    /// failures (the missing entry points, the display connection's own).
    pub fn new(desc: &super::WindowDesc) -> Result<Self, String> {
        if desc.width == 0 || desc.height == 0 {
            return Err(
                "X11Backend: width and height must be positive".into()
            );
        }
        let loader = X11Loader::new()?;
        let display = unsafe {
            (loader.open_display.unwrap())(std::ptr::null())
        };
        if display.is_null() {
            return Err(
                "X11Backend: the display connection could not be opened".into()
            );
        }
        let screen = unsafe { (loader.default_screen.unwrap())(display) };
        let root = unsafe { (loader.root_window.unwrap())(display, screen) };
        let window = unsafe {
            (loader.create_simple_window.unwrap())(
                display,
                root,
                0,
                0,
                desc.width,
                desc.height,
                0,
                0,
                0,
            )
        };
        if window == 0 {
            unsafe { (loader.close_display.unwrap())(display) };
            return Err("X11Backend: the window could not be created".into());
        }
        let title = desc.title.as_bytes();
        let mut title_buf = title.to_vec();
        title_buf.push(0);
        unsafe {
            (loader.store_name.unwrap())(
                display,
                window,
                title_buf.as_ptr() as *const _,
            );
        }
        let delete_atom = unsafe {
            (loader.intern_atom.unwrap())(
                display,
                b"WM_DELETE_WINDOW\0".as_ptr() as *const _,
                0,
            )
        };
        if delete_atom != 0 {
            unsafe {
                (loader.set_wm_protocols.unwrap())(
                    display,
                    window,
                    &delete_atom,
                    1,
                );
            }
        }
        unsafe {
            (loader.select_input.unwrap())(display, window, X_EVENT_MASK);
            (loader.map_window.unwrap())(display, window);
            (loader.flush.unwrap())(display);
        }
        Ok(X11Backend {
            loader,
            display,
            window,
            delete_atom,
            width: desc.width,
            height: desc.height,
            shutdown: false,
            violations: 0,
            closed: false,
        })
    }

    /// The native event queue's one event: the translation the contract's
    /// event carries. Returns None when the queue is empty.
    fn next_event(&mut self) -> Option<Event> {
        let pending = unsafe {
            (self.loader.pending.unwrap())(self.display)
        };
        if pending == 0 {
            return None;
        }
        let mut event = unsafe { std::mem::zeroed::<XEvent>() };
        unsafe {
            (self.loader.next_event.unwrap())(self.display, &mut event);
        }
        let event_type = unsafe { event.event_type };
        match event_type {
            X_KEY_PRESS => {
                let key = unsafe {
                    let mut key_event = event.key;
                    let keysym =
                        (self.loader.lookup_keysym.unwrap())(&mut key_event, 0);
                    key_for_keysym(keysym)
                };
                Some(Event::KeyDown { key })
            }
            X_KEY_RELEASE => {
                let key = unsafe {
                    let mut key_event = event.key;
                    let keysym =
                        (self.loader.lookup_keysym.unwrap())(&mut key_event, 0);
                    key_for_keysym(keysym)
                };
                Some(Event::KeyUp { key })
            }
            X_BUTTON_PRESS => {
                let (button, x, y) = unsafe {
                    (event.button.button, event.button.x, event.button.y)
                };
                let button = match button {
                    X_BUTTON1 => MouseButton::Left,
                    X_BUTTON2 => MouseButton::Middle,
                    X_BUTTON3 => MouseButton::Right,
                    X_BUTTON8 => MouseButton::X1,
                    X_BUTTON9 => MouseButton::X2,
                    X_BUTTON4 | X_BUTTON5 => {
                        // The wheel: the X11 wheel carries the same
                        // ButtonPress as the buttons (the X11 wheel rule).
                        let delta = if button == X_BUTTON4 { 1 } else { -1 };
                        return Some(Event::MouseWheel { delta });
                    }
                    _ => return None,
                };
                Some(Event::MouseDown { button, x, y })
            }
            X_BUTTON_RELEASE => {
                let (button, x, y) = unsafe {
                    (event.button.button, event.button.x, event.button.y)
                };
                let button = match button {
                    X_BUTTON1 => MouseButton::Left,
                    X_BUTTON2 => MouseButton::Middle,
                    X_BUTTON3 => MouseButton::Right,
                    X_BUTTON8 => MouseButton::X1,
                    X_BUTTON9 => MouseButton::X2,
                    // The wheel's release carries no event (the X11
                    // wheel rule: the press is the step, the release
                    // the X11 reports is the same wheel's own).
                    X_BUTTON4 | X_BUTTON5 => return None,
                    _ => return None,
                };
                Some(Event::MouseUp { button, x, y })
            }
            X_MOTION_NOTIFY => {
                let (x, y) = unsafe { (event.motion.x, event.motion.y) };
                Some(Event::MouseMove { x, y })
            }
            X_CONFIGURE_NOTIFY => {
                let (width, height) = unsafe {
                    (event.configure.width, event.configure.height)
                };
                if width > 0 && height > 0 {
                    self.width = width as u32;
                    self.height = height as u32;
                    Some(Event::Resize {
                        width: width as u32,
                        height: height as u32,
                    })
                } else {
                    None
                }
            }
            X_FOCUS_IN => Some(Event::FocusGained),
            X_FOCUS_OUT => Some(Event::FocusLost),
            X_CLIENT_MESSAGE => {
                let (message_type, format, data) = unsafe {
                    (
                        event.client_message.message_type,
                        event.client_message.format,
                        event.client_message.data,
                    )
                };
                if message_type == self.delete_atom
                    && format == 32
                    && data[0] == X_WM_DELETE_WINDOW
                {
                    Some(Event::Close)
                } else {
                    None
                }
            }
            X_DESTROY_NOTIFY => Some(Event::Close),
            _ => None,
        }
    }
}

impl PlatformBackend for X11Backend {
    fn describe(&self) -> String {
        format!(
            "x11 (the window {}x{}, the engine's own Xlib loader)",
            self.width, self.height
        )
    }

    fn poll_events(&mut self) -> Vec<Event> {
        if self.shutdown {
            self.violations += 1;
            return Vec::new();
        }
        let mut events = Vec::new();
        loop {
            match self.next_event() {
                Some(event) => events.push(event),
                None => break,
            }
        }
        events
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn resize(&mut self, width: u32, height: u32) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        if width == 0 || height == 0 {
            self.violations += 1;
            return;
        }
        unsafe {
            (self.loader.resize_window.unwrap())(
                self.display,
                self.window,
                width,
                height,
            );
            (self.loader.flush.unwrap())(self.display);
        }
        self.width = width;
        self.height = height;
    }

    fn native_surface(&self) -> NativeSurface {
        NativeSurface::X11 {
            display: self.display as *mut c_void,
            window: self.window,
        }
    }

    fn shutdown(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.shutdown = true;
        if !self.closed {
            unsafe {
                (self.loader.destroy_window.unwrap())(
                    self.display,
                    self.window,
                );
                (self.loader.close_display.unwrap())(self.display);
            }
            self.closed = true;
        }
        unsafe { libc::dlclose(self.loader.library) };
    }

    fn violation_count(&self) -> u64 {
        self.violations
    }
}
