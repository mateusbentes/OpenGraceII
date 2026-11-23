//! The CoreAudio backend: the macOS hardware output the engine opens
//! directly through the platform's own frameworks (CoreAudio's
//! AudioHardware and AudioUnit surfaces, loaded with dlopen from the
//! system's frameworks). The loader is the engine's own dlopen/dlsym
//! surface (the same rule the ALSA backend carries: no third-party
//! binding, the platform's documented C ABI only).
//!
//! The device path: the default output device through
//! `AudioHardwareGetProperty` (kAudioHardwarePropertyDefaultOutputDevice),
//! the AudioUnit graph through `AudioComponentFindNext` +
//! `AudioComponentInstanceNew` (the kAudioUnitSubType_DefaultOutput
//! HAL output), the stream format set to the engine's own interleaved
//! s16 at the config's rate and channel count, and the mix driven
//! from the app thread with the same model the ALSA backend runs —
//! the OS's AudioUnit pulls the device's frames from its own render
//! thread through the engine's own render callback, which copies the
//! pre-mixed frames the shared mix loop filled.
//!
//! The ABI rule: every constant, structure layout and function
//! signature this file declares is the platform's published ABI
//! (CoreAudio's AudioHardware.h, AudioComponent.h and AUComponent.h),
//! verified against the SDK's headers — the same verification
//! standard the Metal loader records.
#![cfg(all(target_os = "macos", feature = "audio-coreaudio"))]

use super::{AudioBackend, AudioConfig, AudioError, ListenerState, PcmBuffer,
    VoiceHandle, VoicePlacement, VoiceState};
use crate::audio::mix::{self, MixVoice};
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

// The OSStatus is the platforms' own error type (s32).
type OsStatus = i32;
const NO_ERR: OsStatus = 0;

// The published FourCC constants (AudioHardware.h / AudioComponent.h /
// AUComponent.h / CoreAudio's AudioUnitProperties.h).
const fn fourcc(chars: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*chars)
}
// kAudioHardwarePropertyDefaultOutputDevice ('dOut')
const K_AUDIO_HARDWARE_PROPERTY_DEFAULT_OUTPUT_DEVICE: u32 = fourcc(b"dOut");
// kAudioUnitType_Output ('auou') / kAudioUnitSubType_DefaultOutput ('def ')
// (the dbg_ca probe's finding: the 'auol' FourCC is the OfflineEffect
// type's own code, not the output unit's — the find returned null for
// every 'auol' description on the A18 Pro's hardware run).
const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = fourcc(b"auou");
const K_AUDIO_UNIT_SUBTYPE_DEFAULT_OUTPUT: u32 = fourcc(b"def ");
// The AudioUnit's properties are numbered enums, not FourCCs
// (AudioUnitProperties.h — the second hardware run's persistent
// 'who?' error pinned this: the engine was passing the FourCC of a
// selector no AudioUnit property carries).
const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
// The scopes (AudioUnitProperties.h: Global = 0, Input = 1,
// Output = 2 — the first run's 'who?' error also exposed the swapped
// values the engine carried here).
const K_AUDIO_UNIT_SCOPE_GLOBAL: u32 = 0;
const K_AUDIO_UNIT_SCOPE_INPUT: u32 = 1;
const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;
// The playback stream's own bus (element 0 — the output unit's bus
// 0 is the output bus; the reference implementations set the stream
// format at scope Input / element 0 for playback and the render
// callback at scope Global / element 0).
const K_AUDIO_UNIT_ELEMENT_OUTPUT_BUS: u32 = 0;

// The AudioBufferList layout the render callback receives
// (CoreAudio's AudioBufferList: one buffer per call for the HAL
// output's interleaved stream).
#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}
#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

// The AudioStreamBasicDescription (CoreAudioTypes.h).
#[repr(C)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32, // 'lpcm'
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
}
// kAudioFormatLinearPCM ('lpcm')
const K_AUDIO_FORMAT_LINEAR_PCM: u32 = fourcc(b"lpcm");
// kAudioFormatFlagIsSignedInteger (1 << 2) |
// kAudioFormatFlagIsPacked (1 << 3)
const K_AUDIO_FORMAT_FLAG_SIGNED_INTEGER_PACKED: u32 = (1 << 2) | (1 << 3);

// The AURenderCallbackStruct (AUComponent.h).
#[repr(C)]
struct AURenderCallbackStruct {
    input_proc: unsafe extern "C" fn(
        in_ref_con: *mut c_void,
        io_action_flags: *mut u32,
        in_time_stamp: *const AudioTimeStamp,
        in_bus_number: u32,
        in_number_frames: u32,
        io_data: *mut AudioBufferList,
    ) -> OsStatus,
    input_proc_ref_con: *mut c_void,
}

// The AudioTimeStamp (CoreAudioTypes.h): the sample time and the
// flags the render callback reads (the engine's callback reads
// neither, the ABI's own struct shape passes through).
#[repr(C)]
struct AudioTimeStamp {
    sample_time: f64,
    host_time: u64,
    rate_scalar: f64,
    word_clock_time: u64,
    flags: u32,
    reserved: u32,
}

// The AudioComponentDescription (AudioComponent.h).
#[repr(C)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}
// kAudioUnitManufacturer_Apple ('appl')
const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = fourcc(b"appl");

// The resolved function pointers (the C ABI's own signatures, usize
// pointers the dlsym surface hands back; the call sites transmute to
// the documented prototypes).
type FnAudioHardwareGetProperty = unsafe extern "C" fn(
    property: u32,
    size: *mut u32,
    data: *mut c_void,
) -> OsStatus;
type FnAudioComponentFindNext = unsafe extern "C" fn(
    component: *mut c_void,
    description: *const AudioComponentDescription,
) -> *mut c_void;
type FnAudioComponentInstanceNew = unsafe extern "C" fn(
    component: *mut c_void,
    out_instance: *mut *mut c_void,
) -> OsStatus;
type FnAudioComponentInstanceDispose =
    unsafe extern "C" fn(instance: *mut c_void) -> OsStatus;
type FnAudioUnitSetProperty = unsafe extern "C" fn(
    unit: *mut c_void,
    property: u32,
    scope: u32,
    element: u32,
    data: *const c_void,
    size: u32,
) -> OsStatus;
type FnAudioUnitInitialize = unsafe extern "C" fn(unit: *mut c_void) -> OsStatus;
type FnAudioUnitUninitialize =
    unsafe extern "C" fn(unit: *mut c_void) -> OsStatus;
type FnAudioOutputUnitStart =
    unsafe extern "C" fn(unit: *mut c_void) -> OsStatus;
type FnAudioOutputUnitStop =
    unsafe extern "C" fn(unit: *mut c_void) -> OsStatus;

// The shared state the render callback reads: the pre-mixed frames
// the app thread's mix walk fills. The engine's rule is the app-side
// mix loop (the ALSA model), so the callback copies what the last
// mix walk produced and the app thread keeps the buffer ahead of the
// device.
struct CallbackState {
    /// The mixed s16 frames waiting for the device (interleaved).
    pending: Mutex<Vec<i16>>,
    /// The mixed frame count the smoke's checks read.
    mixed: AtomicU64,
    /// The one-shot "the callback ran" flag the smoke's checks read.
    callback_ran: AtomicBool,
    channels: u32,
}

/// The CoreAudio backend: the AudioUnit output plus the shared mix
/// state.
pub struct CoreAudioBackend {
    config: AudioConfig,
    listener: ListenerState,
    voices: Vec<MixVoice>,
    next_handle: u32,
    mixed: u64,
    violations: u64,
    shutdown: bool,
    component: *mut c_void,
    unit: *mut c_void,
    library: usize,
    state: std::sync::Arc<CallbackState>,
}

unsafe impl Send for CoreAudioBackend {}

impl CoreAudioBackend {
    /// Opens the default output device through the engine's dlopen
    /// surface and configures the s16 interleaved AudioUnit stream.
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError(
                "CoreAudioBackend: the sample rate must be positive".to_string(),
            ));
        }
        if config.channels != 1 && config.channels != 2 {
            return Err(AudioError(
                "CoreAudioBackend: the channel count must be 1 or 2".to_string(),
            ));
        }
        if config.period_frames == 0 {
            return Err(AudioError(
                "CoreAudioBackend: the period must carry frames".to_string(),
            ));
        }
        unsafe {
            // The system's frameworks: dlopen needs the full path
            // (dlsym's RTLD_DEFAULT would also see them, but the
            // engine's rule is the explicit load).
            let path =
                b"/System/Library/Frameworks/CoreAudio.framework/CoreAudio\0";
            let library = libc_dlopen(path.as_ptr() as *const c_char);
            if library == 0 {
                return Err(AudioError(
                    "CoreAudioBackend: the CoreAudio framework's load failed"
                        .to_string(),
                ));
            }
            let resolve = |name: &[u8]| -> Result<*mut c_void, AudioError> {
                let symbol = libc_dlsym(library, name.as_ptr() as *const c_char);
                if symbol.is_null() {
                    return Err(AudioError(format!(
                        "CoreAudioBackend: the {} symbol is missing",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    )));
                }
                Ok(symbol)
            };
            let audio_hardware_get_property: FnAudioHardwareGetProperty =
                std::mem::transmute(resolve(b"AudioHardwareGetProperty\0")?);
            let audio_component_find_next: FnAudioComponentFindNext =
                std::mem::transmute(resolve(b"AudioComponentFindNext\0")?);
            let audio_component_instance_new: FnAudioComponentInstanceNew =
                std::mem::transmute(resolve(b"AudioComponentInstanceNew\0")?);
            let audio_component_instance_dispose: FnAudioComponentInstanceDispose =
                std::mem::transmute(
                    resolve(b"AudioComponentInstanceDispose\0")?,
                );
            let audio_unit_set_property: FnAudioUnitSetProperty =
                std::mem::transmute(resolve(b"AudioUnitSetProperty\0")?);
            let audio_unit_initialize: FnAudioUnitInitialize =
                std::mem::transmute(resolve(b"AudioUnitInitialize\0")?);
            let audio_unit_uninitialize: FnAudioUnitUninitialize =
                std::mem::transmute(resolve(b"AudioUnitUninitialize\0")?);
            let audio_output_unit_start: FnAudioOutputUnitStart =
                std::mem::transmute(resolve(b"AudioOutputUnitStart\0")?);
            // The AudioUnit graph: the default-output HAL component.
            let description = AudioComponentDescription {
                component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
                component_sub_type: K_AUDIO_UNIT_SUBTYPE_DEFAULT_OUTPUT,
                component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
                component_flags: 0,
                component_flags_mask: 0,
            };
            let component = audio_component_find_next(
                std::ptr::null_mut(),
                &description,
            );
            if component.is_null() {
                libc_dlclose(library);
                return Err(AudioError(
                    "CoreAudioBackend: the default-output component's find failed"
                        .to_string(),
                ));
            }
            let mut unit: *mut c_void = std::ptr::null_mut();
            let status = audio_component_instance_new(component, &mut unit);
            if status != NO_ERR || unit.is_null() {
                libc_dlclose(library);
                return Err(AudioError(format!(
                    "CoreAudioBackend: the AudioUnit's creation failed ({})",
                    status
                )));
            }
            // The stream format: the engine's interleaved s16 at the
            // config's rate and channels (the HAL output converts).
            let format = AudioStreamBasicDescription {
                sample_rate: config.sample_rate as f64,
                format_id: K_AUDIO_FORMAT_LINEAR_PCM,
                format_flags: K_AUDIO_FORMAT_FLAG_SIGNED_INTEGER_PACKED,
                bytes_per_packet: (config.channels as u32) * 2,
                frames_per_packet: 1,
                bytes_per_frame: (config.channels as u32) * 2,
                channels_per_frame: config.channels,
                bits_per_channel: 16,
            };
            // The unit's input scope carries the stream the callback
            // fills; the output scope carries the device's own.
            let status = audio_unit_set_property(
                unit,
                K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
                K_AUDIO_UNIT_SCOPE_INPUT,
                K_AUDIO_UNIT_ELEMENT_OUTPUT_BUS,
                &format as *const AudioStreamBasicDescription as *const c_void,
                std::mem::size_of::<AudioStreamBasicDescription>() as u32,
            );
            if status != NO_ERR {
                audio_component_instance_dispose(unit);
                libc_dlclose(library);
                return Err(AudioError(format!(
                    "CoreAudioBackend: the stream format's setup failed ({})",
                    status
                )));
            }
            // The render callback: the OS's render thread pulls the
            // frames through the engine's own callback.
            let state = std::sync::Arc::new(CallbackState {
                pending: Mutex::new(Vec::new()),
                mixed: AtomicU64::new(0),
                callback_ran: AtomicBool::new(false),
                channels: config.channels,
            });
            let callback = AURenderCallbackStruct {
                input_proc: render_callback,
                input_proc_ref_con: std::sync::Arc::as_ptr(&state)
                    as *mut c_void,
            };
            let status = audio_unit_set_property(
                unit,
                K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
                K_AUDIO_UNIT_SCOPE_GLOBAL,
                K_AUDIO_UNIT_ELEMENT_OUTPUT_BUS,
                &callback as *const AURenderCallbackStruct as *const c_void,
                std::mem::size_of::<AURenderCallbackStruct>() as u32,
            );
            if status != NO_ERR {
                audio_component_instance_dispose(unit);
                libc_dlclose(library);
                return Err(AudioError(format!(
                    "CoreAudioBackend: the render callback's setup failed ({})",
                    status
                )));
            }
            let status = audio_unit_initialize(unit);
            if status != NO_ERR {
                audio_component_instance_dispose(unit);
                libc_dlclose(library);
                return Err(AudioError(format!(
                    "CoreAudioBackend: the AudioUnit's initialization failed ({})",
                    status
                )));
            }
            let status = audio_output_unit_start(unit);
            if status != NO_ERR {
                audio_unit_uninitialize(unit);
                audio_component_instance_dispose(unit);
                libc_dlclose(library);
                return Err(AudioError(format!(
                    "CoreAudioBackend: the output's start failed ({})",
                    status
                )));
            }
            // The default device query is the describe line's input
            // (the property read is the hardware's own report; a
            // failure names no device and the describe carries the
            // config's numbers alone).
            let mut default_device: u32 = 0;
            let mut size = std::mem::size_of::<u32>() as u32;
            let _ = audio_hardware_get_property(
                K_AUDIO_HARDWARE_PROPERTY_DEFAULT_OUTPUT_DEVICE,
                &mut size,
                &mut default_device as *mut u32 as *mut c_void,
            );
            Ok(CoreAudioBackend {
                config,
                listener: ListenerState::default(),
                voices: Vec::new(),
                next_handle: 1,
                mixed: 0,
                violations: 0,
                shutdown: false,
                component,
                unit,
                library,
                state,
            })
        }
    }
}

impl Drop for CoreAudioBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.unit.is_null() {
                let audio_output_unit_stop: FnAudioOutputUnitStop =
                    std::mem::transmute(libc_dlsym(
                        self.library,
                        b"AudioOutputUnitStop\0".as_ptr() as *const c_char,
                    ));
                let audio_unit_uninitialize: FnAudioUnitUninitialize =
                    std::mem::transmute(libc_dlsym(
                        self.library,
                        b"AudioUnitUninitialize\0".as_ptr() as *const c_char,
                    ));
                let audio_component_instance_dispose:
                    FnAudioComponentInstanceDispose =
                    std::mem::transmute(libc_dlsym(
                        self.library,
                        b"AudioComponentInstanceDispose\0".as_ptr()
                            as *const c_char,
                    ));
                let _ = audio_output_unit_stop(self.unit);
                let _ = audio_unit_uninitialize(self.unit);
                audio_component_instance_dispose(self.unit);
                self.unit = std::ptr::null_mut();
            }
            self.component = std::ptr::null_mut();
            if self.library != 0 {
                libc_dlclose(self.library);
                self.library = 0;
            }
        }
    }
}

impl AudioBackend for CoreAudioBackend {
    fn describe(&self) -> String {
        format!(
            "coreaudio sample_rate={} channels={} period={}",
            self.config.sample_rate, self.config.channels, self.config.period_frames
        )
    }

    fn create_voice(&mut self, buffer: &PcmBuffer) -> Result<VoiceHandle, AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError(
                "create_voice: the backend is shut down".to_string(),
            ));
        }
        if buffer.channels == 0 || buffer.sample_rate == 0 {
            return Err(AudioError(
                "create_voice: the buffer's rates must be positive".to_string(),
            ));
        }
        let handle = VoiceHandle(self.next_handle);
        self.next_handle += 1;
        self.voices.push(MixVoice {
            handle: handle.0,
            buffer: buffer.clone(),
            cursor_frames: 0,
            looping: false,
            state: VoiceState::Ready,
            placement: VoicePlacement::default(),
            released: false,
        });
        Ok(handle)
    }

    fn release_voice(&mut self, voice: VoiceHandle) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        if let Some(found) = self
            .voices
            .iter_mut()
            .find(|v| v.handle == voice.0 && !v.released)
        {
            found.released = true;
        } else {
            self.violations += 1;
        }
    }

    fn play_voice(&mut self, voice: VoiceHandle, looping: bool) -> Result<(), AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError(
                "play_voice: the backend is shut down".to_string(),
            ));
        }
        match self
            .voices
            .iter_mut()
            .find(|v| v.handle == voice.0 && !v.released)
        {
            Some(found) => {
                if found.state == VoiceState::Stopped {
                    found.cursor_frames = 0;
                }
                found.state = VoiceState::Playing;
                found.looping = looping;
                Ok(())
            }
            None => {
                self.violations += 1;
                Err(AudioError("play_voice: the handle is stale".to_string()))
            }
        }
    }

    fn pause_voice(&mut self, voice: VoiceHandle) -> Result<(), AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError(
                "pause_voice: the backend is shut down".to_string(),
            ));
        }
        match self
            .voices
            .iter_mut()
            .find(|v| v.handle == voice.0 && !v.released)
        {
            Some(found) => {
                if found.state == VoiceState::Playing {
                    found.state = VoiceState::Paused;
                }
                Ok(())
            }
            None => {
                self.violations += 1;
                Err(AudioError("pause_voice: the handle is stale".to_string()))
            }
        }
    }

    fn stop_voice(&mut self, voice: VoiceHandle) -> Result<(), AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError(
                "stop_voice: the backend is shut down".to_string(),
            ));
        }
        match self
            .voices
            .iter_mut()
            .find(|v| v.handle == voice.0 && !v.released)
        {
            Some(found) => {
                found.state = VoiceState::Stopped;
                found.cursor_frames = 0;
                Ok(())
            }
            None => {
                self.violations += 1;
                Err(AudioError("stop_voice: the handle is stale".to_string()))
            }
        }
    }

    fn set_voice_placement(&mut self, voice: VoiceHandle, placement: &VoicePlacement) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        if let Some(found) = self
            .voices
            .iter_mut()
            .find(|v| v.handle == voice.0 && !v.released)
        {
            found.placement = *placement;
        } else {
            self.violations += 1;
        }
    }

    fn set_listener(&mut self, listener: &ListenerState) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.listener = *listener;
    }

    fn voice_state(&self, voice: VoiceHandle) -> VoiceState {
        match self
            .voices
            .iter()
            .find(|v| v.handle == voice.0 && !v.released)
        {
            Some(found) => found.state,
            None => VoiceState::Invalid,
        }
    }

    fn mix_frames(&mut self, frames: u32) -> Vec<f32> {
        let channels = self.config.channels as usize;
        let mut peaks = vec![0.0f32; channels];
        if self.shutdown {
            self.violations += 1;
            return peaks;
        }
        let block = mix::run(
            &mut self.voices,
            &self.listener,
            frames,
            channels,
            &mut self.mixed,
        );
        for (channel, &sample) in block.iter().enumerate() {
            let magnitude = sample.abs();
            if magnitude > peaks[channel % channels] {
                peaks[channel % channels] = magnitude;
            }
        }
        // The mixed block converts to the interleaved s16 and queues
        // for the render callback (the OS's pull model; the app's
        // walk runs ahead of the device's own thread).
        let mut s16 = vec![0i16; block.len()];
        for (index, &sample) in block.iter().enumerate() {
            let clamped = sample.clamp(-1.0, 1.0);
            // The symmetric rounding the ALSA backend's s16 path uses.
            let scaled = clamped * 32767.0 + if clamped >= 0.0 { 0.5 } else { -0.5 };
            s16[index] = scaled.max(-32768.0).min(32767.0) as i16;
        }
        if let Ok(mut pending) = self.state.pending.lock() {
            pending.extend_from_slice(&s16);
            // The pending queue's cap: four periods worth of frames
            // (the device's own pull drains it; the cap keeps the
            // latency bounded if the app runs far ahead).
            let cap = self.config.period_frames as usize * 4 * channels;
            if pending.len() > cap {
                let excess = pending.len() - cap;
                pending.drain(0..excess);
            }
        }
        peaks
    }

    fn mixed_frames(&self) -> u64 {
        self.mixed
    }

    fn shutdown(&mut self) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        self.shutdown = true;
        self.voices.clear();
    }

    fn violation_count(&self) -> u64 {
        self.violations
    }
}

/// The engine's own render callback: the OS's AudioUnit render thread
/// pulls the frames from the pending queue the app's mix walk fills.
unsafe extern "C" fn render_callback(
    in_ref_con: *mut c_void,
    io_action_flags: *mut u32,
    _in_time_stamp: *const AudioTimeStamp,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OsStatus {
    let _ = io_action_flags;
    let state = &*(in_ref_con as *const CallbackState);
    let buffers = &mut (*io_data);
    if buffers.number_buffers == 0 {
        return 0;
    }
    let buffer = &mut buffers.buffers[0];
    let channels = state.channels as usize;
    let frames_needed = in_number_frames as usize;
    let bytes_needed = frames_needed * channels * 2;
    if (buffer.data_byte_size as usize) < bytes_needed {
        return 0;
    }
    let destination = buffer.data as *mut i16;
    let taken = {
        match state.pending.lock() {
            Ok(mut pending) => {
                let available_frames = pending.len() / channels;
                let take_frames = available_frames.min(frames_needed);
                let take_samples = take_frames * channels;
                if take_samples > 0 {
                    std::ptr::copy_nonoverlapping(
                        pending.as_ptr(),
                        destination,
                        take_samples,
                    );
                    pending.drain(0..take_samples);
                }
                take_frames
            }
            Err(_) => 0,
        }
    };
    // The short fill: silence the remainder (the device's honest
    // silence instead of the previous audio's repeat).
    if taken < frames_needed {
        let remainder = (frames_needed - taken) * channels;
        std::ptr::write_bytes(
            destination.add(taken * channels),
            0,
            remainder,
        );
    }
    state.mixed.fetch_add(taken as u64, Ordering::Relaxed);
    state.callback_ran.store(true, Ordering::Relaxed);
    NO_ERR
}

// The libc dlopen surface: the same three calls the ALSA backend
// carries (the audio module's own unix-gated helpers).
fn libc_dlopen(path: *const c_char) -> usize {
    unsafe { libc::dlopen(path, libc::RTLD_NOW | libc::RTLD_LOCAL) as usize }
}
fn libc_dlsym(library: usize, name: *const c_char) -> *mut c_void {
    unsafe { libc::dlsym(library as *mut c_void, name) }
}
fn libc_dlclose(library: usize) {
    unsafe {
        if library != 0 {
            libc::dlclose(library as *mut c_void);
        }
    }
}
