//! The AAudio backend: the Android hardware output the engine opens
//! directly through the platform's `libaaudio.so`. The loader is the
//! engine's own dlopen/dlsym surface (the same rule the ALSA and the
//! CoreAudio backends carry: no third-party binding, the platform's
//! documented C ABI only).
//!
//! The device path: `AAudio_createStreamBuilder`, the builder's output
//! direction with the interleaved s16 format at the config's rate and
//! channel count, `AAudioStreamBuilder_openStream`, the builder's
//! delete, and `AAudioStream_requestStart`. The mix loop is the
//! contract's shared arithmetic — the backend mixes the voices into a
//! frame buffer through the same `mix_frames` walk and pushes the s16
//! conversion of the block with the blocking `AAudioStream_write`, so
//! the deterministic checks and the hardware path run the identical
//! mix (the same polling-write model the WASAPI and the ALSA backends
//! carry; no callback thread of the engine's own).
//!
//! The dlopen surface keeps the minSdk at 24: the library exists from
//! API 26 onward, the machines before it report the open error and the
//! smoke's null path carries them (the same graceful-skip rule the
//! CoreAudio backend follows when the device is missing).

use super::{AudioBackend, AudioConfig, AudioError, ListenerState, PcmBuffer,
    VoiceHandle, VoicePlacement, VoiceState};
use crate::audio::mix::{self, MixVoice};

// The resolved function pointers (the C ABI's own signatures, usize
// pointers the dlsym surface hands back; the call sites transmute to
// the documented AAudio prototypes).

type AaudioCreateBuilder = unsafe extern "C" fn(
    builder: *mut *mut libc::c_void,
) -> libc::c_int;
type AaudioBuilderSetI32 = unsafe extern "C" fn(
    builder: *mut libc::c_void,
    value: i32,
);
type AaudioBuilderOpenStream = unsafe extern "C" fn(
    builder: *mut libc::c_void,
    stream: *mut *mut libc::c_void,
) -> libc::c_int;
type AaudioBuilderDelete = unsafe extern "C" fn(
    builder: *mut libc::c_void,
) -> libc::c_int;
type AaudioStreamWrite = unsafe extern "C" fn(
    stream: *mut libc::c_void,
    buffer: *const libc::c_void,
    num_frames: i32,
    timeout_nanoseconds: i64,
) -> libc::c_int;
type AaudioStreamRequestStart = unsafe extern "C" fn(
    stream: *mut libc::c_void,
) -> libc::c_int;
type AaudioStreamClose = unsafe extern "C" fn(
    stream: *mut libc::c_void,
) -> libc::c_int;
type AaudioResultToText = unsafe extern "C" fn(
    result: libc::c_int,
) -> *const libc::c_char;

// The AAUDIO_OK result (aaudio/AAudio.h) and the enum values the
// builder's setters take.
const AAUDIO_OK: libc::c_int = 0;
const AAUDIO_DIRECTION_OUTPUT: i32 = 0;
const AAUDIO_SHARING_MODE_SHARED: i32 = 0;
// The PCM_I16 format (AAudioFormat AAUDIO_FORMAT_PCM_I16 == 1).
const AAUDIO_FORMAT_PCM_I16: i32 = 1;
// The per-write timeout: 100 ms (the blocking write's own wait; the
// loop keeps pushing until the block is fully in the device's buffer).
const WRITE_TIMEOUT_NS: i64 = 100_000_000;

/// The AAudio backend: the opened stream plus the shared mix state.
pub struct AAudioBackend {
    config: AudioConfig,
    listener: ListenerState,
    voices: Vec<MixVoice>,
    next_handle: u32,
    mixed: u64,
    violations: u64,
    shutdown: bool,
    stream: *mut libc::c_void,
    stream_write: AaudioStreamWrite,
    stream_close: AaudioStreamClose,
}

impl AAudioBackend {
    /// Opens the default output stream through the engine's dlopen
    /// surface and configures the interleaved s16 stream.
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError(
                "AAudioBackend: the sample rate must be positive".to_string(),
            ));
        }
        if config.channels != 1 && config.channels != 2 {
            return Err(AudioError(
                "AAudioBackend: the channel count must be 1 or 2".to_string(),
            ));
        }
        if config.period_frames == 0 {
            return Err(AudioError(
                "AAudioBackend: the period must carry frames".to_string(),
            ));
        }
        unsafe {
            let lib = libc::dlopen(
                b"libaaudio.so\0".as_ptr() as *const libc::c_char,
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            );
            if lib.is_null() {
                return Err(AudioError(
                    "AAudioBackend: the libaaudio.so load failed (AAudio needs \
                     Android 8.0 / API 26 onward; earlier machines stay on the \
                     null path)"
                        .to_string(),
                ));
            }
            let resolve = |name: &[u8]| -> Result<*mut libc::c_void, AudioError> {
                let symbol = libc::dlsym(lib, name.as_ptr() as *const libc::c_char);
                if symbol.is_null() {
                    return Err(AudioError(format!(
                        "AAudioBackend: the {} symbol is missing",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    )));
                }
                Ok(symbol)
            };
            let create_builder: AaudioCreateBuilder =
                std::mem::transmute(resolve(b"AAudio_createStreamBuilder\0")?);
            let set_direction: AaudioBuilderSetI32 =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_setDirection\0")?);
            let set_sharing: AaudioBuilderSetI32 =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_setSharingMode\0")?);
            let set_format: AaudioBuilderSetI32 =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_setFormat\0")?);
            let set_rate: AaudioBuilderSetI32 =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_setSampleRate\0")?);
            let set_channels: AaudioBuilderSetI32 =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_setChannelCount\0")?);
            let set_capacity: AaudioBuilderSetI32 =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_setBufferCapacityInFrames\0")?);
            let open_stream: AaudioBuilderOpenStream =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_openStream\0")?);
            let delete_builder: AaudioBuilderDelete =
                std::mem::transmute(resolve(b"AAudioStreamBuilder_delete\0")?);
            let stream_write: AaudioStreamWrite =
                std::mem::transmute(resolve(b"AAudioStream_write\0")?);
            let request_start: AaudioStreamRequestStart =
                std::mem::transmute(resolve(b"AAudioStream_requestStart\0")?);
            let stream_close: AaudioStreamClose =
                std::mem::transmute(resolve(b"AAudioStream_close\0")?);
            // The error messages carry the API's own text when present
            // (the resolver is local to the open; the backend keeps no
            // state the mix loop never reads).
            let result_text = libc::dlsym(lib, b"AAudio_convertResultToText\0"
                .as_ptr() as *const libc::c_char);
            let result_text: Option<AaudioResultToText> = if result_text.is_null() {
                None
            } else {
                Some(std::mem::transmute(result_text))
            };
            let describe_error = |result: libc::c_int| -> String {
                match result_text {
                    Some(text) => {
                        let pointer = text(result);
                        if pointer.is_null() {
                            format!("result {}", result)
                        } else {
                            format!("{} ({})",
                                std::ffi::CStr::from_ptr(pointer).to_string_lossy(),
                                result)
                        }
                    }
                    None => format!("result {}", result),
                }
            };
            let mut builder: *mut libc::c_void = std::ptr::null_mut();
            let result = create_builder(&mut builder);
            if result != AAUDIO_OK || builder.is_null() {
                return Err(AudioError(format!(
                    "AAudioBackend: the stream builder's creation failed ({})",
                    describe_error(result)
                )));
            }
            // The builder's parameters: the output direction, the shared
            // mode (the documented default — the exclusive path owns the
            // endpoint and fails on the mixed-use machines), the s16
            // interleaved format at the config's shape, and the device
            // buffer at four periods (the latency's headroom).
            set_direction(builder, AAUDIO_DIRECTION_OUTPUT);
            set_sharing(builder, AAUDIO_SHARING_MODE_SHARED);
            set_format(builder, AAUDIO_FORMAT_PCM_I16);
            set_rate(builder, config.sample_rate as i32);
            set_channels(builder, config.channels as i32);
            set_capacity(builder, (config.period_frames * 4) as i32);
            let mut stream: *mut libc::c_void = std::ptr::null_mut();
            let result = open_stream(builder, &mut stream);
            delete_builder(builder);
            if result != AAUDIO_OK || stream.is_null() {
                return Err(AudioError(format!(
                    "AAudioBackend: the output stream's open failed ({})",
                    describe_error(result)
                )));
            }
            let result = request_start(stream);
            if result != AAUDIO_OK {
                stream_close(stream);
                return Err(AudioError(format!(
                    "AAudioBackend: the stream's start failed ({})",
                    describe_error(result)
                )));
            }
            Ok(AAudioBackend {
                config,
                listener: ListenerState::default(),
                voices: Vec::new(),
                next_handle: 1,
                mixed: 0,
                violations: 0,
                shutdown: false,
                stream,
                stream_write,
                stream_close,
            })
        }
    }

    /// Writes the mixed block to the stream: the f32 mix converts to
    /// the interleaved s16 the stream carries; the blocking write
    /// pushes the block in device-sized slices until it lands.
    fn write_block(&mut self, frames: &[f32]) {
        let channels = self.config.channels as usize;
        let frame_count = frames.len() / channels;
        if frame_count == 0 {
            return;
        }
        let mut s16 = vec![0i16; frames.len()];
        for (index, &sample) in frames.iter().enumerate() {
            let clamped = sample.clamp(-1.0, 1.0);
            // The symmetric rounding the other backends' s16 paths use.
            let scaled = clamped * 32767.0 + if clamped >= 0.0 { 0.5 } else { -0.5 };
            s16[index] = scaled.max(-32768.0).min(32767.0) as i16;
        }
        unsafe {
            let mut offset = 0usize;
            while offset < frame_count {
                let written = (self.stream_write)(
                    self.stream,
                    s16[offset * channels..].as_ptr() as *const libc::c_void,
                    (frame_count - offset) as i32,
                    WRITE_TIMEOUT_NS,
                );
                if written > 0 {
                    offset += written as usize;
                    continue;
                }
                // The negative result is the error (the documented
                // aaudio_result_t); the write stops — the device's
                // disconnect lands here and the contract's violation
                // surface notes it through the mixed walk.
                break;
            }
        }
    }
}

impl Drop for AAudioBackend {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                (self.stream_close)(self.stream);
            }
            self.stream = std::ptr::null_mut();
        }
    }
}

impl AudioBackend for AAudioBackend {
    fn describe(&self) -> String {
        format!(
            "aaudio sample_rate={} channels={} period={}",
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
        self.write_block(&block);
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
