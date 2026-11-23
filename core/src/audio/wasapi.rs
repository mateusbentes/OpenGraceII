//! The WASAPI backend: the Windows hardware output the engine opens
//! directly through the platform's own COM surface (ole32.dll's
//! CoInitializeEx/CoCreateInstance and mmdevapi.dll's device
//! enumerator). The loader is the engine's own LoadLibraryA surface
//! (the same rule the D3D12 loader carries: no third-party binding,
//! the platform's documented C ABI only).
//!
//! The device path: the shared-mode render client on the default
//! endpoint, the interleaved s16 format at the config's rate and
//! channel count with the auto-convert flag (Windows converts the
//! rate and the channel layout itself), the polling write model —
//! `GetCurrentPadding` reports the device's filled frames, the
//! backend writes what the buffer still has room for and sleeps one
//! millisecond when the buffer is full, the documented shared-mode
//! polling pattern. The mix loop itself is the contract's shared
//! arithmetic — the backend mixes the voices into a frame buffer
//! through the same `mix_frames` walk and writes the s16 conversion
//! of the block to the device, so the deterministic checks and the
//! hardware path run the identical mix.
//!
//! The COM rule: every GUID, vtable slot index and structure layout
//! this file declares is the platform's published ABI (mmdeviceapi.h
//! and audioclient.h), verified against the MinGW-w64 headers — the
//! same verification standard the D3D12 loader records.
#![cfg(all(windows, feature = "audio-wasapi"))]

use super::{AudioBackend, AudioConfig, AudioError, ListenerState, PcmBuffer,
    VoiceHandle, VoicePlacement, VoiceState};
use crate::audio::mix::{self, MixVoice};
use std::os::raw::c_void;

// The COM object handles: opaque pointers the loader never
// dereferences except through the vtable rule.
type ComObject = *mut c_void;

// The published GUIDs (mmdeviceapi.h / audioclient.h).
#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}
const CLSID_MMDEVICE_ENUMERATOR: Guid = Guid {
    data1: 0xBCDE0395, data2: 0xE52F, data3: 0x467C,
    data4: [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
};
const IID_IMMDEVICE_ENUMERATOR: Guid = Guid {
    data1: 0xA95664D2, data2: 0x9614, data3: 0x4F35,
    data4: [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
};
const IID_IAUDIO_CLIENT: Guid = Guid {
    data1: 0x1CB9AD4C, data2: 0xDBFA, data3: 0x4C32,
    data4: [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2],
};
const IID_IAUDIO_RENDER_CLIENT: Guid = Guid {
    data1: 0xF294ACFC, data2: 0x3146, data3: 0x4483,
    data4: [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2],
};

// The published constants (mmdeviceapi.h / audioclient.h).
const COINIT_MULTITHREADED: u32 = 0x0;
const RPC_E_CHANGED_MODE: i32 = 0x80010106u32 as i32;
const CLSCTX_ALL: u32 = 0x17;
const E_DATAFLOW_RENDER: u32 = 0; // eRender
const E_ROLE_MULTIMEDIA: u32 = 1; // eMultimedia
// AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM (0x80000000) |
// AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY (0x08000000).
const AUDCLNT_STREAMFLAGS_AUTOCONVERT: u32 = 0x88000000;
const AUDCLNT_SHAREMODE_SHARED: u32 = 0;

// The published vtable slot indexes (the IUnknown's three entries sit
// at the start of every COM vtable; the order after them is the
// interface declaration's own order).
const SLOT_IMMDEVICE_GET_DEFAULT: usize = 4; // IMMDeviceEnumerator
const SLOT_IMMDEVICE_ACTIVATE: usize = 3; // IMMDevice
const SLOT_AUDIOCLIENT_INITIALIZE: usize = 3;
const SLOT_AUDIOCLIENT_GET_BUFFER_SIZE: usize = 4;
const SLOT_AUDIOCLIENT_GET_CURRENT_PADDING: usize = 6;
const SLOT_AUDIOCLIENT_START: usize = 10;
const SLOT_AUDIOCLIENT_STOP: usize = 11;
const SLOT_AUDIOCLIENT_GET_SERVICE: usize = 14;
const SLOT_RENDER_GET_BUFFER: usize = 3;
const SLOT_RENDER_RELEASE_BUFFER: usize = 4;
const SLOT_RELEASE: usize = 2; // IUnknown

// The untagged PCM WAVEFORMATEX (mmreg.h): the s16 interleaved form
// the write loop fills.
#[repr(C)]
struct WaveFormatEx {
    format_tag: u16, // WAVE_FORMAT_PCM = 1
    channels: u16,
    samples_per_sec: u32,
    avg_bytes_per_sec: u32,
    block_align: u16,
    bits_per_sample: u16,
    extra_size: u16,
}
const WAVE_FORMAT_PCM: u16 = 1;

// The kernel32 surface: the platform's own always-loaded library (the
// D3D12 loader's recorded rule: kernel32 is the platform's own
// library, the same class of surface the dlopen/dlsym pair is on
// Linux). The LoadLibraryA surface loads the two COM dlls; the Sleep
// call paces the polling write model.
#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn Sleep(milliseconds: u32);
}

/// The WASAPI backend: the opened render client plus the shared mix
/// state.
pub struct WasapiAudioBackend {
    config: AudioConfig,
    listener: ListenerState,
    voices: Vec<MixVoice>,
    next_handle: u32,
    mixed: u64,
    violations: u64,
    shutdown: bool,
    device_enumerator: ComObject,
    audio_client: ComObject,
    render_client: ComObject,
    buffer_frames: u32,
    ole32_library: *mut c_void,
    mmdevapi_library: *mut c_void,
    co_uninitialize: Option<unsafe extern "system" fn()>,
}

// The COM objects are thread-affine per apartment but the backend's
// own use is single-threaded (the contract's app-driven mix loop);
// the raw pointers keep the backend Send so the smoke can hold it on
// the main thread like every other backend.
unsafe impl Send for WasapiAudioBackend {}

/// Calls the object's vtable slot at `index` with the argument list
/// the call site transmutes (the COM ABI's rule: the vtable is the
/// object's first word).
unsafe fn vtable_slot(obj: ComObject, index: usize) -> *mut c_void {
    let vtable = *(obj as *mut *mut *mut c_void);
    *vtable.add(index)
}

impl WasapiAudioBackend {
    /// Opens the default render endpoint through the engine's
    /// LoadLibraryA surface and configures the shared-mode s16
    /// stream.
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError(
                "WasapiAudioBackend: the sample rate must be positive".to_string(),
            ));
        }
        if config.channels != 1 && config.channels != 2 {
            return Err(AudioError(
                "WasapiAudioBackend: the channel count must be 1 or 2".to_string(),
            ));
        }
        if config.period_frames == 0 {
            return Err(AudioError(
                "WasapiAudioBackend: the period must carry frames".to_string(),
            ));
        }
        unsafe {
            // The kernel32 surface is the platform's own linked
            // library; the two COM dlls load through LoadLibraryA.
            let ole32_library = LoadLibraryA(b"ole32.dll\0".as_ptr());
            if ole32_library.is_null() {
                return Err(AudioError(
                    "WasapiAudioBackend: the ole32.dll load failed".to_string(),
                ));
            }
            let mmdevapi_library = LoadLibraryA(b"mmdevapi.dll\0".as_ptr());
            if mmdevapi_library.is_null() {
                FreeLibrary(ole32_library);
                return Err(AudioError(
                    "WasapiAudioBackend: the mmdevapi.dll load failed".to_string(),
                ));
            }
            // The symbol resolver: GetProcAddress's ANSI name surface
            // (the byte literal keeps the NUL terminator; concat!'s
            // literal rule means the macro carries the name itself).
            macro_rules! resolve {
                ($library:expr, $name:literal) => {{
                    let symbol = GetProcAddress(
                        $library,
                        concat!($name, "\0").as_ptr(),
                    );
                    if symbol.is_null() {
                        return Err(AudioError(format!(
                            "WasapiAudioBackend: the {} symbol is missing",
                            $name
                        )));
                    }
                    symbol
                }};
            }
            // The two ole32 entry points (the documented exports).
            let co_initialize_ex: unsafe extern "system" fn(
                reserved: *mut c_void,
                model: u32,
            ) -> i32 = {
                let symbol =
                    resolve!(ole32_library, "CoInitializeEx");
                std::mem::transmute(symbol)
            };
            let co_create_instance: unsafe extern "system" fn(
                clsid: *const Guid,
                outer: *mut c_void,
                context: u32,
                iid: *const Guid,
                out: *mut *mut c_void,
            ) -> i32 = {
                let symbol =
                    resolve!(ole32_library, "CoCreateInstance");
                std::mem::transmute(symbol)
            };
            let co_uninitialize: unsafe extern "system" fn() = {
                let symbol =
                    resolve!(ole32_library, "CoUninitialize");
                std::mem::transmute(symbol)
            };
            // The apartment: the MULTITHREADED model (the backend's
            // own thread); an already-initialized different model is
            // the documented benign case.
            let hr = co_initialize_ex(std::ptr::null_mut(), COINIT_MULTITHREADED);
            if hr != 0 && hr != RPC_E_CHANGED_MODE {
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the COM apartment init failed (0x{:08X})",
                    hr as u32
                )));
            }
            let mut device_enumerator: ComObject = std::ptr::null_mut();
            let hr = co_create_instance(
                &CLSID_MMDEVICE_ENUMERATOR,
                std::ptr::null_mut(),
                CLSCTX_ALL,
                &IID_IMMDEVICE_ENUMERATOR,
                &mut device_enumerator,
            );
            if hr != 0 || device_enumerator.is_null() {
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the device enumerator's creation failed \
                     (0x{:08X})",
                    hr as u32
                )));
            }
            // GetDefaultAudioEndpoint(eRender, eMultimedia).
            let mut device: ComObject = std::ptr::null_mut();
            let get_default: unsafe extern "system" fn(
                this: ComObject,
                data_flow: u32,
                role: u32,
                out: *mut ComObject,
            ) -> i32 = std::mem::transmute(vtable_slot(
                device_enumerator,
                SLOT_IMMDEVICE_GET_DEFAULT,
            ));
            let hr = get_default(
                device_enumerator,
                E_DATAFLOW_RENDER,
                E_ROLE_MULTIMEDIA,
                &mut device,
            );
            if hr != 0 || device.is_null() {
                release_com(device_enumerator);
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the default endpoint's query failed \
                     (0x{:08X})",
                    hr as u32
                )));
            }
            // IMMDevice::Activate(IAudioClient).
            let mut audio_client: ComObject = std::ptr::null_mut();
            let activate: unsafe extern "system" fn(
                this: ComObject,
                iid: *const Guid,
                context: u32,
                params: *mut c_void,
                out: *mut ComObject,
            ) -> i32 =
                std::mem::transmute(vtable_slot(device, SLOT_IMMDEVICE_ACTIVATE));
            let hr = activate(
                device,
                &IID_IAUDIO_CLIENT,
                CLSCTX_ALL,
                std::ptr::null_mut(),
                &mut audio_client,
            );
            release_com(device);
            if hr != 0 || audio_client.is_null() {
                release_com(device_enumerator);
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the audio client's activation failed \
                     (0x{:08X})",
                    hr as u32
                )));
            }
            // IAudioClient::Initialize: the shared mode, the s16
            // interleaved format, the auto-convert flags, the
            // config's period worth of buffer.
            let format = WaveFormatEx {
                format_tag: WAVE_FORMAT_PCM,
                channels: config.channels as u16,
                samples_per_sec: config.sample_rate,
                avg_bytes_per_sec: config.sample_rate * config.channels as u32 * 2,
                block_align: (config.channels as u16) * 2,
                bits_per_sample: 16,
                extra_size: 0,
            };
            let buffer_duration: i64 = (config.period_frames as i64 * 4
                * 10_000_000i64
                / config.sample_rate as i64)
                .max(100_000);
            let initialize: unsafe extern "system" fn(
                this: ComObject,
                share_mode: u32,
                stream_flags: u32,
                buffer_duration: i64,
                periodicity: i64,
                format: *const WaveFormatEx,
                guid: *const Guid,
            ) -> i32 = std::mem::transmute(vtable_slot(
                audio_client,
                SLOT_AUDIOCLIENT_INITIALIZE,
            ));
            let hr = initialize(
                audio_client,
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_AUTOCONVERT,
                buffer_duration,
                0, // the periodicity is 0 in the shared mode
                &format,
                std::ptr::null(),
            );
            if hr != 0 {
                release_com(audio_client);
                release_com(device_enumerator);
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the stream's initialization failed \
                     (0x{:08X})",
                    hr as u32
                )));
            }
            // IAudioClient::GetBufferSize.
            let mut buffer_frames: u32 = 0;
            let get_buffer_size: unsafe extern "system" fn(
                this: ComObject,
                out: *mut u32,
            ) -> i32 = std::mem::transmute(vtable_slot(
                audio_client,
                SLOT_AUDIOCLIENT_GET_BUFFER_SIZE,
            ));
            let hr = get_buffer_size(audio_client, &mut buffer_frames);
            if hr != 0 || buffer_frames == 0 {
                release_com(audio_client);
                release_com(device_enumerator);
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the buffer size's query failed (0x{:08X})",
                    hr as u32
                )));
            }
            // IAudioClient::GetService(IAudioRenderClient).
            let mut render_client: ComObject = std::ptr::null_mut();
            let get_service: unsafe extern "system" fn(
                this: ComObject,
                iid: *const Guid,
                out: *mut *mut c_void,
            ) -> i32 = std::mem::transmute(vtable_slot(
                audio_client,
                SLOT_AUDIOCLIENT_GET_SERVICE,
            ));
            let hr = get_service(
                audio_client,
                &IID_IAUDIO_RENDER_CLIENT,
                &mut render_client as *mut ComObject as *mut *mut c_void,
            );
            if hr != 0 || render_client.is_null() {
                release_com(audio_client);
                release_com(device_enumerator);
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the render client's query failed \
                     (0x{:08X})",
                    hr as u32
                )));
            }
            // The stream starts (the polling writes feed it from
            // here).
            let start: unsafe extern "system" fn(this: ComObject) -> i32 =
                std::mem::transmute(vtable_slot(
                    audio_client,
                    SLOT_AUDIOCLIENT_START,
                ));
            let hr = start(audio_client);
            if hr != 0 {
                release_com(render_client);
                release_com(audio_client);
                release_com(device_enumerator);
                co_uninitialize();
                FreeLibrary(mmdevapi_library);
                FreeLibrary(ole32_library);
                return Err(AudioError(format!(
                    "WasapiAudioBackend: the stream's start failed (0x{:08X})",
                    hr as u32
                )));
            }
            Ok(WasapiAudioBackend {
                config,
                listener: ListenerState::default(),
                voices: Vec::new(),
                next_handle: 1,
                mixed: 0,
                violations: 0,
                shutdown: false,
                device_enumerator,
                audio_client,
                render_client,
                buffer_frames,
                ole32_library,
                mmdevapi_library,
                co_uninitialize: Some(co_uninitialize),
            })
        }
    }

    /// Writes the mixed block to the device: the f32 mix converts to
    /// the interleaved s16 the stream carries; the polling model
    /// sleeps while the device's buffer is full.
    fn write_block(&mut self, frames: &[f32]) {
        let channels = self.config.channels as usize;
        let frame_count = frames.len() / channels;
        if frame_count == 0 {
            return;
        }
        let mut s16 = vec![0i16; frames.len()];
        for (index, &sample) in frames.iter().enumerate() {
            let clamped = sample.clamp(-1.0, 1.0);
            // The symmetric rounding the ALSA backend's s16 path uses.
            let scaled = clamped * 32767.0 + if clamped >= 0.0 { 0.5 } else { -0.5 };
            s16[index] = scaled.max(-32768.0).min(32767.0) as i16;
        }
        unsafe {
            let get_current_padding: unsafe extern "system" fn(
                this: ComObject,
                out: *mut u32,
            ) -> i32 = std::mem::transmute(vtable_slot(
                self.audio_client,
                SLOT_AUDIOCLIENT_GET_CURRENT_PADDING,
            ));
            let get_buffer: unsafe extern "system" fn(
                this: ComObject,
                frames_requested: u32,
                out: *mut *mut u8,
            ) -> i32 = std::mem::transmute(vtable_slot(
                self.render_client,
                SLOT_RENDER_GET_BUFFER,
            ));
            let release_buffer: unsafe extern "system" fn(
                this: ComObject,
                frames_written: u32,
                flags: u32,
            ) -> i32 = std::mem::transmute(vtable_slot(
                self.render_client,
                SLOT_RENDER_RELEASE_BUFFER,
            ));
            let mut offset = 0usize;
            while offset < frame_count {
                let mut padding = 0u32;
                if get_current_padding(self.audio_client, &mut padding) != 0 {
                    break;
                }
                let available = self.buffer_frames.saturating_sub(padding) as usize;
                if available == 0 {
                    Sleep(1);
                    continue;
                }
                let chunk = available.min(frame_count - offset);
                let mut destination: *mut u8 = std::ptr::null_mut();
                if get_buffer(
                    self.render_client,
                    chunk as u32,
                    &mut destination,
                ) != 0
                    || destination.is_null()
                {
                    break;
                }
                std::ptr::copy_nonoverlapping(
                    s16[offset * channels..].as_ptr() as *const u8,
                    destination,
                    chunk * channels * 2,
                );
                if release_buffer(self.render_client, chunk as u32, 0) != 0 {
                    break;
                }
                offset += chunk;
            }
        }
    }
}

/// Releases a COM object through the vtable's IUnknown::Release slot.
unsafe fn release_com(obj: ComObject) {
    if !obj.is_null() {
        let release: unsafe extern "system" fn(this: ComObject) -> u32 =
            std::mem::transmute(vtable_slot(obj, SLOT_RELEASE));
        release(obj);
    }
}

impl Drop for WasapiAudioBackend {
    fn drop(&mut self) {
        unsafe {
            // The stream stops and the objects release in reverse.
            let stop: unsafe extern "system" fn(this: ComObject) -> i32 =
                std::mem::transmute(vtable_slot(
                    self.audio_client,
                    SLOT_AUDIOCLIENT_STOP,
                ));
            let _ = stop(self.audio_client);
            release_com(self.render_client);
            release_com(self.audio_client);
            release_com(self.device_enumerator);
            if let Some(co_uninitialize) = self.co_uninitialize.take() {
                co_uninitialize();
            }
            if !self.mmdevapi_library.is_null() {
                FreeLibrary(self.mmdevapi_library);
                self.mmdevapi_library = std::ptr::null_mut();
            }
            if !self.ole32_library.is_null() {
                FreeLibrary(self.ole32_library);
                self.ole32_library = std::ptr::null_mut();
            }
        }
    }
}

impl AudioBackend for WasapiAudioBackend {
    fn describe(&self) -> String {
        format!(
            "wasapi sample_rate={} channels={} period={} buffer={}",
            self.config.sample_rate,
            self.config.channels,
            self.config.period_frames,
            self.buffer_frames
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
