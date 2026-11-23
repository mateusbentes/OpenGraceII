//! The ALSA backend: the Linux hardware output the engine opens
//! directly through the platform's `libasound.so.2`. The loader is the
//! engine's own dlopen/dlsym surface (the same rule the render
//! backends carry: no third-party binding, the platform's documented
//! C ABI only).
//!
//! The device path: `snd_pcm_open` on the default PCM device in
//! blocking-write mode, `snd_pcm_set_params` for the interleaved
//! s16-native-endian format at the config's rate and channel count,
//! and `snd_pcm_writei` pushing each mixed frame block. The mix loop
//! itself is the contract's shared arithmetic — the backend mixes the
//! voices into a frame buffer through the same `mix_frames` walk and
//! writes the s16 conversion of the block to the device, so the
//! deterministic checks and the hardware path run the identical mix.
//!
//! Underruns: `snd_pcm_writei` returns a negative errno-shaped count
//! (`-EPIPE` on the underrun, `-ESTRPIPE` on the suspend); the backend
//! recovers through `snd_pcm_recover` and rewrites the block, the
//! documented resilient-write pattern.

use super::{AudioBackend, AudioConfig, AudioError, ListenerState, PcmBuffer,
    VoiceHandle, VoicePlacement, VoiceState};
use crate::audio::mix::{self, MixVoice};

// The resolved function pointers (the C ABI's own signatures, usize
// pointers the dlsym surface hands back; the call sites transmute to
// the documented ALSA prototypes).

type SndPcmOpen = unsafe extern "C" fn(
    pcm: *mut *mut libc::c_void,
    name: *const libc::c_char,
    stream: libc::c_int,
    mode: libc::c_int,
) -> libc::c_int;
type SndPcmSetParams = unsafe extern "C" fn(
    pcm: *mut libc::c_void,
    format: libc::c_int,
    access: libc::c_int,
    channels: libc::c_uint,
    rate: libc::c_uint,
    soft_resample: libc::c_int,
    latency: libc::c_uint,
) -> libc::c_int;
type SndPcmWritei = unsafe extern "C" fn(
    pcm: *mut libc::c_void,
    buffer: *const libc::c_void,
    size: libc::c_ulong,
) -> libc::c_long;
type SndPcmRecover = unsafe extern "C" fn(
    pcm: *mut libc::c_void,
    err: libc::c_int,
    silent: libc::c_int,
) -> libc::c_int;
type SndPcmClose = unsafe extern "C" fn(pcm: *mut libc::c_void) -> libc::c_int;

// The PCM stream direction (the pcm.h SND_PCM_STREAM_PLAYBACK) and the
// blocking mode (SND_PCM_NONBLOCK == 0).
const SND_PCM_STREAM_PLAYBACK: libc::c_int = 0;
const SND_PCM_NONBLOCK: libc::c_int = 0;
// The interleaved access (SND_PCM_ACCESS_RW_INTERLEAVED).
const SND_PCM_ACCESS_RW_INTERLEAVED: libc::c_int = 3;
// The s16 formats in the pcm.h SND_PCM_FORMAT_* order (S16LE = 2,
// S16BE = 1); the little-endian machines take S16LE.
#[cfg(target_endian = "little")]
const SND_PCM_FORMAT_S16_NATIVE: libc::c_int = 2;
#[cfg(target_endian = "big")]
const SND_PCM_FORMAT_S16_NATIVE: libc::c_int = 1;

/// The ALSA backend: the opened PCM handle plus the shared mix state.
pub struct AlsaAudioBackend {
    config: AudioConfig,
    listener: ListenerState,
    voices: Vec<MixVoice>,
    next_handle: u32,
    mixed: u64,
    violations: u64,
    shutdown: bool,
    handle: *mut libc::c_void,
    pcm_writei: SndPcmWritei,
    pcm_recover: SndPcmRecover,
    pcm_close: SndPcmClose,
}

impl AlsaAudioBackend {
    /// Opens the default playback device through the engine's dlopen
    /// surface and configures the interleaved s16 stream.
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError(
                "AlsaAudioBackend: the sample rate must be positive".to_string(),
            ));
        }
        if config.channels != 1 && config.channels != 2 {
            return Err(AudioError(
                "AlsaAudioBackend: the channel count must be 1 or 2".to_string(),
            ));
        }
        if config.period_frames == 0 {
            return Err(AudioError(
                "AlsaAudioBackend: the period must carry frames".to_string(),
            ));
        }
        unsafe {
            let lib = libc::dlopen(
                b"libasound.so.2\0".as_ptr() as *const libc::c_char,
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            );
            if lib.is_null() {
                return Err(AudioError(
                    "AlsaAudioBackend: the libasound.so.2 load failed (is ALSA \
                     installed?)"
                        .to_string(),
                ));
            }
            let resolve = |name: &[u8]| -> Result<*mut libc::c_void, AudioError> {
                let symbol = libc::dlsym(lib, name.as_ptr() as *const libc::c_char);
                if symbol.is_null() {
                    return Err(AudioError(format!(
                        "AlsaAudioBackend: the {} symbol is missing",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    )));
                }
                Ok(symbol)
            };
            let pcm_open: SndPcmOpen =
                std::mem::transmute(resolve(b"snd_pcm_open\0")?);
            let pcm_set_params: SndPcmSetParams =
                std::mem::transmute(resolve(b"snd_pcm_set_params\0")?);
            let pcm_writei: SndPcmWritei =
                std::mem::transmute(resolve(b"snd_pcm_writei\0")?);
            let pcm_recover: SndPcmRecover =
                std::mem::transmute(resolve(b"snd_pcm_recover\0")?);
            let pcm_close: SndPcmClose =
                std::mem::transmute(resolve(b"snd_pcm_close\0")?);
            let mut handle: *mut libc::c_void = std::ptr::null_mut();
            let result = pcm_open(
                &mut handle,
                b"default\0".as_ptr() as *const libc::c_char,
                SND_PCM_STREAM_PLAYBACK,
                SND_PCM_NONBLOCK,
            );
            if result != 0 || handle.is_null() {
                return Err(AudioError(format!(
                    "AlsaAudioBackend: the default PCM device's open failed ({})",
                    result
                )));
            }
            // The latency in microseconds: the period's frames worth of
            // playback latency (the config's knob mapped onto ALSA's
            // soft-latency parameter).
            let latency =
                (config.period_frames as u64 * 1_000_000 / config.sample_rate as u64)
                    .max(1000) as libc::c_uint;
            let result = pcm_set_params(
                handle,
                SND_PCM_FORMAT_S16_NATIVE,
                SND_PCM_ACCESS_RW_INTERLEAVED,
                config.channels,
                config.sample_rate,
                1, // the soft resample is on (the device's rate mismatch path)
                latency,
            );
            if result != 0 {
                pcm_close(handle);
                return Err(AudioError(format!(
                    "AlsaAudioBackend: the PCM parameters' setup failed ({})",
                    result
                )));
            }
            Ok(AlsaAudioBackend {
                config,
                listener: ListenerState::default(),
                voices: Vec::new(),
                next_handle: 1,
                mixed: 0,
                violations: 0,
                shutdown: false,
                handle,
                pcm_writei,
                pcm_recover,
                pcm_close,
            })
        }
    }

    /// Writes the mixed block to the device: the f32 mix converts to
    /// the interleaved s16 the PCM stream carries; the underrun and
    /// suspend recoveries rewrite the remainder.
    fn write_block(&mut self, frames: &[f32]) {
        let channels = self.config.channels as usize;
        let frame_count = frames.len() / channels;
        if frame_count == 0 {
            return;
        }
        let mut s16 = vec![0i16; frames.len()];
        for (index, &sample) in frames.iter().enumerate() {
            let clamped = sample.clamp(-1.0, 1.0);
            // The symmetric rounding the WAV decoder's s16 path uses.
            let scaled = clamped * 32767.0 + if clamped >= 0.0 { 0.5 } else { -0.5 };
            s16[index] = scaled.max(-32768.0).min(32767.0) as i16;
        }
        unsafe {
            let mut offset = 0usize;
            while offset < frame_count {
                let written = (self.pcm_writei)(
                    self.handle,
                    s16[offset * channels..].as_ptr() as *const libc::c_void,
                    (frame_count - offset) as libc::c_ulong,
                );
                if written > 0 {
                    offset += written as usize;
                    continue;
                }
                let error = -written as libc::c_int;
                // The EPIPE (underrun, 32) and ESTRPIPE (suspend, 86)
                // recoveries; other errors stop the writes (the
                // violation counter notes them through the mixed walk).
                if (self.pcm_recover)(self.handle, error, 1) == 0 {
                    continue;
                }
                break;
            }
        }
    }
}

impl Drop for AlsaAudioBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.pcm_close)(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl AudioBackend for AlsaAudioBackend {
    fn describe(&self) -> String {
        format!(
            "alsa sample_rate={} channels={} period={}",
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
