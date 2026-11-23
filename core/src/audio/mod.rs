//! The audio layer's API-neutral contract: the voice playback the app
//! logic consumes and the device output the per-platform backends
//! implement. The same rule the render and platform contracts carry:
//! the shared module never contains platform-API source — the
//! per-platform implementations live in their own files gated behind
//! their own features, and the contract's types carry only the data
//! the app logic and the backends need.
//!
//! The milestone 10 contract follows the render contract's shape: one
//! trait (`AudioBackend`) with the API-neutral surface, one
//! deterministic null backend the CI's headless checks run through,
//! and the decoders (the WAV and the Ogg Vorbis readers, the engine's
//! own code) feeding the backend's voices.
//!
//! The audio output model: the backend owns the device's mix loop (the
//! platform's own callback path — ALSA's poll, WASAPI's event-driven
//! render client, CoreAudio's render callback, AAudio's callback).
//! The app pushes decoded PCM into voices; the mix loop pulls the
//! voices' samples each callback, applies the per-voice gain and the
//! 3D attenuation, mixes into the device frame and reports the mixed
//! peak through the readback the deterministic checks read.

/// The audio device's opening configuration. The backend validates the
/// sample rate (positive) and the channel count (1 or 2 — the contract's
/// first milestone carries mono and stereo output; the surround
/// expansion follows with the 3D attenuation milestone).
#[derive(Clone, Debug)]
pub struct AudioConfig {
    /// The mix loop's sample rate in Hz (44100 and 48000 are the
    /// devices' common rates).
    pub sample_rate: u32,
    /// The output channel count (1 mono, 2 stereo).
    pub channels: u32,
    /// The mix loop's frames per callback period (the latency's knob;
    /// 512 is the balanced default the backends document).
    pub period_frames: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        AudioConfig {
            sample_rate: 44100,
            channels: 2,
            period_frames: 512,
        }
    }
}

/// A decoded PCM buffer the backend's voice carries. The samples are
/// the contract's own interleaved f32 form (the decoders normalise the
/// WAV's 16-bit and the Vorbis's float scale onto [-1.0, 1.0]).
#[derive(Clone, Debug)]
pub struct PcmBuffer {
    pub sample_rate: u32,
    pub channels: u32,
    /// The interleaved samples (frames * channels).
    pub samples: Vec<f32>,
}

impl PcmBuffer {
    /// The buffer's frame count (the interleaved length divided by the
    /// channel count).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }
}

/// A voice's identifier the backend's creation call hands back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoiceHandle(pub u32);

/// A voice's playback state the readback and the polling report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceState {
    /// The voice exists but never started (the play call moves it).
    Ready,
    /// The voice is mixing into the device frames.
    Playing,
    /// The play call's pause moved the voice out of the mix.
    Paused,
    /// The voice's buffer finished (or the stop call ended it). A
    /// looping voice never reaches this state on its own.
    Stopped,
    /// The voice's handle is stale (released, or the backend shut
    /// down).
    Invalid,
}

/// A voice's spatial position/gain description: the 3D attenuation's
/// inputs. The listener's own state sits on the backend (one listener
/// per device).
#[derive(Clone, Copy, Debug)]
pub struct VoicePlacement {
    /// The source's world position (x, y, z).
    pub position: [f32; 3],
    /// The reference distance (the gain starts falling past it) and
    /// the falloff's max distance.
    pub reference_distance: f32,
    pub max_distance: f32,
    /// The linear gain multiplier [0.0, 1.0+] applied before the 3D
    /// attenuation.
    pub gain: f32,
}

impl Default for VoicePlacement {
    fn default() -> Self {
        VoicePlacement {
            position: [0.0, 0.0, 0.0],
            reference_distance: 1.0,
            max_distance: 100.0,
            gain: 1.0,
        }
    }
}

/// The listener's world state the 3D attenuation reads.
#[derive(Clone, Copy, Debug)]
pub struct ListenerState {
    /// The listener's world position (x, y, z).
    pub position: [f32; 3],
    /// The listener's facing (the forward axis) and up axis.
    pub forward: [f32; 3],
    pub up: [f32; 3],
}

impl Default for ListenerState {
    fn default() -> Self {
        ListenerState {
            position: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
        }
    }
}

/// The audio contract's error type: the open, the decode and the
/// protocol's own failures carry one string each, the same rule the
/// render contract's RenderError follows.
#[derive(Clone, Debug)]
pub struct AudioError(pub String);

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for AudioError {}

impl From<&str> for AudioError {
    fn from(message: &str) -> Self {
        AudioError(message.to_string())
    }
}

impl From<String> for AudioError {
    fn from(message: String) -> Self {
        AudioError(message)
    }
}

/// The audio backend's API-neutral contract. The lifetime carries the
/// device's mix loop's handle ownership the backends record.
pub trait AudioBackend {
    /// The backend's describe line the smoke runs print.
    fn describe(&self) -> String;

    /// Creates a voice carrying the buffer. The voice starts Ready.
    fn create_voice(&mut self, buffer: &PcmBuffer) -> Result<VoiceHandle, AudioError>;

    /// Releases a voice. The stale handle's later calls report the
    /// violation and the Invalid state.
    fn release_voice(&mut self, voice: VoiceHandle);

    /// Starts (or resumes) the voice's playback.
    fn play_voice(&mut self, voice: VoiceHandle, looping: bool) -> Result<(), AudioError>;

    /// Pauses the voice (the resume is play_voice).
    fn pause_voice(&mut self, voice: VoiceHandle) -> Result<(), AudioError>;

    /// Stops the voice and rewinds it to the buffer's start.
    fn stop_voice(&mut self, voice: VoiceHandle) -> Result<(), AudioError>;

    /// Updates a voice's placement (the gain and the 3D position).
    fn set_voice_placement(&mut self, voice: VoiceHandle, placement: &VoicePlacement);

    /// Updates the listener's world state.
    fn set_listener(&mut self, listener: &ListenerState);

    /// The voice's current state.
    fn voice_state(&self, voice: VoiceHandle) -> VoiceState;

    /// Advances the mix loop by the frame count the caller names (the
    /// deterministic path the null backend and the readback checks
    /// drive; the real backends call it from their callback) and
    /// returns the mixed frame's peak magnitude per channel. The
    /// return's length is the config's channel count.
    fn mix_frames(&mut self, frames: u32) -> Vec<f32>;

    /// The mix loop's cursor: the total frames mixed since the open.
    fn mixed_frames(&self) -> u64;

    /// Shuts the backend down (the voices release, the device closes).
    /// Calls after the shutdown report the violation.
    fn shutdown(&mut self);

    /// The violation counter the protocol's checks read (the stale
    /// handle's use, the post-shutdown call).
    fn violation_count(&self) -> u64;
}

/// The deterministic null backend: the CI's headless surface. The mix
/// loop runs the same arithmetic the real backends run (the per-voice
/// cursor walk, the gain/3D attenuation, the peak per channel) but the
/// device's output goes nowhere; the mixed peaks and the voice states
/// carry the checks.
pub struct NullAudioBackend {
    config: AudioConfig,
    listener: ListenerState,
    voices: Vec<mix::MixVoice>,
    next_handle: u32,
    mixed: u64,
    violations: u64,
    shutdown: bool,
}


impl NullAudioBackend {
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError(
                "NullAudioBackend: the sample rate must be positive".to_string(),
            ));
        }
        if config.channels != 1 && config.channels != 2 {
            return Err(AudioError(
                "NullAudioBackend: the channel count must be 1 or 2".to_string(),
            ));
        }
        if config.period_frames == 0 {
            return Err(AudioError(
                "NullAudioBackend: the period must carry frames".to_string(),
            ));
        }
        Ok(NullAudioBackend {
            config,
            listener: ListenerState::default(),
            voices: Vec::new(),
            next_handle: 1,
            mixed: 0,
            violations: 0,
            shutdown: false,
        })
    }

    fn find_voice(&self, handle: VoiceHandle) -> Option<&mix::MixVoice> {
        self.voices
            .iter()
            .find(|v| v.handle == handle.0 && !v.released)
    }

    fn find_voice_mut(&mut self, handle: VoiceHandle) -> Option<&mut mix::MixVoice> {
        self.voices
            .iter_mut()
            .find(|v| v.handle == handle.0 && !v.released)
    }
}

impl AudioBackend for NullAudioBackend {
    fn describe(&self) -> String {
        format!(
            "null audio sample_rate={} channels={} period={}",
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
        let handle = self.next_handle;
        self.next_handle += 1;
        self.voices.push(mix::MixVoice {
            handle,
            buffer: buffer.clone(),
            cursor_frames: 0,
            looping: false,
            state: VoiceState::Ready,
            placement: VoicePlacement::default(),
            released: false,
        });
        Ok(VoiceHandle(handle))
    }

    fn release_voice(&mut self, voice: VoiceHandle) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self.voices.iter_mut().find(|v| v.handle == voice.0) {
            Some(v) => {
                if v.released {
                    self.violations += 1;
                } else {
                    v.released = true;
                }
            }
            None => self.violations += 1,
        }
    }

    fn play_voice(&mut self, voice: VoiceHandle, looping: bool) -> Result<(), AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError("play_voice: the backend is shut down".to_string()));
        }
        match self.find_voice_mut(voice) {
            Some(v) => {
                v.looping = looping;
                if v.state != VoiceState::Playing {
                    v.state = VoiceState::Playing;
                }
                Ok(())
            }
            None => {
                self.violations += 1;
                Err(AudioError("play_voice: the voice is not owned".to_string()))
            }
        }
    }

    fn pause_voice(&mut self, voice: VoiceHandle) -> Result<(), AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError("pause_voice: the backend is shut down".to_string()));
        }
        match self.find_voice_mut(voice) {
            Some(v) => {
                if v.state == VoiceState::Playing {
                    v.state = VoiceState::Paused;
                }
                Ok(())
            }
            None => {
                self.violations += 1;
                Err(AudioError("pause_voice: the voice is not owned".to_string()))
            }
        }
    }

    fn stop_voice(&mut self, voice: VoiceHandle) -> Result<(), AudioError> {
        if self.shutdown {
            self.violations += 1;
            return Err(AudioError("stop_voice: the backend is shut down".to_string()));
        }
        match self.find_voice_mut(voice) {
            Some(v) => {
                v.state = VoiceState::Stopped;
                v.cursor_frames = 0;
                Ok(())
            }
            None => {
                self.violations += 1;
                Err(AudioError("stop_voice: the voice is not owned".to_string()))
            }
        }
    }

    fn set_voice_placement(&mut self, voice: VoiceHandle, placement: &VoicePlacement) {
        if self.shutdown {
            self.violations += 1;
            return;
        }
        match self.find_voice_mut(voice) {
            Some(v) => v.placement = *placement,
            None => self.violations += 1,
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
        match self.find_voice(voice) {
            Some(v) => v.state,
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
        // The peaks: the block's per-channel maxima (the null device's
        // "output" the deterministic checks read).
        for (index, &sample) in block.iter().enumerate() {
            let magnitude = sample.abs();
            if magnitude > peaks[index % channels] {
                peaks[index % channels] = magnitude;
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

pub mod wav;
pub mod ogg;
pub mod vorbis_bitreader;
pub mod mix;
#[cfg(feature = "audio-vorbis")]
pub mod vorbis;
// The ALSA backend is the Linux/Unix hardware path.
#[cfg(all(unix, feature = "audio-alsa"))]
pub mod alsa;
// The WASAPI backend is the Windows hardware path.
#[cfg(all(windows, feature = "audio-wasapi"))]
pub mod wasapi;
// The CoreAudio backend is the macOS hardware path.
#[cfg(all(target_os = "macos", feature = "audio-coreaudio"))]
pub mod coreaudio;
// The AAudio backend is the Android hardware path (the dlopen of
// libaaudio.so keeps the minSdk at 24; the machines before API 26
// report the open error and stay on the null path).
#[cfg(all(target_os = "android", feature = "audio-aaudio"))]
pub mod aaudio;
