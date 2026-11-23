//! The shared mix loop: the per-voice cursor walk, the gain/3D
//! attenuation and the interleaved block the backends' `mix_frames`
//! produce. The null backend and the real device backends (ALSA now,
//! WASAPI/CoreAudio/AAudio with their hardware runs) run this same
//! arithmetic — the deterministic checks and the hardware path mix
//! identically, the milestone rule.

use super::{ListenerState, PcmBuffer, VoiceState, VoicePlacement};

/// A mix loop's voice record (the backend's own storage keeps this
/// shape; the contract's voice API maps onto it).
pub struct MixVoice {
    pub handle: u32,
    pub buffer: PcmBuffer,
    pub cursor_frames: u64,
    pub looping: bool,
    pub state: VoiceState,
    pub placement: VoicePlacement,
    pub released: bool,
}

/// Runs the mix: `frames` output frames at `channels` interleaved,
/// advancing the voices' cursors and the caller's mixed counter, and
/// returning the interleaved block. The attenuation is the contract's
/// linear falloff from the reference distance to the max distance.
pub fn run(
    voices: &mut [MixVoice],
    listener: &ListenerState,
    frames: u32,
    channels: usize,
    mixed: &mut u64,
) -> Vec<f32> {
    let mut block = vec![0.0f32; frames as usize * channels];
    for voice in voices.iter_mut() {
        if voice.released || voice.state != VoiceState::Playing {
            continue;
        }
        let voice_channels = voice.buffer.channels as usize;
        let voice_frames = voice.buffer.frames() as u64;
        if voice_frames == 0 {
            voice.state = VoiceState::Stopped;
            continue;
        }
        // The 3D attenuation: linear falloff from the reference
        // distance to the max distance, clamped to silence past it.
        let dx = voice.placement.position[0] - listener.position[0];
        let dy = voice.placement.position[1] - listener.position[1];
        let dz = voice.placement.position[2] - listener.position[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let attenuation = if distance <= voice.placement.reference_distance {
            1.0
        } else if distance >= voice.placement.max_distance {
            0.0
        } else {
            1.0 - (distance - voice.placement.reference_distance)
                / (voice.placement.max_distance - voice.placement.reference_distance)
        };
        let voice_gain = voice.placement.gain * attenuation;
        for sample_slot in block.chunks_exact_mut(channels) {
            if voice.state != VoiceState::Playing {
                break;
            }
            let frame = voice.cursor_frames % voice_frames;
            for c in 0..channels {
                let source = if voice_channels == channels {
                    (frame as usize) * voice_channels + c
                } else {
                    // Mono into stereo: the same sample both sides.
                    (frame as usize) * voice_channels
                };
                sample_slot[c] += voice.buffer.samples[source] * voice_gain;
            }
            voice.cursor_frames += 1;
            if !voice.looping && voice.cursor_frames >= voice_frames {
                voice.state = VoiceState::Stopped;
            }
        }
    }
    *mixed += frames as u64;
    block
}
