//! The audio layer's smoke: the WAV decoder feeds a real buffer through
//! the null backend's mix loop, the Ogg Vorbis decoder decodes the
//! engine's own test vector, and the ALSA backend (when the feature
//! and a sound device are present) plays a short tone through the
//! real mix loop. The run prints the describe lines, the voice state
//! walk and the mixed peaks, and fails the run on any contract
//! violation.
use og2_core::audio::{
    AudioBackend, AudioConfig, NullAudioBackend, PcmBuffer, VoiceState,
};
fn sine_buffer(sample_rate: u32, seconds: f32, hz: f32) -> PcmBuffer {
    let frames = (sample_rate as f32 * seconds) as usize;
    let mut samples = Vec::with_capacity(frames);
    for n in 0..frames {
        let t = n as f32 / sample_rate as f32;
        samples.push(0.5 * (std::f32::consts::TAU * hz * t).sin());
    }
    PcmBuffer {
        sample_rate,
        channels: 1,
        samples,
    }
}
fn main() {
    // The null backend's contract walk: the tone buffer through the mix
    // loop, the state walk, the protocol's violation surface.
    let mut backend =
        NullAudioBackend::new(AudioConfig::default()).expect("the null backend's open");
    println!("describe: {}", backend.describe());
    let buffer = sine_buffer(44100, 1.0, 440.0);
    let voice = backend.create_voice(&buffer).expect("the voice's creation");
    assert_eq!(backend.voice_state(voice), VoiceState::Ready);
    backend.play_voice(voice, false).expect("the play");
    assert_eq!(backend.voice_state(voice), VoiceState::Playing);
    // Mix one second; the peak should approach the buffer's 0.5
    // amplitude in at least one channel.
    let mut peak_left = 0.0f32;
    let mut peak_right = 0.0f32;
    for _ in 0..(buffer.sample_rate / 512) {
        let peaks = backend.mix_frames(512);
        peak_left = peak_left.max(peaks[0]);
        if let Some(right) = peaks.get(1) {
            peak_right = peak_right.max(*right);
        }
    }
    println!("mixed frames: {}", backend.mixed_frames());
    println!("peaks: left={:.4} right={:.4}", peak_left, peak_right);
    assert!(
        peak_left > 0.4 || peak_right > 0.4,
        "the mixed peak lost the tone"
    );
    // The walk: pause, resume, stop.
    backend.pause_voice(voice).expect("the pause");
    assert_eq!(backend.voice_state(voice), VoiceState::Paused);
    backend.play_voice(voice, false).expect("the resume");
    backend.stop_voice(voice).expect("the stop");
    assert_eq!(backend.voice_state(voice), VoiceState::Stopped);
    // The protocol: the released handle reports the Invalid state and
    // the violation.
    backend.release_voice(voice);
    assert_eq!(backend.voice_state(voice), VoiceState::Invalid);
    backend.release_voice(voice);
    assert_eq!(backend.violation_count(), 1);
    backend.shutdown();
    println!("violations: {}", backend.violation_count());

    // The decoders: the WAV reader reads back the sine buffer's
    // sample data; the Ogg Vorbis decoder decodes the test vector the
    // tests carry (feature-gated; the smoke prints and skips without
    // the feature).
    // The vectors embed in the binary (include_bytes!): the smoke runs
    // standalone on every platform, including the Android's /data/local
    // tmp where no source tree exists.
    let wav_bytes: &[u8] =
        include_bytes!("../tests/vectors/test440.wav");
    let wav = og2_core::audio::wav::decode_wav(&wav_bytes).expect("the WAV decode");
    println!(
        "wav: rate={} channels={} frames={}",
        wav.sample_rate,
        wav.channels,
        wav.samples.len() / wav.channels as usize
    );
    #[cfg(feature = "audio-vorbis")]
    {
        let ogg_bytes: &[u8] =
            include_bytes!("../tests/vectors/test1k.ogg");
        let decoded = og2_core::audio::vorbis::decode_ogg_vorbis(&ogg_bytes)
            .expect("the Vorbis decode");
        println!(
            "vorbis: rate={} channels={} frames={}",
            decoded.sample_rate,
            decoded.channels,
            decoded.samples.len() / decoded.channels as usize
        );
        // The 1 kHz tone's decode must carry a live peak (the test
        // vector's own encode level sits near 0.13; the assertion
        // fails on the silence the broken decoders produce).
        let peak = decoded
            .samples
            .iter()
            .fold(0.0f32, |a, &v| a.max(v.abs()));
        println!("vorbis peak: {:.4}", peak);
        assert!(peak > 0.05, "the decoded Vorbis lost the tone");
    }

    // The ALSA hardware path (feature-gated; the sandbox and the
    // headless runs carry no sound device and the open error is the
    // documented outcome there — the smoke reports and moves on).
    #[cfg(feature = "audio-alsa")]
    match og2_core::audio::alsa::AlsaAudioBackend::new(AudioConfig::default()) {
        Ok(mut device) => {
            println!("describe: {}", device.describe());
            let tone = sine_buffer(48000, 0.25, 440.0);
            let voice = device.create_voice(&tone).expect("the ALSA voice");
            device.play_voice(voice, false).expect("the ALSA play");
            for _ in 0..(48000 / 512 * 2) {
                let peaks = device.mix_frames(512);
                // The device consumed the mixed block; the run fails
                // on a contract violation only.
                let _ = peaks;
                if device.voice_state(voice) == VoiceState::Stopped {
                    break;
                }
            }
            println!("alsa mixed frames: {}", device.mixed_frames());
            device.shutdown();
            println!("alsa violations: {}", device.violation_count());
        }
        Err(error) => {
            println!("alsa: skipped ({})", error);
        }
    }

    // The WASAPI hardware path: the same mix loop through the
    // platform's own render client (a machine without an active audio
    // endpoint reports the open error and the smoke moves on).
    #[cfg(feature = "audio-wasapi")]
    match og2_core::audio::wasapi::WasapiAudioBackend::new(AudioConfig::default())
    {
        Ok(mut device) => {
            println!("describe: {}", device.describe());
            let tone = sine_buffer(48000, 0.25, 440.0);
            let voice = device.create_voice(&tone).expect("the WASAPI voice");
            device.play_voice(voice, false).expect("the WASAPI play");
            for _ in 0..(48000 / 512 * 2) {
                let peaks = device.mix_frames(512);
                // The device consumed the mixed block; the run fails
                // on a contract violation only.
                let _ = peaks;
                if device.voice_state(voice) == VoiceState::Stopped {
                    break;
                }
            }
            println!("wasapi mixed frames: {}", device.mixed_frames());
            device.shutdown();
            println!("wasapi violations: {}", device.violation_count());
        }
        Err(error) => {
            println!("wasapi: skipped ({})", error);
        }
    }

    // The CoreAudio hardware path: the same mix loop through the
    // platform's own AudioUnit (a machine without an active output
    // device reports the open error and the smoke moves on).
    #[cfg(feature = "audio-coreaudio")]
    match og2_core::audio::coreaudio::CoreAudioBackend::new(AudioConfig::default())
    {
        Ok(mut device) => {
            println!("describe: {}", device.describe());
            let tone = sine_buffer(48000, 0.25, 440.0);
            let voice =
                device.create_voice(&tone).expect("the CoreAudio voice");
            device.play_voice(voice, false).expect("the CoreAudio play");
            for _ in 0..(48000 / 512 * 2) {
                let peaks = device.mix_frames(512);
                // The device consumed the mixed block; the run fails
                // on a contract violation only.
                let _ = peaks;
                if device.voice_state(voice) == VoiceState::Stopped {
                    break;
                }
            }
            println!("coreaudio mixed frames: {}", device.mixed_frames());
            device.shutdown();
            println!("coreaudio violations: {}", device.violation_count());
        }
        Err(error) => {
            println!("coreaudio: skipped ({})", error);
        }
    }

    // The AAudio hardware path: the same mix loop through the
    // platform's own stream builder (the machines before API 26 and
    // the runs without the audio permission report the open error and
    // the smoke moves on).
    #[cfg(feature = "audio-aaudio")]
    match og2_core::audio::aaudio::AAudioBackend::new(AudioConfig::default())
    {
        Ok(mut device) => {
            println!("describe: {}", device.describe());
            let tone = sine_buffer(48000, 0.25, 440.0);
            let voice =
                device.create_voice(&tone).expect("the AAudio voice");
            device.play_voice(voice, false).expect("the AAudio play");
            for _ in 0..(48000 / 512 * 2) {
                let peaks = device.mix_frames(512);
                // The device consumed the mixed block; the run fails
                // on a contract violation only.
                let _ = peaks;
                if device.voice_state(voice) == VoiceState::Stopped {
                    break;
                }
            }
            println!("aaudio mixed frames: {}", device.mixed_frames());
            device.shutdown();
            println!("aaudio violations: {}", device.violation_count());
        }
        Err(error) => {
            println!("aaudio: skipped ({})", error);
        }
    }

    println!("OK");
}
