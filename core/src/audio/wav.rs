//! The engine's own WAV reader: the RIFF container's parse and the
//! PCM formats the contract's voices carry (16-bit integer and 32-bit
//! float, mono and stereo). No third-party binding — the chunk walk
//! follows the RIFF specification directly, the same rule the render
//! backends' loaders record.
//!
//! The parser's honesty rules:
//! - the RIFF header's `WAVE` form and the `fmt ` chunk's PCM (1) and
//!   IEEE float (3) tags are the only accepted encodings (the extensible
//!   WAVE_FORMAT_EXTENSIBLE tag follows with the surround milestone);
//! - the `data` chunk's byte count drives the sample count; a truncated
//!   data chunk is an error, not a silent short read;
//! - the samples normalise onto the contract's interleaved f32
//!   [-1.0, 1.0] range (the 16-bit path divides by 32768.0, the float
//!   path passes through).

use super::{AudioError, PcmBuffer};

/// The RIFF little-endian four-character code read.
fn read_fourcc(data: &[u8], offset: usize) -> Option<[u8; 4]> {
    if offset + 4 > data.len() {
        return None;
    }
    let mut tag = [0u8; 4];
    tag.copy_from_slice(&data[offset..offset + 4]);
    Some(tag)
}

/// The little-endian u16/u32 reads the header fields use.
fn read_u16(data: &[u8], offset: usize) -> Option<u16> {
    if offset + 2 > data.len() {
        return None;
    }
    Some(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 > data.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Parses a WAV byte stream into the contract's PCM buffer. The errors
/// name the parse stage the failure happened at (the header, the
/// format, the data), the same diagnostic rule the render loaders
/// carry.
pub fn decode_wav(data: &[u8]) -> Result<PcmBuffer, AudioError> {
    // The RIFF header: "RIFF" + the chunk size + "WAVE".
    match read_fourcc(data, 0) {
        Some([b'R', b'I', b'F', b'F']) => {}
        _ => return Err(AudioError::from("wav: the RIFF header is missing")),
    }
    let riff_size = read_u32(data, 4)
        .ok_or_else(|| AudioError::from("wav: the RIFF size field is truncated"))?;
    let riff_end = (riff_size as usize)
        .checked_add(8)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| AudioError::from("wav: the RIFF size exceeds the buffer"))?;
    match read_fourcc(data, 8) {
        Some([b'W', b'A', b'V', b'E']) => {}
        _ => return Err(AudioError::from("wav: the WAVE form is missing")),
    }
    // The chunk walk: the fmt chunk's fields and the data chunk's bytes.
    let mut offset = 12usize;
    let mut channels = 0u32;
    let mut sample_rate = 0u32;
    let mut format_tag = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data_range: Option<(usize, usize)> = None;
    while offset + 8 <= riff_end {
        let tag = read_fourcc(data, offset)
            .ok_or_else(|| AudioError::from("wav: the chunk walk ran off the riff"))?;
        let size =
            read_u32(data, offset + 4)
                .ok_or_else(|| AudioError::from("wav: the chunk size is truncated"))? as usize;
        let body = offset + 8;
        let body_end = body
            .checked_add(size)
            .filter(|&end| end <= data.len())
            .ok_or_else(|| AudioError::from("wav: the chunk body exceeds the buffer"))?;
        match tag {
            [b'f', b'm', b't', b' '] => {
                if size < 16 {
                    return Err(AudioError::from("wav: the fmt chunk is too short"));
                }
                format_tag = read_u16(data, body)
                    .ok_or_else(|| AudioError::from("wav: the format tag is truncated"))?;
                channels =
                    read_u16(data, body + 2)
                        .ok_or_else(|| AudioError::from("wav: the channel field is truncated"))?
                        as u32;
                sample_rate =
                    read_u32(data, body + 4)
                        .ok_or_else(|| AudioError::from("wav: the rate field is truncated"))?;
                bits_per_sample = read_u16(data, body + 14)
                    .ok_or_else(|| AudioError::from("wav: the bits field is truncated"))?;
            }
            [b'd', b'a', b't', b'a'] => {
                data_range = Some((body, body_end));
            }
            _ => {}
        }
        // The chunks pad to even sizes; the walk honours the pad byte.
        offset = body_end + (size & 1);
    }
    let (data_start, data_end) = data_range
        .ok_or_else(|| AudioError::from("wav: the data chunk is missing"))?;
    if channels != 1 && channels != 2 {
        return Err(AudioError::from(format!(
            "wav: the channel count {} is outside the contract's mono/stereo",
            channels
        )));
    }
    if sample_rate == 0 {
        return Err(AudioError::from("wav: the sample rate must be positive"));
    }
    let bytes_per_sample = match (format_tag, bits_per_sample) {
        (1, 16) => 2usize,
        (3, 32) => 4usize,
        (tag, bits) => {
            return Err(AudioError::from(format!(
                "wav: the format tag {} with {} bits is outside the contract's \
                 16-bit PCM and 32-bit float",
                tag, bits
            )))
        }
    };
    let frame_bytes = bytes_per_sample * channels as usize;
    if (data_end - data_start) % frame_bytes != 0 {
        return Err(AudioError::from(
            "wav: the data chunk's size is not a whole frame count",
        ));
    }
    let frames = (data_end - data_start) / frame_bytes;
    let mut samples = Vec::with_capacity(frames * channels as usize);
    for frame in 0..frames {
        for channel in 0..channels as usize {
            let base = data_start + frame * frame_bytes + channel * bytes_per_sample;
            match format_tag {
                1 => {
                    let raw = i16::from_le_bytes([data[base], data[base + 1]]);
                    samples.push(raw as f32 / 32768.0);
                }
                _ => {
                    let raw = f32::from_le_bytes([
                        data[base],
                        data[base + 1],
                        data[base + 2],
                        data[base + 3],
                    ]);
                    samples.push(raw);
                }
            }
        }
    }
    Ok(PcmBuffer {
        sample_rate,
        channels,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal PCM 16-bit mono WAV byte stream.
    fn build_wav_pcm16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_bytes = samples.len() * 2;
        let mut out = Vec::with_capacity(44 + data_bytes);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + data_bytes) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for sample in samples {
            out.extend_from_slice(&sample.to_le_bytes());
        }
        out
    }

    #[test]
    fn wav_decodes_pcm16_mono_byte_exact() {
        let samples: Vec<i16> = vec![0, 16384, -16384, 32767, -32768];
        let bytes = build_wav_pcm16(&samples, 44100);
        let buffer = decode_wav(&bytes).expect("the decode");
        assert_eq!(buffer.sample_rate, 44100);
        assert_eq!(buffer.channels, 1);
        assert_eq!(buffer.samples.len(), samples.len());
        // The normalisation rule: the 16-bit path divides by 32768.
        assert_eq!(buffer.samples[0], 0.0);
        assert!((buffer.samples[1] - 0.5).abs() < 1e-6);
        assert!((buffer.samples[2] + 0.5).abs() < 1e-6);
        assert!((buffer.samples[3] - (32767.0 / 32768.0)).abs() < 1e-6);
        assert_eq!(buffer.samples[4], -1.0);
    }

    #[test]
    fn wav_rejects_a_truncated_data_chunk() {
        let mut bytes = build_wav_pcm16(&[100, -100], 44100);
        bytes.truncate(bytes.len() - 2);
        assert!(decode_wav(&bytes).is_err());
    }

    #[test]
    fn wav_rejects_a_missing_data_chunk() {
        let mut bytes = build_wav_pcm16(&[100], 44100);
        // Cut the file at the data chunk's header: the walk ends with
        // no data chunk found.
        bytes.truncate(36);
        assert!(decode_wav(&bytes).is_err());
    }

    #[test]
    fn wav_rejects_a_non_wave_form() {
        let mut bytes = build_wav_pcm16(&[100], 44100);
        bytes[8..12].copy_from_slice(b"AVI ");
        assert!(decode_wav(&bytes).is_err());
    }

    #[test]
    fn wav_decodes_float32_passing_through() {
        // A minimal IEEE-float mono WAV: the fmt tag 3 with 32 bits.
        let values = [0.0f32, 0.25, -0.5, 1.0];
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + values.len() * 4) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&48000u32.to_le_bytes());
        out.extend_from_slice(&192000u32.to_le_bytes());
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(values.len() as u32 * 4).to_le_bytes());
        for value in &values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let buffer = decode_wav(&out).expect("the decode");
        assert_eq!(buffer.sample_rate, 48000);
        assert_eq!(buffer.channels, 1);
        assert_eq!(buffer.samples, values.to_vec());
    }
}
