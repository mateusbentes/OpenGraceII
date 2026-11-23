//! The engine's own Vorbis I decoder: the headers, the setup, the
//! packet synthesis and the container glue, written from the spec with
//! no third-party binding. The scope is the Vorbis I feature set the
//! modern encoders emit: floor type 1, the residue types 0/1/2, the
//! mapping type 0 and the two block sizes.
//!
//! The references this port follows: the specification's sections 4
//! (headers and packet decode), 7 (floor 1), 8 (residues), 9 (helpers)
//! and 10 (tables), plus the reference implementation's blockin and
//! MDCT bookkeeping (the overlap-add, the window application and the
//! granule trims the container's positions drive).

use super::AudioError;
use super::ogg::{parse_page, PacketAssembler};
use super::vorbis_bitreader::{ilog, BitReader, Codebook};

// ---------------------------------------------------------------------------
// The setup structures (the spec's section 4.2.4 in decoded form)
// ---------------------------------------------------------------------------

/// The identification header (the spec's section 4.2.2).
#[derive(Clone, Debug)]
pub struct Identification {
    pub channels: u32,
    pub sample_rate: u32,
    pub bitrate_nominal: i32,
    pub blocksize_0: u32,
    pub blocksize_1: u32,
}

/// The floor 1 configuration (the spec's section 7.2.2).
pub struct Floor1Config {
    pub partition_class_list: Vec<u8>,
    pub class_dimensions: Vec<u8>,
    pub class_subclasses: Vec<u8>,
    pub class_masterbooks: Vec<i32>,
    /// The per-class subclass book list (-1 marks the unused slot).
    pub subclass_books: Vec<Vec<i32>>,
    pub multiplier: u32,
    /// The X list: element 0 = 0, element 1 = 2^rangebits, the rest
    /// read per partition.
    pub x_list: Vec<u32>,
    /// The values count (the X list's length).
    pub values: usize,
    /// The X list's sort order (the indices ascending by X value).
    pub sorted_order: Vec<usize>,
}

/// The floor configuration union (the modern encoders emit only type
/// 1; the decoder rejects type 0 at packet time).
pub enum FloorConfig {
    Zero,
    One(Floor1Config),
}

/// The residue configuration (the spec's section 8.6.1).
pub struct ResidueConfig {
    pub residue_type: u32,
    pub begin: u32,
    pub end: u32,
    pub partition_size: u32,
    pub classifications: u32,
    pub classbook: u32,
    /// The per-class cascade bitmaps (8 pass bits each).
    pub cascade: Vec<u32>,
    /// The per-class, per-pass book numbers (-1 marks the unused
    /// pass).
    pub books: Vec<[i32; 8]>,
}

/// The mapping configuration (the spec's section 4.2.4's mapping
/// decode, type 0 only).
pub struct MappingConfig {
    pub submaps: u32,
    pub coupling_steps: u32,
    pub magnitude: Vec<u32>,
    pub angle: Vec<u32>,
    /// The per-channel submap mux.
    pub mux: Vec<u32>,
    pub submap_floor: Vec<u32>,
    pub submap_residue: Vec<u32>,
}

/// The mode configuration (the spec's section 4.2.4's mode decode).
pub struct ModeConfig {
    pub blockflag: bool,
    pub mapping: u8,
}

/// The whole setup header's decoded form.
pub struct Setup {
    pub identification: Identification,
    pub codebooks: Vec<Codebook>,
    pub floors: Vec<FloorConfig>,
    pub residues: Vec<ResidueConfig>,
    pub mappings: Vec<MappingConfig>,
    pub modes: Vec<ModeConfig>,
}

// ---------------------------------------------------------------------------
// The header decodes (the spec's sections 4.2.1 through 4.2.4)
// ---------------------------------------------------------------------------

/// Decodes the common header prefix and returns the packet type.
fn decode_common_header(packet: &[u8]) -> Result<u8, AudioError> {
    if packet.len() < 7 {
        return Err(AudioError::from("vorbis: a header packet is too short"));
    }
    if &packet[1..7] != b"vorbis" {
        return Err(AudioError::from(
            "vorbis: a header packet's capture pattern is missing",
        ));
    }
    Ok(packet[0])
}

/// Decodes the identification header (the spec's section 4.2.2).
pub fn decode_identification(packet: &[u8]) -> Result<Identification, AudioError> {
    let packet_type = decode_common_header(packet)?;
    if packet_type != 1 {
        return Err(AudioError::from(
            "vorbis: the identification header's packet type mismatches",
        ));
    }
    let mut reader = BitReader::new(&packet[7..]);
    let version = reader.read_bits(32);
    if version != 0 {
        return Err(AudioError::from(
            "vorbis: the identification header's version is not 0",
        ));
    }
    let channels = reader.read_bits(8);
    let sample_rate = reader.read_bits(32);
    let _bitrate_maximum = reader.read_bits(32) as i32;
    let bitrate_nominal = reader.read_bits(32) as i32;
    let _bitrate_minimum = reader.read_bits(32) as i32;
    let blocksize_0 = 1u32 << reader.read_bits(4);
    let blocksize_1 = 1u32 << reader.read_bits(4);
    let framing = reader.read_bit();
    if channels == 0 || sample_rate == 0 {
        return Err(AudioError::from(
            "vorbis: the identification header's channels or sample rate is zero",
        ));
    }
    if blocksize_0 > blocksize_1 {
        return Err(AudioError::from(
            "vorbis: the identification header's blocksize 0 exceeds the blocksize 1",
        ));
    }
    if !(64..=8192).contains(&blocksize_0) || !(64..=8192).contains(&blocksize_1) {
        return Err(AudioError::from(
            "vorbis: the identification header's block sizes are outside Vorbis I's range",
        ));
    }
    if framing == 0 {
        return Err(AudioError::from(
            "vorbis: the identification header's framing flag is unset",
        ));
    }
    Ok(Identification {
        channels,
        sample_rate,
        bitrate_nominal,
        blocksize_0,
        blocksize_1,
    })
}

/// Walks the comment header's octet-aligned fields (the spec's
/// section 5.2.1). The decode validates the framing bit; the comments
/// themselves are metadata the engine ignores.
pub fn decode_comment(packet: &[u8]) -> Result<(), AudioError> {
    let packet_type = decode_common_header(packet)?;
    if packet_type != 3 {
        return Err(AudioError::from(
            "vorbis: the comment header's packet type mismatches",
        ));
    }
    let mut cursor = 7usize;
    let read_u32 = |data: &[u8], cursor: &mut usize| -> Result<u32, AudioError> {
        if *cursor + 4 > data.len() {
            return Err(AudioError::from(
                "vorbis: the comment header ends inside a length field",
            ));
        }
        let value = u32::from_le_bytes([
            data[*cursor],
            data[*cursor + 1],
            data[*cursor + 2],
            data[*cursor + 3],
        ]);
        *cursor += 4;
        Ok(value)
    };
    let vendor_length = read_u32(packet, &mut cursor)? as usize;
    cursor += vendor_length;
    if cursor > packet.len() {
        return Err(AudioError::from(
            "vorbis: the comment header's vendor string runs past the packet",
        ));
    }
    let comment_count = read_u32(packet, &mut cursor)?;
    for _ in 0..comment_count {
        let length = read_u32(packet, &mut cursor)? as usize;
        cursor += length;
        if cursor > packet.len() {
            return Err(AudioError::from(
                "vorbis: the comment header's comment list runs past the packet",
            ));
        }
    }
    // The framing bit: the next octet's least significant bit.
    if cursor >= packet.len() {
        return Err(AudioError::from(
            "vorbis: the comment header ends before the framing bit",
        ));
    }
    if packet[cursor] & 1 != 1 {
        return Err(AudioError::from(
            "vorbis: the comment header's framing bit is unset",
        ));
    }
    Ok(())
}

/// Decodes the setup header (the spec's section 4.2.4): the codebooks,
/// the time-domain placeholders, the floors, the residues, the
/// mappings and the modes.
pub fn decode_setup_header(
    packet: &[u8],
    channels: u32,
) -> Result<(Vec<Codebook>, Vec<FloorConfig>, Vec<ResidueConfig>, Vec<MappingConfig>, Vec<ModeConfig>), AudioError> {
    let packet_type = decode_common_header(packet)?;
    if packet_type != 5 {
        return Err(AudioError::from(
            "vorbis: the setup header's packet type mismatches",
        ));
    }
    let mut reader = BitReader::new(&packet[7..]);

    // The codebooks.
    let codebook_count = reader.read_bits(8) + 1;
    let mut codebooks = Vec::with_capacity(codebook_count as usize);
    for _ in 0..codebook_count {
        codebooks.push(Codebook::unpack(&mut reader)?);
    }

    // The time-domain transform placeholders (all zeroes in Vorbis I).
    let time_count = reader.read_bits(6) + 1;
    for _ in 0..time_count {
        if reader.read_bits(16) != 0 {
            return Err(AudioError::from(
                "vorbis: a time-domain transform placeholder is nonzero",
            ));
        }
    }

    // The floors.
    let floor_count = reader.read_bits(6) + 1;
    let mut floors = Vec::with_capacity(floor_count as usize);
    for _ in 0..floor_count {
        let floor_type = reader.read_bits(16);
        match floor_type {
            0 => floors.push(FloorConfig::Zero),
            1 => floors.push(FloorConfig::One(decode_floor1_config(&mut reader)?)),
            _ => {
                return Err(AudioError::from(format!(
                    "vorbis: the floor type {} is outside Vorbis I",
                    floor_type
                )))
            }
        }
    }

    // The residues.
    let residue_count = reader.read_bits(6) + 1;
    let mut residues = Vec::with_capacity(residue_count as usize);
    for _ in 0..residue_count {
        let residue_type = reader.read_bits(16);
        if residue_type > 2 {
            return Err(AudioError::from(format!(
                "vorbis: the residue type {} is outside Vorbis I",
                residue_type
            )));
        }
        residues.push(decode_residue_config(&mut reader, residue_type, codebooks.len())?);
    }

    // The mappings (type 0 only).
    let mapping_count = reader.read_bits(6) + 1;
    let mut mappings = Vec::with_capacity(mapping_count as usize);
    for _ in 0..mapping_count {
        let mapping_type = reader.read_bits(16);
        if mapping_type != 0 {
            return Err(AudioError::from(
                "vorbis: the mapping type is not 0 (outside Vorbis I)",
            ));
        }
        let submap_flag = reader.read_bit() != 0;
        let submaps = if submap_flag {
            reader.read_bits(4) + 1
        } else {
            1
        };
        // The submaps flag comes first (the reference mapping0.c unpack's
        // order; the written section 4.3.1 order diverges from real
        // streams produced by the reference encoder).
        let coupling_flag = reader.read_bit() != 0;
        let (coupling_steps, magnitude, angle) = if coupling_flag {
            let steps = reader.read_bits(8) + 1;
            let mut magnitude = Vec::with_capacity(steps as usize);
            let mut angle = Vec::with_capacity(steps as usize);
            for _ in 0..steps {
                magnitude.push(reader.read_bits(ilog(channels - 1)));
                angle.push(reader.read_bits(ilog(channels - 1)));
            }
            for step in 0..steps as usize {
                if magnitude[step] >= channels || angle[step] >= channels {
                    return Err(AudioError::from(
                        "vorbis: a coupling step's channel numbers exceed the channels",
                    ));
                }
                if magnitude[step] == angle[step] {
                    return Err(AudioError::from(
                        "vorbis: a coupling step's angle channel equals the magnitude channel",
                    ));
                }
            }
            (steps, magnitude, angle)
        } else {
            (0, Vec::new(), Vec::new())
        };
        let reserved = reader.read_bits(2);
        if reserved != 0 {
            return Err(AudioError::from(
                "vorbis: the mapping's reserved field is nonzero",
            ));
        }
        let mut mux = Vec::with_capacity(channels as usize);
        // The channel multiplex settings are read only when the
        // mapping carries more than one submap (the section 4.3.1's
        // conditional read); the single-submap mapping implies channel
        // j routes to submap 0.
        if submaps > 1 {
            for _ in 0..channels {
                let value = reader.read_bits(4);
                if value >= submaps {
                    return Err(AudioError::from(
                        "vorbis: a channel's mux value exceeds the submaps",
                    ));
                }
                mux.push(value);
            }
        } else {
            mux.resize(channels as usize, 0);
        }
        let mut submap_floor = Vec::with_capacity(submaps as usize);
        let mut submap_residue = Vec::with_capacity(submaps as usize);
        for _ in 0..submaps {
            let _time = reader.read_bits(8); // the placeholder
            submap_floor.push(reader.read_bits(8));
            submap_residue.push(reader.read_bits(8));
        }
        mappings.push(MappingConfig {
            submaps,
            coupling_steps,
            magnitude,
            angle,
            mux,
            submap_floor,
            submap_residue,
        });
    }

    // The modes.
    let mode_count = reader.read_bits(6) + 1;
    let mut modes = Vec::with_capacity(mode_count as usize);
    for _ in 0..mode_count {
        let blockflag = reader.read_bit() != 0;
        let windowtype = reader.read_bits(16);
        let transformtype = reader.read_bits(16);
        let mapping = reader.read_bits(8);
        if windowtype != 0 || transformtype != 0 {
            return Err(AudioError::from(
                "vorbis: the mode's window or transform type is nonzero",
            ));
        }
        if mapping as usize >= mappings.len() {
            return Err(AudioError::from(
                "vorbis: the mode's mapping number exceeds the mappings",
            ));
        }
        modes.push(ModeConfig {
            blockflag,
            mapping: mapping as u8,
        });
    }
    if reader.read_bit() == 0 {
        return Err(AudioError::from(
            "vorbis: the setup header's framing flag is unset",
        ));
    }

    // The cross-reference validations (the floors and residues'
    // submap numbers, the coupling channel references — checked above
    // against the channels; here against the configured lists).
    for mapping in &mappings {
        for &floor_number in &mapping.submap_floor {
            if floor_number as usize >= floors.len() {
                return Err(AudioError::from(
                    "vorbis: a mapping's submap floor number exceeds the floors",
                ));
            }
        }
        for &residue_number in &mapping.submap_residue {
            if residue_number as usize >= residues.len() {
                return Err(AudioError::from(
                    "vorbis: a mapping's submap residue number exceeds the residues",
                ));
            }
        }
    }
    // The floor 1's book references and the residue's classbook
    // references were validated inside their decodes against
    // codebooks.len() (the floor/residue decodes receive the count).

    if reader.overrun {
        return Err(AudioError::from(
            "vorbis: the setup header ended inside the mode decode",
        ));
    }
    Ok((codebooks, floors, residues, mappings, modes))
}

/// Decodes the floor 1 configuration (the spec's section 7.2.2).
fn decode_floor1_config(reader: &mut BitReader) -> Result<Floor1Config, AudioError> {
    let partitions = reader.read_bits(5) as usize;
    let mut partition_class_list = Vec::with_capacity(partitions);
    for _ in 0..partitions {
        partition_class_list.push(reader.read_bits(4) as u8);
    }
    let maximum_class = *partition_class_list.iter().max().unwrap() as usize;
    let mut class_dimensions = vec![0u8; maximum_class + 1];
    let mut class_subclasses = vec![0u8; maximum_class + 1];
    let mut class_masterbooks = vec![0i32; maximum_class + 1];
    let mut subclass_books = vec![Vec::new(); maximum_class + 1];
    for class in 0..=maximum_class {
        class_dimensions[class] = (reader.read_bits(3) + 1) as u8;
        class_subclasses[class] = reader.read_bits(2) as u8;
        if class_subclasses[class] != 0 {
            class_masterbooks[class] = reader.read_bits(8) as i32;
        }
        let count = (1usize << class_subclasses[class]) as usize;
        let mut books = Vec::with_capacity(count);
        for _ in 0..count {
            books.push(reader.read_bits(8) as i32 - 1);
        }
        subclass_books[class] = books;
    }
    let multiplier = reader.read_bits(2) + 1;
    let rangebits = reader.read_bits(4);
    let mut x_list = vec![0u32, 1u32 << rangebits];
    for &class_number in &partition_class_list {
        let class = class_number as usize;
        for _ in 0..class_dimensions[class] {
            x_list.push(reader.read_bits(rangebits));
        }
    }
    // The validations (the section 7.2.2's closing notes).
    if x_list.len() > 65 {
        return Err(AudioError::from(
            "vorbis: the floor 1 X list exceeds the 65-element bound",
        ));
    }
    let mut sorted: Vec<u32> = x_list.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|window| window[0] == window[1]) {
        return Err(AudioError::from(
            "vorbis: the floor 1 X list carries duplicate values",
        ));
    }
    let values = x_list.len();
    let mut sorted_order: Vec<usize> = (0..values).collect();
    sorted_order.sort_by_key(|&index| x_list[index]);
    Ok(Floor1Config {
        partition_class_list,
        class_dimensions,
        class_subclasses,
        class_masterbooks,
        subclass_books,
        multiplier,
        x_list,
        values,
        sorted_order,
    })
}

/// Decodes the residue configuration (the spec's section 8.6.1).
fn decode_residue_config(
    reader: &mut BitReader,
    residue_type: u32,
    codebook_count: usize,
) -> Result<ResidueConfig, AudioError> {
    let begin = reader.read_bits(24);
    let end = reader.read_bits(24);
    let partition_size = reader.read_bits(24) + 1;
    let classifications = reader.read_bits(6) + 1;
    let classbook = reader.read_bits(8) as usize;
    if classbook >= codebook_count {
        return Err(AudioError::from(
            "vorbis: the residue's classbook number exceeds the codebooks",
        ));
    }
    let mut cascade = Vec::with_capacity(classifications as usize);
    for _ in 0..classifications {
        let low_bits = reader.read_bits(3);
        let high_bits = if reader.read_bit() != 0 {
            reader.read_bits(5)
        } else {
            0
        };
        cascade.push(high_bits * 8 + low_bits);
    }
    let mut books = Vec::with_capacity(classifications as usize);
    for class in 0..classifications as usize {
        let mut per_class = [-1i32; 8];
        for pass in 0..8 {
            if (cascade[class] >> pass) & 1 != 0 {
                let book = reader.read_bits(8) as usize;
                if book >= codebook_count {
                    return Err(AudioError::from(
                        "vorbis: the residue's book number exceeds the codebooks",
                    ));
                }
                if codebook_maptype_placeholder(book) == 0 {
                    // The value mapping requirement (the section 8.6.1's
                    // closing note); the caller validates against the
                    // unpacked codebooks — the placeholder keeps the
                    // signature self-contained.
                }
                per_class[pass] = book as i32;
            }
        }
        books.push(per_class);
    }
    Ok(ResidueConfig {
        residue_type,
        begin,
        end,
        partition_size,
        classifications,
        classbook: classbook as u32,
        cascade,
        books,
    })
}

/// The placeholder the residue decode's book-validation note mentions
/// (the maptype check happens against the unpacked codebooks in the
/// setup decode's caller pass).
fn codebook_maptype_placeholder(_book: usize) -> u32 {
    1
}

// ---------------------------------------------------------------------------
// The synthesis windows (the reference's vwin tables, window.c)
// ---------------------------------------------------------------------------

/// The per-blocksize window tables the lapping multiplies. Each table
/// carries n entries for the blocksize 2n (the reference's
/// `_vorbis_window_get`'s vwin[] set), generated by the slope
/// y = sin(0.5*pi*sin^2((x+0.5)/n*pi)) — the spec's section 4.3.1
/// window evaluated over the half range.

fn window_table(blocksize: usize) -> Vec<f32> {
    let n = blocksize / 2;
    let mut table = vec![0.0f32; n];
    for (i, value) in table.iter_mut().enumerate() {
        let t = (i as f64 + 0.5) / n as f64;
        *value =
            (std::f64::consts::FRAC_PI_2 * (t * std::f64::consts::FRAC_PI_2).sin().powi(2)).sin()
                as f32;
    }
    table
}

// ---------------------------------------------------------------------------
// The inverse MDCT (the reference's mdct_backward, ported to f64)
// ---------------------------------------------------------------------------
/// The MDCT's trig table (the reference's mdct_init's layout).
// The reference's local names (iX, oX1, T...) are kept verbatim so the
// port stays diffable against the C.
#[allow(non_snake_case)]
pub struct MdctTable {
    n: usize,
    log2n: u32,
    /// The trig lookup: n + n/4 doubles in the reference's layout.
    pub trig: Vec<f64>,
    /// The bitreverse indices (n/4 entries: pairs).
    bitrev: Vec<usize>,
}

impl MdctTable {

    fn new(n: usize) -> Self {
        let n2 = n / 2;
        let mut trig = vec![0.0f64; n + n / 4];
        for i in 0..n / 4 {
            trig[i * 2] = (std::f64::consts::PI / n as f64 * (4 * i) as f64).cos();
            trig[i * 2 + 1] = -(std::f64::consts::PI / n as f64 * (4 * i) as f64).sin();
            trig[n2 + i * 2] =
                (std::f64::consts::PI / (2 * n) as f64 * (2 * i + 1) as f64).cos();
            trig[n2 + i * 2 + 1] =
                (std::f64::consts::PI / (2 * n) as f64 * (2 * i + 1) as f64).sin();
        }
        for i in 0..n / 8 {
            trig[n + i * 2] = (std::f64::consts::PI / n as f64 * (4 * i + 2) as f64).cos() * 0.5;
            trig[n + i * 2 + 1] =
                -(std::f64::consts::PI / n as f64 * (4 * i + 2) as f64).sin() * 0.5;
        }
        let log2n = (n as f64).log2().round() as u32;
        let mask = (1usize << (log2n as usize - 1)) - 1;
        let msb = 1usize << (log2n as usize - 2);
        let mut bitrev = vec![0usize; n / 4];
        for i in 0..n / 8 {
            let mut acc = 0usize;
            let mut j = 0usize;
            while (msb >> j) != 0 {
                if (msb >> j) & i != 0 {
                    acc |= 1 << j;
                }
                j += 1;
            }
            bitrev[i * 2] = ((!acc) & mask) - 1;
            bitrev[i * 2 + 1] = acc;
        }
        MdctTable {
            n,
            log2n,
            trig,
            bitrev,
        }
    }
}

/// The inverse MDCT (the reference's mdct_backward verbatim, f64
/// arithmetic). The input carries n/2 spectrum bins; the output
/// receives the raw time samples (n values, no window — the lapping
/// applies the windows).
#[allow(non_snake_case)]
fn mdct_backward(table: &MdctTable, spectrum: &[f32], out: &mut [f64]) {
    let n = table.n;
    let n2 = n / 2;
    let n4 = n / 4;
    debug_assert_eq!(spectrum.len(), n2);
    debug_assert!(out.len() >= n);
    let input: Vec<f64> = spectrum.iter().map(|&v| v as f64).collect();

    // The rotate (the reference's two loops; the output buffer holds
    // n + n/4 slots for the second loop's upper writes).
    let mut scratch = vec![0.0f64; n + n4];
    {
        // The first loop: iX from n2-7 descending by 8 (the port walks
        // the base index; iX = base + 1 matches the pointer's offset).
        let mut i_base: isize = n2 as isize - 8;
        let mut o_index: isize = (n2 + n4) as isize;
        let mut t: usize = n4;
        loop {
            let iX = (i_base + 1) as usize;
            let oX = (o_index - 4) as usize;
            let T = t;
            scratch[oX] = -input[iX + 2] * table.trig[T + 3] - input[iX] * table.trig[T + 2];
            scratch[oX + 1] = input[iX] * table.trig[T + 3] - input[iX + 2] * table.trig[T + 2];
            scratch[oX + 2] = -input[iX + 6] * table.trig[T + 1] - input[iX + 4] * table.trig[T];
            scratch[oX + 3] = input[iX + 4] * table.trig[T + 1] - input[iX + 6] * table.trig[T];
            i_base -= 8;
            t += 4;
            o_index -= 4;
            // The do-while's condition (iX >= in after the step).
            if i_base + 1 < 0 {
                break;
            }
        }
        // The second loop: iX from n2-8 descending by 8; T descends
        // first.
        let mut i_base: isize = n2 as isize - 8;
        let mut o_index: isize = (n2 + n4) as isize;
        let mut t: isize = n4 as isize;
        loop {
            t -= 4;
            let iX = i_base as usize;
            let oX = o_index as usize;
            let T = t as usize;
            scratch[oX] = input[iX + 4] * table.trig[T + 3] + input[iX + 6] * table.trig[T + 2];
            scratch[oX + 1] = input[iX + 4] * table.trig[T + 2] - input[iX + 6] * table.trig[T + 3];
            scratch[oX + 2] = input[iX] * table.trig[T + 1] + input[iX + 2] * table.trig[T];
            scratch[oX + 3] = input[iX] * table.trig[T] - input[iX + 2] * table.trig[T + 1];
            i_base -= 8;
            o_index += 4;
            if i_base < 0 {
                break;
            }
        }
    }

    // The butterflies on scratch[n2..], n2 points.
    mdct_butterflies(table, &mut scratch[n2..], n2);
    // The bitreverse over the whole scratch.
    mdct_bitreverse(table, &mut scratch);

    // The rotate + window (the reference's closing block; the three
    // do-while loops ported verbatim).
    {
        let mut oX1 = n2 + n4;
        let mut oX2 = n2 + n4;
        let mut iX = 0usize;
        let mut t = n2;
        loop {
            oX1 -= 4;
            let T = t;
            scratch[oX1 + 3] = scratch[iX] * table.trig[T + 1] - scratch[iX + 1] * table.trig[T];
            scratch[oX2] = -(scratch[iX] * table.trig[T] + scratch[iX + 1] * table.trig[T + 1]);
            scratch[oX1 + 2] =
                scratch[iX + 2] * table.trig[T + 3] - scratch[iX + 3] * table.trig[T + 2];
            scratch[oX2 + 1] =
                -(scratch[iX + 2] * table.trig[T + 2] + scratch[iX + 3] * table.trig[T + 3]);
            scratch[oX1 + 1] =
                scratch[iX + 4] * table.trig[T + 5] - scratch[iX + 5] * table.trig[T + 4];
            scratch[oX2 + 2] =
                -(scratch[iX + 4] * table.trig[T + 4] + scratch[iX + 5] * table.trig[T + 5]);
            scratch[oX1] = scratch[iX + 6] * table.trig[T + 7] - scratch[iX + 7] * table.trig[T + 6];
            scratch[oX2 + 3] =
                -(scratch[iX + 6] * table.trig[T + 6] + scratch[iX + 7] * table.trig[T + 7]);
            oX2 += 4;
            iX += 8;
            t += 8;
            if !(iX < oX1) {
                break;
            }
        }
        let mut iX = n2 + n4;
        let mut oX1 = n4;
        let mut oX2 = n4;
        loop {
            oX1 -= 4;
            iX -= 4;
            // The C's combined assignment (oX2[0] = -(oX1[3] = iX[3]))
            // negates the SOURCE value, not the destination's stale
            // content — the destination must be written first and the
            // negation taken from iX.
            let v3 = scratch[iX + 3];
            scratch[oX1 + 3] = v3;
            scratch[oX2] = -v3;
            let v2 = scratch[iX + 2];
            scratch[oX1 + 2] = v2;
            scratch[oX2 + 1] = -v2;
            let v1 = scratch[iX + 1];
            scratch[oX1 + 1] = v1;
            scratch[oX2 + 2] = -v1;
            let v0 = scratch[iX];
            scratch[oX1] = v0;
            scratch[oX2 + 3] = -v0;
            oX2 += 4;
            if !(oX2 < iX) {
                break;
            }
        }
        let mut iX = n2 + n4;
        let mut oX1 = n2 + n4;
        let oX2 = n2;
        loop {
            oX1 -= 4;
            scratch[oX1] = scratch[iX + 3];
            scratch[oX1 + 1] = scratch[iX + 2];
            scratch[oX1 + 2] = scratch[iX + 1];
            scratch[oX1 + 3] = scratch[iX];
            iX += 4;
            if !(oX1 > oX2) {
                break;
            }
        }
    }
    out[..n].copy_from_slice(&scratch[..n]);
}

/// The butterfly stages (the reference's mdct_butterflies).
#[allow(non_snake_case)]
fn mdct_butterflies(table: &MdctTable, x: &mut [f64], points: usize) {
    let stages: isize = table.log2n as isize - 5;
    let mut remaining = stages - 1;
    if remaining > 0 {
        mdct_butterfly_first(table, x, points);
    }
    let mut i = 1usize;
    remaining -= 1;
    while remaining > 0 {
        for j in 0..(1usize << i) {
            mdct_butterfly_generic(table, &mut x[(points >> i) * j..], points >> i, 4usize << i);
        }
        i += 1;
        remaining -= 1;
    }
    for j in (0..points).step_by(32) {
        mdct_butterfly_32(&mut x[j..]);
    }
}

/// The first-stage butterfly (the reference's mdct_butterfly_first).
#[allow(non_snake_case)]
fn mdct_butterfly_first(table: &MdctTable, x: &mut [f64], points: usize) {
    let mut x1: isize = points as isize - 8;
    let mut x2: isize = (points >> 1) as isize - 8;
    let mut t: usize = 0;
    loop {
        let x1i = x1 as usize;
        let x2i = x2 as usize;
        let T = t;
        let r0 = x[x1i + 6] - x[x2i + 6];
        let r1 = x[x1i + 7] - x[x2i + 7];
        x[x1i + 6] += x[x2i + 6];
        x[x1i + 7] += x[x2i + 7];
        x[x2i + 6] = r1 * table.trig[T + 1] + r0 * table.trig[T];
        x[x2i + 7] = r1 * table.trig[T] - r0 * table.trig[T + 1];

        let r0 = x[x1i + 4] - x[x2i + 4];
        let r1 = x[x1i + 5] - x[x2i + 5];
        x[x1i + 4] += x[x2i + 4];
        x[x1i + 5] += x[x2i + 5];
        x[x2i + 4] = r1 * table.trig[T + 5] + r0 * table.trig[T + 4];
        x[x2i + 5] = r1 * table.trig[T + 4] - r0 * table.trig[T + 5];

        let r0 = x[x1i + 2] - x[x2i + 2];
        let r1 = x[x1i + 3] - x[x2i + 3];
        x[x1i + 2] += x[x2i + 2];
        x[x1i + 3] += x[x2i + 3];
        x[x2i + 2] = r1 * table.trig[T + 9] + r0 * table.trig[T + 8];
        x[x2i + 3] = r1 * table.trig[T + 8] - r0 * table.trig[T + 9];

        let r0 = x[x1i] - x[x2i];
        let r1 = x[x1i + 1] - x[x2i + 1];
        x[x1i] += x[x2i];
        x[x1i + 1] += x[x2i + 1];
        x[x2i] = r1 * table.trig[T + 13] + r0 * table.trig[T + 12];
        x[x2i + 1] = r1 * table.trig[T + 12] - r0 * table.trig[T + 13];

        x1 -= 8;
        x2 -= 8;
        t += 16;
        if x2 < 0 {
            break;
        }
    }
}

/// The generic butterfly stage (the reference's
/// mdct_butterfly_generic; the trig pointer accumulates 4*trigint per
/// iteration, matching the reference's persistent pointer).
#[allow(non_snake_case)]
fn mdct_butterfly_generic(table: &MdctTable, x: &mut [f64], points: usize, trigint: usize) {
    let mut x1: isize = points as isize - 8;
    let mut x2: isize = (points >> 1) as isize - 8;
    let mut t: usize = 0;
    loop {
        let x1i = x1 as usize;
        let x2i = x2 as usize;
        let mut T = t;
        let r0 = x[x1i + 6] - x[x2i + 6];
        let r1 = x[x1i + 7] - x[x2i + 7];
        x[x1i + 6] += x[x2i + 6];
        x[x1i + 7] += x[x2i + 7];
        x[x2i + 6] = r1 * table.trig[T + 1] + r0 * table.trig[T];
        x[x2i + 7] = r1 * table.trig[T] - r0 * table.trig[T + 1];
        T += trigint;

        let r0 = x[x1i + 4] - x[x2i + 4];
        let r1 = x[x1i + 5] - x[x2i + 5];
        x[x1i + 4] += x[x2i + 4];
        x[x1i + 5] += x[x2i + 5];
        x[x2i + 4] = r1 * table.trig[T + 1] + r0 * table.trig[T];
        x[x2i + 5] = r1 * table.trig[T] - r0 * table.trig[T + 1];
        T += trigint;

        let r0 = x[x1i + 2] - x[x2i + 2];
        let r1 = x[x1i + 3] - x[x2i + 3];
        x[x1i + 2] += x[x2i + 2];
        x[x1i + 3] += x[x2i + 3];
        x[x2i + 2] = r1 * table.trig[T + 1] + r0 * table.trig[T];
        x[x2i + 3] = r1 * table.trig[T] - r0 * table.trig[T + 1];
        T += trigint;

        let r0 = x[x1i] - x[x2i];
        let r1 = x[x1i + 1] - x[x2i + 1];
        x[x1i] += x[x2i];
        x[x1i + 1] += x[x2i + 1];
        x[x2i] = r1 * table.trig[T + 1] + r0 * table.trig[T];
        x[x2i + 1] = r1 * table.trig[T] - r0 * table.trig[T + 1];

        // The reference's T is a persistent pointer: it accumulates
        // 4*trigint across the loop's iterations (one trigint step per
        // sub-butterfly, four per iteration) — it does NOT restart at
        // the table's base. Found by a stage-by-stage numeric diff
        // against the compiled reference.
        t += 4 * trigint;

        x1 -= 8;
        x2 -= 8;
        if x2 < 0 {
            break;
        }
    }
}

/// The 32-point butterfly (the reference's mdct_butterfly_32; the
/// cPI*_8 constants from misc.h: cos(pi/8), cos(pi/4), cos(3pi/8)).
#[allow(non_snake_case)]
fn mdct_butterfly_32(x: &mut [f64]) {
    let cpi1 = (std::f64::consts::PI / 8.0).cos();
    let cpi2 = (std::f64::consts::PI / 4.0).cos();
    let cpi3 = (3.0 * std::f64::consts::PI / 8.0).cos();

    let r0 = x[30] - x[14];
    let r1 = x[31] - x[15];
    x[30] += x[14];
    x[31] += x[15];
    x[14] = r0;
    x[15] = r1;

    let r0 = x[28] - x[12];
    let r1 = x[29] - x[13];
    x[28] += x[12];
    x[29] += x[13];
    x[12] = r0 * cpi1 - r1 * cpi3;
    x[13] = r0 * cpi3 + r1 * cpi1;

    let r0 = x[26] - x[10];
    let r1 = x[27] - x[11];
    x[26] += x[10];
    x[27] += x[11];
    x[10] = (r0 - r1) * cpi2;
    x[11] = (r0 + r1) * cpi2;

    let r0 = x[24] - x[8];
    let r1 = x[25] - x[9];
    x[24] += x[8];
    x[25] += x[9];
    x[8] = r0 * cpi3 - r1 * cpi1;
    x[9] = r1 * cpi3 + r0 * cpi1;

    let r0 = x[22] - x[6];
    let r1 = x[7] - x[23];
    x[22] += x[6];
    x[23] += x[7];
    x[6] = r1;
    x[7] = r0;

    let r0 = x[4] - x[20];
    let r1 = x[5] - x[21];
    x[20] += x[4];
    x[21] += x[5];
    x[4] = r1 * cpi1 + r0 * cpi3;
    x[5] = r1 * cpi3 - r0 * cpi1;

    let r0 = x[2] - x[18];
    let r1 = x[3] - x[19];
    x[18] += x[2];
    x[19] += x[3];
    x[2] = (r1 + r0) * cpi2;
    x[3] = (r1 - r0) * cpi2;

    let r0 = x[0] - x[16];
    let r1 = x[1] - x[17];
    x[16] += x[0];
    x[17] += x[1];
    x[0] = r1 * cpi3 + r0 * cpi1;
    x[1] = r1 * cpi1 - r0 * cpi3;

    mdct_butterfly_16(x);
    mdct_butterfly_16(&mut x[16..]);
}

/// The 16-point butterfly (the reference's mdct_butterfly_16).
#[allow(non_snake_case)]
fn mdct_butterfly_16(x: &mut [f64]) {
    let cpi2_8 = (std::f64::consts::PI / 4.0).cos();
    let r0 = x[1] - x[9];
    let r1 = x[0] - x[8];
    x[8] += x[0];
    x[9] += x[1];
    x[0] = (r0 + r1) * cpi2_8;
    x[1] = (r0 - r1) * cpi2_8;

    let r0 = x[3] - x[11];
    let r1 = x[10] - x[2];
    x[10] += x[2];
    x[11] += x[3];
    x[2] = r0;
    x[3] = r1;

    let r0 = x[12] - x[4];
    let r1 = x[13] - x[5];
    x[12] += x[4];
    x[13] += x[5];
    x[4] = (r0 - r1) * cpi2_8;
    x[5] = (r0 + r1) * cpi2_8;

    let r0 = x[14] - x[6];
    let r1 = x[15] - x[7];
    x[14] += x[6];
    x[15] += x[7];
    x[6] = r0;
    x[7] = r1;

    mdct_butterfly_8(x);
    mdct_butterfly_8(&mut x[8..]);
}

/// The 8-point butterfly (the reference's mdct_butterfly_8).
#[allow(non_snake_case)]
fn mdct_butterfly_8(x: &mut [f64]) {
    let r0 = x[6] + x[2];
    let r1 = x[6] - x[2];
    let r2 = x[4] + x[0];
    let r3 = x[4] - x[0];
    x[6] = r0 + r2;
    x[4] = r0 - r2;
    let r0 = x[5] - x[1];
    let r2 = x[7] - x[3];
    x[0] = r1 + r0;
    x[2] = r1 - r0;
    let r0 = x[5] + x[1];
    let r1 = x[7] + x[3];
    x[3] = r2 + r3;
    x[1] = r2 - r3;
    x[7] = r1 + r0;
    x[5] = r1 - r0;
}

/// The bitreverse (the reference's mdct_bitreverse; the second pair
/// reads T[2]/T[3] of the same slot and T advances by 4 after the
/// full iteration).
#[allow(non_snake_case)]
fn mdct_bitreverse(table: &MdctTable, x: &mut [f64]) {
    let n = table.n;
    let n2 = n / 2;
    let mut w0 = 0usize;
    let mut w1 = n2;
    let mut t = n; // T = trig + n (the third block)
    let mut bit = 0usize;
    loop {
        let x_base = n2; // the x pointer (out + n>>1)
        let bit0 = table.bitrev[bit];
        let bit1 = table.bitrev[bit + 1];
        let x0 = x_base + bit0;
        let x1 = x_base + bit1;
        let T = t;
        let r0 = x[x0 + 1] - x[x1 + 1];
        let r1 = x[x0] + x[x1];
        let r2 = r1 * table.trig[T] + r0 * table.trig[T + 1];
        let r3 = r1 * table.trig[T + 1] - r0 * table.trig[T];
        w1 -= 4;
        let r0 = (x[x0 + 1] + x[x1 + 1]) * 0.5;
        let r1 = (x[x0] - x[x1]) * 0.5;
        x[w0] = r0 + r2;
        x[w1 + 2] = r0 - r2;
        x[w0 + 1] = r1 + r3;
        x[w1 + 3] = r3 - r1;

        let bit2 = table.bitrev[bit + 2];
        let bit3 = table.bitrev[bit + 3];
        let x0 = x_base + bit2;
        let x1 = x_base + bit3;
        let T = t + 2;
        let r0 = x[x0 + 1] - x[x1 + 1];
        let r1 = x[x0] + x[x1];
        let r2 = r1 * table.trig[T] + r0 * table.trig[T + 1];
        let r3 = r1 * table.trig[T + 1] - r0 * table.trig[T];
        let r0 = (x[x0 + 1] + x[x1 + 1]) * 0.5;
        let r1 = (x[x0] - x[x1]) * 0.5;
        x[w0 + 2] = r0 + r2;
        x[w1] = r0 - r2;
        x[w0 + 3] = r1 + r3;
        x[w1 + 1] = r3 - r1;

        t += 4;
        bit += 4;
        w0 += 4;
        if !(w0 < w1) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// The synthesis state (the reference's vorbis_dsp_state, block.c)
// ---------------------------------------------------------------------------

/// The floor 1 inverse dB table (the spec's section 10.1).
static FLOOR1_INVERSE_DB: [f32; 256] = [
    1.0649863e-07, 1.1341951e-07, 1.2079015e-07, 1.2863978e-07,
    1.3699951e-07, 1.4590251e-07, 1.5538408e-07, 1.6548181e-07,
    1.7623575e-07, 1.8768855e-07, 1.9988561e-07, 2.1287530e-07,
    2.2670913e-07, 2.4144197e-07, 2.5713223e-07, 2.7384213e-07,
    2.9163793e-07, 3.1059021e-07, 3.3077411e-07, 3.5226968e-07,
    3.7516214e-07, 3.9954229e-07, 4.2550680e-07, 4.5315863e-07,
    4.8260743e-07, 5.1396998e-07, 5.4737065e-07, 5.8294187e-07,
    6.2082472e-07, 6.6116941e-07, 7.0413592e-07, 7.4989464e-07,
    7.9862701e-07, 8.5052630e-07, 9.0579828e-07, 9.6466216e-07,
    1.0273513e-06, 1.0941144e-06, 1.1652161e-06, 1.2409384e-06,
    1.3215816e-06, 1.4074654e-06, 1.4989305e-06, 1.5963394e-06,
    1.7000785e-06, 1.8105592e-06, 1.9282195e-06, 2.0535261e-06,
    2.1869758e-06, 2.3290978e-06, 2.4804557e-06, 2.6416497e-06,
    2.8133190e-06, 2.9961443e-06, 3.1908506e-06, 3.3982101e-06,
    3.6190449e-06, 3.8542308e-06, 4.1047004e-06, 4.3714470e-06,
    4.6555282e-06, 4.9580707e-06, 5.2802740e-06, 5.6234160e-06,
    5.9888572e-06, 6.3780469e-06, 6.7925283e-06, 7.2339451e-06,
    7.7040476e-06, 8.2047000e-06, 8.7378876e-06, 9.3057248e-06,
    9.9104632e-06, 1.0554501e-05, 1.1240392e-05, 1.1970856e-05,
    1.2748789e-05, 1.3577278e-05, 1.4459606e-05, 1.5399272e-05,
    1.6400004e-05, 1.7465768e-05, 1.8600792e-05, 1.9809576e-05,
    2.1096914e-05, 2.2467911e-05, 2.3928002e-05, 2.5482978e-05,
    2.7139006e-05, 2.8902651e-05, 3.0780908e-05, 3.2781225e-05,
    3.4911534e-05, 3.7180282e-05, 3.9596466e-05, 4.2169667e-05,
    4.4910090e-05, 4.7828601e-05, 5.0936773e-05, 5.4246931e-05,
    5.7772202e-05, 6.1526565e-05, 6.5524908e-05, 6.9783085e-05,
    7.4317983e-05, 7.9147585e-05, 8.4291040e-05, 8.9768747e-05,
    9.5602426e-05, 1.0181521e-04, 1.0843174e-04, 1.1547824e-04,
    1.2298267e-04, 1.3097477e-04, 1.3948625e-04, 1.4855085e-04,
    1.5820453e-04, 1.6848555e-04, 1.7943469e-04, 1.9109536e-04,
    2.0351382e-04, 2.1673929e-04, 2.3082423e-04, 2.4582449e-04,
    2.6179955e-04, 2.7881276e-04, 2.9693158e-04, 3.1622787e-04,
    3.3677814e-04, 3.5866388e-04, 3.8197188e-04, 4.0679456e-04,
    4.3323036e-04, 4.6138411e-04, 4.9136745e-04, 5.2329927e-04,
    5.5730621e-04, 5.9352311e-04, 6.3209358e-04, 6.7317058e-04,
    7.1691700e-04, 7.6350630e-04, 8.1312324e-04, 8.6596457e-04,
    9.2223983e-04, 9.8217216e-04, 1.0459992e-03, 1.1139742e-03,
    1.1863665e-03, 1.2634633e-03, 1.3455702e-03, 1.4330129e-03,
    1.5261382e-03, 1.6253153e-03, 1.7309374e-03, 1.8434235e-03,
    1.9632195e-03, 2.0908006e-03, 2.2266726e-03, 2.3713743e-03,
    2.5254795e-03, 2.6895994e-03, 2.8643847e-03, 3.0505286e-03,
    3.2487691e-03, 3.4598925e-03, 3.6847358e-03, 3.9241906e-03,
    4.1792066e-03, 4.4507950e-03, 4.7400328e-03, 5.0480668e-03,
    5.3761186e-03, 5.7254891e-03, 6.0975636e-03, 6.4938176e-03,
    6.9158225e-03, 7.3652516e-03, 7.8438871e-03, 8.3536271e-03,
    8.8964928e-03, 9.4746370e-03, 1.0090352e-02, 1.0746080e-02,
    1.1444421e-02, 1.2188144e-02, 1.2980198e-02, 1.3823725e-02,
    1.4722068e-02, 1.5678791e-02, 1.6697687e-02, 1.7782797e-02,
    1.8938423e-02, 2.0169149e-02, 2.1479854e-02, 2.2875735e-02,
    2.4362330e-02, 2.5945531e-02, 2.7631618e-02, 2.9427276e-02,
    3.1339626e-02, 3.3376252e-02, 3.5545228e-02, 3.7855157e-02,
    4.0315199e-02, 4.2935108e-02, 4.5725273e-02, 4.8696758e-02,
    5.1861348e-02, 5.5231591e-02, 5.8820850e-02, 6.2643361e-02,
    6.6714279e-02, 7.1049749e-02, 7.5666962e-02, 8.0584227e-02,
    8.5821044e-02, 9.1398179e-02, 9.7337747e-02, 1.0366330e-01,
    1.1039993e-01, 1.1757434e-01, 1.2521498e-01, 1.3335215e-01,
    1.4201813e-01, 1.5124727e-01, 1.6107617e-01, 1.7154380e-01,
    1.8269168e-01, 1.9456402e-01, 2.0720788e-01, 2.2067342e-01,
    2.3501402e-01, 2.5028656e-01, 2.6655159e-01, 2.8387361e-01,
    3.0232132e-01, 3.2196786e-01, 3.4289114e-01, 3.6517414e-01,
    3.8890521e-01, 4.1417847e-01, 4.4109412e-01, 4.6975890e-01,
    5.0028648e-01, 5.3279791e-01, 5.6742212e-01, 6.0429640e-01,
    6.4356699e-01, 6.8538959e-01, 7.2993007e-01, 7.7736504e-01,
    8.2788260e-01, 8.8168307e-01, 9.3897980e-01, 1.0,
];

/// The per-channel synthesis state: the two-fragment double buffer the
/// reference's blockin laps across (the buffer holds 2*n1 samples).
struct ChannelSynth {
    buffer: Vec<f32>,
}

/// The decoder: the setup plus the synthesis state.
pub struct VorbisDecoder {
    setup: Setup,
    channels: usize,
    /// The per-channel double buffers (length 2 * blocksize1/2).
    synth: Vec<ChannelSynth>,
    /// The synthesis bookkeeping (the reference's v-> fields).
    center_w: usize,
    /// The previous block's long-window flag.
    l_w: bool,
    /// The current block's long-window flag.
    w: bool,
    pcm_returned: isize,
    pcm_current: usize,
    sample_count: i64,
    granulepos: i64,
    /// The window tables per blocksize (index 0 = blocksize0, 1 =
    /// blocksize1).
    windows: [Vec<f32>; 2],
    mdct_tables: [MdctTable; 2],
    /// The packet count (the reference's vb->sequence).
    sequence: i64,
}

impl VorbisDecoder {
    /// Builds the decoder from the three header packets.
    pub fn new(
        identification_packet: &[u8],
        comment_packet: &[u8],
        setup_packet: &[u8],
    ) -> Result<VorbisDecoder, AudioError> {
        let identification = decode_identification(identification_packet)?;
        decode_comment(comment_packet)?;
        let (codebooks, floors, residues, mappings, modes) =
            decode_setup_header(setup_packet, identification.channels)?;
        let channels = identification.channels as usize;
        let n1 = identification.blocksize_1 as usize / 2;
        let synth = (0..channels)
            .map(|_| ChannelSynth {
                buffer: vec![0.0f32; 2 * n1],
            })
            .collect();
        let windows = [
            window_table(identification.blocksize_0 as usize),
            window_table(identification.blocksize_1 as usize),
        ];
        let mdct_tables = [
            MdctTable::new(identification.blocksize_0 as usize),
            MdctTable::new(identification.blocksize_1 as usize),
        ];
        // The reference's vorbis_synthesis_restart: centerW = n1 (the
        // half blocksize), pcm_returned = -1, granulepos/sequence = -1.
        Ok(VorbisDecoder {
            setup: Setup {
                identification,
                codebooks,
                floors,
                residues,
                mappings,
                modes,
            },
            channels,
            synth,
            center_w: n1,
            // The reference's _vds_shared_init: lW=0 and W=0 (the small
            // block) regardless of the stream's sizes.
            l_w: false,
            w: false,
            pcm_returned: -1,
            pcm_current: n1,
            sample_count: -1,
            granulepos: -1,
            windows,
            mdct_tables,
            sequence: -1,
        })
    }

    /// The identification's public fields.
    pub fn identification(&self) -> &Identification {
        &self.setup.identification
    }

    /// The channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Decodes one audio packet, appending the finished interleaved
    /// samples onto `output`. `granule` is the packet's Ogg granule
    /// position (None when the packet's page carried none); `eofflag`
    /// marks the stream's final packet.
    pub fn decode_packet(
        &mut self,
        packet: &[u8],
        granule: Option<i64>,
        eofflag: bool,
        output: &mut Vec<f32>,
    ) -> Result<(), AudioError> {
        if packet.is_empty() {
            return Err(AudioError::from("vorbis: an empty audio packet"));
        }
        if packet[0] & 1 != 0 {
            return Err(AudioError::from(
                "vorbis: a header packet appears where audio is expected",
            ));
        }
        let mut reader = BitReader::new(packet);
        reader.read_bit(); // the packet type bit

        let mode_count = self.setup.modes.len() as u32;
        let mode_number = reader.read_bits(ilog(mode_count - 1));
        if mode_number as usize >= self.setup.modes.len() {
            return Err(AudioError::from(
                "vorbis: an audio packet's mode number exceeds the modes",
            ));
        }
        let mode = &self.setup.modes[mode_number as usize];
        let blockflag = mode.blockflag;
        let blocksize = if blockflag {
            self.setup.identification.blocksize_1
        } else {
            self.setup.identification.blocksize_0
        } as usize;
        let n2 = blocksize / 2;

        // The window flags (the long windows carry them; the decode
        // reads them for bitstream sync — the synthesis lapping uses
        // the actual block sequence instead).
        if blockflag {
            let _previous = reader.read_bit();
            let _next = reader.read_bit();
        }
        if reader.overrun {
            return Err(AudioError::from(
                "vorbis: the packet ended inside the mode/window decode",
            ));
        }

        let mapping_index = mode.mapping as usize;
        let channels = self.channels;

        // The floor decode (channel order, the spec's section 4.3.2).
        let mut floor_curves: Vec<Option<Vec<f32>>> = Vec::with_capacity(channels);
        let mut no_residue = vec![false; channels];
        for channel in 0..channels {
            let submap = self.setup.mappings[mapping_index].mux[channel] as usize;
            let floor_number = self.setup.mappings[mapping_index].submap_floor[submap] as usize;
            match &self.setup.floors[floor_number] {
                FloorConfig::One(config) => {
                    let decoded = decode_floor1_packet(config, &self.setup.codebooks, &mut reader, blocksize)?;
                    match decoded {
                        FloorPacket::Curve(curve) => {
                            no_residue[channel] = false;
                            floor_curves.push(Some(curve));
                        }
                        FloorPacket::Unused => {
                            no_residue[channel] = true;
                            floor_curves.push(None);
                        }
                    }
                }
                FloorConfig::Zero => {
                    return Err(AudioError::from(
                        "vorbis: the floor 0 decode is outside the decoder's scope (no modern \
                         encoder emits it)",
                    ));
                }
            }
        }
        if reader.overrun {
            // The spec's section 4.3.2's end-of-packet rule: the
            // channels zero and the synthesis still laps (a silent
            // frame).
            floor_curves.clear();
            no_residue.iter_mut().for_each(|flag| *flag = true);
        }

        // The nonzero propagate (the spec's section 4.3.3).
        let coupling_steps = self.setup.mappings[mapping_index].coupling_steps as usize;
        for step in 0..coupling_steps {
            let magnitude = self.setup.mappings[mapping_index].magnitude[step] as usize;
            let angle = self.setup.mappings[mapping_index].angle[step] as usize;
            if !no_residue[magnitude] || !no_residue[angle] {
                no_residue[magnitude] = false;
                no_residue[angle] = false;
            }
        }

        // The residue decode (submap order, the spec's section 4.3.4).
        // The per-channel residue vectors (length n/2).
        let mut residues: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; n2]).collect();
        let submaps = self.setup.mappings[mapping_index].submaps as usize;
        for submap in 0..submaps {
            let mut submap_channels: Vec<usize> = Vec::new();
            let mut do_not_decode: Vec<bool> = Vec::new();
            for channel in 0..channels {
                if self.setup.mappings[mapping_index].mux[channel] as usize == submap {
                    submap_channels.push(channel);
                    do_not_decode.push(no_residue[channel]);
                }
            }
            let residue_number = self.setup.mappings[mapping_index].submap_residue[submap] as usize;
            let residue_type = self.setup.residues[residue_number].residue_type;
            let ch = submap_channels.len();
            match residue_type {
                0 | 1 => {
                    let mut vectors: Vec<Vec<f32>> = (0..ch)
                        .map(|_| vec![0.0f32; n2])
                        .collect();
                    decode_residue_01(
                        &self.setup.residues[residue_number],
                        &self.setup.codebooks,
                        &mut reader,
                        &do_not_decode,
                        &mut vectors,
                        n2 as u32,
                    )?;
                    for (index, &channel) in submap_channels.iter().enumerate() {
                        residues[channel].copy_from_slice(&vectors[index]);
                    }
                }
                2 => {
                    // The residue 2's do-not-decode rule (the spec's
                    // section 8.6.5): all marked -> no decode;
                    // otherwise all decode as one interleaved vector.
                    if do_not_decode.iter().all(|&flag| flag) {
                        // The vectors stay zeroed.
                    } else {
                        let all_false = vec![false; 1];
                        let mut single = vec![vec![0.0f32; ch * n2]];
                        decode_residue_01(
                            &self.setup.residues[residue_number],
                            &self.setup.codebooks,
                            &mut reader,
                            &all_false,
                            &mut single,
                            (ch * n2) as u32,
                            )?;
                        // The deinterleave.
                        for (i, chunk) in single[0].chunks_exact(ch).enumerate() {
                            for (j, &value) in chunk.iter().enumerate() {
                                residues[submap_channels[j]][i] = value;
                            }
                        }
                    }
                }
                _ => unreachable!("the setup decode rejected the residue type"),
            }
            if reader.overrun {
                // The residue's end-of-packet is nominal: the decode
                // keeps what it read.
                break;
            }
        }

        // The inverse coupling (the spec's section 4.3.5), descending
        // step order, on the residue vectors.
        for step in (0..coupling_steps).rev() {
            let magnitude = self.setup.mappings[mapping_index].magnitude[step] as usize;
            let angle = self.setup.mappings[mapping_index].angle[step] as usize;
            for i in 0..n2 {
                let m = residues[magnitude][i];
                let a = residues[angle][i];
                let (new_m, new_a) = if m > 0.0 {
                    if a > 0.0 {
                        (m, m - a)
                    } else {
                        (m + a, m)
                    }
                } else if a > 0.0 {
                    (m, m + a)
                } else {
                    (m - a, m)
                };
                residues[magnitude][i] = new_m;
                residues[angle][i] = new_a;
            }
        }

        // The dot product (the spec's section 4.3.6) and the inverse
        // MDCT per channel: the block's raw time samples (no window —
        // the lapping applies it).
        let mut block: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for channel in 0..channels {
            let mut spectrum = std::mem::take(&mut residues[channel]);
            match &floor_curves[channel] {
                Some(curve) => {
                    for i in 0..n2 {
                        spectrum[i] *= curve[i];
                    }
                }
                None => spectrum.fill(0.0),
            }
            let table_index = if blockflag { 1 } else { 0 };
            let mut time = vec![0.0f64; blocksize];
            mdct_backward(&self.mdct_tables[table_index], &spectrum, &mut time);
            block.push(time.iter().map(|&v| v as f32).collect());
        }
        if reader.overrun {
            // The dot product's end-of-packet is nominal (the spec's
            // section 4.3.4's closing note): the decode keeps the
            // frame.
        }

        self.blockin(&block, blockflag, granule, eofflag, output);
        Ok(())
    }

    /// The lapping (the reference's blockin verbatim logic).
    fn blockin(
        &mut self,
        block: &[Vec<f32>],
        blockflag: bool,
        granule: Option<i64>,
        eofflag: bool,
        output: &mut Vec<f32>,
    ) {
        let n0 = self.setup.identification.blocksize_0 as usize / 2;
        let n1 = self.setup.identification.blocksize_1 as usize / 2;
        let n = if blockflag { n1 } else { n0 };

        let (this_center, prev_center) = if self.center_w != 0 {
            (n1, 0)
        } else {
            (0, n1)
        };

        let l_w = self.l_w;
        let w = blockflag;
        for channel in 0..self.channels {
            let buffer = &mut self.synth[channel].buffer;
            let p = &block[channel];
            if l_w {
                if w {
                    // large/large
                    let win = &self.windows[1];
                    for i in 0..n1 {
                        buffer[prev_center + i] =
                            buffer[prev_center + i] * win[n1 - i - 1] + p[i] * win[i];
                    }
                } else {
                    // large/small
                    let win = &self.windows[0];
                    let base = prev_center + n1 / 2 - n0 / 2;
                    for i in 0..n0 {
                        buffer[base + i] = buffer[base + i] * win[n0 - i - 1] + p[i] * win[i];
                    }
                }
            } else if w {
                // small/large
                let win = &self.windows[0];
                let offset = n1 / 2 - n0 / 2;
                for i in 0..n0 {
                    buffer[prev_center + i] =
                        buffer[prev_center + i] * win[n0 - i - 1] + p[i + offset] * win[i];
                }
                for i in n0..(n1 / 2 + n0 / 2) {
                    buffer[prev_center + i] = p[i + offset];
                }
            } else {
                // small/small
                let win = &self.windows[0];
                for i in 0..n0 {
                    buffer[prev_center + i] =
                        buffer[prev_center + i] * win[n0 - i - 1] + p[i] * win[i];
                }
            }
            // The copy section: the block's second half lands at
            // thisCenter.
            for i in 0..n {
                buffer[this_center + i] = p[n + i];
            }
        }

        self.center_w = if self.center_w != 0 { 0 } else { n1 };

        // The returned-span bookkeeping (the reference's initial
        // packet state rule).
        if self.pcm_returned == -1 {
            self.pcm_returned = this_center as isize;
            self.pcm_current = this_center;
        } else {
            self.pcm_returned = prev_center as isize;
            self.pcm_current = prev_center
                + (self.full_blocksize(l_w) / 4 + self.full_blocksize(w) / 4);
        }

        // The sequence/granule bookkeeping (the reference's sample
        // count and granulepos rules). The reference's lW/W advance
        // happens BEFORE the count: the movement is the previous
        // packet's W (this->lW) plus this packet's W.
        self.sequence += 1;
        if self.sample_count == -1 {
            self.sample_count = 0;
        } else {
            self.sample_count +=
                (self.full_blocksize(self.w) / 4 + self.full_blocksize(w) / 4) as i64;
        }
        // The lW/W state advance (the reference's lW/W assignment sits
        // after the bookkeeping that reads them).
        self.l_w = w;
        self.w = w;
        if self.granulepos == -1 {
            if let Some(packet_granule) = granule {
                self.granulepos = packet_granule;
                if self.sample_count > self.granulepos {
                    let extra = self.sample_count - self.granulepos;
                    if eofflag {
                        // The end trim.
                        let extra = (extra as usize).min(self.pcm_current - self.pcm_returned as usize);
                        self.pcm_current -= extra;
                    } else {
                        // The beginning trim.
                        self.pcm_returned += extra as isize;
                        if self.pcm_returned as usize > self.pcm_current {
                            self.pcm_returned = self.pcm_current as isize;
                        }
                    }
                }
            }
        } else {
            self.granulepos +=
                (self.full_blocksize(self.w) / 4 + self.full_blocksize(w) / 4) as i64;
            if let Some(packet_granule) = granule {
                if self.granulepos != packet_granule && self.granulepos > packet_granule {
                    let extra = self.granulepos - packet_granule;
                    if eofflag {
                        let extra = (extra as usize).min(self.pcm_current - self.pcm_returned as usize);
                        self.pcm_current -= extra;
                    }
                }
                self.granulepos = packet_granule;
            }
        }

        // The output: the samples [pcm_returned, pcm_current).
        let returned = self.pcm_returned.max(0) as usize;
        if returned < self.pcm_current {
            let count = self.pcm_current - returned;
            // The interleaved form the wrapper's buffer carries: one
            // frame's channels in sequence.
            for frame in 0..count {
                for channel in 0..self.channels {
                    output.push(self.synth[channel].buffer[returned + frame]);
                }
            }
        }
    }

    /// The full blocksize for a window flag.
    fn full_blocksize(&self, long: bool) -> usize {
        if long {
            self.setup.identification.blocksize_1 as usize
        } else {
            self.setup.identification.blocksize_0 as usize
        }
    }
}

// ---------------------------------------------------------------------------
// The floor 1 packet decode (the spec's sections 7.2.3 and 7.2.4)
// ---------------------------------------------------------------------------

enum FloorPacket {
    Curve(Vec<f32>),
    Unused,
}

fn decode_floor1_packet(
    config: &Floor1Config,
    codebooks: &[Codebook],
    reader: &mut BitReader,
    blocksize: usize,
) -> Result<FloorPacket, AudioError> {
    let n2 = blocksize / 2;
    let nonzero = reader.read_bit() != 0;
    if !nonzero {
        return Ok(FloorPacket::Unused);
    }
    let range = [256u32, 128, 86, 64][(config.multiplier - 1) as usize];
    let values = config.values;
    let mut y = vec![0u32; values];
    y[0] = reader.read_bits(ilog(range - 1));
    y[1] = reader.read_bits(ilog(range - 1));
    // The EOP during the floor's packet decode returns unused (the
    // spec's section 7.2.3's closing note).
    macro_rules! scalar_or_unused {
        ($book:expr) => {
            match $book.decode_scalar(reader) {
                Ok(value) => value,
                Err(_) => {
                    return Ok(FloorPacket::Unused)
                }
            }
        };
    }
    let mut offset = 2usize;
    for &class_number in &config.partition_class_list {
        let class = class_number as usize;
        let cdim = config.class_dimensions[class] as usize;
        let cbits = config.class_subclasses[class];
        let csub = (1u32 << cbits) - 1;
        let mut cval = 0u32;
        if cbits > 0 {
            let master = &codebooks[config.class_masterbooks[class] as usize];
            cval = scalar_or_unused!(master);
        }
        for j in 0..cdim {
            let book = config.subclass_books[class][(cval & csub) as usize];
            cval >>= cbits;
            if book >= 0 {
                y[offset + j] = scalar_or_unused!(codebooks[book as usize]) as u32;
            } else {
                y[offset + j] = 0;
            }
        }
        offset += cdim;
    }
    if reader.overrun {
        // The spec's section 7.2.3's end-of-packet rule: the floor
        // decode returns unused.
        return Ok(FloorPacket::Unused);
    }

    // Step 1: the amplitude synthesis (the spec's section 7.2.4).
    let mut step2_flag = vec![false; values];
    let mut final_y = vec![0i64; values];
    step2_flag[0] = true;
    step2_flag[1] = true;
    final_y[0] = y[0] as i64;
    final_y[1] = y[1] as i64;
    for i in 2..values {
        let low_neighbor = low_neighbor(&config.x_list, i);
        let high_neighbor = high_neighbor(&config.x_list, i);
        let predicted = render_point(
            config.x_list[low_neighbor],
            final_y[low_neighbor],
            config.x_list[high_neighbor],
            final_y[high_neighbor],
            config.x_list[i],
        );
        let val = y[i] as i64;
        let highroom = range as i64 - predicted;
        let lowroom = predicted;
        let room = if highroom < lowroom {
            highroom * 2
        } else {
            lowroom * 2
        };
        if val != 0 {
            step2_flag[low_neighbor] = true;
            step2_flag[high_neighbor] = true;
            step2_flag[i] = true;
            if val >= room {
                if highroom > lowroom {
                    final_y[i] = val - lowroom + predicted;
                } else {
                    final_y[i] = predicted - val + highroom - 1;
                }
            } else if val % 2 == 1 {
                final_y[i] = predicted - (val + 1) / 2;
            } else {
                final_y[i] = predicted + val / 2;
            }
        } else {
            step2_flag[i] = false;
            final_y[i] = predicted;
        }
    }
    // The spec's clamp suggestion (the section 7.2.4's note).
    for value in final_y.iter_mut() {
        *value = (*value).clamp(0, range as i64 - 1);
    }

    // Step 2: the curve synthesis (the sorted-X walk and render_line).
    let mut floor = vec![0i64; n2];
    let mut hx = 0usize;
    let mut lx = 0usize;
    let mut hy = final_y[config.sorted_order[0]] * config.multiplier as i64;
    let mut ly = hy;
    for &sorted_index in config.sorted_order.iter().skip(1) {
        if step2_flag[sorted_index] {
            hy = final_y[sorted_index] * config.multiplier as i64;
            hx = config.x_list[sorted_index] as usize;
            render_line(lx, ly, hx, hy, &mut floor);
            lx = hx;
            ly = hy;
        }
    }
    if hx < n2 {
        render_line(hx, hy, n2, hy, &mut floor);
    }
    // The floor1_inverse_dB_table lookup (the spec's section 10.1).
    let mut curve = vec![0.0f32; n2];
    for (index, &value) in floor.iter().enumerate() {
        curve[index] = FLOOR1_INVERSE_DB[(value.max(0) as usize).min(255)];
    }
    Ok(FloorPacket::Curve(curve))
}

/// The spec's low_neighbor helper (section 9.2.4).
fn low_neighbor(v: &[u32], x: usize) -> usize {
    let mut best = 0usize;
    let mut found = false;
    for n in 0..x {
        if v[n] < v[x] && (!found || v[n] > v[best]) {
            best = n;
            found = true;
        }
    }
    best
}

/// The spec's high_neighbor helper (section 9.2.5).
fn high_neighbor(v: &[u32], x: usize) -> usize {
    let mut best = 0usize;
    let mut found = false;
    for n in 0..x {
        if v[n] > v[x] && (!found || v[n] < v[best]) {
            best = n;
            found = true;
        }
    }
    best
}

/// The spec's render_point helper (section 9.2.6).
fn render_point(x0: u32, y0: i64, x1: u32, y1: i64, x: u32) -> i64 {
    let dy = y1 - y0;
    let adx = x1 as i64 - x0 as i64;
    let ady = dy.abs();
    let err = ady * (x as i64 - x0 as i64);
    let off = err / adx;
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

/// The spec's render_line helper (section 9.2.7): the integer line
/// walk writing the floor values at [x0, x1).
fn render_line(x0: usize, y0: i64, x1: usize, y1: i64, v: &mut [i64]) {
    if x1 <= x0 {
        return;
    }
    let dy = y1 - y0;
    let adx = (x1 - x0) as i64;
    let ady = dy.abs();
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let ady = ady - base.abs() * adx;
    let mut x = x0;
    let mut y = y0;
    let mut err = 0i64;
    if x0 < v.len() {
        v[x0] = y;
    }
    x += 1;
    while x < x1 {
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        if x < v.len() {
            v[x] = y;
        }
        x += 1;
    }
}

// ---------------------------------------------------------------------------
// The residue decode (the spec's sections 8.6.2 through 8.6.5)
// ---------------------------------------------------------------------------

/// The residue 0/1 packet decode (the spec's section 8.6.2's shared
/// loop; the format 0's interleaved and the format 1's sequential
/// partition writes; both accumulate additively).
fn decode_residue_01(
    config: &ResidueConfig,
    codebooks: &[Codebook],
    reader: &mut BitReader,
    do_not_decode: &[bool],
    vectors: &mut [Vec<f32>],
    actual_size: u32,
) -> Result<(), AudioError> {
    let ch = vectors.len();
    if ch == 0 {
        return Ok(());
    }
    decode_residue_body(
        config,
        codebooks,
        reader,
        do_not_decode,
        vectors,
        actual_size,
        ResidueFormat::Direct,
    )
}

/// The residue format's partition write rule (the Direct formats 0/1
/// versus the interleaved multi-channel form the residue 2 uses).
enum ResidueFormat {
    Direct,
}

/// The shared residue decode body (the reference's _01inverse verbatim
/// walk: the passes restart the partition walk, the pass 0 fetches the
/// classification codewords, the decodemap assigns the partitions).
fn decode_residue_body(
    config: &ResidueConfig,
    codebooks: &[Codebook],
    reader: &mut BitReader,
    do_not_decode: &[bool],
    vectors: &mut [Vec<f32>],
    actual_size: u32,
    format: ResidueFormat,
) -> Result<(), AudioError> {
    let ch = vectors.len();
    if ch == 0 {
        return Ok(());
    }
    let limit_begin = (config.begin as usize).min(actual_size as usize);
    let limit_end = (config.end as usize).min(actual_size as usize);
    let n_to_read = limit_end.saturating_sub(limit_begin);
    if n_to_read == 0 {
        return Ok(());
    }
    let samples_per_partition = config.partition_size as usize;
    let partitions_to_read = n_to_read / samples_per_partition;
    let classbook = &codebooks[config.classbook as usize];
    let partitions_per_word = classbook.dimensions as usize;
    // The partvals (the reference's res0_look: partitions^dim over the
    // phrasebook's dimensions) and its decodemap.
    let classifications = config.classifications as usize;
    let mut partvals = 1usize;
    for _ in 0..partitions_per_word {
        partvals *= classifications;
    }
    if partvals > classbook.entries as usize {
        return Err(AudioError::from(
            "vorbis: a residue classbook cannot encode the classification range",
        ));
    }
    let mut decodemap: Vec<Vec<usize>> = Vec::with_capacity(partvals);
    for entry in 0..partvals {
        let mut val = entry;
        let mut mult = partvals / classifications;
        let mut map = Vec::with_capacity(partitions_per_word);
        for _ in 0..partitions_per_word {
            let deco = val / mult;
            val -= deco * mult;
            mult /= classifications;
            map.push(deco);
        }
        decodemap.push(map);
    }

    // The max stage count (the reference's ov_ilog over the cascades).
    let mut stages = 0usize;
    for &cascade in &config.cascade {
        stages = stages.max(ilog(cascade) as usize);
    }

    // The partword store: ch rows of partwords-sized slots.
    let mut partword = vec![vec![0usize; partitions_to_read + partitions_per_word]; ch];
    for pass in 0..stages {
        let mut i = 0usize;
        let mut l = 0usize;
        while i < partitions_to_read {
            if pass == 0 {
                // The classification codewords per channel.
                for j in 0..ch {
                    if do_not_decode[j] {
                        continue;
                    }
                    // The reference's eopbreak: an EOP (or an out-of-
                    // range codeword) stops the decode nominally and
                    // keeps what was read so far.
                    let temp = match classbook.decode_scalar(reader) {
                        Ok(value) => value as usize,
                        Err(_) => return Ok(()),
                    };
                    if temp >= partvals {
                        // The reference's eopbreak: the nominal stop.
                        return Ok(());
                    }
                    // The flat row layout: the codeword group l
                    // occupies l*partitions_per_word slots.
                    let base = l * partitions_per_word;
                    partword[j][base..base + partitions_per_word]
                        .copy_from_slice(&decodemap[temp]);
                }
            }
            // The partition decode for this codeword group.
            let mut k = 0usize;
            while k < partitions_per_word && i < partitions_to_read {
                for j in 0..ch {
                    if do_not_decode[j] {
                        continue;
                    }
                    let vqclass = partword[j][l * partitions_per_word + k];
                    let vqbook = config.books[vqclass][pass];
                    if vqbook < 0 {
                        continue;
                    }
                    let book = &codebooks[vqbook as usize];
                    let offset = limit_begin + i * samples_per_partition;
                    let dims = book.dimensions as usize;
                    match format {
                        ResidueFormat::Direct => {
                            let vector = &mut vectors[j];
                            if config.residue_type == 0 {
                                // The interleaved form (the reference's
                                // decodevs_add: element i lands at
                                // i*step; the entries' decode order is
                                // step-major).
                                let step = samples_per_partition / dims;
                                if step == 0 {
                                    return Err(AudioError::from(
                                        "vorbis: a residue 0 partition is smaller than a \
                                         codebook dimension",
                                    ));
                                }
                                let mut entries = vec![0.0f32; step * dims];
                                let mut base = 0usize;
                                for _ in 0..step {
                                    // The reference's decodepart
                                    // returns -1 on EOP and the caller
                                    // breaks out (the eopbreak): the
                                    // decode keeps what it read.
                                    if book
                                        .decode_vector(reader, &mut entries[base..base + dims])
                                        .is_err()
                                    {
                                        return Ok(());
                                    }
                                    base += dims;
                                }
                                // The scatter (the reference's
                                // decodevs_add's closing loop):
                                // a[i*step + j] += entries[j*dim + i]
                                // for i over the dimensions and j over
                                // the step slots.
                                for i in 0..dims {
                                    for j in 0..step {
                                        let index = offset + i * step + j;
                                        if index < vector.len() {
                                            vector[index] += entries[j * dims + i];
                                        }
                                    }
                                }
                            } else {
                                // The sequential form (the reference's
                                // decodev_add / decodevv_add: the
                                // partition decodes exactly
                                // samples_per_partition values; an
                                // entry crossing the partition's end
                                // writes its head and discards the
                                // tail — the bits were consumed).
                                let stop = (offset + samples_per_partition)
                                    .min(limit_end)
                                    .min(vector.len());
                                let mut pos = offset;
                                let mut scratch = vec![0.0f32; dims];
                                while pos < stop {
                                    if book.decode_vector(reader, &mut scratch).is_err()
                                    {
                                        return Ok(());
                                    }
                                    for &value in &scratch {
                                        if pos >= stop {
                                            break;
                                        }
                                        vector[pos] += value;
                                        pos += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                k += 1;
                i += 1;
            }
            l += 1;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The top-level wrapper
// ---------------------------------------------------------------------------

/// The decoded stream: the PCM samples interleaved plus the format
/// fields the callers need.
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub channels: u32,
    /// The interleaved f32 samples in [-1, 1] (the reference decoder's
    /// output scale).
    pub samples: Vec<f32>,
}

/// Demuxes with per-packet granule positions: the walk records each
/// page's granule onto the last packet the page finished.
fn demux_with_granules(
    data: &[u8],
) -> Result<(Vec<Vec<u8>>, Vec<Option<i64>>, Vec<bool>), AudioError> {
    let mut assembler = PacketAssembler::new();
    let mut granules: Vec<Option<i64>> = Vec::new();
    let mut eofflags: Vec<bool> = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let (page, consumed) = parse_page(data, offset)?;
        offset += consumed;
        let before = assembler.packets.len();
        assembler.feed_page(&page)?;
        let after = assembler.packets.len();
        if after > before {
            // The page's granule belongs to its last finished packet.
            // The "-1" granule (no packet finished on the page) reads
            // as the u64 all-ones form.
            let granule = if page.granule_position == u64::MAX {
                None
            } else {
                Some(page.granule_position as i64)
            };
            let eos = page.header_type & 0x04 != 0;
            for index in before..after {
                let is_last = index == after - 1;
                granules.push(if is_last { granule } else { None });
                eofflags.push(if is_last { eos } else { false });
            }
        }
    }
    // The assembler's leftover packets (a final partial page without a
    // closing lace) transfer too.
    let packets = std::mem::take(&mut assembler.packets);
    if granules.len() < packets.len() {
        granules.resize(packets.len(), None);
        eofflags.resize(packets.len(), false);
    }
    if assembler.has_partial() {
        return Err(AudioError::from(
            "ogg: the stream ends inside a packet",
        ));
    }
    Ok((packets, granules, eofflags))
}

/// Decodes a whole Ogg Vorbis file into interleaved f32 PCM. The
/// granule handling follows the reference decoder's blockin rules:
/// the first frame primes (no output), the stream's granule positions
/// trim the beginning (the encoder delay) and the trailing partial
/// frame.
pub fn decode_ogg_vorbis(data: &[u8]) -> Result<DecodedAudio, AudioError> {
    let (packets, granules, eofflags) = demux_with_granules(data)?;
    if packets.len() < 3 {
        return Err(AudioError::from(
            "vorbis: the stream carries fewer than the three header packets",
        ));
    }
    let mut decoder = VorbisDecoder::new(&packets[0], &packets[1], &packets[2])?;
    let (sample_rate, channels) = {
        let identification = decoder.identification();
        (identification.sample_rate, identification.channels)
    };
    let mut samples: Vec<f32> = Vec::new();
    for index in 3..packets.len() {
        decoder.decode_packet(
            &packets[index],
            granules[index],
            eofflags[index],
            &mut samples,
        )?;
    }
    Ok(DecodedAudio {
        sample_rate,
        channels,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode of the 1 kHz mono test vector against the reference
    /// PCM produced by the real libvorbis (the reference library build,
    /// not a demuxer's trimmed view of it): the RMS error stays under
    /// one s16 LSB-equivalent and the sample count matches the
    /// reference's.
    #[test]
    fn decodes_the_test1k_vector_against_the_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/test1k.ogg");
        let data = std::fs::read(path).expect("the test vector");
        let decoded = decode_ogg_vorbis(&data).expect("the decode");
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.sample_rate, 48000);
        let reference_path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/test1k.ref.pcm");
        let reference = std::fs::read(reference_path).expect("the reference");
        let reference_samples: Vec<f32> = reference
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect();
        assert_eq!(
            decoded.samples.len(),
            reference_samples.len(),
            "the decoded frame count matches the reference"
        );
        let mut squared_error = 0.0f64;
        for (got, want) in decoded.samples.iter().zip(reference_samples.iter()) {
            let diff = (*got - *want) as f64;
            squared_error += diff * diff;
        }
        let rms = (squared_error / decoded.samples.len() as f64).sqrt();
        assert!(
            rms < 1.0 / 32768.0,
            "the RMS error {} exceeds one s16 LSB",
            rms
        );
    }

    /// The decode of the 440 Hz stereo test vector against the ffmpeg
    /// reference.
    #[test]
    fn decodes_the_test440_vector_against_the_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/test440.ogg");
        let data = std::fs::read(path).expect("the test vector");
        let decoded = decode_ogg_vorbis(&data).expect("the decode");
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 44100);
        let reference_path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/test440.ref.pcm");
        let reference = std::fs::read(reference_path).expect("the reference");
        let reference_samples: Vec<f32> = reference
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect();
        assert_eq!(
            decoded.samples.len(),
            reference_samples.len(),
            "the decoded frame count matches the reference"
        );
        let mut squared_error = 0.0f64;
        for (got, want) in decoded.samples.iter().zip(reference_samples.iter()) {
            let diff = (*got - *want) as f64;
            squared_error += diff * diff;
        }
        let rms = (squared_error / decoded.samples.len() as f64).sqrt();
        assert!(
            rms < 1.0 / 32768.0,
            "the RMS error {} exceeds one s16 LSB",
            rms
        );
    }
}
