//! The Vorbis decoder's bit-level machinery: the LSB-first bit reader
//! the spec's bitpacking convention (section 2) defines, and the
//! codebook unpack (section 3.2) the setup header carries.
//!
//! The reader's bit order: the bits arrive least-significant-first
//! within each byte, bytes in stream order. A read past the packet's
//! end marks the reader failed; further reads return zeroes and set
//! the `overrun` flag — the graceful-degradation rule the truncated
//! audio packets rely on (the spec's end-of-packet rule, section 2.1.8).
//!
//! The codebook's Huffman assignment follows the canonical rule
//! (section 3.2.1: each used entry takes the lowest-valued unused
//! codeword of its length, entries in order) implemented with the
//! marker-walk the reference decoder's `_make_words` uses: claiming a
//! node also prunes the imagined subtree below it and advances the
//! shorter-length markers, which keeps the assignment correct for
//! sparse and unordered length lists.

use super::AudioError;

/// The LSB-first bit reader over one packet's bytes.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u32,
    pub overrun: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
            overrun: false,
        }
    }

    /// Reads up to 32 bits LSB-first. A zero-width read is legal (the
    /// spec's "reading zero bits" rule) and returns 0.
    pub fn read_bits(&mut self, count: u32) -> u32 {
        if count == 0 {
            return 0;
        }
        if count > 32 {
            // The spec caps reads at 32 bits; a larger request is a
            // decoder bug and marks the failure rather than wrapping.
            self.overrun = true;
            return 0;
        }
        let mut value: u32 = 0;
        for i in 0..count {
            if self.byte_pos >= self.data.len() {
                self.overrun = true;
                return value;
            }
            let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
            value |= (bit as u32) << i;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        value
    }

    /// Reads one bit.
    #[inline]
    pub fn read_bit(&mut self) -> u32 {
        self.read_bits(1)
    }

    /// The absolute bit cursor (the diagnostics use it).
    pub fn bits_consumed(&self) -> u64 {
        self.byte_pos as u64 * 8 + self.bit_pos as u64
    }
}

/// The codebook's decoded form (the spec's section 3.2 unpack). The
/// Huffman table lists the (length, codeword, entry) triples sorted by
/// (length, codeword) so the decode walk can match the consumed bits;
/// the VQ values are materialised per entry at unpack time.
pub struct Codebook {
    /// The dimensions each lookup entry carries (the VQ vector's
    /// length).
    pub dimensions: u32,
    /// The entries' count (the full declared count, used entries or
    /// not).
    pub entries: u32,
    /// The used entries' count (the sparse books' live subset).
    pub used_entries: u32,
    /// The codeword lengths per entry (0 marks an unused entry).
    pub lengths: Vec<u8>,
    /// The Huffman table: (length, codeword, entry) sorted by (length,
    /// codeword). The codeword's integer form reads MSb-first: the
    /// first bit the packet carries is the codeword's most significant
    /// bit (the spec's section 3.2.1 note).
    pub lookup_table: Vec<(u32, u32, u32)>,
    /// The lookup type: 0 no vector lookup, 1 implicit quantised
    /// values, 2 explicit value list.
    pub lookup_type: u32,
    /// The VQ's dequantised float values, flattened per declared entry
    /// (entry * dimensions; the unused entries carry zeroes).
    pub values: Vec<f32>,
}

impl Codebook {
    /// Unpacks one codebook from the setup header's bit reader (the
    /// spec's section 3.2.1). The caller reads the sync pattern before
    /// the call.
    pub fn unpack(reader: &mut BitReader) -> Result<Codebook, AudioError> {
        let sync = reader.read_bits(24);
        if sync != 0x564342 {
            return Err(AudioError::from(
                "vorbis: a codebook's sync pattern is missing",
            ));
        }
        let dimensions = reader.read_bits(16);
        let entries = reader.read_bits(24);
        if entries == 0 {
            return Err(AudioError::from(
                "vorbis: the codebook's entry count is out of range",
            ));
        }
        if ilog(dimensions) + ilog(entries) > 24 {
            return Err(AudioError::from(
                "vorbis: the codebook's dimension/entry magnitudes overflow the spec's bound",
            ));
        }
        let ordered = reader.read_bit() != 0;
        let mut lengths = vec![0u8; entries as usize];
        if ordered {
            // The ordered length list: run-lengths per length ascending,
            // each run's count read in ilog(entries - current) bits.
            let mut current = 0u32;
            let mut current_length = reader.read_bits(5) + 1;
            while current < entries {
                let number = reader.read_bits(ilog(entries - current));
                if number as u64 > (entries - current) as u64 || current_length > 32 {
                    return Err(AudioError::from(
                        "vorbis: the ordered length list overruns the entries",
                    ));
                }
                for i in 0..number {
                    lengths[(current + i) as usize] = current_length as u8;
                }
                current += number;
                current_length += 1;
            }
        } else {
            let sparse = reader.read_bit() != 0;
            for length in lengths.iter_mut() {
                if sparse {
                    if reader.read_bit() != 0 {
                        *length = (reader.read_bits(5) + 1) as u8;
                    } else {
                        *length = 0; // unused
                    }
                } else {
                    *length = (reader.read_bits(5) + 1) as u8;
                }
            }
        }

        // The canonical codeword assignment (section 3.2.1): each used
        // entry, in entry order, takes the lowest-valued unused codeword
        // of its length. The marker walk the reference decoder carries
        // implements the rule for unordered and sparse length lists:
        // claiming a node prunes the imagined subtree below it and
        // advances the shorter-length markers when the claimed node's
        // sibling path is exhausted.
        let mut lookup_table: Vec<(u32, u32, u32)> = Vec::new();
        let mut used_entries = 0u32;
        {
            let mut marker = [0u32; 33];
            for (index, &length) in lengths.iter().enumerate() {
                if length == 0 {
                    continue;
                }
                used_entries += 1;
                let entry_word = marker[length as usize];
                // The overpopulation check: a claimed node past the
                // length's capacity means the lengths overspecify the
                // tree.
                if length < 32 && (entry_word >> length) != 0 {
                    return Err(AudioError::from(
                        "vorbis: the codebook's lengths overspecify the Huffman tree",
                    ));
                }
                lookup_table.push((length as u32, entry_word, index as u32));
                // Update the markers: walk from this length down; the
                // first odd marker jumps branches (its subtree is full,
                // so the node above takes the next sibling slot).
                let mut j = length as usize;
                while j > 0 {
                    if marker[j] & 1 != 0 {
                        if j == 1 {
                            marker[1] += 1;
                        } else {
                            marker[j] = marker[j - 1] << 1;
                        }
                        break;
                    }
                    marker[j] += 1;
                    j -= 1;
                }
                // Prune the tree: the longer markers dangled from the
                // claimed node; re-dangle them from the new node.
                let mut carried = entry_word;
                for j in (length as usize + 1)..33 {
                    if (marker[j] >> 1) == carried {
                        carried = marker[j];
                        marker[j] = marker[j - 1] << 1;
                    } else {
                        break;
                    }
                }
            }
            // The underpopulation check: any marker still holding bits
            // below its length's capacity means the tree is unfinished.
            // The single-entry codebook (the spec's 20150226 errata) is
            // the tolerated exception: one codeword of length 1 is an
            // underpopulated tree by construction.
            let single_entry = used_entries == 1;
            if !(single_entry && marker[2] == 2) {
                for i in 1..33usize {
                    if marker[i] & (0xffff_ffffu32 >> (32 - i)) != 0 {
                        return Err(AudioError::from(
                            "vorbis: the codebook's lengths underspecify the Huffman tree",
                        ));
                    }
                }
            }
            lookup_table.sort_by_key(|&(length, code, _)| (length, code));
        }

        // The vector lookup (the VQ part, section 3.2.1's lookup
        // decode). The values materialise per declared entry; the
        // unused entries carry zeroes.
        let lookup_type = reader.read_bits(4);
        let dims = dimensions as usize;
        let entries_us = entries as usize;
        let mut values = vec![0.0f32; entries_us * dims];
        match lookup_type {
            0 => {}
            1 | 2 => {
                let minimum = float32_unpack(reader.read_bits(32));
                let delta = float32_unpack(reader.read_bits(32));
                let value_bits = reader.read_bits(4) + 1;
                let sequence_p = reader.read_bit() != 0;
                let lookup_values_count = match lookup_type {
                    1 => lookup1_values(entries, dimensions) as usize,
                    _ => entries_us * dims,
                };
                if reader.overrun {
                    return Err(AudioError::from(
                        "vorbis: the packet ended inside a codebook's lookup header",
                    ));
                }
                let mut multiplicands = vec![0u32; lookup_values_count];
                for slot in multiplicands.iter_mut() {
                    *slot = reader.read_bits(value_bits);
                }
                match lookup_type {
                    1 => {
                        // The lattice VQ: the quantvals build the
                        // square lattice; each entry's vector walks the
                        // multiplicands by the mixed-radix division
                        // (the spec's lookup type 1 rule, the
                        // reference's _book_unquantize case 1).
                        let quantvals = lookup1_values(entries, dimensions) as usize;
                        if quantvals == 0 || dims == 0 {
                            return Err(AudioError::from(
                                "vorbis: a lookup type 1 codebook carries zero quantvals",
                            ));
                        }
                        for entry in 0..entries_us {
                            if lengths[entry] == 0 {
                                continue;
                            }
                            let mut last = 0.0f32;
                            let mut index_div = 1usize;
                            for k in 0..dims {
                                let index = (entry / index_div) % quantvals;
                                let val = (multiplicands[index] as f32).abs() * delta
                                    + minimum
                                    + last;
                                if sequence_p {
                                    last = val;
                                }
                                values[entry * dims + k] = val;
                                index_div *= quantvals;
                            }
                        }
                    }
                    _ => {
                        // The explicit value list: each scalar comes
                        // one-to-one from the multiplicands (the
                        // reference's case 2).
                        for entry in 0..entries_us {
                            if lengths[entry] == 0 {
                                continue;
                            }
                            let mut last = 0.0f32;
                            for k in 0..dims {
                                let index = entry * dims + k;
                                if index >= multiplicands.len() {
                                    break;
                                }
                                let val = (multiplicands[index] as f32).abs() * delta
                                    + minimum
                                    + last;
                                if sequence_p {
                                    last = val;
                                }
                                values[entry * dims + k] = val;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(AudioError::from(format!(
                    "vorbis: the codebook lookup type {} is reserved",
                    lookup_type
                )))
            }
        }

        Ok(Codebook {
            dimensions,
            entries,
            used_entries,
            lengths,
            lookup_table,
            lookup_type,
            values,
        })
    }

    /// Decodes one Huffman symbol: walks the reader's bits one at a
    /// time against the sorted codeword table. The return is the
    /// original entry index. The single-entry codebook (the spec's
    /// 20150226 errata) sinks one bit and always returns its entry.
    pub fn decode_scalar(&self, reader: &mut BitReader) -> Result<u32, AudioError> {
        if self.used_entries == 0 {
            return Err(AudioError::from(
                "vorbis: a decode request hit a codebook with no used entries",
            ));
        }
        if self.used_entries == 1 {
            // The single-entry rule: the walk sinks one bit (either
            // value) and returns the sole entry.
            let _ = reader.read_bit();
            if reader.overrun {
                return Err(AudioError::from(
                    "vorbis: the packet ended inside a Huffman walk",
                ));
            }
            for &(_, _, entry) in &self.lookup_table {
                return Ok(entry);
            }
        }
        let mut word: u32 = 0;
        let mut length: u32 = 0;
        loop {
            let bit = reader.read_bit();
            if reader.overrun {
                return Err(AudioError::from(
                    "vorbis: the packet ended inside a Huffman walk",
                ));
            }
            word = (word << 1) | bit;
            length += 1;
            if length > 32 {
                return Err(AudioError::from(
                    "vorbis: the Huffman walk ran past 32 bits (a corrupt codebook)",
                ));
            }
            // The table's (length, codeword) sort bounds the search to
            // this length's slice. The slice's start comes from the
            // partition point (a binary search may land anywhere inside
            // the equal range; only the partition point is stable).
            let start = self
                .lookup_table
                .partition_point(|&(l, _, _)| l < length);
            let mut end = start;
            while end < self.lookup_table.len() && self.lookup_table[end].0 == length {
                end += 1;
            }
            if start == end {
                continue;
            }
            for &(l, code, entry) in &self.lookup_table[start..end] {
                debug_assert_eq!(l, length);
                if code == word {
                    return Ok(entry);
                }
            }
        }
    }

    /// Decodes one VQ vector into the output slice (the codebook's
    /// dimensions count of floats). The lookup type 0 codebooks carry
    /// no vector values — the entry index itself is the decoded value
    /// in the scalar contexts (the spec's section 3.3 forbids VQ use of
    /// type 0 books; the caller's scalar path handles those).
    pub fn decode_vector(&self, reader: &mut BitReader, output: &mut [f32]) -> Result<(), AudioError> {
        let entry = self.decode_scalar(reader)?;
        let dims = self.dimensions as usize;
        if output.len() < dims {
            return Err(AudioError::from(
                "vorbis: the vector output is smaller than the codebook's dimensions",
            ));
        }
        if self.lookup_type == 0 {
            // The scalar codebook in a VQ position: the entry index is
            // the value (the floor 0 path's convention; the modern
            // streams' floor 1 and residue books always carry maps).
            output[..dims].fill(entry as f32);
        } else {
            let base = entry as usize * dims;
            output[..dims].copy_from_slice(&self.values[base..base + dims]);
        }
        Ok(())
    }
}

/// The spec's ilog helper (section 9.2.1): the position of the highest
/// set bit (the bit count the magnitude needs).
pub(crate) fn ilog(x: u32) -> u32 {
    let mut value = x;
    let mut bits = 0;
    while value > 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

/// The spec's float32_unpack (section 9.2.2): the Vorbis packed float's
/// non-IEEE form (21-bit mantissa, 10-bit biased exponent) onto the
/// host f32. The exponent's bias: exp(bits) - (mantissa bits - 1) -
/// 768.
pub(crate) fn float32_unpack(packed: u32) -> f32 {
    let mantissa = (packed & 0x001f_ffff) as f64;
    let sign = packed & 0x8000_0000;
    let exponent = ((packed & 0x7fe0_0000) >> 21) as i32 - 788;
    let magnitude = mantissa * 2f64.powi(exponent);
    let magnitude = magnitude as f32;
    if sign != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// The spec's lookup1_values helper (section 9.2.3): the largest
/// integer r whose r^dimensions fits the entries count. The reference's
/// _book_maptype1_quantvals computes the float root then verifies by
/// integer means; this port walks the same verification.
pub(crate) fn lookup1_values(entries: u32, dimensions: u32) -> u32 {
    if entries < 1 || dimensions == 0 {
        return 0;
    }
    let entries = entries as i64;
    let dims = dimensions as i64;
    let mut vals = ((entries as f64).powf(1.0 / dims as f64)).floor() as i64;
    if vals < 1 {
        vals = 1;
    }
    loop {
        // The integer verification the reference's
        // _book_maptype1_quantvals carries: the walk builds vals^dims
        // (breaking early when the product would exceed the entries)
        // and (vals+1)^dims, then accepts vals only when
        // vals^dims <= entries < (vals+1)^dims.
        let mut acc: i64 = 1;
        let mut completed = true;
        for _ in 0..dims {
            if entries / vals < acc {
                completed = false;
                break;
            }
            acc *= vals;
        }
        let acc1 = pow_checked(vals + 1, dims);
        if completed && acc <= entries && acc1 > entries {
            return vals as u32;
        }
        if !completed || acc > entries {
            vals -= 1;
        } else {
            vals += 1;
        }
        if vals < 1 {
            return 1;
        }
    }
}

/// The base^dims product saturating at i64's ceiling (the reference's
/// LONG_MAX clamp in the verification walk).
fn pow_checked(base: i64, dims: i64) -> i64 {
    let mut acc: i64 = 1;
    for _ in 0..dims {
        acc = match acc.checked_mul(base) {
            Some(value) => value,
            None => return i64::MAX,
        };
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads one codebook out of a byte buffer (the test helper).
    fn unpack_from(data: &[u8]) -> Codebook {
        let mut reader = BitReader::new(data);
        Codebook::unpack(&mut reader).expect("the codebook unpack")
    }

    /// Packs the spec's section 3.2.1 worked example: lengths
    /// {2,4,4,4,4,2,3,3} unordered, non-sparse, lookup type 0. The
    /// canonical codewords the spec lists drive the assertions.
    #[test]
    fn codewords_match_the_spec_worked_example() {
        // sync 0x564342 (24 bits), dim 0 (16), entries 8 (24), ordered
        // = 0, sparse = 0, lengths 2,4,4,4,4,2,3,3 (5 bits each),
        // lookup type 0 (4 bits).
        let mut bits: Vec<u32> = vec![];
        let push = |bits: &mut Vec<u32>, value: u32, width: u32| {
            for i in 0..width {
                bits.push((value >> i) & 1);
            }
        };
        push(&mut bits, 0x564342, 24);
        push(&mut bits, 0, 16);
        push(&mut bits, 8, 24);
        push(&mut bits, 0, 1); // ordered = 0
        push(&mut bits, 0, 1); // sparse = 0
        for &length in &[2u32, 4, 4, 4, 4, 2, 3, 3] {
            push(&mut bits, length - 1, 5);
        }
        push(&mut bits, 0, 4); // lookup type 0
        // The bits pack LSB-first into bytes.
        let mut data = Vec::new();
        let mut byte = 0u8;
        let mut position = 0u32;
        for &bit in &bits {
            byte |= (bit as u8) << position;
            position += 1;
            if position == 8 {
                data.push(byte);
                byte = 0;
                position = 0;
            }
        }
        if position > 0 {
            data.push(byte);
        }
        let book = unpack_from(&data);
        assert_eq!(book.entries, 8);
        assert_eq!(book.used_entries, 8);
        // The spec's codeword list: entry 0: 00 (0), 1: 0100 (4),
        // 2: 0101 (5), 3: 0110 (6), 4: 0111 (7), 5: 10 (2), 6: 110 (6
        // at length 3), 7: 111 (7 at length 3).
        let expected: [(u32, u32, u32); 8] = [
            // The table sorts by (length, codeword): the two
            // length-2 codewords first, then the length-3 pair, then
            // the length-4 run.
            (2, 0b00, 0),
            (2, 0b10, 5),
            (3, 0b110, 6),
            (3, 0b111, 7),
            (4, 0b0100, 1),
            (4, 0b0101, 2),
            (4, 0b0110, 3),
            (4, 0b0111, 4),
        ];
        for (index, &(length, code, entry)) in expected.iter().enumerate() {
            assert_eq!(
                book.lookup_table[index],
                (length, code, entry),
                "codeword {} mismatches the spec's worked example",
                index
            );
        }
    }

    /// The ordered length path: 4 entries, all length 2, the ordered
    /// run-length encoding (1 run of 4 at length 2).
    #[test]
    fn ordered_lengths_assign_canonical_codewords() {
        let mut bits: Vec<u32> = vec![];
        let push = |bits: &mut Vec<u32>, value: u32, width: u32| {
            for i in 0..width {
                bits.push((value >> i) & 1);
            }
        };
        push(&mut bits, 0x564342, 24);
        push(&mut bits, 1, 16); // dimensions 1
        push(&mut bits, 4, 24); // entries 4
        push(&mut bits, 1, 1); // ordered = 1
        push(&mut bits, 1, 5); // first length 2 (value 1 + 1)
        push(&mut bits, 4, ilog(4)); // 4 entries at this length
        push(&mut bits, 0, 4); // lookup type 0
        let mut data = Vec::new();
        let mut byte = 0u8;
        let mut position = 0u32;
        for &bit in &bits {
            byte |= (bit as u8) << position;
            position += 1;
            if position == 8 {
                data.push(byte);
                byte = 0;
                position = 0;
            }
        }
        if position > 0 {
            data.push(byte);
        }
        let book = unpack_from(&data);
        assert_eq!(book.lookup_table.len(), 4);
        // The canonical assignment: 00, 01, 10, 11.
        for (index, &(length, code, entry)) in book.lookup_table.iter().enumerate() {
            assert_eq!(length, 2);
            assert_eq!(code, index as u32);
            assert_eq!(entry, index as u32);
        }
    }

    /// The lookup type 1 dequantisation: the reference's own self-test
    /// case (the sharedbook.c test4: dim 3, entries 27, quantlist
    /// {0, 7, 2}, q_min -533200896, q_delta 1611661312, 4 value bits,
    /// non-sequential). The expected values come from the reference's
    /// hand-rolled test4_result table.
    #[test]
    fn lookup_type1_matches_the_reference_dequant_case() {
        let mut bits: Vec<u32> = vec![];
        let push = |bits: &mut Vec<u32>, value: u32, width: u32| {
            for i in 0..width {
                bits.push((value >> i) & 1);
            }
        };
        push(&mut bits, 0x564342, 24);
        push(&mut bits, 3, 16); // dimensions 3
        push(&mut bits, 27, 24); // entries 27
        push(&mut bits, 0, 1); // ordered = 0
        push(&mut bits, 0, 1); // sparse = 0
        // A complete Huffman tree over 27 entries: 5 codewords of
        // length 4 and 22 of length 5 (5/16 + 22/32 = 1). The
        // reference's own self-test feeds its codebook struct
        // directly, so it carries no packed lengthlist at all; the
        // unpack test needs one that survives the Huffman
        // validation. The lengths do not affect the dequantised
        // values (they follow the entry index alone).
        for entry in 0..27 {
            push(&mut bits, if entry < 5 { 3 } else { 4 }, 5);
        }
        push(&mut bits, 1, 4); // lookup type 1
        push(&mut bits, (-533200896i64) as u32 & 0xffff_ffff, 32);
        push(&mut bits, 1611661312u32, 32);
        push(&mut bits, 3, 4); // value bits 4 (q_quant - 1)
        push(&mut bits, 0, 1); // sequencep = 0
        // lookup1_values(27, 3) = 3, so 3 multiplicands: {0, 7, 2}.
        for &quant in &[0u32, 7, 2] {
            push(&mut bits, quant, 4);
        }
        let mut data = Vec::new();
        let mut byte = 0u8;
        let mut position = 0u32;
        for &bit in &bits {
            byte |= (bit as u8) << position;
            position += 1;
            if position == 8 {
                data.push(byte);
                byte = 0;
                position = 0;
            }
        }
        if position > 0 {
            data.push(byte);
        }
        let book = unpack_from(&data);
        // The reference's test4_result (the first entries):
        // -3,-3,-3, 4,-3,-3, -1,-3,-3, -3,4,-3, 4,4,-3, -1,4,-3, ...
        let expected: [&[f32]; 4] = [
            &[-3.0, -3.0, -3.0],
            &[4.0, -3.0, -3.0],
            &[-1.0, -3.0, -3.0],
            &[-3.0, 4.0, -3.0],
        ];
        for (entry, values) in expected.iter().enumerate() {
            for (k, &want) in values.iter().enumerate() {
                let got = book.values[entry * 3 + k];
                assert!(
                    (got - want).abs() < 1e-4,
                    "entry {} dim {}: got {} want {}",
                    entry,
                    k,
                    got,
                    want
                );
            }
        }
    }

    /// The single-entry codebook's decode: one used entry of length 1
    /// sinks one bit and returns the entry for either bit value (the
    /// spec's 20150226 errata rule).
    #[test]
    fn single_entry_codebook_sinks_one_bit() {
        let mut bits: Vec<u32> = vec![];
        let push = |bits: &mut Vec<u32>, value: u32, width: u32| {
            for i in 0..width {
                bits.push((value >> i) & 1);
            }
        };
        push(&mut bits, 0x564342, 24);
        push(&mut bits, 1, 16);
        push(&mut bits, 2, 24); // entries 2, one unused (sparse)
        push(&mut bits, 0, 1); // ordered = 0
        push(&mut bits, 1, 1); // sparse = 1
        push(&mut bits, 1, 1); // entry 0 used
        push(&mut bits, 0, 5); // length 1
        push(&mut bits, 0, 1); // entry 1 unused
        push(&mut bits, 0, 4); // lookup type 0
        let mut data = Vec::new();
        let mut byte = 0u8;
        let mut position = 0u32;
        for &bit in &bits {
            byte |= (bit as u8) << position;
            position += 1;
            if position == 8 {
                data.push(byte);
                byte = 0;
                position = 0;
            }
        }
        if position > 0 {
            data.push(byte);
        }
        let book = unpack_from(&data);
        assert_eq!(book.used_entries, 1);
        // Reading with a 0 bit and with a 1 bit both return entry 0 and
        // sink exactly one bit.
        for leading_bit in [0u8, 1u8] {
            let packet = vec![leading_bit];
            let mut reader = BitReader::new(&packet);
            let entry = book.decode_scalar(&mut reader).expect("the decode");
            assert_eq!(entry, 0);
            assert_eq!(reader.bits_consumed(), 1);
        }
    }

    /// The bit reader's LSB-first order: the spec's section 2.1.7
    /// decoding example (the byte 0x3F carries the fields the example
    /// reads).
    #[test]
    fn bit_reader_follows_the_lsb_first_order() {
        // The spec's example stream: byte 0 = 0b00111111 (the 4-bit 12,
        // the 3-bit -1 and the low 1 bit of the 7-bit 17), byte 1 =
        // 0b01001000.
        let data = [0b1111_1100u8, 0b0100_1000u8];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(2), 0b00);
        assert_eq!(reader.read_bits(2), 0b11);
        // The continuation walks the original coded fields (the
        // 3-bit -1 as unsigned 7, then the 7-bit 17).
        assert_eq!(reader.read_bits(3), 0b111);
        assert_eq!(reader.read_bits(7), 17);
        assert!(!reader.overrun);
    }

    /// The ilog helper's spec examples (section 9.2.1).
    #[test]
    fn ilog_matches_the_spec_examples() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
    }
}
