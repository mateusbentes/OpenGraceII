//! The engine's own Ogg container reader: the page-level demuxer the
//! Vorbis decoder consumes. The Ogg page format follows the RFC 3533
//! structure directly — the capture pattern, the header's fields, the
//! segment table and the lacing values — with no third-party binding.
//!
//! The reader's scope is the single-logical-stream file the game assets
//! carry: one Vorbis logical stream multiplexed alone. Chained
//! bitstreams (multiple BOS pages) and multiplexed streams follow with
//! the streaming milestone; the reader reports them as errors instead
//! of silently mis-decoding them.

use super::AudioError;

/// One Ogg page's parsed form: the header's fields plus the packet
/// boundaries the segment table encodes.
pub struct OggPage {
    /// The page's header type flags (0x01 continued, 0x02 BOS, 0x04
    /// EOS).
    pub header_type: u8,
    /// The absolute granule position (-1 when none of the page's
    /// packets finish).
    pub granule_position: u64,
    /// The logical stream's serial number.
    pub serial: u32,
    /// The page sequence number.
    pub sequence: u32,
    /// The page's packet boundaries: the segment table's lacing values
    /// (255 marks a packet continuing onto the next segment).
    pub segments: Vec<u8>,
    /// The page body's bytes.
    pub body: Vec<u8>,
}

impl OggPage {
    /// The total packet byte count the segment table carries.
    pub fn body_len(&self) -> usize {
        self.segments.iter().map(|&s| s as usize).sum()
    }
}

/// Parses one page from the reader's cursor. The `consumed` return
/// names the byte count the page occupied (the caller advances by it).
pub fn parse_page(data: &[u8], offset: usize) -> Result<(OggPage, usize), AudioError> {
    if offset + 27 > data.len() {
        return Err(AudioError::from(
            "ogg: the stream ends inside a page header",
        ));
    }
    if &data[offset..offset + 4] != b"OggS" {
        return Err(AudioError::from(
            "ogg: the capture pattern is missing (a corrupted or misaligned page)",
        ));
    }
    let header_type = data[offset + 5];
    let granule = u64::from_le_bytes([
        data[offset + 6],
        data[offset + 7],
        data[offset + 8],
        data[offset + 9],
        data[offset + 10],
        data[offset + 11],
        data[offset + 12],
        data[offset + 13],
    ]);
    let serial = u32::from_le_bytes([
        data[offset + 14],
        data[offset + 15],
        data[offset + 16],
        data[offset + 17],
    ]);
    let sequence = u32::from_le_bytes([
        data[offset + 18],
        data[offset + 19],
        data[offset + 20],
        data[offset + 21],
    ]);
    let segment_count = data[offset + 26] as usize;
    if offset + 27 + segment_count > data.len() {
        return Err(AudioError::from(
            "ogg: the segment table runs past the buffer",
        ));
    }
    let segments = data[offset + 27..offset + 27 + segment_count].to_vec();
    let body_len: usize = segments.iter().map(|&s| s as usize).sum();
    let body_start = offset + 27 + segment_count;
    if body_start + body_len > data.len() {
        return Err(AudioError::from(
            "ogg: the page body runs past the buffer",
        ));
    }
    let body = data[body_start..body_start + body_len].to_vec();
    let consumed = body_start + body_len - offset;
    Ok((
        OggPage {
            header_type,
            granule_position: granule,
            serial,
            sequence,
            segments,
            body,
        },
        consumed,
    ))
}

/// The packet assembly over the page stream: the segment table's 255
/// laces continue a packet across segments and pages; a lace below 255
/// ends it. The assembler carries the partial packet between pages.
pub struct PacketAssembler {
    partial: Vec<u8>,
    /// The finished packets ready for the caller.
    pub packets: Vec<Vec<u8>>,
    /// The stream's serial the first BOS page carries (the reader
    /// rejects later pages from other serials).
    pub serial: Option<u32>,
    /// The stream's last page's EOS flag.
    pub end_of_stream: bool,
}

impl PacketAssembler {
    pub fn new() -> Self {
        PacketAssembler {
            partial: Vec::new(),
            packets: Vec::new(),
            serial: None,
            end_of_stream: false,
        }
    }

    /// Feeds one page's segments through the assembler. The page's
    /// segments walk the body in order; a continued packet from the
    /// previous page resumes from the partial buffer.
    pub fn feed_page(&mut self, page: &OggPage) -> Result<(), AudioError> {
        if page.header_type & 0x02 != 0 {
            // The BOS page establishes (or confirms) the stream serial.
            match self.serial {
                None => self.serial = Some(page.serial),
                Some(existing) => {
                    if existing != page.serial {
                        return Err(AudioError::from(
                            "ogg: a second logical stream joins the file (multiplexed \
                             streams are outside the reader's scope)",
                        ));
                    }
                }
            }
        }
        if let Some(existing) = self.serial {
            if page.serial != existing {
                return Err(AudioError::from(
                    "ogg: a page from a foreign serial joins the stream",
                ));
            }
        }
        let mut cursor = 0usize;
        for &lace in &page.segments {
            let segment_end = cursor + lace as usize;
            if segment_end > page.body.len() {
                return Err(AudioError::from(
                    "ogg: the segment table exceeds the page's body",
                ));
            }
            self.partial.extend_from_slice(&page.body[cursor..segment_end]);
            cursor = segment_end;
            if lace < 255 {
                // The packet finished.
                let mut finished = std::mem::take(&mut self.partial);
                // A zero-length packet (the lace 0 that finishes an
                // empty partial) is a legal packet per the RFC 3533's
                // lacing rule; it carries no payload but still counts
                // in the packet order.
                self.packets.push(std::mem::take(&mut finished));
            }
        }
        if page.header_type & 0x04 != 0 && !self.partial.is_empty() {
            return Err(AudioError::from(
                "ogg: the EOS page ends inside a packet",
            ));
        }
        if page.header_type & 0x04 != 0 {
            self.end_of_stream = true;
        }
        Ok(())
    }

    /// Whether the assembler holds a partial packet (a truncated
    /// stream's symptom at the file's end).
    pub fn has_partial(&self) -> bool {
        !self.partial.is_empty()
    }
}

/// Demuxes a whole Ogg file into its Vorbis packet sequence. The
/// caller consumes the packets in order (the three headers, then the
/// audio packets).
pub fn demux_vorbis(data: &[u8]) -> Result<Vec<Vec<u8>>, AudioError> {
    let mut assembler = PacketAssembler::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let (page, consumed) = parse_page(data, offset)?;
        assembler.feed_page(&page)?;
        offset += consumed;
    }
    if assembler.has_partial() {
        return Err(AudioError::from(
            "ogg: the stream ends inside a packet",
        ));
    }
    if !assembler.end_of_stream {
        return Err(AudioError::from(
            "ogg: the stream never carried its EOS page",
        ));
    }
    Ok(std::mem::take(&mut assembler.packets))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one page's bytes from its fields (the test helper).
    fn build_page(
        header_type: u8,
        granule: u64,
        serial: u32,
        sequence: u32,
        segments: &[u8],
        body: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"OggS");
        out.push(0); // version
        out.push(header_type);
        out.extend_from_slice(&granule.to_le_bytes());
        out.extend_from_slice(&serial.to_le_bytes());
        out.extend_from_slice(&sequence.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]); // crc placeholder
        out.push(segments.len() as u8);
        out.extend_from_slice(segments);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn page_parses_fields() {
        let bytes = build_page(0x02, 7, 0x1234_5678, 3, &[4, 3], &[1, 2, 3, 4, 5, 6, 7]);
        let (page, consumed) = parse_page(&bytes, 0).expect("the parse");
        assert_eq!(consumed, bytes.len());
        assert_eq!(page.header_type, 0x02);
        assert_eq!(page.granule_position, 7);
        assert_eq!(page.serial, 0x1234_5678);
        assert_eq!(page.sequence, 3);
        assert_eq!(page.segments, vec![4, 3]);
        assert_eq!(page.body, vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(page.body_len(), 7);
    }

    #[test]
    fn assembler_joins_cross_page_packets() {
        // Packet A spans pages (the 255 lace continues), packet B ends
        // inside the second page, and the zero-length third segment
        // legally closes an empty packet per the RFC 3533 lacing rule
        // (a lace below 255 finishes the packet, even at length zero).
        let mut assembler = PacketAssembler::new();
        let body_a: Vec<u8> = (0..255u8).collect();
        let page1 = build_page(0x02, 0, 1, 0, &[255, 100], &[&body_a[..], &[9u8; 100]].concat());
        let body_b: Vec<u8> = (100..155u8).collect();
        let page2 = build_page(0x00, 5, 1, 1, &[55, 0], &[&body_b[..], &[]].concat());
        let (p1, _) = parse_page(&page1, 0).unwrap();
        let (p2, _) = parse_page(&page2, 0).unwrap();
        assembler.feed_page(&p1).unwrap();
        assembler.feed_page(&p2).unwrap();
        assert_eq!(assembler.packets.len(), 3);
        assert_eq!(assembler.packets[0].len(), 255 + 100);
        assert_eq!(assembler.packets[1].len(), 55);
        assert_eq!(assembler.packets[2].len(), 0);
        assert!(assembler.has_partial() == false);
    }

    #[test]
    fn assembler_rejects_a_foreign_serial() {
        let page1 = build_page(0x02, 0, 1, 0, &[1], &[5]);
        let page2 = build_page(0x00, 0, 2, 1, &[1], &[6]);
        let mut assembler = PacketAssembler::new();
        let (p1, _) = parse_page(&page1, 0).unwrap();
        assembler.feed_page(&p1).unwrap();
        let (p2, _) = parse_page(&page2, 0).unwrap();
        assert!(assembler.feed_page(&p2).is_err());
    }

    #[test]
    fn demux_rejects_a_truncated_stream() {
        // One BOS page, no EOS: the demux reports the missing EOS.
        let bytes = build_page(0x02, 0, 1, 0, &[1], &[5]);
        assert!(demux_vorbis(&bytes).is_err());
    }
}
