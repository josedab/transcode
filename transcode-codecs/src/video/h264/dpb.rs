//! Decoded Picture Buffer (DPB) for H.264 reference frame management.

use transcode_core::Frame;
use std::collections::VecDeque;

/// Entry in the decoded picture buffer.
#[derive(Debug, Clone)]
pub struct DpbEntry {
    /// The decoded frame.
    pub frame: Frame,
    /// Frame number.
    pub frame_num: u32,
    /// Picture order count.
    pub poc: i32,
    /// Is this a reference frame.
    pub is_reference: bool,
    /// Is this a long-term reference.
    pub is_long_term: bool,
    /// Long-term frame index.
    pub long_term_frame_idx: Option<u32>,
    /// Has this frame been output.
    pub output: bool,
}

/// Decoded Picture Buffer for managing reference frames.
pub struct DecodedPictureBuffer {
    /// Maximum number of frames in the buffer.
    max_size: usize,
    /// Buffer entries.
    entries: VecDeque<DpbEntry>,
    /// Output queue.
    output_queue: VecDeque<Frame>,
}

impl DecodedPictureBuffer {
    /// Create a new DPB with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: VecDeque::with_capacity(max_size),
            output_queue: VecDeque::new(),
        }
    }

    /// Add a decoded frame to the buffer.
    pub fn add(&mut self, frame: Frame, frame_num: u32, poc: i32, is_reference: bool) {
        // Remove oldest non-reference frame if buffer is full
        while self.entries.len() >= self.max_size {
            if let Some(pos) = self.entries.iter().position(|e| !e.is_reference && e.output) {
                self.entries.remove(pos);
            } else {
                break;
            }
        }

        let entry = DpbEntry {
            frame,
            frame_num,
            poc,
            is_reference,
            is_long_term: false,
            long_term_frame_idx: None,
            output: false,
        };

        self.entries.push_back(entry);
    }

    /// Get reference frames for a list (L0 or L1).
    pub fn get_reference_list(&self, list: u8) -> Vec<&DpbEntry> {
        let mut refs: Vec<_> = self.entries.iter()
            .filter(|e| e.is_reference)
            .collect();

        // Sort by POC for L0, reverse for L1
        if list == 0 {
            refs.sort_by_key(|e| e.poc);
        } else {
            refs.sort_by_key(|e| std::cmp::Reverse(e.poc));
        }

        refs
    }

    /// Get a reference frame by frame number.
    pub fn get_by_frame_num(&self, frame_num: u32) -> Option<&DpbEntry> {
        self.entries.iter().find(|e| e.frame_num == frame_num)
    }

    /// Get a reference frame by POC.
    pub fn get_by_poc(&self, poc: i32) -> Option<&DpbEntry> {
        self.entries.iter().find(|e| e.poc == poc)
    }

    /// Mark a frame as output.
    pub fn mark_output(&mut self, poc: i32) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.poc == poc) {
            entry.output = true;
        }
    }

    /// Bump frames for output in POC order.
    pub fn bump_output(&mut self) -> Vec<Frame> {
        let mut output = Vec::new();

        // Find the minimum POC that hasn't been output
        let mut min_poc = i32::MAX;
        for entry in &self.entries {
            if !entry.output && entry.poc < min_poc {
                min_poc = entry.poc;
            }
        }

        // Output all frames with POC <= min_poc
        for entry in &mut self.entries {
            if !entry.output && entry.poc <= min_poc {
                entry.output = true;
                output.push(entry.frame.clone());
            }
        }

        output
    }

    /// Mark a frame as unused for reference.
    pub fn unmark_reference(&mut self, frame_num: u32) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.frame_num == frame_num) {
            entry.is_reference = false;
        }
    }

    /// Mark all short-term references as unused (for IDR).
    pub fn clear_short_term(&mut self) {
        for entry in &mut self.entries {
            if !entry.is_long_term {
                entry.is_reference = false;
            }
        }
    }

    /// Clear the entire buffer.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.output_queue.clear();
    }

    /// Flush all pending frames.
    pub fn flush(&mut self) -> Vec<Frame> {
        let mut output = Vec::new();

        // Sort by POC and output
        let mut entries: Vec<_> = self.entries.drain(..).collect();
        entries.sort_by_key(|e| e.poc);

        for entry in entries {
            if !entry.output {
                output.push(entry.frame);
            }
        }

        output
    }

    /// Get the number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of reference frames.
    pub fn num_ref_frames(&self) -> usize {
        self.entries.iter().filter(|e| e.is_reference).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{PixelFormat, TimeBase};

    fn make_frame(poc: i32) -> Frame {
        let mut frame = Frame::new(16, 16, PixelFormat::Yuv420p, TimeBase::MPEG);
        frame.poc = poc;
        frame
    }

    #[test]
    fn test_dpb_add() {
        let mut dpb = DecodedPictureBuffer::new(4);
        dpb.add(make_frame(0), 0, 0, true);
        dpb.add(make_frame(2), 1, 2, false);

        assert_eq!(dpb.len(), 2);
        assert_eq!(dpb.num_ref_frames(), 1);
    }

    #[test]
    fn test_dpb_reference_list() {
        let mut dpb = DecodedPictureBuffer::new(4);
        dpb.add(make_frame(0), 0, 0, true);
        dpb.add(make_frame(2), 1, 2, true);
        dpb.add(make_frame(4), 2, 4, true);

        let l0 = dpb.get_reference_list(0);
        assert_eq!(l0.len(), 3);
        assert_eq!(l0[0].poc, 0);
        assert_eq!(l0[2].poc, 4);
    }
}
