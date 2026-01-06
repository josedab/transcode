//! CEA-608 (Line 21) closed caption decoder and encoder.
//!
//! CEA-608 is the analog closed caption standard used in North American
//! television broadcasts. It encodes caption data in the vertical blanking
//! interval (VBI) line 21 of the NTSC video signal.
//!
//! # Format Overview
//!
//! - Two-byte control codes and character pairs
//! - Supports roll-up, pop-on, and paint-on caption modes
//! - Two caption channels (CC1/CC2 and CC3/CC4)
//! - Basic styling (italics, underline, colors)
//! - 32 columns x 15 rows display area

use crate::types::{
    Color, StyledText, SubtitleEvent, SubtitleResult, SubtitleTrack, TextStyle, Timestamp,
};

/// CEA-608 caption channel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionChannel {
    /// Primary caption channel (CC1)
    CC1,
    /// Secondary caption channel (CC2)
    CC2,
    /// Third caption channel (CC3)
    CC3,
    /// Fourth caption channel (CC4)
    CC4,
    /// Text channel 1
    T1,
    /// Text channel 2
    T2,
}

impl Default for CaptionChannel {
    fn default() -> Self {
        CaptionChannel::CC1
    }
}

/// CEA-608 caption mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionMode {
    /// Roll-up captions (1-4 rows)
    RollUp(u8),
    /// Pop-on captions (displayed all at once)
    PopOn,
    /// Paint-on captions (displayed character by character)
    PaintOn,
}

impl Default for CaptionMode {
    fn default() -> Self {
        CaptionMode::PopOn
    }
}

/// CEA-608 basic colors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cea608Color {
    White,
    Green,
    Blue,
    Cyan,
    Red,
    Yellow,
    Magenta,
    Black,
}

impl Cea608Color {
    /// Convert to standard Color
    pub fn to_color(self) -> Color {
        match self {
            Cea608Color::White => Color::WHITE,
            Cea608Color::Green => Color::GREEN,
            Cea608Color::Blue => Color::BLUE,
            Cea608Color::Cyan => Color::CYAN,
            Cea608Color::Red => Color::RED,
            Cea608Color::Yellow => Color::YELLOW,
            Cea608Color::Magenta => Color::MAGENTA,
            Cea608Color::Black => Color::BLACK,
        }
    }

    /// Parse from control code
    pub fn from_code(code: u8) -> Option<Self> {
        match code & 0x07 {
            0 => Some(Cea608Color::White),
            1 => Some(Cea608Color::Green),
            2 => Some(Cea608Color::Blue),
            3 => Some(Cea608Color::Cyan),
            4 => Some(Cea608Color::Red),
            5 => Some(Cea608Color::Yellow),
            6 => Some(Cea608Color::Magenta),
            _ => None,
        }
    }
}

/// CEA-608 character set
fn decode_character(byte: u8) -> Option<char> {
    // Standard characters (0x20-0x7F with some modifications)
    // Special characters override some ASCII positions
    match byte {
        // Special character overrides (CEA-608 specific)
        0x2A => Some('á'),
        0x5C => Some('é'),
        0x5E => Some('í'),
        0x5F => Some('ó'),
        0x60 => Some('ú'),
        0x7B => Some('ç'),
        0x7C => Some('÷'),
        0x7D => Some('Ñ'),
        0x7E => Some('ñ'),
        0x7F => Some('█'), // Block character
        // Standard ASCII range
        0x20..=0x7E => Some(byte as char),
        _ => None,
    }
}

/// Extended character set (special/extended characters)
fn decode_extended_char(byte1: u8, byte2: u8) -> Option<char> {
    // Extended characters are encoded as two-byte sequences
    match (byte1, byte2) {
        // Special characters
        (0x11, 0x30) => Some('®'),
        (0x11, 0x31) => Some('°'),
        (0x11, 0x32) => Some('½'),
        (0x11, 0x33) => Some('¿'),
        (0x11, 0x34) => Some('™'),
        (0x11, 0x35) => Some('¢'),
        (0x11, 0x36) => Some('£'),
        (0x11, 0x37) => Some('♪'),
        (0x11, 0x38) => Some('à'),
        (0x11, 0x39) => Some(' '), // Transparent space
        (0x11, 0x3A) => Some('è'),
        (0x11, 0x3B) => Some('â'),
        (0x11, 0x3C) => Some('ê'),
        (0x11, 0x3D) => Some('î'),
        (0x11, 0x3E) => Some('ô'),
        (0x11, 0x3F) => Some('û'),
        // Spanish/Portuguese
        (0x12, 0x20) => Some('Á'),
        (0x12, 0x21) => Some('É'),
        (0x12, 0x22) => Some('Ó'),
        (0x12, 0x23) => Some('Ú'),
        (0x12, 0x24) => Some('Ü'),
        (0x12, 0x25) => Some('ü'),
        (0x12, 0x26) => Some('\''),
        (0x12, 0x27) => Some('¡'),
        (0x12, 0x28) => Some('*'),
        (0x12, 0x29) => Some('\''),
        (0x12, 0x2A) => Some('—'),
        (0x12, 0x2B) => Some('©'),
        (0x12, 0x2C) => Some('℠'),
        (0x12, 0x2D) => Some('•'),
        (0x12, 0x2E) => Some('"'),
        (0x12, 0x2F) => Some('"'),
        // French
        (0x13, 0x20) => Some('À'),
        (0x13, 0x21) => Some('Â'),
        (0x13, 0x22) => Some('Ç'),
        (0x13, 0x23) => Some('È'),
        (0x13, 0x24) => Some('Ê'),
        (0x13, 0x25) => Some('Ë'),
        (0x13, 0x26) => Some('ë'),
        (0x13, 0x27) => Some('Î'),
        (0x13, 0x28) => Some('Ï'),
        (0x13, 0x29) => Some('ï'),
        (0x13, 0x2A) => Some('Ô'),
        (0x13, 0x2B) => Some('Ù'),
        (0x13, 0x2C) => Some('ù'),
        (0x13, 0x2D) => Some('Û'),
        (0x13, 0x2E) => Some('«'),
        (0x13, 0x2F) => Some('»'),
        _ => None,
    }
}

/// CEA-608 control codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlCode {
    /// Resume caption loading
    RCL,
    /// Backspace
    BS,
    /// Alarm off (not displayed)
    AOF,
    /// Alarm on (not displayed)
    AON,
    /// Delete to end of row
    DER,
    /// Roll-up captions (2 rows)
    RU2,
    /// Roll-up captions (3 rows)
    RU3,
    /// Roll-up captions (4 rows)
    RU4,
    /// Flash on
    FON,
    /// Resume direct captioning
    RDC,
    /// Text restart
    TR,
    /// Resume text display
    RTD,
    /// Erase displayed memory
    EDM,
    /// Carriage return
    CR,
    /// Erase non-displayed memory
    ENM,
    /// End of caption (flip memories)
    EOC,
    /// Tab offset (1-3 columns)
    TO(u8),
}

impl ControlCode {
    /// Parse from byte pair
    pub fn from_bytes(b1: u8, b2: u8) -> Option<Self> {
        // Miscellaneous control codes
        if b1 == 0x14 || b1 == 0x1C {
            match b2 {
                0x20 => Some(ControlCode::RCL),
                0x21 => Some(ControlCode::BS),
                0x22 => Some(ControlCode::AOF),
                0x23 => Some(ControlCode::AON),
                0x24 => Some(ControlCode::DER),
                0x25 => Some(ControlCode::RU2),
                0x26 => Some(ControlCode::RU3),
                0x27 => Some(ControlCode::RU4),
                0x28 => Some(ControlCode::FON),
                0x29 => Some(ControlCode::RDC),
                0x2A => Some(ControlCode::TR),
                0x2B => Some(ControlCode::RTD),
                0x2C => Some(ControlCode::EDM),
                0x2D => Some(ControlCode::CR),
                0x2E => Some(ControlCode::ENM),
                0x2F => Some(ControlCode::EOC),
                _ => None,
            }
        } else if (b1 == 0x17 || b1 == 0x1F) && (0x21..=0x23).contains(&b2) {
            Some(ControlCode::TO(b2 - 0x20))
        } else {
            None
        }
    }

    /// Encode to byte pair
    pub fn to_bytes(self, field: u8) -> (u8, u8) {
        let base = if field == 1 { 0x14 } else { 0x1C };
        match self {
            ControlCode::RCL => (base, 0x20),
            ControlCode::BS => (base, 0x21),
            ControlCode::AOF => (base, 0x22),
            ControlCode::AON => (base, 0x23),
            ControlCode::DER => (base, 0x24),
            ControlCode::RU2 => (base, 0x25),
            ControlCode::RU3 => (base, 0x26),
            ControlCode::RU4 => (base, 0x27),
            ControlCode::FON => (base, 0x28),
            ControlCode::RDC => (base, 0x29),
            ControlCode::TR => (base, 0x2A),
            ControlCode::RTD => (base, 0x2B),
            ControlCode::EDM => (base, 0x2C),
            ControlCode::CR => (base, 0x2D),
            ControlCode::ENM => (base, 0x2E),
            ControlCode::EOC => (base, 0x2F),
            ControlCode::TO(n) => (if field == 1 { 0x17 } else { 0x1F }, 0x20 + n),
        }
    }
}

/// CEA-608 caption buffer (32x15 character grid)
#[derive(Debug, Clone)]
pub struct CaptionBuffer {
    /// Character grid [row][column]
    chars: [[char; 32]; 15],
    /// Style for each cell
    styles: [[TextStyle; 32]; 15],
    /// Current cursor row
    pub cursor_row: usize,
    /// Current cursor column
    pub cursor_col: usize,
}

impl Default for CaptionBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptionBuffer {
    /// Create new empty buffer
    pub fn new() -> Self {
        CaptionBuffer {
            chars: [[' '; 32]; 15],
            styles: std::array::from_fn(|_| std::array::from_fn(|_| TextStyle::default())),
            cursor_row: 14,
            cursor_col: 0,
        }
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        for row in &mut self.chars {
            row.fill(' ');
        }
        for row in &mut self.styles {
            for style in row.iter_mut() {
                *style = TextStyle::default();
            }
        }
    }

    /// Write character at cursor position
    pub fn write_char(&mut self, c: char, style: &TextStyle) {
        if self.cursor_row < 15 && self.cursor_col < 32 {
            self.chars[self.cursor_row][self.cursor_col] = c;
            self.styles[self.cursor_row][self.cursor_col] = style.clone();
            self.cursor_col = (self.cursor_col + 1).min(31);
        }
    }

    /// Move cursor to position
    pub fn set_cursor(&mut self, row: usize, col: usize) {
        self.cursor_row = row.min(14);
        self.cursor_col = col.min(31);
    }

    /// Backspace
    pub fn backspace(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
            self.chars[self.cursor_row][self.cursor_col] = ' ';
        }
    }

    /// Carriage return (roll up)
    pub fn carriage_return(&mut self, rows: u8) {
        let rows = rows.min(4) as usize;
        // Scroll up by one row
        for r in 0..(15 - 1) {
            if r >= 15 - rows {
                self.chars[r] = self.chars[r + 1];
                self.styles[r] = self.styles[r + 1].clone();
            }
        }
        // Clear bottom row
        self.chars[14].fill(' ');
        for style in &mut self.styles[14] {
            *style = TextStyle::default();
        }
        self.cursor_col = 0;
    }

    /// Convert buffer to subtitle text
    pub fn to_styled_text(&self) -> Vec<StyledText> {
        let mut result = Vec::new();
        let mut current_text = String::new();
        let mut current_style = TextStyle::default();
        let mut has_content = false;

        for row in 0..15 {
            // Find start and end of content in row
            let row_str: String = self.chars[row].iter().collect();
            let trimmed = row_str.trim();

            if trimmed.is_empty() {
                if has_content {
                    // Add newline between content rows
                    if !current_text.is_empty() {
                        result.push(StyledText::new(current_text.clone(), current_style.clone()));
                        current_text.clear();
                    }
                    result.push(StyledText::plain("\n"));
                }
                continue;
            }

            has_content = true;

            // Find first non-space column
            let start = self.chars[row].iter().position(|&c| c != ' ').unwrap_or(0);
            let end = self.chars[row]
                .iter()
                .rposition(|&c| c != ' ')
                .map(|p| p + 1)
                .unwrap_or(0);

            for col in start..end {
                let c = self.chars[row][col];
                let style = &self.styles[row][col];

                if *style != current_style && !current_text.is_empty() {
                    result.push(StyledText::new(current_text.clone(), current_style.clone()));
                    current_text.clear();
                }

                current_style = style.clone();
                current_text.push(c);
            }

            if row < 14 {
                if !current_text.is_empty() {
                    result.push(StyledText::new(current_text.clone(), current_style.clone()));
                    current_text.clear();
                }
                result.push(StyledText::plain("\n"));
            }
        }

        if !current_text.is_empty() {
            result.push(StyledText::new(current_text, current_style));
        }

        // Clean up empty/whitespace-only entries
        result.retain(|st| !st.text.trim().is_empty() || st.text.contains('\n'));

        result
    }
}

/// CEA-608 decoder state
#[derive(Debug)]
pub struct Cea608Decoder {
    /// Current channel
    channel: CaptionChannel,
    /// Caption mode
    mode: CaptionMode,
    /// Displayed buffer
    displayed: CaptionBuffer,
    /// Non-displayed buffer (for pop-on)
    non_displayed: CaptionBuffer,
    /// Current text style
    current_style: TextStyle,
    /// Last control code (for duplicate detection)
    last_control: Option<(u8, u8)>,
    /// Current timestamp
    current_pts: u64,
    /// Accumulated events
    events: Vec<SubtitleEvent>,
    /// Roll-up base row
    rollup_base_row: usize,
}

impl Default for Cea608Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea608Decoder {
    /// Create new decoder
    pub fn new() -> Self {
        Cea608Decoder {
            channel: CaptionChannel::CC1,
            mode: CaptionMode::PopOn,
            displayed: CaptionBuffer::new(),
            non_displayed: CaptionBuffer::new(),
            current_style: TextStyle::default(),
            last_control: None,
            current_pts: 0,
            events: Vec::new(),
            rollup_base_row: 14,
        }
    }

    /// Set target channel
    pub fn set_channel(&mut self, channel: CaptionChannel) {
        self.channel = channel;
    }

    /// Process a byte pair
    pub fn process_pair(&mut self, pts: u64, b1: u8, b2: u8) {
        self.current_pts = pts;

        // Apply parity (strip high bits in some cases)
        let b1 = b1 & 0x7F;
        let b2 = b2 & 0x7F;

        // Skip null pairs
        if b1 == 0 && b2 == 0 {
            return;
        }

        // Check for control code
        if (0x10..=0x1F).contains(&b1) {
            self.process_control(b1, b2);
        } else if b1 >= 0x20 {
            // Text characters
            self.process_text(b1, b2);
        }
    }

    /// Process control code
    fn process_control(&mut self, b1: u8, b2: u8) {
        // Duplicate control code detection
        if let Some((last_b1, last_b2)) = self.last_control {
            if last_b1 == b1 && last_b2 == b2 {
                self.last_control = None;
                return; // Ignore duplicate
            }
        }
        self.last_control = Some((b1, b2));

        // Parse control code
        if let Some(code) = ControlCode::from_bytes(b1, b2) {
            self.handle_control_code(code);
            return;
        }

        // Preamble address code (PAC) - position and style
        if (0x10..=0x17).contains(&b1) && (0x40..=0x7F).contains(&b2) {
            self.handle_preamble(b1, b2);
            return;
        }

        // Mid-row code - style changes
        if (b1 == 0x11 || b1 == 0x19) && (0x20..=0x2F).contains(&b2) {
            self.handle_mid_row(b2);
            return;
        }

        // Extended characters
        if let Some(c) = decode_extended_char(b1, b2) {
            self.write_char(c);
        }
    }

    /// Handle control code
    fn handle_control_code(&mut self, code: ControlCode) {
        match code {
            ControlCode::RCL => {
                self.mode = CaptionMode::PopOn;
            }
            ControlCode::BS => {
                self.active_buffer_mut().backspace();
            }
            ControlCode::DER => {
                // Delete to end of row
                let buf = self.active_buffer_mut();
                for col in buf.cursor_col..32 {
                    buf.chars[buf.cursor_row][col] = ' ';
                }
            }
            ControlCode::RU2 => {
                self.mode = CaptionMode::RollUp(2);
                self.rollup_base_row = 14;
            }
            ControlCode::RU3 => {
                self.mode = CaptionMode::RollUp(3);
                self.rollup_base_row = 14;
            }
            ControlCode::RU4 => {
                self.mode = CaptionMode::RollUp(4);
                self.rollup_base_row = 14;
            }
            ControlCode::RDC => {
                self.mode = CaptionMode::PaintOn;
            }
            ControlCode::EDM => {
                // Emit current displayed content before erasing
                self.emit_displayed();
                self.displayed.clear();
            }
            ControlCode::CR => {
                if let CaptionMode::RollUp(rows) = self.mode {
                    self.emit_displayed();
                    self.displayed.carriage_return(rows);
                }
            }
            ControlCode::ENM => {
                self.non_displayed.clear();
            }
            ControlCode::EOC => {
                // Emit current displayed content
                self.emit_displayed();
                // Swap buffers
                std::mem::swap(&mut self.displayed, &mut self.non_displayed);
                self.non_displayed.clear();
            }
            ControlCode::TO(n) => {
                let buf = self.active_buffer_mut();
                buf.cursor_col = (buf.cursor_col + n as usize).min(31);
            }
            _ => {}
        }
    }

    /// Handle preamble address code
    fn handle_preamble(&mut self, b1: u8, b2: u8) {
        // Decode row
        let row = match b1 & 0x07 {
            0 => {
                if b2 & 0x20 != 0 {
                    1
                } else {
                    0
                }
            }
            1 => {
                if b2 & 0x20 != 0 {
                    3
                } else {
                    2
                }
            }
            2 => {
                if b2 & 0x20 != 0 {
                    5
                } else {
                    4
                }
            }
            3 => {
                if b2 & 0x20 != 0 {
                    7
                } else {
                    6
                }
            }
            4 => {
                if b2 & 0x20 != 0 {
                    9
                } else {
                    8
                }
            }
            5 => {
                if b2 & 0x20 != 0 {
                    11
                } else {
                    10
                }
            }
            6 => {
                if b2 & 0x20 != 0 {
                    13
                } else {
                    12
                }
            }
            7 => 14,
            _ => 14,
        };

        // Decode column/indent
        let col = if b2 & 0x10 != 0 {
            ((b2 & 0x0E) >> 1) as usize * 4
        } else {
            0
        };

        // Set cursor position
        let buf = self.active_buffer_mut();
        buf.set_cursor(row, col);

        // Decode style
        let underline = b2 & 0x01 != 0;
        let style_bits = (b2 & 0x0E) >> 1;

        self.current_style = TextStyle::default();
        self.current_style.underline = underline;

        if b2 & 0x10 == 0 {
            // Color/style code
            match style_bits {
                0 => self.current_style.color = Some(Color::WHITE),
                1 => self.current_style.color = Some(Color::GREEN),
                2 => self.current_style.color = Some(Color::BLUE),
                3 => self.current_style.color = Some(Color::CYAN),
                4 => self.current_style.color = Some(Color::RED),
                5 => self.current_style.color = Some(Color::YELLOW),
                6 => self.current_style.color = Some(Color::MAGENTA),
                7 => self.current_style.italic = true,
                _ => {}
            }
        }
    }

    /// Handle mid-row style code
    fn handle_mid_row(&mut self, b2: u8) {
        let underline = b2 & 0x01 != 0;
        let style_bits = (b2 & 0x0E) >> 1;

        self.current_style.underline = underline;

        match style_bits {
            0 => self.current_style.color = Some(Color::WHITE),
            1 => self.current_style.color = Some(Color::GREEN),
            2 => self.current_style.color = Some(Color::BLUE),
            3 => self.current_style.color = Some(Color::CYAN),
            4 => self.current_style.color = Some(Color::RED),
            5 => self.current_style.color = Some(Color::YELLOW),
            6 => self.current_style.color = Some(Color::MAGENTA),
            7 => self.current_style.italic = true,
            _ => {}
        }

        // Write a space to mark style change
        self.write_char(' ');
    }

    /// Process text bytes
    fn process_text(&mut self, b1: u8, b2: u8) {
        self.last_control = None;

        if let Some(c) = decode_character(b1) {
            self.write_char(c);
        }

        if b2 >= 0x20 {
            if let Some(c) = decode_character(b2) {
                self.write_char(c);
            }
        }
    }

    /// Write character to active buffer
    fn write_char(&mut self, c: char) {
        let style = self.current_style.clone();
        self.active_buffer_mut().write_char(c, &style);
    }

    /// Get active buffer based on mode
    fn active_buffer_mut(&mut self) -> &mut CaptionBuffer {
        match self.mode {
            CaptionMode::PopOn => &mut self.non_displayed,
            _ => &mut self.displayed,
        }
    }

    /// Emit displayed content as event
    fn emit_displayed(&mut self) {
        let text = self.displayed.to_styled_text();
        if text.is_empty() || text.iter().all(|s| s.text.trim().is_empty()) {
            return;
        }

        let start = Timestamp::from_millis(self.current_pts);
        // Default duration of 0 - will be updated when next caption arrives
        let end = start;

        let event = SubtitleEvent::with_styled_text(start, end, text);
        self.events.push(event);
    }

    /// Finalize and get subtitle track
    pub fn finish(mut self) -> SubtitleTrack {
        // Emit any remaining content
        self.emit_displayed();

        // Fix up end times
        for i in 0..self.events.len() {
            if i + 1 < self.events.len() {
                self.events[i].end = self.events[i + 1].start;
            } else {
                // Default 3 second duration for last caption
                self.events[i].end =
                    Timestamp::from_millis(self.events[i].start.milliseconds + 3000);
            }
        }

        SubtitleTrack::with_events(self.events)
    }
}

/// CEA-608 encoder
#[derive(Debug)]
pub struct Cea608Encoder {
    /// Target channel
    channel: CaptionChannel,
    /// Output pairs
    output: Vec<(u64, u8, u8)>,
}

impl Default for Cea608Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea608Encoder {
    /// Create new encoder
    pub fn new() -> Self {
        Cea608Encoder {
            channel: CaptionChannel::CC1,
            output: Vec::new(),
        }
    }

    /// Set target channel
    pub fn set_channel(&mut self, channel: CaptionChannel) {
        self.channel = channel;
    }

    /// Encode subtitle track
    pub fn encode(&mut self, track: &SubtitleTrack) -> Vec<(u64, u8, u8)> {
        self.output.clear();

        for event in &track.events {
            self.encode_event(event);
        }

        self.output.clone()
    }

    /// Encode single event
    fn encode_event(&mut self, event: &SubtitleEvent) {
        let pts = event.start.milliseconds;
        let field = 1u8;

        // Resume caption loading
        let (b1, b2) = ControlCode::RCL.to_bytes(field);
        self.output.push((pts, b1, b2));
        self.output.push((pts, b1, b2)); // Duplicate for reliability

        // Erase non-displayed memory
        let (b1, b2) = ControlCode::ENM.to_bytes(field);
        self.output.push((pts, b1, b2));
        self.output.push((pts, b1, b2));

        // Position cursor at bottom center (row 14, indent 0)
        self.output.push((pts, 0x14, 0x70));
        self.output.push((pts, 0x14, 0x70));

        // Encode text
        let text = event.plain_text();
        self.encode_text(pts, &text);

        // End of caption
        let (b1, b2) = ControlCode::EOC.to_bytes(field);
        self.output.push((pts, b1, b2));
        self.output.push((pts, b1, b2));

        // Erase at end time
        let end_pts = event.end.milliseconds;
        let (b1, b2) = ControlCode::EDM.to_bytes(field);
        self.output.push((end_pts, b1, b2));
        self.output.push((end_pts, b1, b2));
    }

    /// Encode text string
    fn encode_text(&mut self, pts: u64, text: &str) {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c1 = chars[i];
            let b1 = self.encode_char(c1).unwrap_or(0x20);

            let b2 = if i + 1 < chars.len() {
                let c2 = chars[i + 1];
                if c2 == '\n' {
                    i += 1;
                    // Carriage return for newline
                    self.output.push((pts, b1, 0x00));
                    self.output.push((pts, 0x14, 0x2D)); // CR
                    self.output.push((pts, 0x14, 0x2D));
                    i += 1;
                    continue;
                }
                self.encode_char(c2).unwrap_or(0x00)
            } else {
                0x00
            };

            self.output.push((pts, b1, b2));
            i += if b2 != 0 { 2 } else { 1 };
        }
    }

    /// Encode single character
    fn encode_char(&self, c: char) -> Option<u8> {
        let byte = c as u32;
        if (0x20..=0x7E).contains(&byte) {
            Some(byte as u8)
        } else {
            None
        }
    }
}

/// Parse CEA-608 caption data
pub fn parse(data: &[(u64, u8, u8)]) -> SubtitleResult<SubtitleTrack> {
    let mut decoder = Cea608Decoder::new();

    for &(pts, b1, b2) in data {
        decoder.process_pair(pts, b1, b2);
    }

    Ok(decoder.finish())
}

/// Write CEA-608 caption data
pub fn write(track: &SubtitleTrack) -> Vec<(u64, u8, u8)> {
    let mut encoder = Cea608Encoder::new();
    encoder.encode(track)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_codes() {
        assert_eq!(ControlCode::from_bytes(0x14, 0x20), Some(ControlCode::RCL));
        assert_eq!(ControlCode::from_bytes(0x14, 0x2F), Some(ControlCode::EOC));
        assert_eq!(ControlCode::from_bytes(0x17, 0x21), Some(ControlCode::TO(1)));

        let (b1, b2) = ControlCode::EOC.to_bytes(1);
        assert_eq!(b1, 0x14);
        assert_eq!(b2, 0x2F);
    }

    #[test]
    fn test_character_decode() {
        assert_eq!(decode_character(0x41), Some('A'));
        assert_eq!(decode_character(0x20), Some(' '));
        assert_eq!(decode_character(0x7D), Some('Ñ'));
    }

    #[test]
    fn test_caption_buffer() {
        let mut buf = CaptionBuffer::new();
        buf.set_cursor(14, 0);
        buf.write_char('H', &TextStyle::default());
        buf.write_char('i', &TextStyle::default());

        let text = buf.to_styled_text();
        let combined: String = text.iter().map(|s| s.text.as_str()).collect();
        assert!(combined.contains("Hi"));
    }

    #[test]
    fn test_decoder_basic() {
        let mut decoder = Cea608Decoder::new();

        // RCL
        decoder.process_pair(0, 0x14, 0x20);
        // ENM
        decoder.process_pair(0, 0x14, 0x2E);
        // PAC - row 14
        decoder.process_pair(0, 0x14, 0x70);
        // Text "Hi"
        decoder.process_pair(0, 0x48, 0x69); // 'H', 'i'
        // EOC
        decoder.process_pair(0, 0x14, 0x2F);

        let track = decoder.finish();
        assert!(!track.events.is_empty());
    }

    #[test]
    fn test_encoder_roundtrip() {
        let mut track = SubtitleTrack::new();
        track.add_event(SubtitleEvent::new(
            Timestamp::from_millis(0),
            Timestamp::from_millis(2000),
            "Hello",
        ));

        let encoded = write(&track);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_cea608_color() {
        assert_eq!(Cea608Color::White.to_color(), Color::WHITE);
        assert_eq!(Cea608Color::Red.to_color(), Color::RED);
        assert_eq!(Cea608Color::from_code(0), Some(Cea608Color::White));
    }
}
