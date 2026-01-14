//! CEA-708 (DTVCC) closed caption decoder and encoder.
//!
//! CEA-708 is the digital closed caption standard for ATSC digital television
//! in North America. It provides enhanced capabilities over CEA-608 including:
//!
//! - Up to 63 caption services
//! - Unicode character support
//! - Advanced positioning and styling
//! - Multiple windows per service
//!
//! # Format Overview
//!
//! CEA-708 captions are carried in DTVCC (Digital Television Closed Caption)
//! packets within the video elementary stream's user data.

use crate::types::{
    Alignment, Color, Position, StyledText, SubtitleEvent, SubtitleResult, SubtitleTrack,
    TextStyle, Timestamp,
};
use std::collections::HashMap;

/// Maximum number of windows per service
pub const MAX_WINDOWS: usize = 8;

/// Maximum characters per row
pub const MAX_COLUMNS: usize = 42;

/// Maximum rows per window
pub const MAX_ROWS: usize = 15;

/// CEA-708 service number (1-63)
pub type ServiceNumber = u8;

/// CEA-708 window ID (0-7)
pub type WindowId = u8;

/// Caption window anchor point
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnchorPoint {
    #[default]
    TopLeft,
    TopCenter,
    TopRight,
    MiddleLeft,
    MiddleCenter,
    MiddleRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl AnchorPoint {
    /// Parse from 4-bit value
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => AnchorPoint::TopLeft,
            1 => AnchorPoint::TopCenter,
            2 => AnchorPoint::TopRight,
            3 => AnchorPoint::MiddleLeft,
            4 => AnchorPoint::MiddleCenter,
            5 => AnchorPoint::MiddleRight,
            6 => AnchorPoint::BottomLeft,
            7 => AnchorPoint::BottomCenter,
            8 => AnchorPoint::BottomRight,
            _ => AnchorPoint::BottomLeft,
        }
    }

    /// Convert to alignment
    pub fn to_alignment(self) -> Alignment {
        match self {
            AnchorPoint::TopLeft => Alignment::TopLeft,
            AnchorPoint::TopCenter => Alignment::TopCenter,
            AnchorPoint::TopRight => Alignment::TopRight,
            AnchorPoint::MiddleLeft => Alignment::MiddleLeft,
            AnchorPoint::MiddleCenter => Alignment::MiddleCenter,
            AnchorPoint::MiddleRight => Alignment::MiddleRight,
            AnchorPoint::BottomLeft => Alignment::BottomLeft,
            AnchorPoint::BottomCenter => Alignment::BottomCenter,
            AnchorPoint::BottomRight => Alignment::BottomRight,
        }
    }
}

/// Window justification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Justification {
    #[default]
    Left,
    Right,
    Center,
    Full,
}

impl Justification {
    /// Parse from 2-bit value
    pub fn from_u8(value: u8) -> Self {
        match value & 0x03 {
            0 => Justification::Left,
            1 => Justification::Right,
            2 => Justification::Center,
            3 => Justification::Full,
            _ => Justification::Left,
        }
    }
}

/// Print direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrintDirection {
    #[default]
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

impl PrintDirection {
    /// Parse from 2-bit value
    pub fn from_u8(value: u8) -> Self {
        match value & 0x03 {
            0 => PrintDirection::LeftToRight,
            1 => PrintDirection::RightToLeft,
            2 => PrintDirection::TopToBottom,
            3 => PrintDirection::BottomToTop,
            _ => PrintDirection::LeftToRight,
        }
    }
}

/// Scroll direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollDirection {
    #[default]
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

impl ScrollDirection {
    /// Parse from 2-bit value
    pub fn from_u8(value: u8) -> Self {
        match value & 0x03 {
            0 => ScrollDirection::LeftToRight,
            1 => ScrollDirection::RightToLeft,
            2 => ScrollDirection::TopToBottom,
            3 => ScrollDirection::BottomToTop,
            _ => ScrollDirection::LeftToRight,
        }
    }
}

/// Window display effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayEffect {
    #[default]
    Snap,
    Fade,
    Wipe,
}

impl DisplayEffect {
    /// Parse from 2-bit value
    pub fn from_u8(value: u8) -> Self {
        match value & 0x03 {
            0 => DisplayEffect::Snap,
            1 => DisplayEffect::Fade,
            2 => DisplayEffect::Wipe,
            _ => DisplayEffect::Snap,
        }
    }
}

/// Border type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderType {
    #[default]
    None,
    Raised,
    Depressed,
    Uniform,
    ShadowLeft,
    ShadowRight,
}

impl BorderType {
    /// Parse from 3-bit value
    pub fn from_u8(value: u8) -> Self {
        match value & 0x07 {
            0 => BorderType::None,
            1 => BorderType::Raised,
            2 => BorderType::Depressed,
            3 => BorderType::Uniform,
            4 => BorderType::ShadowLeft,
            5 => BorderType::ShadowRight,
            _ => BorderType::None,
        }
    }
}

/// Pen style attributes
#[derive(Debug, Clone, Default)]
pub struct PenAttributes {
    /// Pen size (0=small, 1=standard, 2=large)
    pub pen_size: u8,
    /// Font style (0-7)
    pub font_style: u8,
    /// Text tag (0-15)
    pub text_tag: u8,
    /// Offset (0=subscript, 1=normal, 2=superscript)
    pub offset: u8,
    /// Italics
    pub italics: bool,
    /// Underline
    pub underline: bool,
    /// Edge type (0-7)
    pub edge_type: u8,
}

/// Pen color attributes
#[derive(Debug, Clone)]
pub struct PenColor {
    /// Foreground color
    pub foreground: Color,
    /// Foreground opacity (0=solid, 1=flash, 2=translucent, 3=transparent)
    pub fg_opacity: u8,
    /// Background color
    pub background: Color,
    /// Background opacity
    pub bg_opacity: u8,
    /// Edge color
    pub edge_color: Color,
}

impl Default for PenColor {
    fn default() -> Self {
        PenColor {
            foreground: Color::WHITE,
            fg_opacity: 0,
            background: Color::BLACK,
            bg_opacity: 0,
            edge_color: Color::BLACK,
        }
    }
}

/// Window attributes
#[derive(Debug, Clone, Default)]
pub struct WindowAttributes {
    /// Fill color
    pub fill_color: Color,
    /// Fill opacity
    pub fill_opacity: u8,
    /// Border color
    pub border_color: Color,
    /// Border type
    pub border_type: BorderType,
    /// Word wrap
    pub word_wrap: bool,
    /// Print direction
    pub print_direction: PrintDirection,
    /// Scroll direction
    pub scroll_direction: ScrollDirection,
    /// Justification
    pub justify: Justification,
    /// Display effect
    pub effect: DisplayEffect,
    /// Effect direction
    pub effect_direction: u8,
    /// Effect speed
    pub effect_speed: u8,
}

/// Caption window
#[derive(Debug, Clone)]
pub struct Window {
    /// Window ID
    pub id: WindowId,
    /// Visible
    pub visible: bool,
    /// Priority (0-7)
    pub priority: u8,
    /// Anchor point
    pub anchor_point: AnchorPoint,
    /// Anchor vertical position (0-74, percentage)
    pub anchor_v: u8,
    /// Anchor horizontal position (0-209, percentage)
    pub anchor_h: u8,
    /// Row count (1-15)
    pub row_count: u8,
    /// Column count (1-42)
    pub col_count: u8,
    /// Row lock
    pub row_lock: bool,
    /// Column lock
    pub col_lock: bool,
    /// Pen style (0-7)
    pub pen_style: u8,
    /// Window style (0-7)
    pub window_style: u8,
    /// Window attributes
    pub attributes: WindowAttributes,
    /// Current pen attributes
    pub pen_attributes: PenAttributes,
    /// Current pen color
    pub pen_color: PenColor,
    /// Text buffer [row][column]
    pub text: Vec<Vec<char>>,
    /// Style buffer
    pub styles: Vec<Vec<TextStyle>>,
    /// Cursor row
    pub cursor_row: usize,
    /// Cursor column
    pub cursor_col: usize,
}

impl Window {
    /// Create new window
    pub fn new(id: WindowId) -> Self {
        let row_count = MAX_ROWS;
        let col_count = MAX_COLUMNS;

        Window {
            id,
            visible: false,
            priority: 0,
            anchor_point: AnchorPoint::BottomLeft,
            anchor_v: 74,
            anchor_h: 0,
            row_count: row_count as u8,
            col_count: col_count as u8,
            row_lock: false,
            col_lock: false,
            pen_style: 0,
            window_style: 0,
            attributes: WindowAttributes::default(),
            pen_attributes: PenAttributes::default(),
            pen_color: PenColor::default(),
            text: vec![vec![' '; col_count]; row_count],
            styles: vec![vec![TextStyle::default(); col_count]; row_count],
            cursor_row: 0,
            cursor_col: 0,
        }
    }

    /// Clear window contents
    pub fn clear(&mut self) {
        for row in &mut self.text {
            row.fill(' ');
        }
        for row in &mut self.styles {
            for style in row {
                *style = TextStyle::default();
            }
        }
        self.cursor_row = 0;
        self.cursor_col = 0;
    }

    /// Write character at cursor
    pub fn write_char(&mut self, c: char) {
        if self.cursor_row < self.text.len() && self.cursor_col < self.text[0].len() {
            self.text[self.cursor_row][self.cursor_col] = c;
            self.styles[self.cursor_row][self.cursor_col] = self.current_style();
            self.cursor_col += 1;
            if self.cursor_col >= self.col_count as usize {
                self.cursor_col = 0;
                self.cursor_row += 1;
            }
        }
    }

    /// Get current text style
    pub fn current_style(&self) -> TextStyle {
        TextStyle {
            bold: false,
            italic: self.pen_attributes.italics,
            underline: self.pen_attributes.underline,
            strikethrough: false,
            color: Some(self.pen_color.foreground),
            background_color: if self.pen_color.bg_opacity < 3 {
                Some(self.pen_color.background)
            } else {
                None
            },
            font_name: None,
            font_size: None,
        }
    }

    /// Set cursor position
    pub fn set_cursor(&mut self, row: usize, col: usize) {
        self.cursor_row = row.min(self.row_count as usize - 1);
        self.cursor_col = col.min(self.col_count as usize - 1);
    }

    /// Carriage return
    pub fn carriage_return(&mut self) {
        self.cursor_col = 0;
        self.cursor_row += 1;
        if self.cursor_row >= self.row_count as usize {
            // Scroll up
            self.text.remove(0);
            self.text.push(vec![' '; self.col_count as usize]);
            self.styles.remove(0);
            self.styles.push(vec![TextStyle::default(); self.col_count as usize]);
            self.cursor_row = self.row_count as usize - 1;
        }
    }

    /// Horizontal carriage return
    pub fn horizontal_carriage_return(&mut self) {
        self.cursor_col = 0;
    }

    /// Form feed (clear window)
    pub fn form_feed(&mut self) {
        self.clear();
    }

    /// Backspace
    pub fn backspace(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        }
    }

    /// Convert to styled text
    pub fn to_styled_text(&self) -> Vec<StyledText> {
        let mut result = Vec::new();
        let mut current_text = String::new();
        let mut current_style = TextStyle::default();
        let mut has_content = false;

        for row in 0..self.row_count as usize {
            let row_str: String = self.text[row].iter().collect();
            let trimmed = row_str.trim();

            if trimmed.is_empty() {
                if has_content && !current_text.is_empty() {
                    result.push(StyledText::new(current_text.clone(), current_style.clone()));
                    current_text.clear();
                    result.push(StyledText::plain("\n"));
                }
                continue;
            }

            has_content = true;

            let start = self.text[row].iter().position(|&c| c != ' ').unwrap_or(0);
            let end = self.text[row]
                .iter()
                .rposition(|&c| c != ' ')
                .map(|p| p + 1)
                .unwrap_or(0);

            for col in start..end {
                let c = self.text[row][col];
                let style = &self.styles[row][col];

                if *style != current_style && !current_text.is_empty() {
                    result.push(StyledText::new(current_text.clone(), current_style.clone()));
                    current_text.clear();
                }

                current_style = style.clone();
                current_text.push(c);
            }

            if row + 1 < self.row_count as usize {
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

        result.retain(|st| !st.text.trim().is_empty() || st.text.contains('\n'));
        result
    }
}

/// CEA-708 command codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Null (no operation)
    NUL,
    /// End of text
    ETX,
    /// Backspace
    BS,
    /// Form feed
    FF,
    /// Carriage return
    CR,
    /// Horizontal carriage return
    HCR,
    /// Clear windows
    CLW(u8),
    /// Display windows
    DSW(u8),
    /// Hide windows
    HDW(u8),
    /// Toggle windows
    TGW(u8),
    /// Delete windows
    DLW(u8),
    /// Delay
    DLY(u8),
    /// Delay cancel
    DLC,
    /// Reset
    RST,
    /// Set pen attributes
    SPA,
    /// Set pen color
    SPC,
    /// Set pen location
    SPL,
    /// Set window attributes
    SWA,
    /// Define window 0-7
    DF(WindowId),
}

impl Command {
    /// Parse from byte
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Command::NUL),
            0x03 => Some(Command::ETX),
            0x08 => Some(Command::BS),
            0x0C => Some(Command::FF),
            0x0D => Some(Command::CR),
            0x0E => Some(Command::HCR),
            0x88 => Some(Command::CLW(byte & 0x07)),
            0x89 => Some(Command::DSW(byte & 0x07)),
            0x8A => Some(Command::HDW(byte & 0x07)),
            0x8B => Some(Command::TGW(byte & 0x07)),
            0x8C => Some(Command::DLW(byte & 0x07)),
            0x8D => Some(Command::DLY(0)),
            0x8E => Some(Command::DLC),
            0x8F => Some(Command::RST),
            0x90 => Some(Command::SPA),
            0x91 => Some(Command::SPC),
            0x92 => Some(Command::SPL),
            0x97 => Some(Command::SWA),
            0x98..=0x9F => Some(Command::DF(byte - 0x98)),
            _ => None,
        }
    }

    /// Get number of parameter bytes
    pub fn param_count(&self) -> usize {
        match self {
            Command::CLW(_)
            | Command::DSW(_)
            | Command::HDW(_)
            | Command::TGW(_)
            | Command::DLW(_) => 1,
            Command::DLY(_) => 1,
            Command::SPA => 2,
            Command::SPC => 3,
            Command::SPL => 2,
            Command::SWA => 4,
            Command::DF(_) => 6,
            _ => 0,
        }
    }
}

/// CEA-708 decoder state for a service
#[derive(Debug)]
pub struct ServiceDecoder {
    /// Service number
    #[allow(dead_code)]
    service_number: ServiceNumber,
    /// Windows
    windows: [Option<Window>; MAX_WINDOWS],
    /// Current window
    current_window: WindowId,
    /// Current timestamp
    current_pts: u64,
    /// Accumulated events
    events: Vec<SubtitleEvent>,
}

impl ServiceDecoder {
    /// Create new service decoder
    pub fn new(service_number: ServiceNumber) -> Self {
        ServiceDecoder {
            service_number,
            windows: Default::default(),
            current_window: 0,
            current_pts: 0,
            events: Vec::new(),
        }
    }

    /// Process service block data
    pub fn process(&mut self, pts: u64, data: &[u8]) {
        self.current_pts = pts;
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            i += 1;

            // C0 control codes (0x00-0x1F)
            if byte <= 0x1F {
                self.process_c0(byte);
                continue;
            }

            // G0 characters (0x20-0x7F)
            if (0x20..=0x7F).contains(&byte) {
                self.write_char(byte as char);
                continue;
            }

            // C1 control codes (0x80-0x9F)
            if (0x80..=0x9F).contains(&byte) {
                let params_needed = self.c1_param_count(byte);
                if i + params_needed <= data.len() {
                    let params = &data[i..i + params_needed];
                    self.process_c1(byte, params);
                    i += params_needed;
                }
                continue;
            }

            // G1 characters (0xA0-0xFF) - Latin-1 supplement
            if byte >= 0xA0 {
                let c = char::from_u32(byte as u32).unwrap_or(' ');
                self.write_char(c);
            }
        }
    }

    /// Get parameter count for C1 command
    fn c1_param_count(&self, byte: u8) -> usize {
        match byte {
            0x88..=0x8C => 1, // CLW, DSW, HDW, TGW, DLW
            0x8D => 1,        // DLY
            0x8E | 0x8F => 0, // DLC, RST
            0x90 => 2,        // SPA
            0x91 => 3,        // SPC
            0x92 => 2,        // SPL
            0x97 => 4,        // SWA
            0x98..=0x9F => 6, // DF0-DF7
            _ => 0,
        }
    }

    /// Process C0 control code
    fn process_c0(&mut self, code: u8) {
        match code {
            0x00 => {} // NUL
            0x03 => {
                // ETX - end of text, emit caption
                self.emit_current();
            }
            0x08 => {
                // BS
                if let Some(ref mut window) = self.windows[self.current_window as usize] {
                    window.backspace();
                }
            }
            0x0C => {
                // FF
                if let Some(ref mut window) = self.windows[self.current_window as usize] {
                    window.form_feed();
                }
            }
            0x0D => {
                // CR
                if let Some(ref mut window) = self.windows[self.current_window as usize] {
                    window.carriage_return();
                }
            }
            0x0E => {
                // HCR
                if let Some(ref mut window) = self.windows[self.current_window as usize] {
                    window.horizontal_carriage_return();
                }
            }
            _ => {}
        }
    }

    /// Process C1 control code
    fn process_c1(&mut self, code: u8, params: &[u8]) {
        match code {
            0x88 => {
                // CLW - clear windows
                if !params.is_empty() {
                    let bitmap = params[0];
                    // First emit all windows that need clearing
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 && self.windows[i].is_some() {
                            self.emit_window(i);
                        }
                    }
                    // Then clear them
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 {
                            if let Some(ref mut window) = self.windows[i] {
                                window.clear();
                            }
                        }
                    }
                }
            }
            0x89 => {
                // DSW - display windows
                if !params.is_empty() {
                    let bitmap = params[0];
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 {
                            if let Some(ref mut window) = self.windows[i] {
                                window.visible = true;
                            }
                        }
                    }
                }
            }
            0x8A => {
                // HDW - hide windows
                if !params.is_empty() {
                    let bitmap = params[0];
                    // First emit all windows that will be hidden
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 && self.windows[i].is_some() {
                            self.emit_window(i);
                        }
                    }
                    // Then hide them
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 {
                            if let Some(ref mut window) = self.windows[i] {
                                window.visible = false;
                            }
                        }
                    }
                }
            }
            0x8B => {
                // TGW - toggle windows
                if !params.is_empty() {
                    let bitmap = params[0];
                    // First collect which windows will become hidden
                    let mut to_emit = Vec::new();
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 {
                            if let Some(ref window) = self.windows[i] {
                                if window.visible {
                                    // Will become hidden
                                    to_emit.push(i);
                                }
                            }
                        }
                    }
                    // Emit windows that will become hidden
                    for i in to_emit {
                        self.emit_window(i);
                    }
                    // Then toggle visibility
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 {
                            if let Some(ref mut window) = self.windows[i] {
                                window.visible = !window.visible;
                            }
                        }
                    }
                }
            }
            0x8C => {
                // DLW - delete windows
                if !params.is_empty() {
                    let bitmap = params[0];
                    for i in 0..8 {
                        if bitmap & (1 << i) != 0 {
                            self.emit_window(i);
                            self.windows[i] = None;
                        }
                    }
                }
            }
            0x8F => {
                // RST - reset
                for i in 0..MAX_WINDOWS {
                    self.emit_window(i);
                    self.windows[i] = None;
                }
            }
            0x90 => {
                // SPA - set pen attributes
                if params.len() >= 2 {
                    if let Some(ref mut window) = self.windows[self.current_window as usize] {
                        window.pen_attributes.pen_size = params[0] & 0x03;
                        window.pen_attributes.offset = (params[0] >> 2) & 0x03;
                        window.pen_attributes.text_tag = (params[0] >> 4) & 0x0F;
                        window.pen_attributes.font_style = params[1] & 0x07;
                        window.pen_attributes.italics = params[1] & 0x08 != 0;
                        window.pen_attributes.underline = params[1] & 0x10 != 0;
                        window.pen_attributes.edge_type = (params[1] >> 5) & 0x07;
                    }
                }
            }
            0x91 => {
                // SPC - set pen color
                if params.len() >= 3 {
                    let fg_color = Self::parse_color_static(params[0]);
                    let fg_opacity = (params[0] >> 6) & 0x03;
                    let bg_color = Self::parse_color_static(params[1]);
                    let bg_opacity = (params[1] >> 6) & 0x03;
                    let edge_color = Self::parse_color_static(params[2]);
                    if let Some(ref mut window) = self.windows[self.current_window as usize] {
                        window.pen_color.foreground = fg_color;
                        window.pen_color.fg_opacity = fg_opacity;
                        window.pen_color.background = bg_color;
                        window.pen_color.bg_opacity = bg_opacity;
                        window.pen_color.edge_color = edge_color;
                    }
                }
            }
            0x92 => {
                // SPL - set pen location
                if params.len() >= 2 {
                    if let Some(ref mut window) = self.windows[self.current_window as usize] {
                        let row = (params[0] & 0x0F) as usize;
                        let col = (params[1] & 0x3F) as usize;
                        window.set_cursor(row, col);
                    }
                }
            }
            0x97 => {
                // SWA - set window attributes
                if params.len() >= 4 {
                    let fill_color = Self::parse_color_static(params[0]);
                    let fill_opacity = (params[0] >> 6) & 0x03;
                    let border_type = BorderType::from_u8(params[1] & 0x07);
                    let border_color = Self::parse_color_static(params[1] >> 2);
                    let print_direction = PrintDirection::from_u8(params[2] & 0x03);
                    let scroll_direction = ScrollDirection::from_u8((params[2] >> 2) & 0x03);
                    let justify = Justification::from_u8((params[2] >> 4) & 0x03);
                    let word_wrap = params[2] & 0x40 != 0;
                    let effect = DisplayEffect::from_u8(params[3] & 0x03);
                    let effect_direction = (params[3] >> 2) & 0x03;
                    let effect_speed = (params[3] >> 4) & 0x0F;
                    if let Some(ref mut window) = self.windows[self.current_window as usize] {
                        window.attributes.fill_color = fill_color;
                        window.attributes.fill_opacity = fill_opacity;
                        window.attributes.border_type = border_type;
                        window.attributes.border_color = border_color;
                        window.attributes.print_direction = print_direction;
                        window.attributes.scroll_direction = scroll_direction;
                        window.attributes.justify = justify;
                        window.attributes.word_wrap = word_wrap;
                        window.attributes.effect = effect;
                        window.attributes.effect_direction = effect_direction;
                        window.attributes.effect_speed = effect_speed;
                    }
                }
            }
            0x98..=0x9F => {
                // DF0-DF7 - define window
                let window_id = code - 0x98;
                if params.len() >= 6 {
                    let mut window = Window::new(window_id);
                    window.visible = params[0] & 0x20 != 0;
                    window.row_lock = params[0] & 0x10 != 0;
                    window.col_lock = params[0] & 0x08 != 0;
                    window.priority = params[0] & 0x07;
                    window.anchor_point = AnchorPoint::from_u8((params[1] >> 4) & 0x0F);
                    window.anchor_v = params[1] & 0x7F;
                    window.anchor_h = params[2];
                    window.row_count = (params[3] & 0x0F) + 1;
                    window.col_count = (params[4] & 0x3F) + 1;
                    window.pen_style = params[3] >> 4;
                    window.window_style = params[5] & 0x07;

                    // Resize text buffers
                    let rows = window.row_count as usize;
                    let cols = window.col_count as usize;
                    window.text = vec![vec![' '; cols]; rows];
                    window.styles = vec![vec![TextStyle::default(); cols]; rows];

                    self.windows[window_id as usize] = Some(window);
                    self.current_window = window_id;
                }
            }
            _ => {}
        }
    }

    /// Parse 6-bit color to Color (static version)
    fn parse_color_static(byte: u8) -> Color {
        let r = (byte & 0x03) * 85;
        let g = ((byte >> 2) & 0x03) * 85;
        let b = ((byte >> 4) & 0x03) * 85;
        Color::rgb(r, g, b)
    }

    /// Write character to current window
    fn write_char(&mut self, c: char) {
        if let Some(ref mut window) = self.windows[self.current_window as usize] {
            window.write_char(c);
        }
    }

    /// Emit current visible content
    fn emit_current(&mut self) {
        for i in 0..MAX_WINDOWS {
            if let Some(ref window) = self.windows[i] {
                if window.visible {
                    self.emit_window(i);
                }
            }
        }
    }

    /// Emit window content as event
    fn emit_window(&mut self, window_id: usize) {
        if let Some(ref window) = self.windows[window_id] {
            let text = window.to_styled_text();
            if text.is_empty() || text.iter().all(|s| s.text.trim().is_empty()) {
                return;
            }

            let start = Timestamp::from_millis(self.current_pts);
            let mut event = SubtitleEvent::with_styled_text(start, start, text);

            // Set position based on anchor
            let x = window.anchor_h as f32 / 209.0 * 100.0;
            let y = window.anchor_v as f32 / 74.0 * 100.0;
            event.position = Some(Position::new(x, y, window.anchor_point.to_alignment()));

            self.events.push(event);
        }
    }

    /// Finalize and get events
    pub fn finish(mut self) -> Vec<SubtitleEvent> {
        // Emit any remaining visible content
        self.emit_current();

        // Fix up end times
        for i in 0..self.events.len() {
            if i + 1 < self.events.len() {
                self.events[i].end = self.events[i + 1].start;
            } else {
                self.events[i].end =
                    Timestamp::from_millis(self.events[i].start.milliseconds + 3000);
            }
        }

        self.events
    }
}

/// CEA-708 decoder
#[derive(Debug)]
pub struct Cea708Decoder {
    /// Service decoders
    services: HashMap<ServiceNumber, ServiceDecoder>,
    /// Target service (None = all services)
    target_service: Option<ServiceNumber>,
}

impl Default for Cea708Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea708Decoder {
    /// Create new decoder
    pub fn new() -> Self {
        Cea708Decoder {
            services: HashMap::new(),
            target_service: None,
        }
    }

    /// Set target service
    pub fn set_service(&mut self, service: ServiceNumber) {
        self.target_service = Some(service);
    }

    /// Process DTVCC packet
    pub fn process_packet(&mut self, pts: u64, data: &[u8]) {
        let mut i = 0;

        while i < data.len() {
            // Service block header
            if i + 1 > data.len() {
                break;
            }

            let header = data[i];
            i += 1;

            let service_number = (header >> 5) & 0x07;
            let block_size = (header & 0x1F) as usize;

            if service_number == 0 {
                continue; // Null service
            }

            if service_number == 7 {
                // Extended service number
                if i >= data.len() {
                    break;
                }
                let _ext_service = data[i] & 0x3F;
                i += 1;
                // Skip extended service data for now
                continue;
            }

            if i + block_size > data.len() {
                break;
            }

            // Check if we want this service
            if let Some(target) = self.target_service {
                if service_number != target {
                    i += block_size;
                    continue;
                }
            }

            // Get or create service decoder
            let service = self
                .services
                .entry(service_number)
                .or_insert_with(|| ServiceDecoder::new(service_number));

            // Process service block
            service.process(pts, &data[i..i + block_size]);
            i += block_size;
        }
    }

    /// Finalize and get subtitle track
    pub fn finish(self) -> SubtitleTrack {
        let mut all_events = Vec::new();

        for (_, service) in self.services {
            all_events.extend(service.finish());
        }

        // Sort by start time
        all_events.sort_by_key(|e| e.start.milliseconds);

        SubtitleTrack::with_events(all_events)
    }
}

/// Parse CEA-708 caption data
pub fn parse(data: &[(u64, Vec<u8>)]) -> SubtitleResult<SubtitleTrack> {
    let mut decoder = Cea708Decoder::new();

    for (pts, packet) in data {
        decoder.process_packet(*pts, packet);
    }

    Ok(decoder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_point() {
        assert_eq!(AnchorPoint::from_u8(0), AnchorPoint::TopLeft);
        assert_eq!(AnchorPoint::from_u8(8), AnchorPoint::BottomRight);
        assert_eq!(
            AnchorPoint::BottomCenter.to_alignment(),
            Alignment::BottomCenter
        );
    }

    #[test]
    fn test_window_creation() {
        let window = Window::new(0);
        assert_eq!(window.id, 0);
        assert!(!window.visible);
        assert_eq!(window.cursor_row, 0);
        assert_eq!(window.cursor_col, 0);
    }

    #[test]
    fn test_window_write() {
        let mut window = Window::new(0);
        window.write_char('H');
        window.write_char('i');

        assert_eq!(window.text[0][0], 'H');
        assert_eq!(window.text[0][1], 'i');
        assert_eq!(window.cursor_col, 2);
    }

    #[test]
    fn test_service_decoder() {
        let mut decoder = ServiceDecoder::new(1);

        // Define window
        decoder.process_c1(0x98, &[0x20, 0x60, 0x00, 0x03, 0x1F, 0x00]);

        // Write some text
        decoder.write_char('T');
        decoder.write_char('e');
        decoder.write_char('s');
        decoder.write_char('t');

        // End of text
        decoder.process_c0(0x03);

        let events = decoder.finish();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_decoder() {
        let decoder = Cea708Decoder::new();
        let track = decoder.finish();
        assert!(track.events.is_empty());
    }

    #[test]
    fn test_command_parsing() {
        assert_eq!(Command::from_byte(0x00), Some(Command::NUL));
        assert_eq!(Command::from_byte(0x08), Some(Command::BS));
        assert_eq!(Command::from_byte(0x0D), Some(Command::CR));
        assert_eq!(Command::from_byte(0x98), Some(Command::DF(0)));
        assert_eq!(Command::from_byte(0x9F), Some(Command::DF(7)));
    }

    #[test]
    fn test_justification() {
        assert_eq!(Justification::from_u8(0), Justification::Left);
        assert_eq!(Justification::from_u8(2), Justification::Center);
    }

    #[test]
    fn test_pen_color() {
        let color = PenColor::default();
        assert_eq!(color.foreground, Color::WHITE);
        assert_eq!(color.background, Color::BLACK);
    }
}
