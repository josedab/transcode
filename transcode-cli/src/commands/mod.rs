//! CLI subcommand implementations.

pub mod codecs;
pub mod completions;
pub mod doctor;
pub mod info;
pub mod presets;

pub use codecs::CmdCodecs;
pub use completions::CmdCompletions;
pub use doctor::CmdDoctor;
pub use info::CmdInfo;
pub use presets::CmdPresets;
