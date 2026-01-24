//! Shell completion generation command.

use clap::{Args, CommandFactory};
use clap_complete::{generate, Shell};
use std::io;

/// Generate shell completions.
#[derive(Args, Debug)]
pub struct CmdCompletions {
    /// Shell to generate completions for.
    #[arg(value_enum)]
    pub shell: Shell,
}

impl CmdCompletions {
    /// Execute the completions command.
    pub fn run<C: CommandFactory>(&self) -> anyhow::Result<()> {
        let mut cmd = C::command();
        generate(self.shell, &mut cmd, "transcode", &mut io::stdout());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    #[command(name = "test")]
    struct TestCli {
        #[command(subcommand)]
        command: Option<TestCommands>,
    }

    #[derive(clap::Subcommand)]
    enum TestCommands {
        Completions(CmdCompletions),
    }

    #[test]
    fn test_shell_variants() {
        // Just verify that Shell enum is usable
        let _bash = Shell::Bash;
        let _zsh = Shell::Zsh;
        let _fish = Shell::Fish;
    }
}
