#[cfg(all(unix, not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use colored::*;
use group_similar::{group_similar, Config, Threshold};
use std::collections::HashMap;
use std::io::{self, Read};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "group-similar",
    about = "Group similar names",
    setting = structopt::clap::AppSettings::ColoredHelp
)]
pub struct Flags {
    /// Configure a threshold, a value from 0 to 1, to dictate how strict (or permissive) the
    /// matching algorithm is
    #[structopt(default_value, long)]
    pub threshold: Threshold,

    /// Include all records rather than just results that have similar values
    #[structopt(long)]
    pub all: bool,

    /// Render results in JSON format
    #[structopt(long)]
    pub json: bool,
}

fn read_from_stdin() -> io::Result<String> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

fn main() -> io::Result<()> {
    let flags = Flags::from_args();
    let stdin = read_from_stdin()?;
    let input = stdin.lines().collect::<Vec<_>>();

    let config: Config<&str> = Config::jaro_winkler(flags.threshold.clone());

    if flags.json {
        println!(
            "{}",
            serde_json::to_string(
                &group_similar(&input, &config)
                    .iter()
                    .filter(|(_, v)| flags.all || !v.is_empty())
                    .collect::<HashMap<_, _>>()
            )
            .unwrap()
        );
    } else {
        for (k, vs) in group_similar(&input, &config)
            .iter()
            .filter(|(_, v)| flags.all || !v.is_empty())
        {
            println!("{}", k.green().bold());

            for v in vs {
                println!("   {}", v.dimmed().italic());
            }

            println!("");
        }
    }

    Ok(())
}
