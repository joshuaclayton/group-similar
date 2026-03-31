#[cfg(all(unix, not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use colored::*;
use group_similar::{group_similar, normalize, Config, Threshold};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read};
use std::str::FromStr;
use structopt::StructOpt;

#[derive(Debug, Clone, Copy)]
pub enum Metric {
    Jaro,
    Cosine,
    CosinePos,
}

impl FromStr for Metric {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "jaro" => Ok(Metric::Jaro),
            "cosine" => Ok(Metric::Cosine),
            "cosine-pos" => Ok(Metric::CosinePos),
            other => Err(format!(
                "metric must be one of: jaro, cosine, cosine-pos; got '{}'",
                other
            )),
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "group-similar",
    about = "Hierarchical clustering by similarity, with batteries-included string metrics",
    setting = structopt::clap::AppSettings::ColoredHelp
)]
pub struct Flags {
    /// Threshold (0-1) controlling how strict matching is. The right value
    /// depends on the chosen metric: cosine-pos ~0.3-0.4 (default), cosine
    /// ~0.5, jaro ~0.1.
    #[structopt(long, default_value = "0.3")]
    pub threshold: Threshold,

    /// Include singletons (records that didn't cluster with anything else) in
    /// the output. By default only clusters of 2+ are shown.
    #[structopt(long)]
    pub all: bool,

    /// Render results in JSON format
    #[structopt(long)]
    pub json: bool,

    /// Disable normalization. By default, embedded IDs, hex addresses, bare
    /// hex tokens, and quoted timestamps are collapsed to placeholders before
    /// deduplication.
    #[structopt(long = "no-normalize")]
    pub no_normalize: bool,

    /// Similarity metric: jaro (Jaro-Winkler), cosine (IDF-weighted token
    /// cosine), or cosine-pos (cosine with leading-token position boost,
    /// default).
    #[structopt(long, default_value = "cosine-pos")]
    pub metric: Metric,

    /// Disable q-gram blocking. By default, blocking is on (candidate
    /// filtering + per-component complete-link clustering, with the same or
    /// near-identical output partition).
    #[structopt(long = "no-blocked")]
    pub no_blocked: bool,

    /// Tuning parameter for the q-gram blocking filter. Higher values reject
    /// more candidate pairs (faster but risks dropping real matches). Ignored
    /// when --no-blocked is set. Typical range: 0.1-0.4.
    #[structopt(long, default_value = "0.3")]
    pub tau: f64,
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

    let mut config: Config<&str> = match flags.metric {
        Metric::Jaro => Config::jaro_winkler(flags.threshold.clone()),
        Metric::Cosine => Config::token_cosine(&input, flags.threshold.clone()),
        Metric::CosinePos => Config::token_cosine_positional(&input, flags.threshold.clone()),
    };
    if !flags.no_normalize {
        config = config.with_normalizer(normalize::default_normalizer());
    }
    config = if flags.no_blocked {
        config.without_blocking()
    } else {
        config.with_blocking(flags.tau)
    };

    let results: BTreeMap<&&str, Vec<&&str>> = group_similar(&input, &config);

    if flags.json {
        println!(
            "{}",
            serde_json::to_string(
                &results
                    .iter()
                    .filter(|(_, v)| flags.all || !v.is_empty())
                    .collect::<HashMap<_, _>>()
            )
            .unwrap()
        );
    } else {
        for (k, vs) in results.iter().filter(|(_, v)| flags.all || !v.is_empty()) {
            println!("{}", k.green().bold());

            for v in vs {
                println!("   {}", v.dimmed().italic());
            }

            println!();
        }
    }

    Ok(())
}
