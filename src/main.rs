use group_similar::{group_similar, Config, Named, Threshold};
use serde::{Serialize, Serializer};
use std::collections::HashMap;
use std::io::{self, Read};
use structopt::StructOpt;

#[derive(Eq, PartialEq, std::hash::Hash, Debug)]
struct Wrapper(String);

impl Serialize for Wrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

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
}

impl Wrapper {
    fn new(i: String) -> Self {
        Wrapper(i)
    }
}

impl std::fmt::Display for Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Named for Wrapper {
    fn name(&self) -> &str {
        &self.0
    }
}

fn read_from_stdin() -> io::Result<String> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

fn main() -> io::Result<()> {
    let flags = Flags::from_args();
    let input = read_from_stdin()?
        .lines()
        .map(|v| Wrapper::new(v.into()))
        .collect::<Vec<Wrapper>>();

    let config: Config<Wrapper> = Config::jaro_winkler(flags.threshold.clone());

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

    Ok(())
}
