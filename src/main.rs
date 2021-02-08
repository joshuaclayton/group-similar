use group_similar::{group_similar, Config, Named, Threshold};
use std::io::{self, Read};
use structopt::StructOpt;

#[derive(Eq, PartialEq, std::hash::Hash, Debug)]
struct Wrapper(String);

#[derive(Debug, StructOpt)]
#[structopt(
    name = "group-similar",
    about = "Group similar names",
    setting = structopt::clap::AppSettings::ColoredHelp
)]
pub struct Flags {
    /// Threshold
    #[structopt(default_value, long)]
    pub threshold: Threshold,
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

    println!("processing {} records", input.len());
    let config: Config<Wrapper> = Config::jaro_winkler(flags.threshold);

    for (primary, secondary) in group_similar(&input, &config) {
        if !secondary.is_empty() {
            println!("{}", primary);
            for n in secondary {
                println!("  {}", n);
            }
            println!("");
        }
    }

    Ok(())
}
