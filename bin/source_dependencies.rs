extern crate csv;
extern crate serde;

use csv::StringRecord;
use serde::Deserialize;

use std::error::Error;
use std::fs::File;
use std::io::Write;

// The order of the fields must be the same of the input file.
#[derive(Debug, Deserialize)]
struct Dependency {
    id: u32,
    platform: String,
    project_name: String,
    project_id: u32,
    version_number: String,
    version_id: u32,
    dependency_name: String,
    dependency_platform: String,
    dependency_kind: String,
    optional_dependency: bool,
    dependency_requirements: String,
    dependency_project_id: String,
}

fn source_dependencies(path: &str, platform: &str) -> Result<(), Box<dyn Error>> {
    let dependencies_file = File::open(path)?;

    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(dependencies_file);

    for result in rdr
        .records()
        .filter_map(|e| e.ok())
        .filter(by_platform(platform))
        .take(10)
    {
        let record: Dependency = result.deserialize(None)?;
        println!("{:?}", record);
    }

    Ok(())
}

fn by_platform<'a>(platform: &'a str) -> Box<FnMut(&StringRecord) -> bool + 'a> {
    Box::new(move |e| e[1] == *platform)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        writeln!(
            std::io::stderr(),
            "Usage: {} <PATH-TO-CSV-FILE> <PLATFORM>",
            &args[0]
        )
        .unwrap();
        writeln!(
            std::io::stderr(),
            r#"Example: {} ~/Downloads/Libraries.io-open-data-1.4.0.tar.gz 
            ~/Downloads/libraries-1.4.0-2018-12-22/dependencies-1.4.0-2018-12-22.csv
            Cargo"#,
            &args[0]
        )
        .unwrap();
    }

    // Read the path to the file from the args
    let (path, platform) = (&args[1], &args[2]);

    source_dependencies(path, platform)
}
