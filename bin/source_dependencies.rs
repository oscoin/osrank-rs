extern crate csv;
extern crate serde;

use csv::StringRecord;
use serde::Deserialize;

use std::collections::HashSet;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;

type ProjectId = u32;

// The order of the fields must be the same of the input file.
#[derive(Debug, Deserialize)]
struct Dependency {
    id: u32,
    platform: String,
    project_name: String,
    project_id: ProjectId,
    version_number: String,
    version_id: u32,
    dependency_name: String,
    dependency_platform: String,
    dependency_kind: String,
    optional_dependency: bool,
    dependency_requirements: Option<String>,
    dependency_project_id: Option<ProjectId>,
}

type UniqueProjects = HashSet<ProjectId>;
type UniqueDependencies = HashSet<(ProjectId, ProjectId)>;

fn source_dependencies(path: &str, platform: &str) -> Result<(), Box<dyn Error>> {
    let dependencies_file = File::open(path)?;

    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(dependencies_file);
    let mut dependencies = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(format!("data/{}_dependencies.csv", platform.to_lowercase()).as_str())?;

    let mut dependencies_meta = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(format!("data/{}_dependencies_meta.csv", platform.to_lowercase()).as_str())?;

    let mut unique_projects: UniqueProjects = HashSet::new();
    let mut unique_dependencies: UniqueDependencies = HashSet::new();

    //Write the header(s)
    dependencies.write_all(b"FROM_ID,TO_ID\n")?;
    dependencies_meta.write_all(b"ID,NAME,PLATFORM\n")?;

    for result in rdr
        .records()
        .filter_map(|e| e.ok())
        .filter(by_platform(platform))
    {
        let dependency: Dependency = result.deserialize(None)?;
        extract_dependency(&mut dependencies, &dependency, &mut unique_dependencies)?;
        extract_metadata(
            &mut unique_projects,
            platform,
            &mut dependencies_meta,
            &dependency,
        )?;
    }

    Ok(())
}

fn extract_dependency(
    dependencies: &mut File,
    dependency: &Dependency,
    unique_dependencies: &mut UniqueDependencies,
) -> Result<(), Box<dyn Error>> {
    match dependency.dependency_project_id {
        None => (),
        Some(pid) => {
            if unique_dependencies
                .get(&(dependency.project_id, pid))
                .is_none()
            {
                unique_dependencies.insert((dependency.project_id, pid));
                dependencies
                    .write_all(format!("{},{}\n", dependency.project_id, pid).as_bytes())?;
            }
        }
    }

    Ok(())
}

fn extract_metadata(
    unique_projects: &mut UniqueProjects,
    platform: &str,
    dependencies_meta: &mut File,
    dependency: &Dependency,
) -> Result<(), Box<dyn Error>> {
    // Remember if we visited this project before, so that we write only
    // unique meta information to the file.
    if unique_projects.get(&dependency.project_id).is_none() {
        unique_projects.insert(dependency.project_id);
        dependencies_meta.write_all(
            format!(
                "{},{},{}\n",
                dependency.project_id, dependency.project_name, platform
            )
            .as_bytes(),
        )?;
    }

    Ok(())
}

fn by_platform<'a>(platform: &'a str) -> Box<dyn FnMut(&StringRecord) -> bool + 'a> {
    Box::new(move |e| e[1] == *platform)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <PATH-TO-CSV-FILE> <PLATFORM>", &args[0]);
        eprintln!(
            r#"Example: {} ~/Downloads/Libraries.io-open-data-1.4.0.tar.gz 
            ~/Downloads/libraries-1.4.0-2018-12-22/dependencies-1.4.0-2018-12-22.csv
            Cargo"#,
            &args[0]
        );
    }

    // Read the path to the file from the args
    let (path, platform) = (&args[1], &args[2]);

    source_dependencies(path, platform)
}
