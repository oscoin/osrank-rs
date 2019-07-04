extern crate csv;
extern crate reqwest;
extern crate serde;

use csv::StringRecord;
use reqwest::Url;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::{thread, time};

use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};

enum HttpMethod {
    Get,
}

// The order of the fields must be the same of the input file.
#[derive(Debug)]
struct Project<'a> {
    id: u32,
    platform: &'a str,
    project_name: &'a str,
    repository_url: &'a str,
    repository_fork: bool,
    repository_display_name: &'a str,
}

#[derive(Debug, Deserialize)]
struct GithubContribution {
    total: u64,
    author: GithubUser,
    weeks: Vec<GithubWeek>,
}

#[derive(Debug, Deserialize)]
struct GithubWeek {
    // Unix timestamp of the beginning of this week.
    w: u64,
}

#[derive(Debug, Deserialize)]
struct GithubUser {
    login: String,
    id: u64,
}

type UniqueProjects = HashSet<String>;

fn call_github<T>(http_method: HttpMethod, token: &str, url_path: &str) -> Result<T, Box<dyn Error>>
where
    T: DeserializeOwned,
{
    let http_client = reqwest::Client::new();
    let bearer = format!("Bearer {}", token);
    match http_method {
        HttpMethod::Get => {
            let url: Url = format!("{}{}", GITHUB_BASE_URL, url_path)
                .as_str()
                .parse()?;
            let res = http_client
                .get(url)
                .header(reqwest::header::AUTHORIZATION, bearer.as_str())
                .send()
                .and_then(|x| x.error_for_status())
                .and_then(move |mut x| x.text_with_charset("utf-8"))?;
            match serde_json::from_str(&res) {
                Err(err) => Err(Box::new(std::io::Error::new(
                    ErrorKind::Other,
                    format!("{:#?}, original json: {}", err, res),
                ))),
                Ok(v) => Ok(v),
            }
        }
    }
}

fn deserialise_project<'a>(sr: &'a StringRecord) -> Option<Project<'a>> {
    if let Some(Ok(pid)) = sr.get(0).map(|s: &str| s.parse::<u32>()) {
        let platform = sr.get(1);
        let project_name = sr.get(2);
        let repository_url = sr.get(9);
        let repository_fork = sr.get(24).and_then(|s: &str| match s {
            "t" => Some(true),
            "f" => Some(false),
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        });
        let repository_display_name = sr.get(54);

        match (
            platform,
            project_name,
            repository_url,
            repository_fork,
            repository_display_name,
        ) {
            (Some(pl), Some(pn), Some(ru), Some(rf), Some(dn)) => Some(Project {
                id: pid,
                platform: pl,
                project_name: pn,
                repository_url: ru,
                repository_fork: rf,
                repository_display_name: dn,
            }),
            _ => None,
        }
    } else {
        None
    }
}

fn source_contributors(
    github_token: &str,
    path: &str,
    platform: &str,
) -> Result<(), Box<dyn Error>> {
    let projects_file = File::open(path)?;

    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(projects_file);
    let mut contributions = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(format!("data/{}_contributions.csv", platform.to_lowercase()).as_str())?;

    let mut unique_projects = HashSet::new();

    //Write the header
    contributions.write(b"MAINTAINER,REPO,CONTRIBUTIONS,NAME\n")?;

    for result in rdr
        .records()
        .filter_map(|e| e.ok())
        .filter(by_platform(platform))
    {
        if let Some(project) = deserialise_project(&result) {
            extract_contribution(
                &mut contributions,
                &mut unique_projects,
                project,
                github_token,
            )?;
        }
    }

    Ok(())
}

const GITHUB_BASE_URL: &'static str = "https://api.github.com";

fn extract_github_owner_and_repo(repo_url: &str) -> Option<(&str, &str)> {
    match repo_url.split('/').collect::<Vec<&str>>().as_slice() {
        [_, "", "github.com", owner, repo] => Some((owner, repo)),
        _ => None,
    }
}

// Extract the contribution relative to this project. For now only GitHub is
// supported.
fn extract_contribution(
    contributions: &mut File,
    unique_projects: &mut UniqueProjects,
    project: Project,
    auth_token: &str,
) -> Result<(), Box<dyn Error>> {
    // If this is an authentic project and not a fork, proceed.
    if !project.repository_fork && unique_projects.get(project.repository_url) == None {
        match extract_github_owner_and_repo(project.repository_url) {
            None => {
                let err = Box::new(std::io::Error::new(
                    ErrorKind::Other,
                    "couldn't extract project metadata",
                ));
                println!("Skipping {} due to {:#?}", &project.repository_url, err);
                Ok(())
            }
            Some((owner, name)) => {
                unique_projects.insert(String::from(project.repository_url));
                println!("Processing {}/{}", owner, name);
                let res: Result<Vec<GithubContribution>, Box<dyn Error>> = call_github(
                    HttpMethod::Get,
                    auth_token,
                    format!("/repos/{}/{}/stats/contributors", owner, name).as_str(),
                );

                match res {
                    Err(err) => {
                        println!("Skipping {} due to {:#?}", &project.repository_url, err);
                    }
                    Ok(stats) => {
                        for contribution in stats {
                            if is_maintainer(&owner, &contribution) {
                                contributions.write(
                                    format!(
                                        "github@{},{},{},{}\n",
                                        contribution.author.login,
                                        project.repository_url,
                                        contribution.total,
                                        project.project_name
                                    )
                                    .as_bytes(),
                                )?;
                            }
                        }
                    }
                }

                //Wait 5 seconds to not overload Github and not hit the quota limit.
                let delay = time::Duration::from_secs(5);
                thread::sleep(delay);
                Ok(())
            }
        }
    } else {
        Ok(())
    }
}

// FIXME(adn) Totally arbitrary choice: consider a maintainer
// for a project a user that has been contributed for more
// than 6 months. Furthermore, it needs to have a somewhat steady contribution
// history.
fn is_maintainer(owner: &str, stat: &GithubContribution) -> bool {
    stat.author.login == owner || { stat.total > 50 }
}

fn by_platform<'a>(platform: &'a str) -> Box<FnMut(&StringRecord) -> bool + 'a> {
    Box::new(move |e| e[1] == *platform)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let github_token = env::var("OSRANK_GITHUB_TOKEN")?;

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
            ~/Downloads/libraries-1.4.0-2018-12-22/projects_with_repository_fields-1.4.0-2018-12-22.csv
            Cargo"#,
            &args[0]
        )
        .unwrap();
    }

    // Read the path to the file from the args
    let (path, platform) = (&args[1], &args[2]);

    source_contributors(&github_token, path, platform)
}
