extern crate csv;
extern crate reqwest;
extern crate serde;

extern crate failure;
#[macro_use]
extern crate failure_derive;

use csv::StringRecord;
use reqwest::{Client, Url};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::{thread, time};

use std::collections::HashSet;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;

enum HttpMethod {
    Get,
}

#[derive(Debug, Fail)]
enum AppError {
    // Returned when we couldn't extract an owner and a repo from the repository URL.
    #[fail(display = "Couldn't extract project metadata for {}", repo_url)]
    MetadataExtractionFailed { repo_url: String },

    // Returned in case of generic I/O error.
    #[fail(display = "i/o error when reading/writing on the CSV file {}", _0)]
    IOError(std::io::Error),

    // Returned when the OSRANK_GITHUB_TOKEN is not present as an env var.
    #[fail(display = "Couldn't find OSRANK_GITHUB_TOKEN in your env vars: {}", _0)]
    GithubTokenNotFound(std::env::VarError),

    // Returned when we failed to issue the HTTP request.
    #[fail(display = "Request to Github failed: {}", _0)]
    GithubAPIRequestFailed(reqwest::Error),

    // Returned when the Github API returned a non-2xx status code.
    #[fail(display = "Github returned non-200 {} with body {}", _0, _1)]
    GithubAPINotOK(reqwest::StatusCode, String),

    // Returned when the parsing of the http URL to query Github failed.
    #[fail(display = "Github URL failed parsing into a valid HTTP URL: {}", _0)]
    GithubUrlParsingFailed(reqwest::UrlError),

    // Returned when the Github API returned a non-2xx status code.
    #[fail(display = "Couldn't deserialise the JSON returned by Github: {}", _0)]
    DeserialisationFailure(reqwest::Error),

    // Returned when the Github API returned a non-2xx status code.
    #[fail(display = "No more retries.")]
    NoRetriesLeft,
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> AppError {
        AppError::IOError(err)
    }
}

impl From<std::env::VarError> for AppError {
    fn from(err: std::env::VarError) -> AppError {
        AppError::GithubTokenNotFound(err)
    }
}

impl From<reqwest::Error> for AppError {
    fn from(err: reqwest::Error) -> AppError {
        AppError::GithubAPIRequestFailed(err)
    }
}

impl From<reqwest::UrlError> for AppError {
    fn from(err: reqwest::UrlError) -> AppError {
        AppError::GithubUrlParsingFailed(err)
    }
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

struct Retries {
    retries_num: u8,
}

impl Retries {
    fn new(retries_num: u8) -> Self {
        Retries { retries_num }
    }
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

/// Calls the Github API using the given HttpMethod and url_path. Due to the
/// fact some endpoints like the statistics one use cached information and
/// might return a 202 with an empty JSON as the stats are computed, we need
/// to wait a little bit and retry, up to a certain number of times.
fn call_github<T>(
    http_client: &Client,
    http_method: HttpMethod,
    token: &str,
    url_path: &str,
    retries: Retries,
) -> Result<T, AppError>
where
    T: DeserializeOwned,
{
    let retries_left = retries.retries_num;
    if retries_left == 0 {
        Err(AppError::NoRetriesLeft)
    } else {
        let bearer = format!("Bearer {}", token);
        match http_method {
            HttpMethod::Get => {
                let url: Url = format!("{}{}", GITHUB_BASE_URL, url_path)
                    .as_str()
                    .parse()?;
                let mut res = http_client
                    .get(url)
                    .header(reqwest::header::AUTHORIZATION, bearer.as_str())
                    .send()?;
                match res.status() {
                    reqwest::StatusCode::OK => res
                        .json()
                        .or_else(|e| Err(AppError::DeserialisationFailure(e))),
                    // Github needs a bit more time to compute the stats.
                    // We retry.
                    reqwest::StatusCode::ACCEPTED => {
                        println!("Retrying, only {} retries left ...", retries_left);
                        thread::sleep(time::Duration::from_secs(1));
                        call_github(
                            http_client,
                            http_method,
                            token,
                            url_path,
                            Retries::new(retries_left - 1),
                        )
                    }
                    err => {
                        let body = res.text()?;
                        Err(AppError::GithubAPINotOK(err, body))
                    }
                }
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

fn source_contributors(github_token: &str, path: &str, platform: &str) -> Result<(), AppError> {
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
    let http_client = reqwest::Client::new();

    //Write the header
    contributions.write(b"MAINTAINER,REPO,CONTRIBUTIONS,NAME\n")?;

    for result in rdr
        .records()
        .filter_map(|e| e.ok())
        .filter(by_platform(platform))
    {
        if let Some(project) = deserialise_project(&result) {
            extract_contribution(
                &http_client,
                &mut contributions,
                &mut unique_projects,
                project,
                github_token,
            )?;
        }
    }

    Ok(())
}

fn extract_github_owner_and_repo(repo_url: &str) -> Result<(&str, &str), AppError> {
    match repo_url.split('/').collect::<Vec<&str>>().as_slice() {
        [_, "", "github.com", owner, repo] => Ok((owner, repo)),
        _ => Err(AppError::MetadataExtractionFailed {
            repo_url: repo_url.to_string(),
        }),
    }
}

// Extract the contribution relative to this project. For now only GitHub is
// supported.
fn extract_contribution(
    http_client: &Client,
    contributions: &mut File,
    unique_projects: &mut UniqueProjects,
    project: Project,
    auth_token: &str,
) -> Result<(), AppError> {
    // If this is an authentic project and not a fork, proceed.
    if !project.repository_fork && unique_projects.get(project.repository_url) == None {
        match extract_github_owner_and_repo(project.repository_url) {
            Err(err) => {
                println!("Skipping {} due to {:#?}", &project.repository_url, err);
                Ok(())
            }
            Ok((owner, name)) => {
                unique_projects.insert(String::from(project.repository_url));
                println!("Processing {}/{}", owner, name);
                let res: Result<Vec<GithubContribution>, AppError> = call_github(
                    &http_client,
                    HttpMethod::Get,
                    auth_token,
                    format!("/repos/{}/{}/stats/contributors", owner, name).as_str(),
                    Retries::new(5),
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

                // Wait 800 ms to not overload Github and not hit the quota limit.
                // GH allows us 5000 requests per hour. If we wait 800ms, we
                // aim for the theoretical limit, while preserving a certain
                // slack.
                let delay = time::Duration::from_millis(800);
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

const GITHUB_BASE_URL: &'static str = "https://api.github.com";

fn main() -> Result<(), AppError> {
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
