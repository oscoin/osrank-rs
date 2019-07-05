
For now there is very little to see. This is a work-in-progress, unstable,
pre-alpha implementation of `osrank` in Rust.

# Sourcing the data

This project provides a bunch of binaries to source the data necessary to
compute things like an adjacency matrix locally, bypassing the Jupyter notebook.
In particular:

* `osrank-source-dependencies` can be used to produce a CSV file in the same
  format of the one produced by the Jupyter notebook of all the projects and
  its dependencies for a given ecosystem, and can be parameterised by platform
  to generate multiple CSV files.

* `osrank-source-contributions` can be used to produce a CSV file of a list of
  maintainers, alongside the projects they maintain and the number of
  contributions. It can be parameterised by platform to generate multiple CSV
  files.

## Before starting

For the sake of not committing bit objects into `git`, we do not store these
.csv files into the git history (apart from rare exceptions), but they rather
need to be generated from the binaries (or uploaded to a place like S3 for
quicker retrieval). In order to do so, there are a bunch of preliminary
operations a user must do:

* Download the (fairly big) dataset from [libraries.io](https://zenodo.org/record/2536573#.XR7_7ZMzZTY) which includes a
  bunch of interesting datasets we need to operate on;

* Setup a [Github authentication token](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line) if one desires to run
  `osrank-source-contributions`. You don't need to set any permission for this one
  (i.e. you don't need to check any checkbox in the menu, when creating one).

## osrank-source-dependencies

It's warmly recommended to compile the binary in release mode by typing:

```
cargo build --release --features build-binary --bin osrank-source-dependencies
```

The `--features build-binary` is a compilation flag used to minimise the dependency
footprint of the project, making sure certain libraries are compiled and
downloaded only for these binaries, but not for library code.

Once the compilation finished, one can proceed running the script like so
(for example):

```
./target/release/osrank-source-dependencies \
~/Downloads/libraries-1.4.0-2018-12-22/dependencies-1.4.0-2018-12-22.csv Cargo
```

This will produce a `data/cargo_dependencies.csv` and a `data/cargo_dependencies_meta.csv`
csv files on the local filesystem.

## osrank-source-contributions

Same process applies for this binary, with the exception that a valid Github
API token needs to be supplied as a valid env-var. For example:

```
OSRANK_GITHUB_TOKEN=<VALID_TOKEN> \
./target/release/osrank-source-contributions \
~/Downloads/libraries-1.4.0-2018-12-22/projects_with_repository_fields-1.4.0-2018-12-22.csv Cargo
```

This script will take a while to run as it is throttled to ensure we do not
hit Github's Quota Limit, as authenticated users are allowed to only perform
5000 requests per hour. At the end of the process, this will produce a 
`data/cargo_contributions.csv` file on disk.