
![Buildkite](https://badge.buildkite.com/0342a83f273c9ff90762c676def36d4f987f62483d7ae1b333.svg)
[![codecov](https://codecov.io/gh/oscoin/osrank-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/oscoin/osrank-rs)
![rustc_version](https://img.shields.io/badge/rustc-1.38.0-orange)

This repo contains a work-in-progress, unstable, pre-alpha implementation 
of `osrank` in Rust.

Table of Contents
=================

   * [Table of Contents](#table-of-contents)
   * [Getting started](#getting-started)
   * [Building the project](#building-the-project)
   * [Running the tests](#running-the-tests)
   * [Running the benchmarks](#running-the-benchmarks)
      * [Running the dev benchmarks](#running-the-dev-benchmarks)
      * [Running the nightly benchmarks](#running-the-nightly-benchmarks)
   * [Code organisation](#code-organisation)
   * [(Binaries only) Sourcing the data](#binaries-only-sourcing-the-data)
      * [Before starting](#before-starting)
      * [osrank-source-dependencies](#osrank-source-dependencies)
      * [osrank-source-contributions](#osrank-source-contributions)
         * [Resuming work](#resuming-work)
      * [osrank-adjacency-matrix](#osrank-adjacency-matrix)

# Getting started

**If you are new to the project, you might want to start by reading the
[specification](https://github.com/oscoin/osrank-spec/raw/master/osrank_spec.pdf)**.
This document contains information about the osrank's basic model, the set of
open questions remaining to the answered as well as a general discussion
about a possible API.

# Building the project

`osrank-rs` has been successfully compiled locally and on CI using the following
`rustc` version:

```
rustc 1.38.0 (625451e37 2019-09-23)
```

# Running the tests

Tests for the libraries can be run via:

```
cargo test --all
```

There are also tests associated with most binaries. To run them, simply run:

```
cargo test --features build-binary --bin <selected_binary>
```

# Running the benchmarks

We provide benchmarks for the (naive only for now) algorithm. In order to build
(but not run) the benchmarks, simply do:

```
cargo bench --no-run
```

We also provide a filter to select which flavour of benchmarks one wants to run.
In particular, the `dev` benchmarks use a small number of iterations and are
useful for "local" development, as they are fairly fast to run. Conversely,
the nightly benchmarks are much slower and they are meant to be run as part of
CI.

## Running the dev benchmarks

```
cargo bench -- dev
```

## Running the nightly benchmarks

```
cargo bench -- nightly
```

# Code organisation

The code is split into a library and a set of binaries, which can be used to
perform data transformations, import & export graphs and more. We also have
a set of benchmarks.

# (Binaries only) Sourcing the data

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

* `osrank-adjacency-matrix` can be used to calculate the adjancency matrix
  for a whole network using the formula of the basic model.

* `osrank-export-to-gephi` can be used to export a `Graph` into both `.gexf`
  and `.graphml` formats, to be used with a data visualiser like Gephi.

* `osrank-rank` can be used to run simulations, by specifying some initial
  `.csv` files for the selected ecosystem as well as overriding any meaningful
  parameter for the simulation. Refer to `osrank-rank --help` for a full
  breakdown of the supported options. It generates a `.csv` file with the
  sorted `osrank`s.

## Before starting

For the sake of not committing bit objects into `git`, we do not store these
.csv files into the git history. There are two options available to the user:

1. (Easy) Use one of the pre-generated `.csv` files stored in the
   [osrank-rs-ecosystems](https://github.com/oscoin/osrank-rs-ecosystems) repo.

2. (Hard) Generate the files from the binaries. In order to do so, there are a bunch 
   of preliminary operations a user must do:

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
~/Downloads/libraries-1.4.0-2018-12-22/dependencies-1.4.0-2018-12-22.csv <Chosen_Platform>
```

(p.s. You can discover which `<Chosen_Platform>`s are available by opening one
on those big `.csv` files and searching there directly, or refer to the Libraries.io
documentation).

This will produce a `data/<Chosen_Platform>_dependencies.csv` and a 
`data/<Chosen_Platform>_dependencies_meta.csv` csv files on the local filesystem.

## osrank-source-contributions

Same process applies for this binary, with the exception that a valid Github
API token needs to be supplied as a valid env-var. For example:

```
OSRANK_GITHUB_TOKEN=<VALID_TOKEN> \
./target/release/osrank-source-contributions \
~/Downloads/libraries-1.4.0-2018-12-22/projects_with_repository_fields-1.4.0-2018-12-22.csv <Chosen_Platform>
```

This script will take a while to run as it is throttled to ensure we do not
hit Github's Quota Limit, as authenticated users are allowed to only perform
5000 requests per hour. At the end of the process, this will produce a 
`data/cargo_contributions.csv` file on disk.

### Resuming work

If the dataset is big, chances are the script will need to run for many days.
Luckily enough, we support a `--resume-from <url>` parameter which can be used
to pass as input the URL of the *last* visited project, and the script will
automatically resume fetching data from there.

## osrank-adjacency-matrix

This script is largely superseded by the `osrank-rank` algorithm, but it's
still useful as it performs only the `pagerank` step, by actually calling the
non-incremental algorithm. This means the result will be much more precise and
the sum of all the ranks will be exactly a probability distribution, but it 
won't scale for large graphs.
