use itertools::Itertools;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Debug, Display)]
pub enum CsvExporterError {
    IOError(std::io::Error),
}

impl From<std::io::Error> for CsvExporterError {
    fn from(err: std::io::Error) -> CsvExporterError {
        CsvExporterError::IOError(err)
    }
}

/// Given a (id,rank) iterator, write into a `.csv` file the (sorted) rank,
/// from the highest to the lowest.
pub fn export_rank_to_csv<K, V>(
    annotator: impl Iterator<Item = (K, V)>,
    to_f64: Box<dyn Fn(V) -> f64>,
    out_path: &str,
) -> Result<(), CsvExporterError>
where
    V: PartialOrd,
    K: std::fmt::Display,
{
    let mut output_csv = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)?;

    for (node_id, rank) in annotator.sorted_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap()) {
        output_csv.write_all(
            format!("{} {:.32}\n", node_id, to_f64(rank))
                .as_str()
                .as_bytes(),
        )?;
    }

    Ok(())
}
