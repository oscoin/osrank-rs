RUSTFLAGS="-Ccodegen-units=1 -Cinline-threshold=0 -Clink-dead-code -Coverflow-checks=off" cargo cov test --all
zip -0 ccov.zip `find . \( -name "osrank*.gc*" \) -print`;
grcov ccov.zip -s . -t lcov > lcov0.info

# Remove a bit of noise from the report
lcov --remove lcov0.info "*rustc*" -o lcov1.info
lcov --remove lcov1.info "*registry*" -o lcov.info
rm lcov0.info
rm lcov1.info

mkdir -p coverage-report
genhtml -o coverage-report/ --show-details --highlight --ignore-errors source --legend lcov.info

# clean up
rm ccov.zip
cargo cov clean
