# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-07-28

### Added

- Standalone installable Python package.
- `cell-registration-dashboard` and `db-builder` command-line entry points.
- Windows, Linux, and macOS installation helpers.
- Offline frontend dependencies.
- Dynamic HDF5 root discovery with `Database` as the preferred schema root.
- Auto-cleaning by spatial-footprint area and positive temporal activity, with
  distributions for choosing both thresholds.
- Revision-scoped Overview caching to avoid reloading unchanged all-session
  data when switching between a single session and Overview.
- Existing analysis regression tests and cross-platform installation checks.
- Five-session example database for manual browser validation.
