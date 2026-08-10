# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## [0.1.0] - 2026-08-10

### Added

- Standalone installable Python package.
- `cell-registration-dashboard` and `db-builder` command-line entry points.
- Windows, Linux, and macOS installation helpers.
- Reliable Windows installation in Local AppData with visible progress and
  repository-root launchers.
- Offline frontend dependencies.
- Dynamic HDF5 root discovery with `Database` as the preferred schema root.
- Auto-cleaning by spatial-footprint area and positive temporal activity, with
  distributions for choosing both thresholds.
- Revision-scoped Overview caching to avoid reloading unchanged all-session
  data when switching between a single session and Overview.
- Existing analysis regression tests and cross-platform installation checks.
- Five-session example database for manual browser validation.
- Documentation screenshots covering cleaning, alignment, cell matching, and
  database building.
- Lean CaImAn, Minian, and MIN1PIPE imports with shared cross-pipeline quality
  metrics and optional native quality evidence.
- Interactive quality review with disabled-by-default thresholds and explicit
  confirmation before discarding selected cells.

### Fixed

- Keep processed databases and matching-result files visible when saving on
  Windows mapped drives.
