# Calcium Imaging Dashboard

Calcium Imaging Dashboard is a local, browser-based workflow for:

- building a portable HDF5 database from CaImAn or Minian analysis output;
- cleaning and merging segmented cells;
- aligning fields of view across sessions; and
- matching cells across sessions.

The software runs on your computer and opens its interface in your default web
browser. It does not require a hosted web service or an internet connection
after installation.

## Requirements

- Python 3.11 or 3.12
- A modern web browser
- Windows, Linux, or macOS

Python 3.13 is not yet supported because some scientific dependencies may not
provide compatible builds.

## Install after cloning

Clone the repository, enter its directory, and run the installer for your
operating system:

### Windows

Double-click `install-windows.bat`, or run:

```powershell
.\install-windows.bat
```

The installer shows progress and waits for confirmation before closing. It
installs the isolated Python environment under
`%LOCALAPPDATA%\CalciumImagingDashboard\venv`, so cloning the repository to a
mapped or network drive does not slow down the installed application.

### Linux

```bash
chmod +x install-linux.sh
./install-linux.sh
```

### macOS

Double-click `install-macos.command`, or run:

```bash
chmod +x install-macos.command
./install-macos.command
```

On Linux and macOS, the installers create an isolated `.venv` in the repository.
On Windows, the environment is stored in Local AppData. Running an installer
again refreshes its dedicated environment; it does not alter databases or
analysis files.

If an earlier version created `.venv` inside the repository, the new Windows
launchers no longer use it. You may remove that old folder manually after
confirming that both new launchers work.

## Launch

On Windows, double-click one of the launchers in the repository:

- `launch-dashboard.bat`
- `launch-db-builder.bat`

On Linux or macOS, activate the environment and run either command:

```bash
source .venv/bin/activate
cell-registration-dashboard
```

To open a database immediately:

```powershell
.\launch-dashboard.bat --database "C:\path\to\Database.mat"
```

On Linux or macOS, the equivalent commands are:

```bash
cell-registration-dashboard --database /path/to/Database.mat
db-builder
```

If the default local port is occupied, use `--port`, for example
`cell-registration-dashboard --port 8012`.

## Database format and safe editing

New databases use the top-level HDF5 group `Database`. The dashboard also reads
an older file when it contains one unambiguous non-system root group.

The dashboard keeps the selected raw database unchanged. Saving creates or
updates a sibling file whose name ends in `_processed`.

## Example data

`examples/Mouse56_5sessions.mat` is a small five-session database for manually
checking the browser workflow. See `examples/README.md` for its separate data
license.

## Development and tests

Install development dependencies and run the migrated regression tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The repository deliberately has no general-purpose `scripts/` directory.
Installation entry points live at the repository root so users can find them
immediately.

## Citation

Please cite the software using `CITATION.cff`. GitHub can render this file as a
ready-to-copy citation.

## License

The software is licensed under the BSD 3-Clause License. The example database
has its own CC BY 4.0 license, described in `examples/README.md`.
