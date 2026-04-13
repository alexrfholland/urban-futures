# First Machine Setup

Run these steps once after cloning the repo onto a new machine.

## 1. Bootstrap the repo shell

Windows PowerShell:

```powershell
. .\scripts\dev-shell.ps1
```

macOS / Linux shell:

```bash
. ./scripts/dev-shell.sh
```

This puts the repo-local toolchain on `PATH` for the current shell:

- repo-root `uv` wrapper
- repo-local `.venv`
- Windows Mediaflux JRE + client bin when present

## 2. Verify `uv`

Windows:

```powershell
uv --version
```

macOS / Linux:

```bash
uv --version
```

If `uv` is still missing:

- Windows: the repo expects `.tools/uv/uv.exe`
- macOS / Linux: install `uv` on the machine or set `UV_BIN` before using the repo wrapper

## 3. Create or refresh the repo environment

Core environment:

```bash
uv sync
```

Full environment used across this repo:

```bash
uv sync --extra world-construction --extra visuals --extra blender --extra rhino-legacy
```

## 4. Verify the repo interpreter

```bash
uv run python -V
uv run python -c "import sys; print(sys.executable)"
```

## 5. Windows-only Mediaflux bootstrap

The repo shell script sets the local client/JRE paths for the current shell.
You still need valid user credentials in:

- `%USERPROFILE%\.Arcitecta\mflux.cfg`

Then verify:

```powershell
uv run python -m mediafluxsync project-config --project-dir .
uv run python -m mediafluxsync project-path pipeline --project-dir .
```

## 6. bV2 runs

Use the repo launcher instead of hand-writing long Blender commands:

Windows:

```powershell
.\scripts\run-bv2.ps1 -Site trimmed-parade -Mode timeline -SimRoot 4.10 -DataBundleRoot .\_data-refactored\model-outputs\generated-states\4.10\output
```

macOS / Linux:

```bash
./scripts/run-bv2.sh --site trimmed-parade --mode timeline --sim-root 4.10 --data-bundle-root ./_data-refactored/model-outputs/generated-states/4.10/output
```
