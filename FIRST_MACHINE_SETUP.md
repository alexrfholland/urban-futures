# First Machine Setup

Run these steps once after cloning the repo onto a new machine.

## 1. Install and verify `uv`

Install `uv` on the machine and ensure it is on `PATH`.

```bash
uv --version
```

## 2. Create or refresh the repo environment

Core environment:

```bash
uv sync
```

Full environment used across this repo:

```bash
uv sync --extra world-construction --extra visuals --extra blender --extra rhino-legacy
```

## 3. Verify the repo interpreter

```bash
uv run python -V
uv run python -c "import sys; print(sys.executable)"
```

## 4. Windows-only Mediaflux bootstrap

Set the local client/JRE paths in the current shell, then verify.
You still need valid user credentials in:

- `%USERPROFILE%\.Arcitecta\mflux.cfg`

```powershell
$env:PATH = "D:\2026 Arboreal Futures\urban-futures\.tools\mediaflux-bin\unimelb-mf-clients-0.8.5\jre\bin;$env:PATH"
$env:MEDIAFLUX_CLIENT_BIN_DIR = "D:\2026 Arboreal Futures\urban-futures\.tools\mediaflux-bin\unimelb-mf-clients-0.8.5\bin\windows"
```

Then verify:

```powershell
uv run python -m mediafluxsync project-config --project-dir .
uv run python -m mediafluxsync project-path pipeline --project-dir .
```

## 5. bV2 runs

For bV2 invocation, use the authoritative run contract:
[bV2_run-instructions.md](_documentation-refactored/blenderv2/bV2_run-instructions.md)
