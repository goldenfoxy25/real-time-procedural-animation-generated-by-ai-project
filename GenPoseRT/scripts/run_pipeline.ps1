# Run pipeline helper
# Usage: right-click -> Run with PowerShell, or open a PowerShell and run:
#   powershell -ExecutionPolicy Bypass -File .\GenPoseRT\scripts\run_pipeline.ps1 -PackedInput 'C:/Users/mathb/Downloads/Packed' -ConvertedOut 'C:/Users/mathb/OneDrive/Bureau/GenPoseRT/data/converted_packed'

param(
    [string]$PackedInput = "",
    [string]$ConvertedOut = "",
    [string]$BlenderExe = "C:\Program Files\Blender Foundation\Blender\blender.exe",
    [string]$TargetModel = "",
    [switch]$RunBlenderConversions = $true,
    [int]$MaxFoldersToConvert = 0 # 0 = no limit (convert all)
)

Write-Host "--- RUN PIPELINE ---"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Compute repository root (one level above the scripts folder)
$repoRootPath = Join-Path $scriptDir ".."
try {
    $repoRoot = (Resolve-Path $repoRootPath -ErrorAction Stop).Path
} catch {
    # Fallback: use script dir's parent string
    $repoRoot = $repoRootPath
}

# Set portable defaults if parameters not provided
if ([string]::IsNullOrWhiteSpace($PackedInput)) {
    # Default: look for an input folder relative to repository (user can override)
    $PackedInput = Join-Path $repoRoot 'data\Packed'
}
if ([string]::IsNullOrWhiteSpace($ConvertedOut)) {
    $ConvertedOut = Join-Path $repoRoot 'data\converted_packed'
}
if ([string]::IsNullOrWhiteSpace($TargetModel)) {
    $defaultModel = Join-Path $repoRoot 'test\BASEmodel.obj'
    if (Test-Path $defaultModel) { $TargetModel = $defaultModel }
}

# 1) Create venv if not exists (place .venv at repo root)
$venvPath = Join-Path $repoRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating venv at $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "Venv already exists at $venvPath"
}

# Prefer python from the created venv for subsequent commands (fallback to system python)
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = 'python'
}

# 2) Activate venv
$activate = Join-Path $venvPath "Scripts/Activate.ps1"
if (Test-Path $activate) {
    Write-Host "Activating venv"
    & $activate
} else {
    Write-Host "Could not find Activate.ps1 at $activate"
}

# 3) Install requirements
$req = Join-Path $repoRoot "requirements.txt"
if (Test-Path $req) {
    Write-Host "Installing requirements from $req"
    & $pythonExe -m pip install -r $req
} else {
    Write-Host "No requirements.txt found at $req"
}

# 4) Prepare Packed commands
Write-Host "Preparing Packed conversion commands"
$prepScript = Join-Path $repoRoot "scripts\prepare_packed.py"
if (-not (Test-Path $prepScript)) {
    Write-Host "Could not find prepare_packed.py at $prepScript" -ForegroundColor Red
} else {
    # Build args
    $prepArgs = @('--input', $PackedInput, '--output', $ConvertedOut)
    if ($TargetModel -ne "") { $prepArgs += @('--target_model', $TargetModel) }
    Write-Host "Running: $pythonExe $prepScript $($prepArgs -join ' ')"
    $prepOutput = & $pythonExe $prepScript @prepArgs
    Write-Host $prepOutput
}

# 5) Optionally run Blender conversions (requires Blender installed)
if ($RunBlenderConversions) {
    Write-Host "Running Blender conversions (requires Blender installed at $BlenderExe)"
    if (-not (Test-Path $BlenderExe)) {
        Write-Host "Blender executable not found at $BlenderExe. Skipping conversions." -ForegroundColor Yellow
    } else {
        # Generate the commands again, capturing them
        $raw = & $pythonExe $prepScript --input $PackedInput --output $ConvertedOut
        $cmds = $raw | Select-String 'blender' | ForEach-Object {$_.Line}
        if ($MaxFoldersToConvert -gt 0) {
            $cmds = $cmds | Select-Object -First $MaxFoldersToConvert
            Write-Host "Limiting conversions to first $MaxFoldersToConvert folders (set -MaxFoldersToConvert 0 to convert all)"
        } else {
            Write-Host "Converting all folders (this may take a long time)"
        }
        foreach ($c in $cmds) {
            Write-Host "Running: $c"
            try {
                Invoke-Expression $c
            } catch {
                Write-Host "Conversion command failed: $_" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "Skipping Blender conversions (set -RunBlenderConversions to run)"
}

# 6) Test loader
Write-Host "Testing data loader"
$loader = Join-Path $repoRoot "src\data_loader.py"
if (Test-Path $loader) {
    Write-Host "Running: $pythonExe $loader $ConvertedOut"
    & $pythonExe $loader $ConvertedOut
} else {
    Write-Host "Could not find data_loader at $loader" -ForegroundColor Yellow
}

# 7) Run model demo
Write-Host "Running model demo"
$model = Join-Path $repoRoot "src\model.py"
if (Test-Path $model) {
    $modelArgs = @()
    if ($TargetModel -ne "") { $modelArgs += @('--target_model', $TargetModel) }
    Write-Host "Running: $pythonExe $model $($modelArgs -join ' ')"
    & $pythonExe $model @modelArgs
} else {
    Write-Host "Could not find model script at $model" -ForegroundColor Yellow
}

Write-Host "--- PIPELINE DONE ---"
