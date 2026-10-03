<#
.SYNOPSIS
    Train / continue training one 3-mic variation of the UaVirBASE SSL model.

.DESCRIPTION
    Picks a microphone trio and regression-target config, then runs main.py via
    --json so exactly one variation trains per invocation. Automatically resumes
    from the latest checkpoint in the run's saved_models folder (full
    incremental state: model + optimizer + scheduler + RNG). After every test
    pass (start of each epoch + one final post-training test) a line is
    appended to the run's results file (tab-separated .txt).

    Mic trios (0-based channel indices; lower ring N=0 E=1 S=2 W=3, upper ring NE=4..NW=7):
      401    -> channels [4,0,1]  (NE upper + N/E lower), targets: distance + height + azimuth
      610    -> channels [6,1,0]  (SW upper + N/E lower), targets: distance + height + azimuth
      090270 -> channels [0,1,3]  (N/E/W lower ring),    targets: hypotenuse (slant range) + azimuth

.EXAMPLE
    .\run_variation.ps1 -Mic 401 -Epochs 5            # fresh start or resume, 5 more epochs
    .\run_variation.ps1 -Mic 610 -Epochs 3 -Reset     # archive old run, start from scratch
    .\run_variation.ps1 -Mic 090270 -Epochs 1         # first epoch of the hypotenuse+azimuth model
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("401", "610", "090270")]
    [string]$Mic,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1000)]
    [int]$Epochs,

    # Archive the run's checkpoints + results and start from scratch
    [switch]$Reset,

    # Dataset root (must contain train\ [, val\] and test\). Default: config.py.
    # Use D:\UNIFinalprojectYaser\h\divided_datasets for the local copy with the
    # val split (train=68 / val=32 / test=32) made by make_validation_split.py.
    [string]$DatasetPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# --- Per-mic configuration ---------------------------------------------------
$feature = 'mel_96000_2048_1024_0_128_0_0'   # run-name prefix used by main.py
switch ($Mic) {
    "401" {
        $channels = @(4, 0, 1)
        $chTag = "ch401"
        $targets = '"use_distance": true, "use_height": true, "use_hypotenuse": false, "use_side": false'
        $targetTag = "tDH--"
    }
    "610" {
        $channels = @(6, 1, 0)
        $chTag = "ch610"
        $targets = '"use_distance": true, "use_height": true, "use_hypotenuse": false, "use_side": false'
        $targetTag = "tDH--"
    }
    "090270" {
        $channels = @(0, 1, 3)
        $chTag = "ch013"
        $targets = '"use_distance": false, "use_height": false, "use_hypotenuse": true, "use_side": false'
        $targetTag = "t--R-"
    }
}

$runName   = "$feature`_$chTag`_$targetTag"
$runDir    = Join-Path "saved_models" $runName
$resultsFile = Join-Path "results" "$runName.txt"

# --- Optional reset: archive, never delete -----------------------------------
if ($Reset) {
    foreach ($dir in @($runDir)) {
        if (Test-Path $dir) {
            $archive = Join-Path "saved_models" ("_archive\{0}_{1}" -f $runName, (Get-Date -Format "yyyyMMdd_HHmmss"))
            New-Item -ItemType Directory -Force -Path (Split-Path $archive) | Out-Null
            Move-Item $dir $archive
            Write-Host "Archived checkpoints: $dir -> $archive"
        }
    }
    if (Test-Path $resultsFile) {
        $rArchive = Join-Path "results" ("_archive\{0}_{1}.txt" -f $runName, (Get-Date -Format "yyyyMMdd_HHmmss"))
        New-Item -ItemType Directory -Force -Path (Split-Path $rArchive) | Out-Null
        Move-Item $resultsFile $rArchive
        Write-Host "Archived results: $resultsFile -> $rArchive"
    }
}

# --- Resume detection: continue from the highest <N>_final.pth ---------------
$pretrained = $false
$modelPath = ""
$startEpoch = 0
if (Test-Path $runDir) {
    $latest = Get-ChildItem $runDir -Filter "*_final.pth" |
        ForEach-Object { if ($_.Name -match '^(\d+)_final\.pth$') { [int]$Matches[1] } } |
        Sort-Object -Descending | Select-Object -First 1
    if ($null -ne $latest) {
        $pretrained = $true
        $modelPath = Join-Path $runDir "$($latest)_final.pth"
        $startEpoch = $latest
    }
}
$totalEpochs = $startEpoch + $Epochs

Write-Host "==============================================================="
Write-Host " Mic trio      : $Mic (channels $($channels -join ',')) -> $chTag"
Write-Host " Targets       : $targetTag"
Write-Host " Run name      : $runName"
if ($pretrained) {
    Write-Host " Resuming from : $modelPath (epoch $startEpoch done)"
} else {
    Write-Host " Starting      : fresh training (no checkpoint found)"
}
Write-Host " Epochs        : $Epochs more -> total $totalEpochs"
if ($DatasetPath -ne "") {
    Write-Host " Dataset       : $DatasetPath"
    Write-Host " Val split     : $(if (Test-Path (Join-Path $DatasetPath 'val')) {'YES -> best.pth by val_loss, test untouched (evaluate_test.py after)'} else {'no -> legacy test-driven mode'})"
}
Write-Host " Results file  : $resultsFile (one line per eval pass)"
Write-Host "==============================================================="

# --- Build the JSON override (replaces config['variations']) -----------------
# Written to a temp file and passed via --json-file: PowerShell 5.1 mangles
# embedded double quotes when passing JSON strings directly to native commands.
$chJson = ($channels -join ",")
$datasetJson = ""
if ($DatasetPath -ne "") {
    # Forward slashes so JSON stays valid without escaping backslashes
    $datasetJson = ' "dataset_path": "{0}",' -f ($DatasetPath -replace '\\', '/')
}
$json = @"
{"variations": [
    {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 1024,
     "n_mfcc": 0, "n_mels": 128, "n_lfcc": 0, "n_barks": 0,
     "channels": [$chJson], $targets}
 ],
 "epochs": $totalEpochs,$datasetJson
 "checkpoint_path": "saved_models/$runName",
 "pretrained": $($pretrained.ToString().ToLower()),
 "model_path": "$($modelPath -replace '\\', '/')",
 "results_file": "$($resultsFile -replace '\\', '/')"
}
"@

$jsonFile = Join-Path $env:TEMP "uavirbase_run_variation.json"
$json | Out-File -FilePath $jsonFile -Encoding ascii
python main.py --json-file $jsonFile
$code = $LASTEXITCODE
Remove-Item $jsonFile -ErrorAction SilentlyContinue
if ($code -ne 0) {
    Write-Host "Training failed with exit code $code" -ForegroundColor Red
    exit $code
}
Write-Host "Done. MAE/RMSE per epoch: $resultsFile"
