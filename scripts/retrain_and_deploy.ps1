<#
.SYNOPSIS
    Retrain the NSL model and ship it to a connected phone, end to end.

.DESCRIPTION
    Runs the whole RETRAIN.md chain as one queued sequence, aborting the moment any
    step fails. Aborting matters here: results/model.keras, model.tflite,
    ood_stats.npz, ood.json and model_meta.json are ONE atomic set, and a
    half-finished run leaves the open-set gate describing an embedding space the
    model does not have -- which rejects real signs wholesale instead of crashing.

    Order differs from RETRAIN.md in one deliberate way: train_eval.py runs LAST.
    It only produces reportable metrics and is not part of the shipping chain, so
    running it at the end gets a working APK onto the phone ~15 minutes sooner
    while the leave-one-signer-out folds grind away afterwards.

.PARAMETER SkipDataset
    Skip find_seq_len + build_dataset. Use for a hyperparameter-only change, where
    the tensors on disk are still correct.

.PARAMETER SkipTrain
    Skip train_model. Use to re-export and redeploy the model already in results/.

.PARAMETER SkipEval
    Skip train_eval. Saves ~10-15 min, but leaves results/metrics.json describing
    whatever model was trained before -- do not quote it in a report afterwards.

.PARAMETER SkipInstall
    Build the APK but do not touch the device.

.PARAMETER AppOnly
    Dart/Kotlin change only: analyze, test, build, install. No Python at all.

.PARAMETER SplitPerAbi
    Build per-ABI APKs instead of one fat APK, and install the -Abi one directly.
    Smaller download; note that `flutter install` cannot be used with these (see
    the comment at Install-Apk).

.PARAMETER Abi
    Which per-ABI APK to install when -SplitPerAbi is set. Default arm64-v8a.

.PARAMETER BackupDir
    Where to copy the current model artifacts before overwriting them. Defaults to
    a timestamped folder beside the repo. model.keras and model.tflite are
    gitignored, so this is the only way back.

.EXAMPLE
    .\scripts\retrain_and_deploy.ps1
    Full run: new recordings -> trained model -> APK -> phone -> metrics.

.EXAMPLE
    .\scripts\retrain_and_deploy.ps1 -SkipDataset -SkipEval
    Fast hyperparameter iteration.

.EXAMPLE
    .\scripts\retrain_and_deploy.ps1 -AppOnly
    After editing only Dart or Kotlin.
#>

[CmdletBinding()]
param(
    [switch]$SkipDataset,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$SkipInstall,
    [switch]$AppOnly,
    [switch]$SplitPerAbi,
    [string]$Abi = 'arm64-v8a',
    [string]$BackupDir
)

$ErrorActionPreference = 'Stop'

# --- paths ---------------------------------------------------------------

$RepoRoot = Split-Path -Parent $PSScriptRoot
$AppDir   = Join-Path $RepoRoot 'app'
$Python   = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Results  = Join-Path $RepoRoot 'results'
$Assets   = Join-Path $AppDir  'assets\models'

$script:StepNumber = 0
$script:Timings = New-Object System.Collections.ArrayList

# --- helpers -------------------------------------------------------------

function Write-Head {
    param([string]$Text)
    $script:StepNumber++
    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
    Write-Host ("  [{0}] {1}" -f $script:StepNumber, $Text) -ForegroundColor Cyan
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
}

function Write-Skip {
    param([string]$Text)
    Write-Host ("  -- skipped: {0}" -f $Text) -ForegroundColor DarkGray
}

# Runs a native executable and aborts on a non-zero exit code.
#
# stderr is deliberately NOT redirected: in Windows PowerShell 5.1, piping a native
# command's stderr wraps each line in an ErrorRecord and can flip $? to false even
# when the process returned 0. $LASTEXITCODE is the only trustworthy signal.
function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$What,
        [switch]$Advisory
    )
    if (-not $What) { $What = (Split-Path -Leaf $Exe) }

    $started = Get-Date
    if ($WorkingDirectory) { Push-Location $WorkingDirectory }
    try {
        Write-Host ("  > {0} {1}" -f (Split-Path -Leaf $Exe), ($Arguments -join ' ')) -ForegroundColor DarkGray
        & $Exe @Arguments
        $code = $LASTEXITCODE
    }
    finally {
        if ($WorkingDirectory) { Pop-Location }
    }

    $elapsed = (Get-Date) - $started
    [void]$script:Timings.Add([pscustomobject]@{ Step = $What; Seconds = [math]::Round($elapsed.TotalSeconds, 1) })

    if ($code -ne 0) {
        if ($Advisory) {
            Write-Host ("  ! {0} reported issues (exit {1}) -- continuing, this step is advisory." -f $What, $code) -ForegroundColor Yellow
            return
        }
        Write-Host ''
        Write-Host ("ABORTED at '{0}' (exit code {1})." -f $What, $code) -ForegroundColor Red
        Write-Host 'Nothing further has run. results/ may now be internally inconsistent --' -ForegroundColor Red
        Write-Host 'fix the cause and re-run, or restore from the backup printed above.' -ForegroundColor Red
        exit $code
    }
    Write-Host ("  ok ({0:n1}s)" -f $elapsed.TotalSeconds) -ForegroundColor DarkGreen
}

function Find-Adb {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Android\sdk\platform-tools\adb.exe'),
        (Join-Path $env:ProgramFiles 'Android\android-sdk\platform-tools\adb.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $onPath = Get-Command adb -ErrorAction Ignore
    if ($onPath) { return $onPath.Source }
    return $null
}

function Get-AttachedDevice {
    param([string]$Adb)
    if (-not $Adb) { return $null }
    $lines = & $Adb devices
    foreach ($line in $lines) {
        if ($line -match '^(\S+)\s+device$') { return $Matches[1] }
    }
    return $null
}

# --- preflight -----------------------------------------------------------

Write-Head 'Preflight'

if (-not (Test-Path $Python)) {
    throw "No venv python at $Python. Create it:  python -m venv .venv ; .venv\Scripts\activate ; pip install -r requirements.txt"
}
$flutter = Get-Command flutter -ErrorAction Ignore
if (-not $flutter) { throw 'flutter is not on PATH.' }
Write-Host ("  python  {0}" -f $Python) -ForegroundColor DarkGray
Write-Host ("  flutter {0}" -f $flutter.Source) -ForegroundColor DarkGray

$adb = Find-Adb
$device = $null
if (-not $SkipInstall) {
    $device = Get-AttachedDevice -Adb $adb
    if ($device) {
        Write-Host ("  device  {0}" -f $device) -ForegroundColor DarkGray
    }
    else {
        # A warning, not a failure: building is still useful, and the phone can be
        # plugged in before the install step is reached several minutes from now.
        Write-Host '  device  none detected -- plug the phone in (and accept the USB debugging prompt)' -ForegroundColor Yellow
        Write-Host '          before this run reaches the install step, or re-run with -SkipInstall.' -ForegroundColor Yellow
    }
}

$plan = if ($AppOnly) { 'app only (analyze, test, build, install)' } else { 'full pipeline' }
Write-Host ("  plan    {0}" -f $plan) -ForegroundColor DarkGray

# --- 0. backup -----------------------------------------------------------

if (-not $AppOnly) {
    Write-Head 'Back up current model artifacts'
    if (-not $BackupDir) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $BackupDir = Join-Path (Split-Path -Parent $RepoRoot) ("nsl-backup\" + $stamp)
    }
    if (-not (Test-Path $BackupDir)) {
        New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
    }
    $toBackup = @('model.keras', 'model.tflite', 'model_meta.json', 'ood.json', 'ood_stats.npz', 'metrics.json')
    $saved = 0
    foreach ($name in $toBackup) {
        $src = Join-Path $Results $name
        if (Test-Path $src) {
            Copy-Item $src -Destination $BackupDir -Force
            $saved++
        }
    }
    Write-Host ("  saved {0} file(s) -> {1}" -f $saved, $BackupDir) -ForegroundColor Green
    Write-Host '  (model.keras and model.tflite are gitignored -- this is the only way back)' -ForegroundColor DarkGray
}

# --- 1-2. dataset --------------------------------------------------------

if ($AppOnly -or $SkipDataset) {
    Write-Head 'Dataset'
    Write-Skip 'find_seq_len + build_dataset (existing data/processed reused)'
}
else {
    Write-Head 'Derive SEQ_LEN from clip lengths'
    Invoke-Native -Exe $Python -Arguments @('-u', 'scripts\find_seq_len.py') -WorkingDirectory $RepoRoot -What 'find_seq_len'

    Write-Head 'Build normalized fixed-length tensors'
    Invoke-Native -Exe $Python -Arguments @('-u', 'scripts\build_dataset.py') -WorkingDirectory $RepoRoot -What 'build_dataset'
}

# --- 3. train the deployable model --------------------------------------

if ($AppOnly -or $SkipTrain) {
    Write-Head 'Train deployable model'
    Write-Skip 'train_model (reusing results/model.keras)'
}
else {
    Write-Head 'Train the deployable model'
    Write-Host '  Watch for "real signs wrongly rejected: ~1%". Much higher means the' -ForegroundColor DarkGray
    Write-Host '  open-set gate is misfitted and the app will answer unknown to everything.' -ForegroundColor DarkGray
    Invoke-Native -Exe $Python -Arguments @('-u', 'scripts\train_model.py', '--verbose', '2') -WorkingDirectory $RepoRoot -What 'train_model'
}

# --- 4. export -----------------------------------------------------------

if ($AppOnly) {
    Write-Head 'Export TFLite'
    Write-Skip 'export_tflite'
}
else {
    Write-Head 'Export to TFLite (+ portable OOD stats)'
    Write-Host '  Must print "argmax agreement 100%" -- the export refuses to ship otherwise.' -ForegroundColor DarkGray
    Invoke-Native -Exe $Python -Arguments @('scripts\export_tflite.py') -WorkingDirectory $RepoRoot -What 'export_tflite'
}

# --- 5. copy assets ------------------------------------------------------

if ($AppOnly) {
    Write-Head 'Copy assets into the app'
    Write-Skip 'asset copy (no new model)'
}
else {
    Write-Head 'Copy model assets into the app'
    if (-not (Test-Path $Assets)) { New-Item -ItemType Directory -Force -Path $Assets | Out-Null }
    foreach ($name in @('model.tflite', 'ood.json', 'model_meta.json')) {
        $src = Join-Path $Results $name
        if (-not (Test-Path $src)) { throw "missing $src -- did export_tflite run?" }
        Copy-Item $src -Destination $Assets -Force
        Write-Host ("  {0} -> assets/models/" -f $name) -ForegroundColor DarkGray
    }
    # The two MediaPipe .task bundles are static; only copy them if absent.
    foreach ($name in @('pose_landmarker_full.task', 'hand_landmarker.task')) {
        $dst = Join-Path $Assets $name
        if (-not (Test-Path $dst)) {
            $src = Join-Path $RepoRoot ('models\tasks\' + $name)
            if (-not (Test-Path $src)) { throw "missing $src -- the app cannot extract landmarks without it" }
            Copy-Item $src -Destination $Assets -Force
            Write-Host ("  {0} -> assets/models/ (was missing)" -f $name) -ForegroundColor Yellow
        }
    }
    $meta = Get-Content (Join-Path $Assets 'model_meta.json') -Raw | ConvertFrom-Json
    Write-Host ("  app now carries seq_len={0}, {1} classes" -f $meta.seq_len, $meta.class_names.Count) -ForegroundColor Green
}

# --- 6. re-pin the Dart port -------------------------------------------

if ($AppOnly) {
    Write-Head 'Regenerate golden fixtures'
    Write-Skip 'make_golden (model unchanged)'
}
else {
    Write-Head 'Re-pin the Dart port to this model'
    Write-Host '  The app re-implements preprocess.py and ood.py in Dart. A mismatch does' -ForegroundColor DarkGray
    Write-Host '  NOT throw -- the model just gets features it never trained on. This is the' -ForegroundColor DarkGray
    Write-Host '  only step that catches it.' -ForegroundColor DarkGray
    Invoke-Native -Exe $Python -Arguments @('scripts\make_golden.py') -WorkingDirectory $RepoRoot -What 'make_golden'
}

# --- 7. analyze + test -------------------------------------------------

Write-Head 'Static analysis (advisory)'
Invoke-Native -Exe $flutter.Source -Arguments @('analyze') -WorkingDirectory $AppDir -What 'flutter analyze' -Advisory

Write-Head 'Verify Dart matches the Python pipeline'
Invoke-Native -Exe $flutter.Source -Arguments @('test') -WorkingDirectory $AppDir -What 'flutter test'

# --- 8. build ----------------------------------------------------------

Write-Head 'Build the release APK'
if ($SplitPerAbi) {
    Invoke-Native -Exe $flutter.Source -Arguments @('build', 'apk', '--release', '--split-per-abi') -WorkingDirectory $AppDir -What 'flutter build (split)'
    $apk = Join-Path $AppDir ("build\app\outputs\flutter-apk\app-{0}-release.apk" -f $Abi)
}
else {
    Invoke-Native -Exe $flutter.Source -Arguments @('build', 'apk', '--release') -WorkingDirectory $AppDir -What 'flutter build'
    $apk = Join-Path $AppDir 'build\app\outputs\flutter-apk\app-release.apk'
}
if (-not (Test-Path $apk)) { throw "expected APK not found: $apk" }
$sizeMb = [math]::Round((Get-Item $apk).Length / 1MB, 1)
Write-Host ("  {0}  ({1} MB)" -f (Split-Path -Leaf $apk), $sizeMb) -ForegroundColor Green

# --- 9. install --------------------------------------------------------

Write-Head 'Install on the phone'
if ($SkipInstall) {
    Write-Skip 'install (-SkipInstall)'
    Write-Host ("  APK is at: {0}" -f $apk) -ForegroundColor DarkGray
}
else {
    $device = Get-AttachedDevice -Adb $adb
    if (-not $adb) {
        Write-Host '  ! adb not found; falling back to `flutter install`.' -ForegroundColor Yellow
        if ($SplitPerAbi) {
            throw 'flutter install cannot install a split APK -- it always picks app-release.apk. Install with adb, or re-run without -SplitPerAbi.'
        }
        Invoke-Native -Exe $flutter.Source -Arguments @('install', '--release') -WorkingDirectory $AppDir -What 'flutter install'
    }
    elseif (-not $device) {
        Write-Host '  ! No device detected -- skipping install.' -ForegroundColor Yellow
        Write-Host ("    Plug the phone in, then:  {0} install -r -d `"{1}`"" -f $adb, $apk) -ForegroundColor Yellow
    }
    else {
        # adb rather than `flutter install`, always, and by explicit path.
        # `flutter install` only ever installs app-release.apk, which
        # --split-per-abi does NOT refresh -- so it silently installs a stale
        # build. That cost real debugging time once.
        #
        # -r reinstalls keeping app data; -d allows the version-code downgrade you
        # hit going from a split APK (version code 2001) back to a fat one (1).
        Invoke-Native -Exe $adb -Arguments @('install', '-r', '-d', $apk) -What 'adb install'
        Write-Host ("  installed on {0}" -f $device) -ForegroundColor Green
    }
}

# --- 10. reportable metrics (last: not part of shipping) ---------------

if ($AppOnly -or $SkipEval) {
    Write-Head 'Signer-independent evaluation'
    Write-Skip 'train_eval'
    Write-Host '  results/metrics.json still describes the PREVIOUS model -- do not quote it.' -ForegroundColor Yellow
}
else {
    Write-Head 'Signer-independent evaluation (leave-one-signer-out)'
    Write-Host '  ~10-15 min. This is the number for your report; train_model''s "best val' -ForegroundColor DarkGray
    Write-Host '  accuracy" is NOT signer-independent. The app is already installed, so you' -ForegroundColor DarkGray
    Write-Host '  can test on the phone while this runs.' -ForegroundColor DarkGray
    Invoke-Native -Exe $Python -Arguments @('-u', 'scripts\train_eval.py', '--verbose', '2') -WorkingDirectory $RepoRoot -What 'train_eval'
}

# --- summary -----------------------------------------------------------

Write-Host ''
Write-Host ('=' * 74) -ForegroundColor Green
Write-Host '  DONE' -ForegroundColor Green
Write-Host ('=' * 74) -ForegroundColor Green

$script:Timings | Format-Table -AutoSize | Out-String | Write-Host

if (Test-Path (Join-Path $Results 'metrics.json')) {
    try {
        $m = Get-Content (Join-Path $Results 'metrics.json') -Raw | ConvertFrom-Json
        # Read from the file, NOT measured by this run -- it may predate the model
        # just built if -SkipEval/-AppOnly was used. The seq_len is printed so a
        # mismatch with the shipped model is visible at a glance.
        Write-Host ("  results/metrics.json says: LOSO mean {0:p2}, pooled {1:p2}, seq_len {2}" -f `
            $m.signer_independent.mean_accuracy, $m.signer_independent.pooled_accuracy, $m.config.seq_len) -ForegroundColor Cyan
        if ($SkipEval -or $AppOnly) {
            Write-Host '  (read from disk, not measured by this run -- check the seq_len matches)' -ForegroundColor DarkGray
        }
    }
    catch {
        Write-Host '  (could not parse metrics.json)' -ForegroundColor DarkGray
    }
}

Write-Host ''
Write-Host '  On the phone: open the app (offline mode is the default), record one sign.' -ForegroundColor White
Write-Host '  Healthy telemetry: offline+gate green, pose/hands green while signing,' -ForegroundColor DarkGray
Write-Host '  cam/conv/frames all climbing, fps >= 14, detect well under 62 ms.' -ForegroundColor DarkGray
Write-Host '  Whichever of cam/conv/frames stops climbing is the failing stage.' -ForegroundColor DarkGray
Write-Host ''
