<#
.SYNOPSIS
Build the Android APK with the server URL and API key compiled in.

.DESCRIPTION
Reads TUNNEL_HOSTNAME and NSL_API_KEY from .env and passes them to Flutter as
--dart-define values, so a freshly installed APK connects with no setup screen.

Use this with a NAMED tunnel only. A quick tunnel's hostname is regenerated on
every restart, so a baked-in one is stale the moment you restart the stack.

Release by default: the frame encoder is pure Dart (YUV -> RGB -> JPEG) and is
the client-side bottleneck. Debug Dart is far slower, and the model reads frame
count as duration, so a slow client measurably costs accuracy.

ASCII only on purpose - Windows PowerShell 5.1 reads .ps1 as ANSI, and a stray
em dash becomes a parse error.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1 -Install

.EXAMPLE
# Override without touching .env (e.g. a LAN address for one build)
powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1 -Url http://192.168.1.15:8000
#>
param(
    [string]$Url,
    [string]$ApiKey,
    [switch]$DebugBuild,
    [switch]$Install,
    [switch]$SplitPerAbi,
    [string]$Device
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repo '.env'

# --- read .env (KEY=VALUE, ignoring comments and blanks) --------------------
$settings = @{}
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -lt 1) { continue }
        $settings[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim()
    }
} else {
    Write-Warning "No .env at $envFile - copy .env.example to .env first."
}

if (-not $Url) {
    $tunnelHost = $settings['TUNNEL_HOSTNAME']
    if ($tunnelHost) {
        # Tolerate someone pasting a full URL into TUNNEL_HOSTNAME.
        $tunnelHost = $tunnelHost -replace '^https?://', '' -replace '/+$', ''
        $Url = "https://$tunnelHost"
    }
}
if (-not $ApiKey) { $ApiKey = $settings['NSL_API_KEY'] }

if (-not $Url) {
    Write-Host ''
    Write-Host 'No server URL.'
    Write-Host 'Set TUNNEL_HOSTNAME in .env (e.g. nsl.yourdomain.com), or pass'
    Write-Host '-Url http://<laptop-ip>:8000 for a LAN build.'
    Write-Host ''
    exit 1
}
if (-not $ApiKey) {
    Write-Warning 'NSL_API_KEY is empty - the app will connect without a key.'
}

$mode = '--release'
if ($DebugBuild) { $mode = '--debug' }

Write-Host ''
Write-Host "  server   $Url"
if ($ApiKey) {
    Write-Host "  api key  set ($($ApiKey.Length) chars)"
} else {
    Write-Host '  api key  (none)'
}
Write-Host "  mode     $mode"
Write-Host ''

Push-Location (Join-Path $repo 'app')
try {
    $buildArgs = @(
        'build', 'apk', $mode,
        "--dart-define=NSL_SERVER_URL=$Url",
        "--dart-define=NSL_API_KEY=$ApiKey"
    )
    # One APK per CPU architecture instead of a fat one carrying all three.
    # The arm64 slice is roughly a third the size, which matters when the APK
    # has to be sent to the phone over a chat app rather than a cable.
    if ($SplitPerAbi) { $buildArgs += '--split-per-abi' }
    & flutter @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "flutter build failed ($LASTEXITCODE)" }

    if ($Install) {
        # NOTE: this uninstalls the existing app first, clearing SharedPreferences.
        # That is what makes the baked-in URL take effect over any saved override.
        $installArgs = @('install', $mode)
        if ($Device) { $installArgs += @('-d', $Device) }
        & flutter @installArgs
        if ($LASTEXITCODE -ne 0) { throw "flutter install failed ($LASTEXITCODE)" }
    }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Done. The app opens straight to the camera, with no setup screen.'
Write-Host ''
$apkDir = Join-Path $repo 'app\build\app\outputs\flutter-apk'
Get-ChildItem -Path $apkDir -Filter '*release*.apk' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { "  {0,-40} {1,6:N1} MB" -f $_.Name, ($_.Length / 1MB) }
