<#
.SYNOPSIS
Health-check the named Cloudflare tunnel end to end.

.DESCRIPTION
The named-tunnel equivalent of tunnel_url.sh (which exists to *discover* a quick
tunnel's random URL). Here the hostname is already known, so the only question is
whether the whole path works: cloudflared connected, and /health answering
through it.

Checks the local server first, so a failure tells you which half is broken.

ASCII only on purpose - Windows PowerShell 5.1 reads .ps1 as ANSI.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\check_tunnel.ps1
#>
param([string]$TunnelHostname)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot

if (-not $TunnelHostname) {
    $envFile = Join-Path $repo '.env'
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            $t = $line.Trim()
            if ($t -match '^TUNNEL_HOSTNAME\s*=\s*(.+)$') { $TunnelHostname = $Matches[1].Trim() }
        }
    }
}
if (-not $TunnelHostname) {
    Write-Host 'No TUNNEL_HOSTNAME in .env, and none passed. See .env.example.'
    exit 1
}
$TunnelHostname = $TunnelHostname -replace '^https?://', '' -replace '/+$', ''

Write-Host ''
Write-Host '1. containers'
$ps = & docker compose -f (Join-Path $repo 'docker-compose.yml') ps --format '{{.Name}} {{.Status}}'
if ($ps) { $ps | ForEach-Object { "   $_" } } else { Write-Host '   nothing running' }

Write-Host ''
Write-Host '2. server on localhost'
try {
    $local = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 10
    Write-Host "   ok - backend=$($local.backend) seq_len=$($local.seq_len) auth=$($local.auth_required)"
} catch {
    Write-Host "   FAILED - $($_.Exception.Message)"
    Write-Host '   The server itself is down; the tunnel cannot help.'
    Write-Host '   docker compose logs server'
    exit 1
}

Write-Host ''
Write-Host "3. through the tunnel  https://$TunnelHostname"
try {
    $remote = Invoke-RestMethod -Uri "https://$TunnelHostname/health" -TimeoutSec 25
    Write-Host "   ok - backend=$($remote.backend) seq_len=$($remote.seq_len)"
} catch {
    Write-Host "   FAILED - $($_.Exception.Message)"
    Write-Host ''
    Write-Host '   Server is healthy locally, so this is the tunnel. Check that:'
    Write-Host '     - you started it with:  docker compose --profile named up -d'
    Write-Host '     - the public hostname points at http://server:8000, not localhost'
    Write-Host '     - docker compose logs tunnel-named   shows "Registered tunnel connection"'
    exit 1
}

Write-Host ''
Write-Host 'Good. Build the app against it:  scripts\build_app.ps1 -Install'
Write-Host ''
