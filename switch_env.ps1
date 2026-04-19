param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("dev", "prod")]
    [string]$Profile,

    [switch]$Force,
    [switch]$List
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envDev = Join-Path $repoRoot ".env.dev"
$envProd = Join-Path $repoRoot ".env.prod"
$targetEnv = Join-Path $repoRoot ".env"

if ($List) {
    Write-Host "Profils disponibles:" -ForegroundColor Cyan
    if (Test-Path $envDev) { Write-Host "- dev  (.env.dev)" }
    if (Test-Path $envProd) { Write-Host "- prod (.env.prod)" }
    Write-Host ""
    Write-Host "Usage: .\switch_env.ps1 -Profile dev|prod [-Force]"
    exit 0
}

if (-not $Profile) {
    Write-Host "Usage: .\switch_env.ps1 -Profile dev|prod [-Force]" -ForegroundColor Yellow
    Write-Host "       .\switch_env.ps1 -List"
    exit 1
}

$sourceEnv = switch ($Profile) {
    "dev" { $envDev }
    "prod" { $envProd }
}

if (-not (Test-Path $sourceEnv)) {
    Write-Error "Profil introuvable: $sourceEnv"
    exit 1
}

if ((Test-Path $targetEnv) -and (-not $Force)) {
    $confirmation = Read-Host ".env existe déjà. Écraser avec le profil '$Profile' ? (y/N)"
    if ($confirmation -notin @("y", "Y", "yes", "YES")) {
        Write-Host "Opération annulée." -ForegroundColor Yellow
        exit 0
    }
}

Copy-Item $sourceEnv $targetEnv -Force
Write-Host "Profil '$Profile' appliqué dans .env" -ForegroundColor Green
Write-Host "Source: $sourceEnv"
Write-Host "Cible : $targetEnv"
