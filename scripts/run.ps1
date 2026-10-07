<#
.SYNOPSIS
    PowerShell script to start/stop the Milvus RAG stack via Docker or Podman on Windows.
.EXAMPLE
    .\scripts\run.ps1 -Engine docker -Action up
    .\scripts\run.ps1 -Engine podman -Action up
    .\scripts\run.ps1 -Action down
#>
param (
    [ValidateSet("docker", "podman")]
    [string]$Engine = "docker",

    [ValidateSet("up", "down", "logs", "restart")]
    [string]$Action = "up"
)

$RootDir = Split-Path -Parent $PSScriptRoot

# Create data & volume folders
$Folders = @(
    "volumes/etcd",
    "volumes/minio",
    "volumes/milvus",
    "volumes/model_cache",
    "data/raw",
    "data/processed",
    "data/embeddings",
    "data/eval"
)

foreach ($folder in $Folders) {
    $fullPath = Join-Path $RootDir $folder
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

# Ensure .env exists
$envFile = Join-Path $RootDir ".env"
$envExample = Join-Path $RootDir ".env.example"
if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
    Copy-Item $envExample $envFile
    Write-Host ">>> Created .env from .env.example" -ForegroundColor Green
}

Set-Location $RootDir

if ($Engine -eq "docker") {
    $composeFile = "docker-compose.yml"
    if ($Action -eq "up") {
        Write-Host ">>> Starting Milvus RAG stack via Docker Compose..." -ForegroundColor Cyan
        docker compose -f $composeFile up -d
        Write-Host ">>> Services launched!" -ForegroundColor Green
        Write-Host "    - Milvus Standalone: localhost:19530"
        Write-Host "    - Attu Web UI:       http://localhost:3000"
        Write-Host "    - Streamlit UI:      http://localhost:8502"
        Write-Host "    - RAG FastAPI Docs:  http://localhost:8000/docs"
    } elseif ($Action -eq "down") {
        Write-Host ">>> Stopping Milvus RAG stack via Docker Compose..." -ForegroundColor Yellow
        docker compose -f $composeFile down
    } elseif ($Action -eq "logs") {
        docker compose -f $composeFile logs -f
    } elseif ($Action -eq "restart") {
        docker compose -f $composeFile restart
    }
} else {
    $composeFile = "compose.yaml"
    if ($Action -eq "up") {
        Write-Host ">>> Starting Milvus RAG stack via Podman..." -ForegroundColor Cyan
        podman compose -f $composeFile up -d
    } elseif ($Action -eq "down") {
        podman compose -f $composeFile down
    } elseif ($Action -eq "logs") {
        podman compose -f $composeFile logs -f
    } elseif ($Action -eq "restart") {
        podman compose -f $composeFile restart
    }
}
