# deploy.ps1 — Level 1 Manual Deployment Script
#
# This script exemplifies the ad-hoc, fragile deployment process
# typical of Level 1 maturity. Each step requires manual verification
# and there are no safety nets.
#
# Usage: .\scripts\deploy.ps1 -ModelPath ".\models\churn_model_2026-06-20.pkl" -Server "prod-serve-01"

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [Parameter(Mandatory=$true)]
    [string]$Server,

    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/home/model-serving/models/"
)

# === STEP 1: Verify model exists ===
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found at $ModelPath"
    exit 1
}

Write-Host "=== LEVEL 1 MANUAL DEPLOYMENT ===" -ForegroundColor Yellow
Write-Host "Model: $ModelPath"
Write-Host "Server: $Server"
Write-Host ""

# === STEP 2: SCP the model to production server ===
Write-Host "[1/3] Copying model to $Server..." -ForegroundColor Cyan
# NOTE: Requires SSH keys to be set up manually.
#       No inventory of which servers have which keys.
try {
    scp -q $ModelPath "${Server}:${RemotePath}"
    Write-Host "      Done." -ForegroundColor Green
} catch {
    Write-Error "SCP failed. Check SSH keys and server connectivity."
    exit 1
}

# === STEP 3: SSH in and restart the model service ===
Write-Host "[2/3] Restarting model service on $Server..." -ForegroundColor Cyan
try {
    ssh $Server "sudo systemctl restart model-serving"
    Write-Host "      Service restarted." -ForegroundColor Green
} catch {
    Write-Error "SSH command failed. Service may be down."
    exit 1
}

# === STEP 4: Quick smoke test ===
Write-Host "[3/3] Running smoke test via SSH..." -ForegroundColor Cyan
try {
    $result = ssh $Server "curl -s -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{\"features\": [35, 12, 75, 906, 0, 0, 1, 0, 45, 3]}'"
    Write-Host "      Response: $result" -ForegroundColor Green
} catch {
    Write-Warning "Smoke test failed. Check service manually."
}

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Yellow
Write-Host "WARNING: No rollback was configured. Keep the old .pkl file handy."
Write-Host "WARNING: No monitoring is in place. Check logs manually."
