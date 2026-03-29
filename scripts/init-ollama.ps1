<#
.SYNOPSIS
    Uni_Vision — Ollama Model Initializer (Windows)

.DESCRIPTION
    Pulls the Qwen 3.5 9B Q4_K_M base model, then creates three custom
    Modelfile variants: OCR, Adjudicator, and Navarasa (Indian contextualizer).

    The Navarasa model pulls its weights directly from HuggingFace GGUF
    (no separate base-model pull required).

    Prerequisites: Ollama must be installed and running locally.
        https://ollama.com/download

.EXAMPLE
    .\scripts\init-ollama.ps1
    .\scripts\init-ollama.ps1 -OllamaHost "http://localhost:11434"
#>
[CmdletBinding()]
param(
    [string]$OllamaHost = "http://localhost:11434",
    [string]$ModelfileDir = "$PSScriptRoot\..\config\ollama"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    Write-Host "[init-ollama] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[init-ollama] $Message" -ForegroundColor Green
}

function Write-Err {
    param([string]$Message)
    Write-Host "[init-ollama] ERROR: $Message" -ForegroundColor Red
}

# ── Wait for Ollama ───────────────────────────────────────────────
function Wait-ForOllama {
    Write-Log "Waiting for Ollama at $OllamaHost ..."
    $maxRetries = 30
    for ($i = 1; $i -le $maxRetries; $i++) {
        try {
            $null = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 5
            Write-Success "Ollama is ready."
            return
        } catch {
            Write-Log "  retry $i/$maxRetries ..."
            Start-Sleep -Seconds 5
        }
    }
    Write-Err "Ollama did not become ready within $($maxRetries * 5)s."
    exit 1
}

# ── Pull base model ──────────────────────────────────────────────
function Pull-BaseModel {
    $model = "qwen3.5:9b-q4_K_M"
    Write-Log "Pulling base model: $model (this may take several minutes on first run) ..."

    try {
        $body = @{ name = $model; stream = $false } | ConvertTo-Json
        $null = Invoke-RestMethod -Uri "$OllamaHost/api/pull" `
            -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1800
        Write-Success "Base model $model pulled successfully."
    } catch {
        Write-Err "Failed to pull $model : $_"
        exit 1
    }
}

# ── Create custom model ──────────────────────────────────────────
function New-CustomModel {
    param(
        [string]$Name,
        [string]$ModelfilePath
    )

    $resolvedPath = Resolve-Path $ModelfilePath -ErrorAction SilentlyContinue
    if (-not $resolvedPath) {
        Write-Err "Modelfile not found: $ModelfilePath"
        exit 1
    }

    Write-Log "Creating model: $Name from $resolvedPath ..."
    $content = Get-Content -Raw -Path $resolvedPath

    try {
        $body = @{ name = $Name; modelfile = $content; stream = $false } | ConvertTo-Json -Depth 5
        $null = Invoke-RestMethod -Uri "$OllamaHost/api/create" `
            -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
        Write-Success "Model $Name created successfully."
    } catch {
        Write-Err "Failed to create model ${Name}: $_"
        exit 1
    }
}

# ── Validate model ────────────────────────────────────────────────
function Test-Model {
    param([string]$Name)
    Write-Log "Validating model: $Name ..."
    try {
        $body = @{
            model = $Name
            prompt = "test"
            stream = $false
            options = @{ num_predict = 1 }
        } | ConvertTo-Json -Depth 3
        $null = Invoke-RestMethod -Uri "$OllamaHost/api/generate" `
            -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
        Write-Success "Model $Name is valid and loadable."
    } catch {
        Write-Log "WARNING: Validation for $Name returned an error (model may still be loading)."
    }
}

# ── List models ───────────────────────────────────────────────────
function Show-Models {
    Write-Log "Installed models:"
    try {
        $tags = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get
        foreach ($m in $tags.models) {
            $sizeGb = [math]::Round($m.size / 1GB, 1)
            Write-Host "  - $($m.name)  ($sizeGb GB)"
        }
    } catch {
        Write-Log "Could not list models."
    }
}

# ── Main ──────────────────────────────────────────────────────────
Write-Host ""
Write-Log "======================================================"
Write-Log "  Uni_Vision — Ollama Model Initialization"
Write-Log "======================================================"
Write-Host ""

Wait-ForOllama

# Step 1: Pull Qwen 3.5 9B Q4_K_M
Pull-BaseModel

# Step 2: Create OCR variant
New-CustomModel -Name "uni-vision-ocr" -ModelfilePath "$ModelfileDir\Modelfile.ocr"

# Step 3: Create Adjudicator variant
New-CustomModel -Name "uni-vision-adjudicator" -ModelfilePath "$ModelfileDir\Modelfile.adjudicator"

# Step 4: Create Navarasa Indian contextualizer (downloads GGUF from HuggingFace)
Write-Log "Navarasa 2.0 7B will be downloaded from HuggingFace on first creation — this may take a while."
New-CustomModel -Name "uni-vision-navarasa" -ModelfilePath "$ModelfileDir\Modelfile.navarasa"

# Step 5: Validate
Test-Model -Name "uni-vision-ocr"
Test-Model -Name "uni-vision-adjudicator"
Test-Model -Name "uni-vision-navarasa"

# Step 6: Show results
Show-Models

Write-Host ""
Write-Success "======================================================"
Write-Success "  Initialization complete."
Write-Success "======================================================"
Write-Host ""
