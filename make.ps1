<#
.SYNOPSIS
    Build script for relaynet2 thesis experiments (replaces Makefile).
.EXAMPLE
    .\make.ps1 list
    .\make.ps1 quick
    .\make.ps1 exp -Section 7.17
    .\make.ps1 clean
#>
param(
    [Parameter(Position=0)]
    [ValidateSet("help","list","quick","full","full-cpu","charts","exp","exp-quick",
                 "retrain","inference","test","test-quick",
                 "clean","clean-results","clean-weights","clean-logs","clean-pycache")]
    [string]$Target = "help",

    [Alias("s")]
    [string]$Section
)

$ErrorActionPreference = "Stop"
$Python = "python"
$Runner = "run_experiments.py"
$Seed   = 42

switch ($Target) {
    "help" {
        Write-Host ""
        Write-Host "  relaynet2 Experiment Runner" -ForegroundColor Cyan
        Write-Host "  ===========================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  .\make.ps1 list             List available experiments"
        Write-Host "  .\make.ps1 quick            Run all experiments (quick mode)"
        Write-Host "  .\make.ps1 full             Run all experiments (full, GPU)"
        Write-Host "  .\make.ps1 full-cpu         Run all experiments (full, CPU)"
        Write-Host "  .\make.ps1 charts           Regenerate charts from JSON"
        Write-Host "  .\make.ps1 exp -s 7.17      Run specific experiment(s)"
        Write-Host "  .\make.ps1 exp-quick -s 7.17  Run specific experiment(s) quick"
        Write-Host "  .\make.ps1 retrain -s 7.17  Force retrain experiment(s)"
        Write-Host "  .\make.ps1 inference        Inference only (load weights)"
        Write-Host "  .\make.ps1 test             Run pytest suite"
        Write-Host "  .\make.ps1 test-quick       Run pytest (fail-fast)"
        Write-Host "  .\make.ps1 clean            Clean all outputs"
        Write-Host "  .\make.ps1 clean-results    Clean results only"
        Write-Host "  .\make.ps1 clean-weights    Clean weights only"
        Write-Host "  .\make.ps1 clean-logs       Clean logs only"
        Write-Host "  .\make.ps1 clean-pycache    Clean __pycache__"
        Write-Host ""
    }
    "list" {
        & $Python $Runner --list
    }
    "quick" {
        & $Python $Runner --all --quick --seed $Seed
    }
    "full" {
        & $Python $Runner --all --gpu --seed $Seed
    }
    "full-cpu" {
        & $Python $Runner --all --seed $Seed
    }
    "charts" {
        & $Python $Runner --regen-charts
    }
    "exp" {
        if (-not $Section) { Write-Error "Usage: .\make.ps1 exp -s 7.17"; return }
        $sections = $Section -split '\s+'
        & $Python $Runner --exp @sections --gpu --seed $Seed
    }
    "exp-quick" {
        if (-not $Section) { Write-Error "Usage: .\make.ps1 exp-quick -s 7.17"; return }
        $sections = $Section -split '\s+'
        & $Python $Runner --exp @sections --quick --seed $Seed
    }
    "retrain" {
        if (-not $Section) { Write-Error "Usage: .\make.ps1 retrain -s 7.17"; return }
        $sections = $Section -split '\s+'
        & $Python $Runner --exp @sections --retrain --gpu --seed $Seed
    }
    "inference" {
        & $Python $Runner --all --inference-only --seed $Seed
    }
    "test" {
        & $Python -m pytest tests/ -v --tb=short
    }
    "test-quick" {
        & $Python -m pytest tests/ -v --tb=short -x -q
    }
    "clean" {
        & $PSCommandPath clean-results
        & $PSCommandPath clean-weights
        & $PSCommandPath clean-logs
        Write-Host "All outputs cleaned." -ForegroundColor Green
    }
    "clean-results" {
        $dirs = @(
            "results\bpsk_comparison", "results\normalized_3k", "results\modulation",
            "results\qam16_activation", "results\layernorm", "results\activation_comparison",
            "results\csi", "results\e2e", "results\all_relays_16class", "results\channel_analysis"
        )
        foreach ($d in $dirs) {
            if (Test-Path $d) { Remove-Item $d -Recurse -Force; Write-Host "  Removed $d" }
        }
        if (Test-Path "results\master_ber_comparison.png") {
            Remove-Item "results\master_ber_comparison.png" -Force
        }
        Write-Host "Results cleaned." -ForegroundColor Green
    }
    "clean-weights" {
        if (Test-Path "weights") { Remove-Item "weights" -Recurse -Force }
        Write-Host "Weights cleaned." -ForegroundColor Green
    }
    "clean-logs" {
        if (Test-Path "results\logs") { Remove-Item "results\logs" -Recurse -Force }
        Write-Host "Logs cleaned." -ForegroundColor Green
    }
    "clean-pycache" {
        Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" |
            Remove-Item -Recurse -Force
        Write-Host "Cache cleaned." -ForegroundColor Green
    }
}
