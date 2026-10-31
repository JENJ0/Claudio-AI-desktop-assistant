<#
.SYNOPSIS
    Start CLAUDIO at Windows login (Task Scheduler), or remove that start.

.DESCRIPTION
    Registers the task "Claudio" for the current user: 30 s after login it runs
    .venv\Scripts\pythonw.exe -m claudio.supervisor (no console window) from the project
    folder. The supervisor starts CLAUDIO and starts it again after a crash (Task Scheduler's
    own restart only covers a program that cannot start). No admin rights needed. Running it
    again replaces the task; -Uninstall twice is fine.
    Same task as `python -m claudio.autostart install` (src/claudio/autostart.py).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$DryRun  # show what would happen, change nothing
)

$ErrorActionPreference = 'Stop'
$TaskName = 'Claudio'
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project '.venv\Scripts\pythonw.exe'
$User = "$env:USERDOMAIN\$env:USERNAME"

if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        Write-Host "Task '$TaskName' is not installed: nothing to remove."
        exit 0
    }
    if ($DryRun) {
        Write-Host "Would remove the task '$TaskName'."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Task '$TaskName' removed: CLAUDIO no longer starts at login."
    exit 0
}

# Fail before registering: a task pointing nowhere starts and dies silently.
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Error "$Python not found: run 'uv sync' in $Project first."
}
if (-not (Test-Path -LiteralPath (Join-Path $Project 'src\claudio') -PathType Container)) {
    Write-Error "$Project is not the CLAUDIO project folder."
}

$action = New-ScheduledTaskAction -Execute $Python -Argument '-m claudio.supervisor' `
    -WorkingDirectory $Project
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$trigger.Delay = 'PT30S'  # the audio device is often not ready at the first second
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

if ($DryRun) {
    Write-Host "Would register '$TaskName' for $User :"
    Write-Host "  run     : $Python -m claudio.supervisor"
    Write-Host "  in      : $Project"
    Write-Host "  when    : at login, after 30 s; after a crash the supervisor starts it again"
    exit 0
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description 'CLAUDIO voice assistant, started at login.' -Force | Out-Null
Write-Host "Task '$TaskName' registered: CLAUDIO starts 30 s after you log in."
