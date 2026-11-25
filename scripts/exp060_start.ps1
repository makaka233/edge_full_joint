param(
    [ValidateRange(1, 6)][int]$Workers = 4,
    [switch]$Third
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$results = Join-Path $root 'runs/results/EXP060'
$stage = if ($Third) { 'third' } else { 'initial' }
$record = Join-Path $results "launch.$stage.json"
$complete = Join-Path $results "$stage-complete.json"
if (Test-Path -LiteralPath $complete) { throw "EXP060 stage already complete: $complete" }
if (!(Test-Path -LiteralPath (Join-Path $results 'smoke.json'))) { throw 'Run the registered EXP060 smoke first.' }
if ($Third -and !(Test-Path -LiteralPath (Join-Path $results 'summary.initial.json'))) {
    throw 'Generate and check the registered initial summary before starting Seed14.'
}
if (Test-Path -LiteralPath $record) {
    $prior = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $previousProcess = Get-Process -Id $prior.pid -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.StartTime.ToUniversalTime().Ticks -eq $prior.startTicks) {
        throw "EXP060 stage already running under PID $($prior.pid): $($prior.stdout)"
    }
}

$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $results "unattended.$stage.$stamp.log"
$stderr = Join-Path $results "unattended.$stage.$stamp.stderr.log"
$python = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$argsForPython = @('-u','-m','scripts.exp060_run','unattended','--workers',$Workers)
if ($Third) { $argsForPython += '--third' }
$launched = Start-Process -FilePath $python -ArgumentList $argsForPython -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$info = [ordered]@{
    stage = $stage
    pid = $launched.Id
    start = $launched.StartTime.ToString('o')
    startTicks = $launched.StartTime.ToUniversalTime().Ticks
    workers = $Workers
    stdout = $stdout
    stderr = $stderr
    completionFile = $complete
}
$info | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding UTF8
$info | ConvertTo-Json
