param([Parameter(Mandatory=$true)][ValidateSet("train","s-only","eval","d120","plot")][string]$Mode,[Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs)
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
$Parts=@($Root,"$Root\r12","$Root\r9","$Root\r41","$Root\audit","$Root\baseline\edge_round31\source")
$env:PYTHONPATH=($Parts -join [IO.Path]::PathSeparator)
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
$Entrypoints=@{ "train"="joint.py"; "s-only"="s_only.py"; "eval"="eval_min.py"; "d120"="d120_min.py"; "plot"="plot_min.py" }
Push-Location "$Root\baseline\edge_round31\source"
try { & python "$Root\r12\$($Entrypoints[$Mode])" @ExtraArgs; exit $LASTEXITCODE } finally { Pop-Location }
