param(
  [Parameter(ValueFromRemainingArguments=$true)][string[]] $Arguments
)
python -m edge_d.cli @Arguments
exit $LASTEXITCODE
