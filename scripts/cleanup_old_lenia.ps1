\
# cleanup_old_lenia.ps1
# Deletes common leftover files from older LeniaDynamics versions in the current folder.
# Run from inside the LeniaDynamics directory:
#   powershell -ExecutionPolicy Bypass -File .\scripts\cleanup_old_lenia.ps1

$old = @(
  ".\src\cuda_optional.jl"
)

foreach ($p in $old) {
  if (Test-Path $p) {
    Write-Host "Removing $p"
    Remove-Item -Force $p
  }
}

Write-Host "Done. If you still see old-files issues, delete the whole directory and unpack fresh."
