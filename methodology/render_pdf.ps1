param(
  [string]$Html = "$PSScriptRoot\Retail_Pulse_AI_Methodology.html",
  [string]$Pdf  = "$PSScriptRoot\Retail_Pulse_AI_Methodology.pdf"
)

# Render in a space-free temp dir. The project path contains a space ("OpenWork Chat"),
# which Windows argument quoting mangles for both Start-Process and headless Chrome.
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$stage  = Join-Path $env:TEMP ("rprender_" + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$srcHtml = Join-Path $stage "doc.html"
$outPdf  = Join-Path $stage "doc.pdf"
Copy-Item -LiteralPath $Html -Destination $srcHtml -Force

$err = Join-Path $stage "err.txt"
$proc = Start-Process -FilePath $chrome -ArgumentList @(
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--disable-extensions",
  "--no-first-run",
  "--no-pdf-header-footer",
  "--print-to-pdf=$outPdf",
  "file:///$($srcHtml -replace '\\','/')"
) -PassThru -Wait -NoNewWindow -RedirectStandardError $err

$fail = Get-Content $err -ErrorAction SilentlyContinue | Select-String "ERROR" | Select-Object -First 1
if (-not (Test-Path $outPdf)) {
  if ($fail) { Write-Output "chrome: $($fail.Line)" }
  Write-Output "FAILED (exit $($proc.ExitCode)) - no pdf produced"
  Remove-Item -LiteralPath $stage -Recurse -Force
  exit 1
}

Copy-Item -LiteralPath $outPdf -Destination $Pdf -Force
$kb = [math]::Round((Get-Item $Pdf).Length / 1KB, 1)
Remove-Item -LiteralPath $stage -Recurse -Force
Write-Output ("rendered -> {0}  ({1} KB, exit {2})" -f $Pdf, $kb, $proc.ExitCode)