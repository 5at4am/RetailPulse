param([string]$Html = "$PSScriptRoot\Retail_Pulse_AI_Methodology.html")

# Screenshot the HTML at A4-proportioned width so the layout can be eyeballed.
# The PDF itself cannot be visually inspected in this environment.
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$stage  = Join-Path $env:TEMP ("rpshot_" + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $stage -Force | Out-Null
$src = Join-Path $stage "doc.html"
Copy-Item -LiteralPath $Html -Destination $src -Force

$shots = @(
  @{ name = "page1"; h = 1100 },
  @{ name = "page2"; h = 2000 },
  @{ name = "page3"; h = 2000 },
  @{ name = "page4"; h = 2000 }
)

foreach ($s in $shots) {
  $png = Join-Path $stage "$($s.name).png"
  Start-Process -FilePath $chrome -ArgumentList @(
    "--headless=new","--disable-gpu","--no-sandbox","--disable-extensions",
    "--no-first-run","--hide-scrollbars",
    "--user-data-dir=$stage\ud",
    "--force-device-scale-factor=1.4",
    "--window-size=794,$($s.h)",
    "--screenshot=$png",
    "file:///$($src -replace '\\','/')"
  ) -Wait -NoNewWindow | Out-Null
  if (Test-Path $png) {
    Copy-Item -LiteralPath $png -Destination "$PSScriptRoot\$($s.name).png" -Force
    Write-Output ("{0}.png  {1:N0} KB" -f $s.name, ((Get-Item $png).Length/1KB))
  } else {
    Write-Output "$($s.name): screenshot failed"
  }
}
Remove-Item -LiteralPath $stage -Recurse -Force