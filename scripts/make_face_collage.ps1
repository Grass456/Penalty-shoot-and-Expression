param(
    [string]$InputDir = "data/faces",
    [string]$OutputPath = "outputs/face_crops_collage.png",
    [int]$SamplesPerSubject = 1,
    [int]$Seed = 2026
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$inputRoot = (Resolve-Path $InputDir).Path
$outputFile = [System.IO.Path]::GetFullPath($OutputPath)
$outputDir = [System.IO.Path]::GetDirectoryName($outputFile)
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$subjects = Get-ChildItem -Path $inputRoot -Directory | Sort-Object Name
$random = [System.Random]::new($Seed)
$selected = [System.Collections.Generic.List[System.IO.FileInfo]]::new()

foreach ($subject in $subjects) {
    $crops = @(Get-ChildItem -Path $subject.FullName -File |
        Where-Object { $_.Extension -in ".jpg", ".jpeg", ".png" -and $_.BaseName -notlike "*_FULLFRAME" } |
        Sort-Object Name)
    if ($crops.Count -eq 0) { continue }

    $take = [Math]::Min($SamplesPerSubject, $crops.Count)
    $available = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
    foreach ($crop in $crops) { $available.Add($crop) }
    for ($i = 0; $i -lt $take; $i++) {
        $index = $random.Next(0, $available.Count)
        $selected.Add($available[$index])
        $available.RemoveAt($index)
    }
}

if ($selected.Count -eq 0) {
    throw "No face crops found under $inputRoot"
}

$columns = 8
$rows = [Math]::Ceiling($selected.Count / [double]$columns)
$tile = 180
$gap = 10
$marginX = 36
$topBand = 82
$bottomBand = 34
$canvasWidth = $marginX * 2 + $columns * $tile + ($columns - 1) * $gap
$canvasHeight = $topBand + $rows * $tile + ($rows - 1) * $gap + $bottomBand

$canvas = [System.Drawing.Bitmap]::new($canvasWidth, $canvasHeight, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$graphics = [System.Drawing.Graphics]::FromImage($canvas)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
$graphics.Clear([System.Drawing.Color]::FromArgb(247, 248, 250))

$titleFont = [System.Drawing.Font]::new("Arial", 22, [System.Drawing.FontStyle]::Bold)
$subtitleFont = [System.Drawing.Font]::new("Arial", 10, [System.Drawing.FontStyle]::Regular)
$titleBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(28, 35, 46))
$subtitleBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(94, 105, 120))
$borderPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(218, 222, 229), 1)

$graphics.DrawString("Face crops", $titleFont, $titleBrush, $marginX, 14)
$graphics.DrawString("Representative samples from $($selected.Count) subjects", $subtitleFont, $subtitleBrush, $marginX, 48)

for ($i = 0; $i -lt $selected.Count; $i++) {
    $column = $i % $columns
    $row = [Math]::Floor($i / $columns)
    $x = $marginX + $column * ($tile + $gap)
    $y = $topBand + $row * ($tile + $gap)
    $destination = [System.Drawing.Rectangle]::new($x, $y, $tile, $tile)

    $source = $null
    try {
        $source = [System.Drawing.Image]::FromFile($selected[$i].FullName)
        $side = [Math]::Min($source.Width, $source.Height)
        $sourceX = [Math]::Floor(($source.Width - $side) / 2)
        $sourceY = [Math]::Floor(($source.Height - $side) / 2)
        $sourceRect = [System.Drawing.Rectangle]::new($sourceX, $sourceY, $side, $side)
        $graphics.DrawImage($source, $destination, $sourceRect, [System.Drawing.GraphicsUnit]::Pixel)
    }
    finally {
        if ($null -ne $source) { $source.Dispose() }
    }
    $graphics.DrawRectangle($borderPen, $destination)
}

$footerY = $canvasHeight - $bottomBand + 10
$graphics.DrawString("One crop per subject • seed $Seed", $subtitleFont, $subtitleBrush, $marginX, $footerY)

$canvas.Save($outputFile, [System.Drawing.Imaging.ImageFormat]::Png)

$borderPen.Dispose()
$titleBrush.Dispose()
$subtitleBrush.Dispose()
$titleFont.Dispose()
$subtitleFont.Dispose()
$graphics.Dispose()
$canvas.Dispose()

Write-Host "Created $outputFile with $($selected.Count) face crops."
