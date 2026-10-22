# Baixa as tablebases Syzygy de vitória/empate/derrota (.rtbw) até 5 peças
# para a pasta syzygy/ ao lado deste script. Uso, no PowerShell:
#   powershell -ExecutionPolicy Bypass -File baixar_syzygy.ps1
# Arquivos já baixados são pulados, então dá para rodar de novo se cair.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # a barra de progresso deixa o download muito lento
$destino = Join-Path $PSScriptRoot 'syzygy'
New-Item -ItemType Directory -Force $destino | Out-Null

function Links($url) {
    $r = Invoke-WebRequest $url -UseBasicParsing
    $r.Links.href | ForEach-Object { [Uri]::new([Uri]$url, $_).AbsoluteUri }
}

# O lichess separa as pastas em .../3-4-5-wdl/ (ou .../3-4-5/); o espelho
# da sesse.net fica de reserva.
$fontes = @('https://tablebase.lichess.ovh/tables/standard/', 'http://tablebase.sesse.net/syzygy/')
$arquivos = @()
foreach ($fonte in $fontes) {
    try {
        $pastas = Links $fonte | Where-Object { $_ -match '/3-4-5[^/]*/$' -and $_ -notmatch 'dtz' }
        foreach ($p in $pastas) { $arquivos += Links $p | Where-Object { $_ -like '*.rtbw' } }
        if ($arquivos.Count -gt 0) { Write-Host "Fonte: $fonte ($($pastas -join ', '))"; break }
        Write-Host "Nenhum .rtbw em $fonte. Pastas encontradas:"; Links $fonte | Write-Host
    } catch {
        Write-Host "Falhou em ${fonte}: $($_.Exception.Message)"
    }
}
if ($arquivos.Count -eq 0) { throw 'Não achei as tabelas. Mande esta saída inteira.' }

$arquivos = $arquivos | Sort-Object -Unique
$i = 0
foreach ($a in $arquivos) {
    $i++
    $nome = [IO.Path]::GetFileName(([Uri]$a).LocalPath)
    $alvo = Join-Path $destino $nome
    if (Test-Path $alvo) { continue }
    Write-Host "[$i/$($arquivos.Count)] $nome"
    Invoke-WebRequest $a -OutFile "$alvo.parcial" -UseBasicParsing
    Move-Item "$alvo.parcial" $alvo
}
Write-Host "Pronto: $($arquivos.Count) arquivos em $destino"
