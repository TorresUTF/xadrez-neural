# Baixa as tablebases Syzygy de vitória/empate/derrota (.rtbw) até 5 peças
# para a pasta syzygy/ ao lado deste script. Uso, no PowerShell:
#   powershell -ExecutionPolicy Bypass -File baixar_syzygy.ps1
# Com -Dtz, baixa as tabelas de distância (.rtbz, ~0,6 GB), que o comando
# "progresso" usa:
#   powershell -ExecutionPolicy Bypass -File baixar_syzygy.ps1 -Dtz
# Arquivos já baixados são pulados, então dá para rodar de novo se cair.
param([switch]$Dtz)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # a barra de progresso deixa o download muito lento
$destino = Join-Path $PSScriptRoot 'syzygy'
New-Item -ItemType Directory -Force $destino | Out-Null

function Links($url) {
    $r = Invoke-WebRequest $url -UseBasicParsing
    $r.Links.href | ForEach-Object { [Uri]::new([Uri]$url, $_).AbsoluteUri }
}

# O lichess separa as pastas em .../3-4-5-wdl/ e .../3-4-5-dtz/ (ou junta
# tudo em .../3-4-5/); o espelho da sesse.net fica de reserva.
$extensao = if ($Dtz) { '*.rtbz' } else { '*.rtbw' }
$fontes = @('https://tablebase.lichess.ovh/tables/standard/', 'http://tablebase.sesse.net/syzygy/')
$arquivos = @()
foreach ($fonte in $fontes) {
    try {
        $outra = if ($Dtz) { 'wdl' } else { 'dtz' }
        $pastas = Links $fonte | Where-Object { $_ -match '/3-4-5[^/]*/$' -and $_ -notmatch $outra }
        foreach ($p in $pastas) { $arquivos += Links $p | Where-Object { $_ -like $extensao } }
        if ($arquivos.Count -gt 0) { Write-Host "Fonte: $fonte ($($pastas -join ', '))"; break }
        Write-Host "Nenhum $extensao em $fonte. Pastas encontradas:"; Links $fonte | Write-Host
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
