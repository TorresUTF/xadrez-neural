// =====================================================================
// EXTERNO — conversar com outro motor de xadrez pelo protocolo UCI
// =====================================================================
//
// Etapa 4: medir a nossa rede contra adversários de verdade, como o
// Maia-1100 (uma rede treinada para jogar como jogadores de ~1100 no
// Lichess, que roda dentro do motor lc0). Qualquer motor UCI serve.
//
// O protocolo é texto, linha por linha, pela entrada e saída padrão:
//     nós: uci                      motor: ... uciok
//     nós: setoption name X value Y
//     nós: isready                  motor: readyok
//     nós: position startpos moves e2e4 e7e5
//     nós: go nodes 1               motor: ... bestmove g1f3
//
// No torneio, o jogador externo é escrito assim:
//     "uci=<comando com argumentos>;go=<parâmetros do go>;Opcao=valor"
// por exemplo:
//     "uci=lc0\lc0.exe --weights=lc0\maia-1100.pb.gz;go=nodes 1"
// (o comando é separado por espaços: o caminho não pode ter espaços)
//
// Cada thread do torneio abre o seu próprio processo do motor e o usa
// em todas as partidas que joga.
// =====================================================================

use shakmaty::{uci::UciMove, CastlingMode, Chess, Move};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Clone, Debug, PartialEq)]
pub struct Externo {
    pub comando: Vec<String>,
    pub go: String,
    pub opcoes: Vec<(String, String)>,
}

// "uci=<comando>;go=nodes 1;Threads=1"  (sem o "uci=")
pub fn ler(texto: &str) -> Option<Externo> {
    let mut partes = texto.split(';');
    let comando: Vec<String> = partes.next()?.split_whitespace().map(String::from).collect();
    if comando.is_empty() {
        return None;
    }
    let mut go = "nodes 1".to_string();
    let mut opcoes = Vec::new();
    for p in partes {
        let (chave, valor) = p.split_once('=')?;
        if chave == "go" {
            go = valor.to_string();
        } else {
            opcoes.push((chave.to_string(), valor.to_string()));
        }
    }
    Some(Externo { comando, go, opcoes })
}

struct Motor {
    filho: Child,
    entrada: ChildStdin,
    saida: BufReader<ChildStdout>,
}

impl Motor {
    fn abrir(cfg: &Externo) -> Motor {
        let mut filho = Command::new(&cfg.comando[0])
            .args(&cfg.comando[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("não consegui abrir o motor \"{}\": {e}", cfg.comando[0]));
        let entrada = filho.stdin.take().unwrap();
        let saida = BufReader::new(filho.stdout.take().unwrap());
        let mut m = Motor { filho, entrada, saida };
        m.enviar("uci");
        m.esperar("uciok");
        for (chave, valor) in &cfg.opcoes {
            m.enviar(&format!("setoption name {chave} value {valor}"));
        }
        m.enviar("isready");
        m.esperar("readyok");
        m
    }

    fn enviar(&mut self, linha: &str) {
        writeln!(self.entrada, "{linha}").and_then(|_| self.entrada.flush()).expect("o motor externo fechou");
    }

    // lê linhas até uma que comece com `prefixo`, e devolve essa linha
    fn esperar(&mut self, prefixo: &str) -> String {
        let mut linha = String::new();
        loop {
            linha.clear();
            let lidos = self.saida.read_line(&mut linha).expect("erro lendo o motor externo");
            if lidos == 0 {
                panic!("o motor externo fechou antes de responder \"{prefixo}\"");
            }
            if linha.trim_start().starts_with(prefixo) {
                return linha.trim().to_string();
            }
        }
    }

    fn lance(&mut self, pos: &Chess, lances: &[Move], go: &str) -> Move {
        let texto: Vec<String> = lances.iter().map(|&m| UciMove::from_move(m, CastlingMode::Standard).to_string()).collect();
        if texto.is_empty() {
            self.enviar("position startpos");
        } else {
            self.enviar(&format!("position startpos moves {}", texto.join(" ")));
        }
        self.enviar(&format!("go {go}"));
        let resposta = self.esperar("bestmove");
        let uci = resposta.split_whitespace().nth(1).unwrap_or("");
        uci.parse::<UciMove>()
            .ok()
            .and_then(|u| u.to_move(pos).ok())
            .unwrap_or_else(|| panic!("o motor externo respondeu um lance inválido: \"{resposta}\""))
    }
}

impl Drop for Motor {
    fn drop(&mut self) {
        let _ = writeln!(self.entrada, "quit");
        let _ = self.entrada.flush();
        if !matches!(self.filho.try_wait(), Ok(Some(_))) {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let _ = self.filho.kill();
        }
        let _ = self.filho.wait();
    }
}

thread_local! {
    // um processo por motor e por thread, reaproveitado entre partidas
    static MOTORES: RefCell<HashMap<String, Motor>> = RefCell::new(HashMap::new());
}

// O lance do motor externo nesta posição. `lances` = a partida desde o
// começo (o UCI recebe a partida inteira, não só a posição).
pub fn lance(cfg: &Externo, pos: &Chess, lances: &[Move]) -> Move {
    let chave = format!("{:?}", cfg);
    MOTORES.with(|motores| {
        let mut motores = motores.borrow_mut();
        let motor = motores.entry(chave).or_insert_with(|| Motor::abrir(cfg));
        motor.lance(pos, lances, &cfg.go)
    })
}

#[cfg(test)]
mod testes {
    use super::*;
    use shakmaty::Position;

    #[test]
    fn le_a_configuracao() {
        let e = ler(r"lc0\lc0.exe --weights=lc0\maia-1100.pb.gz;go=nodes 1;Threads=1").unwrap();
        assert_eq!(e.comando, vec![r"lc0\lc0.exe".to_string(), r"--weights=lc0\maia-1100.pb.gz".to_string()]);
        assert_eq!(e.go, "nodes 1");
        assert_eq!(e.opcoes, vec![("Threads".to_string(), "1".to_string())]);
        assert_eq!(ler("motor").unwrap().go, "nodes 1");
        assert!(ler("").is_none());
        assert!(ler("motor;semigual").is_none());
    }

    #[test]
    fn joga_contra_o_proprio_motor_pelo_uci() {
        // O nosso próprio programa no modo `uci` serve de motor externo
        // (o executável fica uma pasta acima do executável dos testes).
        let pasta = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
        let exe = pasta.join(format!("xadrez-neural{}", std::env::consts::EXE_SUFFIX));
        if !exe.exists() {
            eprintln!("({} não existe: rode cargo build antes; teste pulado)", exe.display());
            return;
        }
        let cfg = ler(&format!("{} uci material;go=depth 2", exe.display())).unwrap();
        let mut pos = Chess::default();
        let mut lances = Vec::new();
        for _ in 0..6 {
            let m = lance(&cfg, &pos, &lances);
            assert!(pos.is_legal(m));
            pos.play_unchecked(m);
            lances.push(m);
        }
    }
}
