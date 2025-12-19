import requests
import json
import os
import glob
import sys
import time
import shutil
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

# --- Configurações ---
DATASETS_DIR = "datasets"
FEAR_FILE = "fear_index.json"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
FEAR_API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
LIMIT_PER_REQUEST = 1000
REQUEST_DELAY_SEC = 0.2
JSON_INDENT = 4

# Cria diretório se não existir
os.makedirs(DATASETS_DIR, exist_ok=True)

# ---------------------------------------------------------
# MÓDULO 1: FEAR & GREED (Sempre Sobrescreve)
# ---------------------------------------------------------
def update_fear_index():
    print(f"\n--- Atualizando Fear & Greed Index ---")
    try:
        response = requests.get(FEAR_API_URL)
        data = response.json()
        
        fear_data = []
        for item in data['data']:
            fear_data.append({
                'timestamp': item['timestamp'],
                'fear_index': int(item['value'])
            })
        
        # Salva sobrescrevendo
        path = os.path.join(DATASETS_DIR, FEAR_FILE)
        with open(path, 'w') as f:
            json.dump(fear_data, f, indent=JSON_INDENT)
            
        print(f"  -> ✅ Fear Index salvo com sucesso ({len(fear_data)} registros).")
        
    except Exception as e:
        print(f"  -> ❌ Erro ao baixar Fear Index: {e}")

# ---------------------------------------------------------
# MÓDULO 2: FUNÇÕES AUXILIARES DE CANDLES
# ---------------------------------------------------------
def fetch_candles_range(symbol, start_ts_ms):
    """
    Busca candles na Binance a partir de start_ts_ms até o momento atual.
    Retorna uma lista de candles formatados.
    """
    end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_start = start_ts_ms
    new_candles = []
    
    total_duration = end_ts_ms - current_start
    if total_duration <= 0:
        return []

    print(f"  -> Buscando {symbol} a partir de {datetime.utcfromtimestamp(start_ts_ms/1000)}...")
    
    with tqdm(total=total_duration, unit="ms", leave=False) as pbar:
        while current_start < end_ts_ms:
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": current_start,
                "limit": LIMIT_PER_REQUEST
            }
            
            try:
                r = requests.get(BINANCE_API_URL, params=params)
                data = r.json()
            except Exception as e:
                print(f"\n  [Erro Rede] {symbol}: {e}")
                break

            if not isinstance(data, list) or not data:
                break
            
            # Processa o lote
            for c in data:
                new_candles.append({
                    "timestamp": datetime.utcfromtimestamp(c[0] / 1000).isoformat(),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })

            # Atualiza ponteiro de tempo
            last_open_time = data[-1][0]
            step = last_open_time - current_start + 60000 # Avança baseado no que baixou
            current_start = last_open_time + 60000
            
            pbar.update(step)
            time.sleep(REQUEST_DELAY_SEC)
            
    return new_candles

def save_json(filepath, data):
    """Salva o JSON de forma segura (sobrescreve)."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=JSON_INDENT)
        print(f"  -> ✅ Salvo: {os.path.basename(filepath)} ({len(data)} candles)")
    except Exception as e:
        print(f"  -> ❌ Erro ao salvar {filepath}: {e}")

# ---------------------------------------------------------
# MÓDULO 3: LÓGICA DE EXECUÇÃO (HISTÓRICO vs UPDATE)
# ---------------------------------------------------------
def run_history_mode(days, symbols):
    print(f"\n=== MODO HISTÓRICO: Baixando últimos {days} dias ===")
    
    # Calcula timestamp de início (Agora - Dias) em MS
    start_dt = datetime.now(timezone.utc) - timedelta(days=days)
    start_ts_ms = int(start_dt.timestamp() * 1000)
    
    for raw_symbol in symbols:
        symbol = f"{raw_symbol.upper()}USDT"
        file_path = os.path.join(DATASETS_DIR, f"{raw_symbol.upper()}_val.json")
        
        candles = fetch_candles_range(symbol, start_ts_ms)
        if candles:
            save_json(file_path, candles)
        else:
            print(f"  -> ⚠️ Nenhum dado encontrado para {symbol}.")

def run_update_mode(symbols):
    print(f"\n=== MODO UPDATE: Atualizando incrementalmente ===")
    
    for raw_symbol in symbols:
        symbol = f"{raw_symbol.upper()}USDT"
        file_path = os.path.join(DATASETS_DIR, f"{raw_symbol.upper()}_val.json")
        
        if not os.path.exists(file_path):
            print(f"  -> [Pular] Arquivo não existe para {symbol}. Use o modo histórico primeiro.")
            continue
            
        # 1. Carregar existente
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"  -> ❌ Erro ao ler {file_path}: {e}")
            continue

        if not existing_data:
            print(f"  -> Arquivo vazio. Pulando.")
            continue
            
        # 2. Descobrir último timestamp
        try:
            last_ts_str = existing_data[-1]['timestamp']
            last_dt = datetime.fromisoformat(last_ts_str)
            # Começa 1 minuto depois do último
            start_ts_ms = int(last_dt.replace(tzinfo=timezone.utc).timestamp() * 1000) + 60000
        except Exception as e:
            print(f"  -> Erro ao processar data do arquivo: {e}")
            continue
            
        # 3. Baixar delta
        new_candles = fetch_candles_range(symbol, start_ts_ms)
        
        if new_candles:
            print(f"  -> {len(new_candles)} novos candles encontrados. Integrando...")
            existing_data.extend(new_candles)
            save_json(file_path, existing_data)
        else:
            print(f"  -> {symbol} já está atualizado.")

# ---------------------------------------------------------
# MÓDULO 4: LIMPEZA DE CACHE
# ---------------------------------------------------------
def cleanup_feathers():
    print(f"\n--- Limpeza de Cache (.feather) ---")
    files = glob.glob(os.path.join(DATASETS_DIR, "*.feather"))
    if not files:
        print("  -> Nenhum arquivo .feather para deletar.")
        return

    for f in files:
        try:
            os.remove(f)
            print(f"  -> Deletado: {os.path.basename(f)}")
        except Exception as e:
            print(f"  -> Erro ao deletar {f}: {e}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Uso:")
        print("  Modo Histórico: python get_data.py <DIAS> <SYMBOL1> <SYMBOL2> ...")
        print("  Modo Update:    python get_data.py update <SYMBOL1> <SYMBOL2> ...")
        print("Exemplo: python get_data.py 360 BTC ETH ADA")
        return

    mode_arg = sys.argv[1]
    symbols = sys.argv[2:]
    
    # 1. Atualiza Fear Index (Comum a ambos)
    update_fear_index()
    
    # 2. Executa modo específico
    if mode_arg.lower() == 'update':
        run_update_mode(symbols)
    else:
        # Tenta interpretar como número de dias
        try:
            days = int(mode_arg)
            run_history_mode(days, symbols)
        except ValueError:
            print(f"Erro: O primeiro argumento deve ser um número (dias) ou 'update'. Recebido: {mode_arg}")
            return

    # 3. Limpa arquivos feather antigos
    cleanup_feathers()
    
    print("\n✅ Processo concluído.")

if __name__ == "__main__":
    main()
