# main_runner.py

import os
import json
import sys
import multiprocessing
import time
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from param_generator import generate_and_save_parameters
from vectorized_backtest import run_vectorized_backtest
from results_consolidator import consolidate_and_present_results, generate_monthly_log

df_btc_global = None
df_asset_global = None
df_fear_global = None

def convert_and_load_candles(json_path_str):
    json_path = Path(json_path_str)
    feather_path = json_path.with_suffix('.feather')
    if feather_path.exists():
        print(f"⚡ Loading optimized file: {feather_path.name}")
        return pd.read_feather(feather_path)
    else:
        print(f"🐌 Loading slow JSON file for the first time: {json_path.name}")
        try:
            df = pd.read_json(json_path_str)
            # Handle timestamp column names
            if 'timestamp' in df.columns and 'fear_index' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').asfreq('1min', method='ffill').reset_index()
            
            print(f"💾 Converting to fast Feather format for future use...")
            df.to_feather(feather_path)
            print("✅ Conversion complete.")
            return df
        except Exception as e:
            print(f"❌ ERROR loading or processing candles from '{json_path_str}': {e}")
            return None

def init_worker(btc_candles, asset_candles, fear_candles):
    global df_btc_global, df_asset_global, df_fear_global
    df_btc_global = btc_candles
    df_asset_global = asset_candles
    df_fear_global = fear_candles

def worker(task_args):
    params, index, log_file, lock = task_args
    # Passes global dataframe to backtest function
    return run_vectorized_backtest(params, df_btc_global, df_asset_global, df_fear_global, index, log_file=log_file, lock=lock)

def main():
    start_time_total = time.time()
    
    if len(sys.argv) < 2:
        print("\nUsage: python main_runner.py <config.json> [days_to_test] [processes] [log]")
        sys.exit(1)

    config_file_name = sys.argv[1]
    days_to_test = int(sys.argv[2]) if len(sys.argv) > 2 else None
    n_processes = int(sys.argv[3]) if len(sys.argv) > 3 else multiprocessing.cpu_count()
    log_enabled = 'log' in sys.argv

    log_file_name = 'backtest_trades.log' if log_enabled else None
    if log_enabled and os.path.exists(log_file_name):
        os.remove(log_file_name)
    
    total_combinations = generate_and_save_parameters(config_file_name)
    print(f"  - ✅ {total_combinations} parameter combinations generated.")

    if input("\n🤔 Do you wish to continue execution? (y/n): ").lower().strip() != 'y':
        sys.exit("🛑 Execution cancelled by user.")

    print("\n⏳ Checking and loading candle files...")
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    path_btc = config['candles_file_BTC'][0]
    path_asset = config['candles_file_asset'][0]
    path_fear = config['candles_file_fear'][0]
    
    df_btc = convert_and_load_candles(path_btc)
    df_asset = convert_and_load_candles(path_asset)
    df_fear = convert_and_load_candles(path_fear)

    if df_btc is None or df_asset is None or df_fear is None: sys.exit("Failed to load candle files.")

    if days_to_test is not None:
        print(f"⏳ Filtering data for the last {days_to_test} days...")
        end_date = min(df_btc['timestamp'].max(), df_asset['timestamp'].max())
        start_date = end_date - pd.to_timedelta(days_to_test, unit='D')
        df_btc = df_btc[df_btc['timestamp'] >= start_date].reset_index(drop=True)
        df_asset = df_asset[df_asset['timestamp'] >= start_date].reset_index(drop=True)
        print("✅ Data filtered successfully.")

    params_to_test = json.load(open('test_params.json'))
    manager = multiprocessing.Manager()
    log_lock = manager.Lock() if log_enabled else None
    all_tasks = [(params, idx, log_file_name, log_lock) for idx, params in enumerate(params_to_test)]

    print(f"\n🚀 Processing {len(all_tasks)} tasks with {n_processes} processes...")
    
    init_args = (df_btc, df_asset, df_fear)
    
    with multiprocessing.Pool(processes=n_processes, initializer=init_worker, initargs=init_args) as pool:
        all_returns = list(tqdm(pool.imap_unordered(worker, all_tasks), total=len(all_tasks), desc="Overall Progress"))

    print("\n✅ Processing complete. Consolidating results...")
    
    final_results = [ret[0] for ret in all_returns if ret and ret[0]]
    monthly_logs = {ret[0]['test_id']: ret[1] for ret in all_returns if ret and ret[0] and ret[1] is not None}
    
    with open('consolidated_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    try:
        consolidate_and_present_results(config_file_name)
    except Exception as e:
        print(f"\n❌ A fatal error occurred during result consolidation:")
        print(f"   Error: {e}\n")
    
    if log_enabled:
        print(f"✅ Detailed trade log saved to '{log_file_name}'")
        if final_results:
            best_results = sorted(final_results, key=lambda x: x.get('final_balance', 0), reverse=True)
            if best_results:
                best_result = best_results[0]
                best_id = best_result.get('test_id')
                best_log = monthly_logs.get(best_id)
                
                if best_log is not None:
                    print("📝 Generating monthly performance log...")
                    initial_balance_config = config.get('initial_balance', [1000.0])[0]
                    generate_monthly_log(best_log, df_asset, initial_balance_config)
                    print(f"✅ Monthly performance log saved to 'monthly_performance.txt'")

    total_duration = time.time() - start_time_total
    print(f"\n🏁 PROCESS COMPLETE! Total time: {int(total_duration)}s 🏁")

if __name__ == "__main__":
    main()
