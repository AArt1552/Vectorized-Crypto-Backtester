# results_consolidator.py

import json
import sys
import pandas as pd

def load_consolidated_results():
    """
    Loads consolidated results from a JSON file.
    """
    try:
        with open('consolidated_results.json', 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def format_cell(text, width, align='<'):
    """
    Formats a string for a table cell with fixed width.
    """
    text_str = str(text)
    if len(text_str) > width:
        return text_str[:width-1] + '…'
    return f"{text_str:{align}{width}}"

def display_final_results(results_list, title):
    """
    Displays a formatted table with final backtest results.
    """
    print(f"\n{title}\n")
    if not results_list:
        print("No results to display.\n")
        return

    # Column definitions and widths
    columns = {
        'Balance': 14, '% Final': 8, 'Hold': 9, 'Trades': 6, 'Winrate': 8, 'W.(BE/L/R1/2)': 20,
        'W.SP(B/R)': 14,
        'WR Fear(S/H/L)': 18,
        '% Pos': 7, 
        'Risk(Ent%/NoSL)': 16,
        'BTC Min(E/W/%)': 16,
        'Asset Min(E/W/%)': 18,
        
        # Expanded Column for Dynamic Leverage (BTC/RE/SP)
        'Leverage Fear(Std | High | Low)': 45,

        'H.BTC E/L': 9, 'H.Re-Ent(1/2)': 12,
        'H.SP(B/R)': 9,
        'H.BTC S/L/P': 14,
        'H.TP(T/U)': 9,
        'H.SL/H/AP': 10,
        'H.Fear(H/L/S)': 15,
        'SP(A/Bv/p/Rp/w)': 20,
        'TP(D;T/U/M/SLc/Ec/SL+/C)': 41,
        'SL(v/c)/H/AP(v/p)': 24,
        'RA(T/R/S/B)': 16,
        'RA2(T/W/R/A)': 18,
        'RA1 Act/Blk': 12,
        'RA2 Act/Blk': 12,
        'BTC E-L(v/p)': 18,
        'BTC S-L-P(v/p)': 28,
        'Re-ent1-2(p/w)': 21,
        'F.HL(v/o)': 12,
        'F.H BTC E(v/p/v/p)': 22, 'F.H RE(p/w/p/w)': 22,
        'F.L BTC E(v/p/v/p)': 22, 'F.L RE(p/w/p/w)': 22,
        'F.BTC Exit(H/L v/p)': 20,
        'F.EL Exit(H/L v/p)': 20,
        'F.Stop(v/e/x)': 10,
    }

    # Print header
    header = " | ".join([format_cell(k, v) for k, v in columns.items()])
    print(header)
    print("-" * len(header))

    for r in results_list:
        if not r: continue

        h_sl_h = f"{r.get('stop_loss_hits',0)}/{r.get('stop_loss_hard_hits',0)}/{r.get('asset_exit_panic_hits',0)}"
        h_reent = f"{r.get('reentry_hits',0)}/{r.get('reentry_2_hits',0)}"
        h_btc_e = f"{r.get('btc_entry_hits',0)}/{r.get('btc_entry_long_hits',0)}"
        h_btc_s = f"{r.get('btc_exit_hits',0)}/{r.get('btc_exit_long_hits',0)}/{r.get('btc_exit_panic_hits',0)}"
        
        h_sp_br = f"{r.get('sp_be_hits',0)}/{r.get('sp_r_hits',0)}"
        
        wr_sp_be = r.get('win_rate_sp_be', 0)
        wr_sp_r = r.get('win_rate_sp_r', 0)
        winrates_sp_detailed = f"{wr_sp_be:.1f}/{wr_sp_r:.1f}%"
        
        sp_a = r.get('leverage_SP', 0)
        sp_b_v = r.get('sp_btc_entry_trigger_value', 0)
        sp_b_p = r.get('sp_btc_entry_trigger_period', 0)
        sp_r_p = r.get('sp_reentry_pct_values', 0)
        sp_r_w = r.get('sp_reentry_window_values', 0)
        sp_config = f"{sp_a}/{sp_b_v}/{sp_b_p}/{sp_r_p}/{sp_r_w}"

        wr_be, wr_bel = r.get('win_rate_btc_entry', 0), r.get('win_rate_btc_entry_long', 0)
        wr_r1, wr_r2 = r.get('win_rate_reentry', 0), r.get('win_rate_reentry_2', 0)
        winrates_detailed = f"{wr_be:.1f}/{wr_bel:.1f}/{wr_r1:.1f}/{wr_r2:.1f}%"

        # --- LEVERAGE FORMATTING ---
        # Format: BTC / REENTRY / SP
        alav_std = f"S:{r.get('leverage_BTC',0)}/{r.get('leverage_Reentry',0)}/{r.get('leverage_SP',0)}"
        alav_high = f"H:{r.get('fear_high_leverage_BTC',0)}/{r.get('fear_high_leverage_Reentry',0)}/{r.get('fear_high_leverage_SP',0)}"
        alav_low = f"L:{r.get('fear_low_leverage_BTC',0)}/{r.get('fear_low_leverage_Reentry',0)}/{r.get('fear_low_leverage_SP',0)}"
        lev_full = f"{alav_std} | {alav_high} | {alav_low}"

        btc_e_config = (f"{r.get('btc_entry_trigger_value',0)}/{r.get('btc_entry_trigger_period',0)} | "
                        f"{r.get('btc_entry_long_trigger_value',0)}/{r.get('btc_entry_long_trigger_period',0)}")

        btc_s_config = (f"{r.get('btc_exit_trigger_value',0)}/{r.get('btc_exit_trigger_period',0)} | "
                        f"{r.get('btc_exit_long_trigger_value',0)}/{r.get('btc_exit_long_trigger_period',0)} | "
                        f"{r.get('btc_exit_panic_trigger_value',0)}/{r.get('btc_exit_panic_trigger_period',0)}")

        reent_config = (f"{r.get('reentry_pct_values',0)}/{r.get('reentry_window_values',0)} | "
                        f"{r.get('reentry_pct_values_2',0)}/{r.get('reentry_window_values_2',0)}")

        h_tp_tu = f"{r.get('take_profit_hits', 0)}/{r.get('total_tp_updates', 0)}"
        h_fear_combined = f"{r.get('fear_high_hits', 0)}/{r.get('fear_low_hits', 0)}/{r.get('fear_stop_hits', 0)}"

        fh_override = 'T' if r.get('fear_high_override', False) else 'F'
        fl_override = 'T' if r.get('fear_low_override', False) else 'F'
        f_hl_config = f"{r.get('fear_high_value', 0)}/{fh_override}/{r.get('fear_low_value', 0)}/{fl_override}"

        fh_btc_e = (f"{r.get('fear_high_btc_entry_trigger_value',0)}/{r.get('fear_high_btc_entry_trigger_period',0)}/"
                    f"{r.get('fear_high_btc_entry_long_trigger_value',0)}/{r.get('fear_high_btc_entry_long_trigger_period',0)}")
        fh_re = (f"{r.get('fear_high_reentry_pct_values',0)}/{r.get('fear_high_reentry_window_values',0)}/"
                 f"{r.get('fear_high_reentry_pct_values_2',0)}/{r.get('fear_high_reentry_window_values_2',0)}")

        fl_btc_e = (f"{r.get('fear_low_btc_entry_trigger_value',0)}/{r.get('fear_low_btc_entry_trigger_period',0)}/"
                    f"{r.get('fear_low_btc_entry_long_trigger_value',0)}/{r.get('fear_low_btc_entry_long_trigger_period',0)}")
        fl_re = (f"{r.get('fear_low_reentry_pct_values',0)}/{r.get('fear_low_reentry_window_values',0)}/"
                 f"{r.get('fear_low_reentry_pct_values_2',0)}/{r.get('fear_low_reentry_window_values_2',0)}")

        f_btc_exit_config = (f"{r.get('fear_high_btc_exit_trigger_value',0)}/{r.get('fear_high_btc_exit_trigger_period',0)}/"
                             f"{r.get('fear_low_btc_exit_trigger_value',0)}/{r.get('fear_low_btc_exit_trigger_period',0)}")

        f_btc_el_exit_config = (f"{r.get('fear_high_btc_exit_long_trigger_value',0)}/{r.get('fear_high_btc_exit_long_trigger_period',0)}/"
                                f"{r.get('fear_low_btc_exit_long_trigger_value',0)}/{r.get('fear_low_btc_exit_long_trigger_period',0)}")

        fs_bt_entry = 'T' if r.get('fear_stop_btc_entry', False) else 'F'
        fs_exit_pos = 'T' if r.get('fear_stop_exit_position', False) else 'F'
        fs_config = f"{r.get('fear_stop_value', 0)}/{fs_bt_entry}/{fs_exit_pos}"

        wr_s = r.get('win_rate_fear_standard', 0)
        wr_h = r.get('win_rate_fear_high', 0)
        wr_l = r.get('win_rate_fear_low', 0)
        winrate_fear_detailed = f"{wr_s:.1f}/{wr_h:.1f}/{wr_l:.1f}%"

        tp_d = 'T' if r.get('take_profit_dynamic', False) else 'F'
        tp_t = r.get('take_profit_trigger', 0)
        tp_u = r.get('take_profit_update', 0)
        tp_m = r.get('take_profit_multiplier', 0)
        tp_slc = r.get('take_profit_confirmation_sl_pct', 0)
        tp_ec = r.get('take_profit_confirmation_end_pct', 0)
        tp_sla = r.get('take_profit_update_add_to_sl', 0)
        tp_c = r.get('cooldown_tp', 0)
        tp_config = f"{tp_d};{tp_t}/{tp_u}/{tp_m}/{tp_slc}/{tp_ec}/{tp_sla}/{tp_c}"

        sl_v = r.get('stop_loss', 0)
        sl_c = r.get('stop_loss_cooldown', 0)
        slh_v = r.get('stop_loss_hard', 0)
        ap_v = r.get('asset_exit_panic_trigger_value', 0)
        ap_p = r.get('asset_exit_panic_trigger_period', 0)
        sl_config = f"{sl_v}/{sl_c}/{slh_v}/{ap_v}/{ap_p}"

        ra_t = r.get('red_alert_trigger_pct', 0)
        ra_r = r.get('red_alert_reset_pct', 0)
        ra_s = 'T' if r.get('exit_position_red_alert', False) else 'F'
        ra_b = 'T' if r.get('red_alert_block_btc_entries', False) else 'F' 
        ra_config = f"{ra_t}/{ra_r}/{ra_s}/{ra_b}"

        ra2_t = r.get('red_alert2_trigger_pct', 0)
        ra2_w = r.get('red_alert2_window', 0)
        ra2_r = r.get('red_alert2_reset_pct', 0)
        ra2_a = 'T' if r.get('red_alert2_reset_auto_entry', False) else 'F'
        ra2_config = f"{ra2_t}/{ra2_w}/{ra2_r}/{ra2_a}"
        
        ra1_act_blk = f"{r.get('ra1_activation_count', 0)}/{r.get('ra1_blocked_count', 0)}"
        ra2_act_blk = f"{r.get('ra2_activation_count', 0)}/{r.get('ra2_blocked_count', 0)}"
        
        entry_pct = r.get('entry_percentage', 0)
        no_sl = 'T' if r.get('disable_stop_loss', False) else 'F'
        risk_config = f"{entry_pct}%/{no_sl}"

        btc_min_enable = 'T' if r.get('btc_minimum_enable', False) else 'F'
        if btc_min_enable == 'T':
            btc_min_config = f"{btc_min_enable}/{r.get('btc_minimum_window',0)}/{r.get('btc_minimum_pct_to_entry',0)*100:.1f}%"
        else:
            btc_min_config = "F"

        asset_min_enable = 'T' if r.get('asset_minimum_enable', False) else 'F'
        if asset_min_enable == 'T':
            asset_min_config = f"{asset_min_enable}/{r.get('asset_minimum_window',0)}/{r.get('asset_minimum_pct_to_entry',0)*100:.1f}%"
        else:
            asset_min_config = "F"

        row_data = [
            format_cell(f"{r.get('final_balance', 0):.2f}", columns['Balance']),
            format_cell(f"{r.get('final_balance_pct', 0):+.2f}%", columns['% Final'], align='>'),
            format_cell(f"{r.get('final_hold_balance', 0):.2f}", columns['Hold']),
            format_cell(r.get('total_trades', 0), columns['Trades']),
            format_cell(f"{r.get('win_rate', 0):.2f}%", columns['Winrate']),
            format_cell(winrates_detailed, columns['W.(BE/L/R1/2)']),
            format_cell(winrates_sp_detailed, columns['W.SP(B/R)']),
            format_cell(winrate_fear_detailed, columns['WR Fear(S/H/L)']),
            format_cell(f"{r.get('time_in_pos_pct', 0):.2f}%", columns['% Pos']),
            
            format_cell(risk_config, columns['Risk(Ent%/NoSL)']),
            format_cell(btc_min_config, columns['BTC Min(E/W/%)']),
            format_cell(asset_min_config, columns['Asset Min(E/W/%)']),
            
            # Updated Column
            format_cell(lev_full, columns['Leverage Fear(Std | High | Low)']),
            
            format_cell(h_btc_e, columns['H.BTC E/L']),
            format_cell(h_reent, columns['H.Re-Ent(1/2)']),
            format_cell(h_sp_br, columns['H.SP(B/R)']),
            format_cell(h_btc_s, columns['H.BTC S/L/P']),
            format_cell(h_tp_tu, columns['H.TP(T/U)']),
            format_cell(h_sl_h, columns['H.SL/H/AP']),
            format_cell(h_fear_combined, columns['H.Fear(H/L/S)']),
            format_cell(sp_config, columns['SP(A/Bv/p/Rp/w)']),
            format_cell(tp_config, columns['TP(D;T/U/M/SLc/Ec/SL+/C)']),
            format_cell(sl_config, columns['SL(v/c)/H/AP(v/p)']),
            format_cell(ra_config, columns['RA(T/R/S/B)']),
            format_cell(ra2_config, columns['RA2(T/W/R/A)']),
            format_cell(ra1_act_blk, columns['RA1 Act/Blk']),
            format_cell(ra2_act_blk, columns['RA2 Act/Blk']),
            format_cell(btc_e_config, columns['BTC E-L(v/p)']),
            format_cell(btc_s_config, columns['BTC S-L-P(v/p)']),
            format_cell(reent_config, columns['Re-ent1-2(p/w)']),
            format_cell(f_hl_config, columns['F.HL(v/o)']),
            format_cell(fh_btc_e, columns['F.H BTC E(v/p/v/p)']),
            format_cell(fh_re, columns['F.H RE(p/w/p/w)']),
            format_cell(fl_btc_e, columns['F.L BTC E(v/p/v/p)']),
            format_cell(fl_re, columns['F.L RE(p/w/p/w)']),
            format_cell(f_btc_exit_config, columns['F.BTC Exit(H/L v/p)']),
            format_cell(f_btc_el_exit_config, columns['F.EL Exit(H/L v/p)']),
            format_cell(fs_config, columns['F.Stop(v/e/x)']),
        ]

        print(" | ".join(row_data))

def generate_monthly_log(trades_df, df_asset, initial_balance):
    """
    Generates a monthly performance report from trade data.
    """
    if trades_df is None or trades_df.empty:
        print("⚠️ No trade data to generate monthly log.")
        return

    df_asset_close = df_asset.rename(columns={'close': 'close_asset'}).set_index('timestamp')
    trades_df = trades_df.set_index('exit_time')
    
    # Using 'ME' (Month End) to avoid FutureWarning
    monthly_profit = trades_df['profit'].resample('ME').sum()
    monthly_asset_close = df_asset_close['close_asset'].resample('ME').last()

    current_balance = initial_balance
    initial_hold_qty = initial_balance / df_asset_close['close_asset'].iloc[0] if not df_asset_close.empty else 0

    with open('monthly_performance.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nMonthly Performance Report\n" + "="*60 + "\n\n")
        f.write(f"Initial Balance: ${initial_balance:,.2f}\n")
        if not df_asset_close.empty:
            f.write(f"Asset Initial Price: ${df_asset_close['close_asset'].iloc[0]:,.4f}\n\n")

        header = f"{'Month/Year':<10} | {'Profit/Loss':>16} | {'Final Balance':>16} | {'Hold Value':>16}\n"
        f.write(header + "-" * len(header) + "\n")

        month_end_hold_value = initial_balance
        for month, profit in monthly_profit.items():
            current_balance += profit
            if month in monthly_asset_close.index:
                month_end_hold_value = initial_hold_qty * monthly_asset_close.get(month, 0)
            f.write(f"{month.strftime('%Y-%m'):<10} | ${profit:>15,.2f} | ${current_balance:>15,.2f} | ${month_end_hold_value:>15,.2f}\n")

        f.write("-" * len(header) + f"\n\nTotal Final Balance: ${current_balance:,.2f}\nFinal Hold Value: ${month_end_hold_value:,.2f}\n" + "="*60 + "\n")

def consolidate_and_present_results(config_file_name):
    """
    Main function that loads, processes, and displays the results.
    """
    results = load_consolidated_results()
    if not results:
        print("⚠️ No results to consolidate.")
        return

    results = [r for r in results if r is not None]
    if not results:
        print("⚠️ No valid results found after filtering.")
        return

    top_results = sorted(results, key=lambda x: x.get('final_balance', 0), reverse=True)

    display_final_results(top_results[:20], "🏆 Top 20 Best Combinations (Final Format) 🏆")

    with open('full_report.txt', 'w', encoding='utf-8') as f:
        json.dump(top_results, f, indent=4)
    print("\n✅ Full report (JSON) saved to 'full_report.txt'")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    consolidate_and_present_results(config)
