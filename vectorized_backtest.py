# vectorized_backtest.py

import pandas as pd
import logging
import os
import numpy as np
from numba import njit

@njit
def numba_jit_simulation(
    timestamps_int, day_of_year_arr, close_asset, close_btc, fear_index_array,
    p_initial_balance, p_fixed_fee,
    p_leverage_BTC, p_leverage_Reentry,
    p_fear_high_leverage_BTC, p_fear_high_leverage_Reentry,
    p_fear_low_leverage_BTC, p_fear_low_leverage_Reentry,
    p_red_alert_trigger_pct, p_red_alert_reset_pct,
    p_red_alert_reset_auto_entry,
    p_red_alert2_trigger_pct, p_red_alert2_window, p_red_alert2_reset_pct, p_red_alert2_reset_auto_entry,
    p_red_alert_block_btc_entries,
    p_max_entries_day, p_max_entries_day_btc, p_stop_loss_hard, p_cooldown_sl,
    p_stop_loss, p_stop_loss_cooldown, p_exit_position_red_alert,
    p_take_profit_dynamic, p_take_profit_trigger, p_take_profit_update_enable,
    p_take_profit_update, p_take_profit_multiplier, p_cooldown_tp,
    p_take_profit_confirmation_sl_pct, p_take_profit_confirmation_end_pct,
    p_take_profit_update_add_to_sl,
    p_btc_entry_trigger_value, p_btc_entry_trigger_period,
    p_btc_exit_trigger_value, p_btc_exit_trigger_period,
    p_btc_entry_long_trigger_value, p_btc_entry_long_trigger_period,
    p_btc_exit_long_trigger_value, p_btc_exit_long_trigger_period,
    p_reentry_pct_values, p_reentry_window_values,
    p_reentry_pct_values_2, p_reentry_window_values_2,
    p_btc_exit_panic_trigger_value, p_btc_exit_panic_trigger_period,
    p_asset_exit_panic_trigger_value, p_asset_exit_panic_trigger_period,
    p_leverage_SP, p_fear_high_leverage_SP, p_fear_low_leverage_SP,
    p_sp_btc_entry_trigger_value, p_sp_btc_entry_trigger_period,
    p_sp_reentry_pct_values, p_sp_reentry_window_values,
    p_fear_high_override, p_fear_high_value,
    p_fear_high_btc_entry_trigger_value, p_fear_high_btc_entry_trigger_period,
    p_fear_high_btc_exit_trigger_value, p_fear_high_btc_exit_trigger_period,
    p_fear_high_btc_entry_long_trigger_value, p_fear_high_btc_entry_long_trigger_period,
    p_fear_high_btc_exit_long_trigger_value, p_fear_high_btc_exit_long_trigger_period,
    p_fear_high_reentry_pct_values, p_fear_high_reentry_window_values,
    p_fear_high_reentry_pct_values_2, p_fear_high_reentry_window_values_2,
    p_fear_low_override, p_fear_low_value,
    p_fear_low_btc_entry_trigger_value, p_fear_low_btc_entry_trigger_period,
    p_fear_low_btc_exit_trigger_value, p_fear_low_btc_exit_trigger_period,
    p_fear_low_btc_entry_long_trigger_value, p_fear_low_btc_entry_long_trigger_period,
    p_fear_low_btc_exit_long_trigger_value, p_fear_low_btc_exit_long_trigger_period,
    p_fear_low_reentry_pct_values, p_fear_low_reentry_window_values,
    p_fear_low_reentry_pct_values_2, p_fear_low_reentry_window_values_2,
    p_fear_stop, p_fear_stop_value, p_fear_stop_btc_entry, p_fear_stop_exit_position,
    p_entry_percentage, p_disable_stop_loss,
    p_btc_minimum_enable, p_btc_minimum_pct_to_entry, p_btc_minimum_window,
    p_asset_minimum_enable, p_asset_minimum_pct_to_entry, p_asset_minimum_window,
    log_enabled
):
    balance = p_initial_balance
    is_in_position = False
    entry_price = 0.0
    asset_quantity = 0.0
    cooldown_timer = 0
    total_trades, total_wins = 0, 0
    total_tp_updates = 0
    current_day_of_year = 0
    daily_entry_count = 0
    red_alert_active = False
    last_exit_price = 0.0
    lowest_price_since_alert = np.inf
    red_alert2_active = False
    lowest_price_ra2 = np.inf
    highest_price_since_entry = 0.0
    minutes_in_position = 0
    hits = np.zeros(18, dtype=np.int64)
    entry_triggers_map = {'BE': 0, 'BEL': 1, 'R': 2, 'R2': 3, 'SP_BE': 4, 'SP_R': 5}
    entry_type_trades = np.zeros(6, dtype=np.int64)
    entry_type_wins = np.zeros(6, dtype=np.int64)
    entry_reason_idx = -1
    base_trade_capital = 0.0
    leveraged_capital = 0.0
    current_leverage = 1.0
    entry_timestamp_idx = -1
    tp_level, tp_profit_floor_pct = 0.0, 0.0
    tp_confirmation_mode_active, tp_confirmation_sl_level, tp_confirmation_end_level, tp_pending_floor_pct = False, 0.0, 0.0, 0.0
    fear_high_hits, fear_low_hits, fear_stop_hits = 0, 0, 0
    eff_btc_entry_val, eff_btc_entry_per = 0.0, 0
    eff_btc_exit_val, eff_btc_exit_per = 0.0, 0
    eff_btc_entry_long_val, eff_btc_entry_long_per = 0.0, 0
    eff_btc_exit_long_val, eff_btc_exit_long_per = 0.0, 0
    eff_reentry_pct, eff_reentry_win = 0.0, 0
    eff_reentry_pct_2, eff_reentry_win_2 = 0.0, 0
    is_fear_stop_active_today = False
    fear_state_trades = np.zeros(3, dtype=np.int64)
    fear_state_wins = np.zeros(3, dtype=np.int64)
    current_fear_state = 0
    trade_entry_fear_state = -1
    fear_index_no_trade = -1.0
    reserved_balance = 0.0 

    # Dynamic Leverage Variables
    current_lev_btc = 1
    current_lev_re = 1
    current_lev_sp = 1

    if log_enabled:
        trade_log_data = np.zeros((10000, 8), dtype=np.float64)
    else:
        trade_log_data = np.zeros((1, 8), dtype=np.float64)
    trade_log_idx = 0

    ra1_activation_count = 0
    ra1_blocked_count = 0
    ra2_activation_count = 0
    ra2_blocked_count = 0

    for i in range(len(close_asset)):
        day_of_year = day_of_year_arr[i]
        
        if day_of_year != current_day_of_year:
            current_day_of_year = day_of_year
            daily_entry_count = 0
            current_fear_index = fear_index_array[i]
            is_fear_stop_active_today = p_fear_stop and current_fear_index != -1 and current_fear_index <= p_fear_stop_value
            if is_fear_stop_active_today: fear_stop_hits += 1
            is_high_fear = p_fear_high_override and current_fear_index != -1 and current_fear_index >= p_fear_high_value
            is_low_fear = p_fear_low_override and current_fear_index != -1 and current_fear_index <= p_fear_low_value
            current_fear_state = 0
            if is_high_fear: current_fear_state = 1
            elif is_low_fear: current_fear_state = 2
            
            # --- Parameter Configuration Based on Fear State ---
            if is_high_fear:
                fear_high_hits += 1
                # Triggers
                eff_btc_entry_val, eff_btc_entry_per = p_fear_high_btc_entry_trigger_value, p_fear_high_btc_entry_trigger_period
                eff_btc_exit_val, eff_btc_exit_per = p_fear_high_btc_exit_trigger_value, p_fear_high_btc_exit_trigger_period
                eff_btc_entry_long_val, eff_btc_entry_long_per = p_fear_high_btc_entry_long_trigger_value, p_fear_high_btc_entry_long_trigger_period
                eff_btc_exit_long_val, eff_btc_exit_long_per = p_fear_high_btc_exit_long_trigger_value, p_fear_high_btc_exit_long_trigger_period
                eff_reentry_pct, eff_reentry_win = p_fear_high_reentry_pct_values, p_fear_high_reentry_window_values
                eff_reentry_pct_2, eff_reentry_win_2 = p_fear_high_reentry_pct_values_2, p_fear_high_reentry_window_values_2
                # Leverage Override
                current_lev_btc = p_fear_high_leverage_BTC
                current_lev_re = p_fear_high_leverage_Reentry
                current_lev_sp = p_fear_high_leverage_SP
            elif is_low_fear:
                fear_low_hits += 1
                # Triggers
                eff_btc_entry_val, eff_btc_entry_per = p_fear_low_btc_entry_trigger_value, p_fear_low_btc_entry_trigger_period
                eff_btc_exit_val, eff_btc_exit_per = p_fear_low_btc_exit_trigger_value, p_fear_low_btc_exit_trigger_period
                eff_btc_entry_long_val, eff_btc_entry_long_per = p_fear_low_btc_entry_long_trigger_value, p_fear_low_btc_entry_long_trigger_period
                eff_btc_exit_long_val, eff_btc_exit_long_per = p_fear_low_btc_exit_long_trigger_value, p_fear_low_btc_exit_long_trigger_period
                eff_reentry_pct, eff_reentry_win = p_fear_low_reentry_pct_values, p_fear_low_reentry_window_values
                eff_reentry_pct_2, eff_reentry_win_2 = p_fear_low_reentry_pct_values_2, p_fear_low_reentry_window_values_2
                # Leverage Override
                current_lev_btc = p_fear_low_leverage_BTC
                current_lev_re = p_fear_low_leverage_Reentry
                current_lev_sp = p_fear_low_leverage_SP
            else:
                # Triggers
                eff_btc_entry_val, eff_btc_entry_per = p_btc_entry_trigger_value, p_btc_entry_trigger_period
                eff_btc_exit_val, eff_btc_exit_per = p_btc_exit_trigger_value, p_btc_exit_trigger_period
                eff_btc_entry_long_val, eff_btc_entry_long_per = p_btc_entry_long_trigger_value, p_btc_entry_long_trigger_period
                eff_btc_exit_long_val, eff_btc_exit_long_per = p_btc_exit_long_trigger_value, p_btc_exit_long_trigger_period
                eff_reentry_pct, eff_reentry_win = p_reentry_pct_values, p_reentry_window_values
                eff_reentry_pct_2, eff_reentry_win_2 = p_reentry_pct_values_2, p_reentry_window_values_2
                # Standard Leverage
                current_lev_btc = p_leverage_BTC
                current_lev_re = p_leverage_Reentry
                current_lev_sp = p_leverage_SP

        if cooldown_timer > 0: cooldown_timer -= 1
        if is_in_position: minutes_in_position += 1
        
        # --- RA 1 ---
        if not is_in_position and last_exit_price > 0:
            drop_from_exit = (last_exit_price - close_asset[i]) / last_exit_price
            if not red_alert_active and drop_from_exit >= p_red_alert_trigger_pct:
                red_alert_active = True
                ra1_activation_count += 1
                lowest_price_since_alert = close_asset[i]
            
            if red_alert_active:
                lowest_price_since_alert = min(lowest_price_since_alert, close_asset[i])
                rise_from_low = (close_asset[i] - lowest_price_since_alert) / lowest_price_since_alert if lowest_price_since_alert > 0 else 0
                if rise_from_low >= p_red_alert_reset_pct:
                    red_alert_active = False
                    last_exit_price = 0.0
                    lowest_price_since_alert = np.inf
                    if p_red_alert_reset_auto_entry and not is_in_position and cooldown_timer == 0 and balance > 0:
                        limit_reached = daily_entry_count >= p_max_entries_day
                        fear_stop_blocks = is_fear_stop_active_today 
                        if not limit_reached and not fear_stop_blocks:
                            entry_reason_idx = entry_triggers_map['R'] 
                            entry_type_trades[entry_reason_idx] += 1
                            hits[entry_reason_idx] += 1
                            is_in_position, daily_entry_count, entry_price, entry_timestamp_idx, highest_price_since_entry = True, daily_entry_count + 1, close_asset[i], i, close_asset[i]
                            fear_index_no_trade = fear_index_array[i]
                            trade_entry_fear_state = current_fear_state
                            
                            current_leverage = current_lev_re
                            
                            if p_entry_percentage > 0:
                                base_trade_capital = balance * (p_entry_percentage / 100.0)
                                reserved_balance = balance - base_trade_capital
                            else:
                                base_trade_capital = balance
                                reserved_balance = 0.0
                            
                            leveraged_capital = base_trade_capital * current_leverage
                            asset_quantity = (leveraged_capital * (1 - p_fixed_fee)) / entry_price
                            balance = 0.0
                            continue 

        # --- RA 2 ---
        if not red_alert2_active and i >= p_red_alert2_window:
            price_window_start = close_asset[i - p_red_alert2_window]
            if price_window_start > 0:
                drop_window = (price_window_start - close_asset[i]) / price_window_start
                if drop_window >= p_red_alert2_trigger_pct:
                    red_alert2_active = True
                    ra2_activation_count += 1
                    lowest_price_ra2 = close_asset[i]

        if red_alert2_active:
            lowest_price_ra2 = min(lowest_price_ra2, close_asset[i])
            rise_from_low_ra2 = 0.0
            if lowest_price_ra2 > 0:
                rise_from_low_ra2 = (close_asset[i] - lowest_price_ra2) / lowest_price_ra2
            if rise_from_low_ra2 >= p_red_alert2_reset_pct:
                red_alert2_active = False
                lowest_price_ra2 = np.inf
                if p_red_alert2_reset_auto_entry and not is_in_position and cooldown_timer == 0 and balance > 0:
                    limit_reached = daily_entry_count >= p_max_entries_day
                    fear_stop_blocks = is_fear_stop_active_today
                    blocked_by_ra1 = red_alert_active 
                    if not limit_reached and not fear_stop_blocks and not blocked_by_ra1:
                        entry_reason_idx = entry_triggers_map['R']
                        entry_type_trades[entry_reason_idx] += 1
                        hits[entry_reason_idx] += 1
                        is_in_position, daily_entry_count, entry_price, entry_timestamp_idx, highest_price_since_entry = True, daily_entry_count + 1, close_asset[i], i, close_asset[i]
                        fear_index_no_trade = fear_index_array[i]
                        trade_entry_fear_state = current_fear_state
                        
                        current_leverage = current_lev_re
                        
                        if p_entry_percentage > 0:
                            base_trade_capital = balance * (p_entry_percentage / 100.0)
                            reserved_balance = balance - base_trade_capital
                        else:
                            base_trade_capital = balance
                            reserved_balance = 0.0

                        leveraged_capital = base_trade_capital * current_leverage
                        asset_quantity = (leveraged_capital * (1 - p_fixed_fee)) / entry_price
                        balance = 0.0
                        continue

        if is_in_position:
            highest_price_since_entry = max(highest_price_since_entry, close_asset[i])
            exited, exit_reason_idx, cooldown = False, -1, 0
            minutes_since_entry = i - entry_timestamp_idx
            current_profit_pct = (close_asset[i] - entry_price) / entry_price if entry_price > 0 else 0
            
            # --- Liquidation Stop ---
            if current_leverage > 1.0:
                liquidation_threshold = 1.0 / current_leverage
                current_drop_pct = (entry_price - close_asset[i]) / entry_price if entry_price > 0 else 0
                if current_drop_pct >= liquidation_threshold: 
                    exited, exit_reason_idx, cooldown = True, 12, p_cooldown_sl
            
            if not exited and is_fear_stop_active_today and p_fear_stop_exit_position: exited, exit_reason_idx, cooldown = True, 11, p_cooldown_sl
            
            if not exited:
                btc_exit_signal = False
                if i >= eff_btc_exit_per:
                    if (close_btc[i] - close_btc[i - eff_btc_exit_per]) / close_btc[i - eff_btc_exit_per] <= -eff_btc_exit_val: btc_exit_signal = True
                btc_exit_long_signal = False
                if i >= eff_btc_exit_long_per:
                    if (close_btc[i] - close_btc[i - eff_btc_exit_long_per]) / close_btc[i - eff_btc_exit_long_per] <= -eff_btc_exit_long_val: btc_exit_long_signal = True
                btc_exit_panic_signal = False
                if i >= p_btc_exit_panic_trigger_period:
                    if (close_btc[i] - close_btc[i - p_btc_exit_panic_trigger_period]) / close_btc[i - p_btc_exit_panic_trigger_period] <= -p_btc_exit_panic_trigger_value: btc_exit_panic_signal = True
                asset_exit_panic_signal = False
                if i >= p_asset_exit_panic_trigger_period:
                    if (close_asset[i] - close_asset[i - p_asset_exit_panic_trigger_period]) / close_asset[i - p_asset_exit_panic_trigger_period] <= -p_asset_exit_panic_trigger_value: asset_exit_panic_signal = True
                combined_btc_exit_signal = btc_exit_signal or btc_exit_long_signal
                
                # --- Optional SL Logic ---
                if p_stop_loss_hard > 0 and (entry_price - close_asset[i]) / entry_price >= p_stop_loss_hard: 
                     if not p_disable_stop_loss:
                        exited, exit_reason_idx, cooldown = True, 9, p_cooldown_sl
                
                elif btc_exit_panic_signal: exited, exit_reason_idx, cooldown = True, 13, p_cooldown_sl
                elif asset_exit_panic_signal: exited, exit_reason_idx, cooldown = True, 14, p_cooldown_sl
                elif p_exit_position_red_alert:
                    if p_red_alert_trigger_pct < 999 and highest_price_since_entry > 0:
                        if (highest_price_since_entry - close_asset[i]) / highest_price_since_entry >= p_red_alert_trigger_pct: exited, exit_reason_idx, cooldown = True, 10, p_cooldown_sl
                elif combined_btc_exit_signal: exited, exit_reason_idx, cooldown = True, (6 if btc_exit_long_signal else 5), p_cooldown_sl
                
                elif not p_disable_stop_loss and p_stop_loss > 0 and (entry_price - close_asset[i]) / entry_price >= p_stop_loss and minutes_since_entry >= p_stop_loss_cooldown: 
                    exited, exit_reason_idx, cooldown = True, 8, p_cooldown_sl
            
            if not exited:
                if p_take_profit_dynamic:
                    if tp_confirmation_mode_active:
                        if close_asset[i] <= tp_confirmation_sl_level: exited, exit_reason_idx, cooldown = True, 7, p_cooldown_tp
                        elif close_asset[i] >= tp_confirmation_end_level:
                            tp_profit_floor_pct = tp_pending_floor_pct
                            base_tp_level = entry_price * (1 + tp_profit_floor_pct)
                            sl_increment = entry_price * p_take_profit_update_add_to_sl
                            tp_level = base_tp_level + sl_increment
                            if p_take_profit_update > 0: total_tp_updates += round((tp_profit_floor_pct - (tp_profit_floor_pct-p_take_profit_update)) / p_take_profit_update)
                            tp_confirmation_mode_active, tp_confirmation_sl_level, tp_confirmation_end_level, tp_pending_floor_pct = False, 0.0, 0.0, 0.0
                    else:
                        if close_asset[i] <= tp_level and tp_level > 0: exited, exit_reason_idx, cooldown = True, 7, p_cooldown_tp
                        else:
                            target_profit_floor_pct = 0.0
                            if p_take_profit_trigger > 0 and current_profit_pct >= p_take_profit_trigger:
                                profit_above_trigger = current_profit_pct - p_take_profit_trigger
                                num_steps = int(profit_above_trigger / p_take_profit_multiplier) if p_take_profit_multiplier > 0 else 0
                                target_profit_floor_pct = p_take_profit_trigger + (num_steps * p_take_profit_update)
                            if target_profit_floor_pct > tp_profit_floor_pct:
                                tp_confirmation_mode_active = True
                                tp_pending_floor_pct = target_profit_floor_pct
                                target_tp_level = entry_price * (1 + target_profit_floor_pct)
                                tp_confirmation_sl_level = target_tp_level * (1 - p_take_profit_confirmation_sl_pct)
                                tp_confirmation_end_level = target_tp_level * (1 + p_take_profit_confirmation_end_pct)
                else:
                    if p_take_profit_trigger > 0 and current_profit_pct >= p_take_profit_trigger: exited, exit_reason_idx, cooldown = True, 7, p_cooldown_tp
            if exited:
                exit_price = close_asset[i]
                gross_sell_value = asset_quantity * exit_price
                net_sell_value = gross_sell_value * (1 - p_fixed_fee)
                trade_profit = net_sell_value - leveraged_capital
                if log_enabled and trade_log_idx < len(trade_log_data):
                    trade_log_data[trade_log_idx] = [timestamps_int[entry_timestamp_idx], timestamps_int[i], entry_price, exit_price, trade_profit, entry_reason_idx, exit_reason_idx, fear_index_no_trade]
                    trade_log_idx += 1
                
                # --- Balance Recomposition ---
                balance = reserved_balance + base_trade_capital + trade_profit
                
                if balance < 0: balance = 0.0
                total_trades += 1
                if trade_profit > 0:
                    total_wins += 1
                    if entry_reason_idx != -1: entry_type_wins[entry_reason_idx] += 1
                if trade_entry_fear_state != -1:
                    fear_state_trades[trade_entry_fear_state] += 1
                    if trade_profit > 0: fear_state_wins[trade_entry_fear_state] += 1
                hits[exit_reason_idx] += 1
                is_in_position, cooldown_timer, last_exit_price, red_alert_active, lowest_price_since_alert = False, cooldown, exit_price, False, np.inf
                tp_level, tp_profit_floor_pct = 0.0, 0.0
                trade_entry_fear_state = -1
                tp_confirmation_mode_active, tp_confirmation_sl_level, tp_confirmation_end_level, tp_pending_floor_pct = False, 0.0, 0.0, 0.0
                reserved_balance = 0.0 

        if not is_in_position and cooldown_timer == 0 and balance > 0:
            btc_entry_signal, btc_entry_long_signal = False, False
            if i >= eff_btc_entry_per:
                 if (close_btc[i] - close_btc[i - eff_btc_entry_per]) / close_btc[i - eff_btc_entry_per] >= eff_btc_entry_val: btc_entry_signal = True
            if i >= eff_btc_entry_long_per:
                 if (close_btc[i] - close_btc[i - eff_btc_entry_long_per]) / close_btc[i - eff_btc_entry_long_per] >= eff_btc_entry_long_val: btc_entry_long_signal = True
            reentry_signal, reentry_signal_2 = False, False
            if i >= eff_reentry_win:
                 if (close_asset[i] - close_asset[i - eff_reentry_win]) / close_asset[i - eff_reentry_win] >= eff_reentry_pct: reentry_signal = True
            if i >= eff_reentry_win_2:
                 if (close_asset[i] - close_asset[i - eff_reentry_win_2]) / close_asset[i - eff_reentry_win_2] >= eff_reentry_pct_2: reentry_signal_2 = True
            sp_btc_entry_signal = False
            if i >= p_sp_btc_entry_trigger_period:
                 if (close_btc[i] - close_btc[i - p_sp_btc_entry_trigger_period]) / close_btc[i - p_sp_btc_entry_trigger_period] >= p_sp_btc_entry_trigger_value: sp_btc_entry_signal = True
            sp_reentry_signal = False
            if i >= p_sp_reentry_window_values:
                 if (close_asset[i] - close_asset[i - p_sp_reentry_window_values]) / close_asset[i - p_sp_reentry_window_values] >= p_sp_reentry_pct_values: sp_reentry_signal = True

            # --- FILTER 1: BTC Minimum Condition ---
            if p_btc_minimum_enable:
                btc_condition_met = False
                if i >= p_btc_minimum_window:
                    price_past = close_btc[i - p_btc_minimum_window]
                    if price_past > 0:
                        change = (close_btc[i] - price_past) / price_past
                        if change >= p_btc_minimum_pct_to_entry:
                            btc_condition_met = True
                if not btc_condition_met:
                    reentry_signal = False
                    reentry_signal_2 = False
                    sp_reentry_signal = False

            # --- FILTER 2: Asset Minimum Condition ---
            if p_asset_minimum_enable:
                asset_condition_met = False
                if i >= p_asset_minimum_window:
                    asset_past = close_asset[i - p_asset_minimum_window]
                    if asset_past > 0:
                        asset_change = (close_asset[i] - asset_past) / asset_past
                        if asset_change >= p_asset_minimum_pct_to_entry:
                            asset_condition_met = True
                if not asset_condition_met:
                    btc_entry_signal = False
                    btc_entry_long_signal = False
                    sp_btc_entry_signal = False

            if is_fear_stop_active_today:
                reentry_signal = False
                reentry_signal_2 = False
                sp_reentry_signal = False
                if p_fear_stop_btc_entry:
                    btc_entry_signal = False
                    btc_entry_long_signal = False
                    sp_btc_entry_signal = False

            entry_signal = btc_entry_signal or btc_entry_long_signal or reentry_signal or reentry_signal_2 or sp_btc_entry_signal or sp_reentry_signal

            if entry_signal:
                is_btc_entry_flag = btc_entry_signal or btc_entry_long_signal or sp_btc_entry_signal
                
                red_alert_blocks_entry = False
                
                if red_alert_active:
                    if p_red_alert_block_btc_entries:
                        red_alert_blocks_entry = True
                        ra1_blocked_count += 1
                    elif not is_btc_entry_flag:
                        red_alert_blocks_entry = True
                        ra1_blocked_count += 1
                
                if red_alert2_active:
                    blocked_by_ra2 = False
                    if p_red_alert_block_btc_entries:
                        blocked_by_ra2 = True
                    elif not is_btc_entry_flag:
                        blocked_by_ra2 = True
                    
                    if blocked_by_ra2:
                        red_alert_blocks_entry = True
                        ra2_blocked_count += 1
                
                if not red_alert_blocks_entry:
                    limit_reached = daily_entry_count >= p_max_entries_day
                    if not (limit_reached and (p_max_entries_day_btc or not is_btc_entry_flag)):
                        if sp_btc_entry_signal: entry_reason_idx = entry_triggers_map['SP_BE']
                        elif sp_reentry_signal: entry_reason_idx = entry_triggers_map['SP_R']
                        elif btc_entry_signal: entry_reason_idx = entry_triggers_map['BE']
                        elif btc_entry_long_signal: entry_reason_idx = entry_triggers_map['BEL']
                        elif reentry_signal: entry_reason_idx = entry_triggers_map['R']
                        elif reentry_signal_2: entry_reason_idx = entry_triggers_map['R2']

                        if entry_reason_idx != -1: entry_type_trades[entry_reason_idx] += 1
                        if entry_reason_idx <= 3: hits[entry_reason_idx] += 1
                        elif entry_reason_idx == 4: hits[15] += 1
                        elif entry_reason_idx == 5: hits[16] += 1
                        
                        is_in_position, daily_entry_count, entry_price, entry_timestamp_idx, highest_price_since_entry = True, daily_entry_count + 1, close_asset[i], i, close_asset[i]
                        fear_index_no_trade = fear_index_array[i]
                        trade_entry_fear_state = current_fear_state

                        # --- DYNAMIC LEVERAGE SELECTION ---
                        if entry_reason_idx == 4 or entry_reason_idx == 5: current_leverage = current_lev_sp
                        elif entry_reason_idx <= 1: current_leverage = current_lev_btc
                        else: current_leverage = current_lev_re
                        
                        # --- Fractional Entry Logic ---
                        if p_entry_percentage > 0:
                            base_trade_capital = balance * (p_entry_percentage / 100.0)
                            reserved_balance = balance - base_trade_capital
                        else:
                            base_trade_capital = balance
                            reserved_balance = 0.0

                        leveraged_capital = base_trade_capital * current_leverage
                        asset_quantity = (leveraged_capital * (1 - p_fixed_fee)) / entry_price
                        balance = 0.0

    if is_in_position:
        exit_price = close_asset[-1]
        net_sell_value = (asset_quantity * exit_price) * (1 - p_fixed_fee)
        trade_profit = net_sell_value - leveraged_capital
        if log_enabled and trade_log_idx < len(trade_log_data):
            trade_log_data[trade_log_idx] = [timestamps_int[entry_timestamp_idx], timestamps_int[-1], entry_price, exit_price, trade_profit, entry_reason_idx, -2, fear_index_no_trade]
            trade_log_idx += 1
        
        # --- Final Balance Recomposition ---
        balance = reserved_balance + base_trade_capital + trade_profit
        
        if balance < 0: balance = 0.0
        total_trades += 1
        if trade_profit > 0:
            total_wins += 1
            if entry_reason_idx != -1: entry_type_wins[entry_reason_idx] += 1
        if trade_entry_fear_state != -1:
            fear_state_trades[trade_entry_fear_state] += 1
            if trade_profit > 0: fear_state_wins[trade_entry_fear_state] += 1

    return (balance, total_trades, total_wins, minutes_in_position, hits, total_tp_updates, entry_type_trades, entry_type_wins, trade_log_data[:trade_log_idx], fear_high_hits, fear_low_hits, fear_stop_hits, fear_state_trades, fear_state_wins, ra1_activation_count, ra1_blocked_count, ra2_activation_count, ra2_blocked_count)

def run_vectorized_backtest(params, btc_candles, asset_candles, fear_candles, index, log_file=None, lock=None):
    p = {key: value[0] for key, value in params.items()}

    if not pd.api.types.is_datetime64_any_dtype(btc_candles['timestamp']):
        btc_candles['timestamp'] = pd.to_datetime(btc_candles['timestamp'], utc=True)
    btc_candles_prep = btc_candles.set_index('timestamp').resample('1min').ffill().reset_index()

    if not pd.api.types.is_datetime64_any_dtype(asset_candles['timestamp']):
        asset_candles['timestamp'] = pd.to_datetime(asset_candles['timestamp'], utc=True)
    asset_candles_prep = asset_candles.set_index('timestamp').resample('1min').ffill().reset_index()
    
    df = pd.merge(btc_candles_prep.rename(columns={'close': 'close_btc'}), asset_candles_prep.rename(columns={'close': 'close_asset'}), on='timestamp', how='inner')

    if df.empty: return None, None

    df['date'] = df['timestamp'].dt.date
    fear_candles['date'] = fear_candles['timestamp'].dt.date
    df = pd.merge(df, fear_candles[['date', 'fear_index']], on='date', how='left')
    df['fear_index'] = df['fear_index'].fillna(-1)

    day_of_year_array = df['timestamp'].dt.dayofyear.to_numpy(dtype=np.int32)
    timestamps_int_array = (df['timestamp'].astype(np.int64) // 10**9).to_numpy()
    close_asset_array = df['close_asset'].to_numpy(dtype=np.float64)
    close_btc_array = df['close_btc'].to_numpy(dtype=np.float64)
    fear_index_array = df['fear_index'].to_numpy(dtype=np.int32)

    balance, total_trades, total_wins, minutes_in_position, hits, total_tp_updates, entry_trades, entry_wins, trade_log_array, fear_high_hits, fear_low_hits, fear_stop_hits, fear_state_trades, fear_state_wins, ra1_acts, ra1_blks, ra2_acts, ra2_blks = numba_jit_simulation(
        timestamps_int_array, day_of_year_array, close_asset_array, close_btc_array, fear_index_array,
        p.get('initial_balance', 1000.0), p.get('fixed_fee', 0.001),
        p.get('leverage_BTC', 1), p.get('leverage_Reentry', 1),
        
        # Passing new leverage parameters to JIT
        p.get('fear_high_leverage_BTC', 1), p.get('fear_high_leverage_Reentry', 1),
        p.get('fear_low_leverage_BTC', 1), p.get('fear_low_leverage_Reentry', 1),
        
        p.get('red_alert_trigger_pct', 999), p.get('red_alert_reset_pct', 999),
        p.get('red_alert_reset_auto_entry', False), 
        p.get('red_alert2_trigger_pct', 0.085), p.get('red_alert2_window', 500), p.get('red_alert2_reset_pct', 0.03), p.get('red_alert2_reset_auto_entry', False),
        p.get('red_alert_block_btc_entries', False),
        p.get('max_entries_day', 999),
        p.get('max_entries_day_btc', True), p.get('stop_loss_hard', 0), p.get('cooldown_sl', 0),
        p.get('stop_loss', 0), p.get('stop_loss_cooldown', 0), p.get('exit_position_red_alert', False),
        p.get('take_profit_dynamic', False), p.get('take_profit_trigger', 0),
        p.get('take_profit_update_enable', False), p.get('take_profit_update', 0),
        p.get('take_profit_multiplier', 0), p.get('cooldown_tp', 0),
        p.get('take_profit_confirmation_sl_pct', 0), p.get('take_profit_confirmation_end_pct', 0),
        p.get('take_profit_update_add_to_sl', 0), 
        p['btc_entry_trigger_value'], p['btc_entry_trigger_period'],
        p['btc_exit_trigger_value'], p['btc_exit_trigger_period'],
        p['btc_entry_long_trigger_value'], p['btc_entry_long_trigger_period'],
        p['btc_exit_long_trigger_value'], p['btc_exit_long_trigger_period'],
        p['reentry_pct_values'], p['reentry_window_values'],
        p['reentry_pct_values_2'], p['reentry_window_values_2'],
        p['btc_exit_panic_trigger_value'], p['btc_exit_panic_trigger_period'],
        p['asset_exit_panic_trigger_value'], p['asset_exit_panic_trigger_period'],
        p.get('leverage_SP', 1),
        # New SP fear params
        p.get('fear_high_leverage_SP', 1), p.get('fear_low_leverage_SP', 1),

        p.get('sp_btc_entry_trigger_value', 0), p.get('sp_btc_entry_trigger_period', 0),
        p.get('sp_reentry_pct_values', 0), p.get('sp_reentry_window_values', 0),
        p['fear_high_override'], p['fear_high_value'],
        p['fear_high_btc_entry_trigger_value'], p['fear_high_btc_entry_trigger_period'],
        p['fear_high_btc_exit_trigger_value'], p['fear_high_btc_exit_trigger_period'],
        p['fear_high_btc_entry_long_trigger_value'], p['fear_high_btc_entry_long_trigger_period'],
        p['fear_high_btc_exit_long_trigger_value'], p['fear_high_btc_exit_long_trigger_period'],
        p['fear_high_reentry_pct_values'], p['fear_high_reentry_window_values'],
        p['fear_high_reentry_pct_values_2'], p['fear_high_reentry_window_values_2'],
        p['fear_low_override'], p['fear_low_value'],
        p['fear_low_btc_entry_trigger_value'], p['fear_low_btc_entry_trigger_period'],
        p['fear_low_btc_exit_trigger_value'], p['fear_low_btc_exit_trigger_period'],
        p['fear_low_btc_entry_long_trigger_value'], p['fear_low_btc_entry_long_trigger_period'],
        p['fear_low_btc_exit_long_trigger_value'], p['fear_low_btc_exit_long_trigger_period'],
        p['fear_low_reentry_pct_values'], p['fear_low_reentry_window_values'],
        p['fear_low_reentry_pct_values_2'], p['fear_low_reentry_window_values_2'],
        p.get('fear_stop', False), p.get('fear_stop_value', 0), p.get('fear_stop_btc_entry', False), p.get('fear_stop_exit_position', False),
        p.get('entry_percentage', 0), p.get('disable_stop_loss', False),
        
        p.get('btc_minimum_enable', False), p.get('btc_minimum_pct_to_entry', 0.0), p.get('btc_minimum_window', 1),
        p.get('asset_minimum_enable', False), p.get('asset_minimum_pct_to_entry', 0.0), p.get('asset_minimum_window', 1),

        log_file is not None
    )

    log_df = None
    if trade_log_array.shape[0] > 0:
        log_df = pd.DataFrame(trade_log_array, columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'entry_reason', 'exit_reason', 'fear_index_entry'])
        log_df['entry_time'] = pd.to_datetime(log_df['entry_time'], unit='s')
        log_df['exit_time'] = pd.to_datetime(log_df['exit_time'], unit='s')

        if log_file and lock:
            entry_map = {0: 'BE', 1: 'BEL', 2: 'R1', 3: 'R2', 4: 'SP_BE', 5: 'SP_R'}
            exit_map = {5: 'BX', 6: 'BXL', 7: 'TP', 8: 'SL', 9: 'SLH', 10: 'RA', 11: 'FS', 12: 'LIQ', 13: 'BXP', 14: 'AXP', -2: 'END'}
            with lock:
                with open(log_file, 'a', encoding='utf-8') as f:
                    for _, row in log_df.iterrows():
                        f.write(f"ID:{index+1};ENTRY:{row['entry_time']};EXIT:{row['exit_time']};E_PRICE:{row['entry_price']:.4f};X_PRICE:{row['exit_price']:.4f};PROFIT:{row['profit']:.4f};E_REASON:{entry_map.get(row['entry_reason'], 'UKN')};X_REASON:{exit_map.get(row['exit_reason'], 'UKN')};FEAR_ENTRY:{int(row['fear_index_entry'])}\n")

    total_minutes_backtest = len(df)
    time_in_pos_pct = (minutes_in_position / total_minutes_backtest) * 100 if total_minutes_backtest > 0 else 0

    win_rate_be, win_rate_bel, win_rate_r1, win_rate_r2 = 0, 0, 0, 0
    if entry_trades[0] > 0: win_rate_be = (entry_wins[0] / entry_trades[0]) * 100
    if entry_trades[1] > 0: win_rate_bel = (entry_wins[1] / entry_trades[1]) * 100
    if entry_trades[2] > 0: win_rate_r1 = (entry_wins[2] / entry_trades[2]) * 100
    if entry_trades[3] > 0: win_rate_r2 = (entry_wins[3] / entry_trades[3]) * 100
    
    win_rate_sp_be = 0
    win_rate_sp_r = 0
    if entry_trades[4] > 0: win_rate_sp_be = (entry_wins[4] / entry_trades[4]) * 100
    if entry_trades[5] > 0: win_rate_sp_r = (entry_wins[5] / entry_trades[5]) * 100

    win_rate_standard = (fear_state_wins[0] / fear_state_trades[0]) * 100 if fear_state_trades[0] > 0 else 0
    win_rate_high = (fear_state_wins[1] / fear_state_trades[1]) * 100 if fear_state_trades[1] > 0 else 0
    win_rate_low = (fear_state_wins[2] / fear_state_trades[2]) * 100 if fear_state_trades[2] > 0 else 0

    result = {
        'test_id': int(index + 1), 'final_balance': round(balance, 2),
        'final_balance_pct': round(((balance - p.get('initial_balance', 1000.0)) / p.get('initial_balance', 1000.0)) * 100, 2),
        'final_hold_balance': round((p.get('initial_balance', 1000.0) / df['close_asset'].iloc[0]) * df['close_asset'].iloc[-1] if df['close_asset'].iloc[0] > 0 else p.get('initial_balance', 1000.0), 2),
        'total_trades': int(total_trades), 'win_rate': round((total_wins / total_trades) * 100 if total_trades > 0 else 0, 2),
        'time_in_pos_pct': round(time_in_pos_pct, 2),
        'btc_entry_hits': int(hits[0]), 'reentry_hits': int(hits[2]), 'reentry_2_hits': int(hits[3]),
        'btc_exit_hits': int(hits[5]), 'btc_exit_long_hits': int(hits[6]),
        'take_profit_hits': int(hits[7]), 'total_tp_updates': int(total_tp_updates),
        'stop_loss_hits': int(hits[8]), 'stop_loss_hard_hits': int(hits[9]),
        'ra_exit_pos_hits': int(hits[10]), 'btc_entry_long_hits': int(hits[1]),
        'fear_stop_exit_hits': int(hits[11]),
        'liquidation_hits': int(hits[12]),
        'btc_exit_panic_hits': int(hits[13]),
        'asset_exit_panic_hits': int(hits[14]),
        'sp_be_hits': int(hits[15]),
        'sp_r_hits': int(hits[16]),
        'win_rate_btc_entry': round(win_rate_be, 2), 'win_rate_btc_entry_long': round(win_rate_bel, 2),
        'win_rate_reentry': round(win_rate_r1, 2), 'win_rate_reentry_2': round(win_rate_r2, 2),
        'win_rate_sp_be': round(win_rate_sp_be, 2),
        'win_rate_sp_r': round(win_rate_sp_r, 2),
        'fear_high_hits': int(fear_high_hits), 'fear_low_hits': int(fear_low_hits),
        'fear_stop_hits': int(fear_stop_hits),
        'win_rate_fear_standard': round(win_rate_standard, 2),
        'win_rate_fear_high': round(win_rate_high, 2),
        'win_rate_fear_low': round(win_rate_low, 2),
        'ra1_activation_count': int(ra1_acts),
        'ra1_blocked_count': int(ra1_blks),
        'ra2_activation_count': int(ra2_acts),
        'ra2_blocked_count': int(ra2_blks),
    }
    result.update({k: v for k, v in p.items()})

    return result, log_df
