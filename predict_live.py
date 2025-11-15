import torch
import joblib
import time
from mexc_client import MEXCClient
from model import CryptoPredicter
from exchange import Exchange
import numpy as np
import pandas as pd
import subprocess
import logging
from datetime import datetime
import json
import os
import sys
import shutil

# ═══════════════════════════════════════════════════════════
# 📝 LOGGING & STATS CONFIGURATION
# ═══════════════════════════════════════════════════════════
LOG_FILE = "trading_bot.log"
STATS_FILE = "trading_stats.json"
TRADES_LOG_FILE = "trades_log.json"  # Separate file for individual trade logs
MODEL_DIR = "../model"  # Directory for saving/loading models and scalers
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# 🧪 TEST MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════
# Set TEST = True to simulate trades without using real money
# Set TEST = False for live trading with real money
TEST = True  # ⚠️ CHANGE THIS TO False FOR LIVE TRADING
# ═══════════════════════════════════════════════════════════

                                    # ═══════════════════════════════════════════════════════════
                                    # ⚠️ FUTURES TRADING RISK CONFIGURATION
                                    # ═══════════════════════════════════════════════════════════
                                    # Futures trading uses leverage and is EXTREMELY risky!
                                    # Only trade with very high confidence to minimize risk
CONFIDENCE_THRESHOLD_TRADE  = 0.70  # Only trade when confidence > 70%
CONFIDENCE_THRESHOLD_TEST   = 0.70  # Trigger model testing when below 70%
MAX_POSITION_RISK           = 0.10  # Max 10% of balance at risk per trade
MAX_LEVERAGE                = 75    # Maximum leverage to use (lower = safer)
                                    # Early stop parameters
EARLY_STOP_MAX_TIME_MINUTES = 120   # Close trade if it's been red for this long (minutes) - 2 hours
EARLY_STOP_OPPOSITE_SIGNAL  = True  # Close losing trades if model predicts opposite signal
                                    # Early stop triggers when BOTH conditions are met: time limit AND opposite signal
                                    # Model refinement parameters
REFINEMENT_INTERVAL_SECONDS = 3600  # Trigger model refinement every 1 hour (3600 seconds)
                                    # ═══════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging to file and console."""
    # Using basicConfig to set up handlers, ensuring it runs only once
    if not logging.getLogger().handlers:
        # Use append mode for log file
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set format for both handlers
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

def main():
    setup_logging()
    logging.info("🚀 Starting trading bot...")
    
    model = CryptoPredicter()
    test_model = CryptoPredicter()
    client = MEXCClient()
    exchange = Exchange()
    
    look_back = 60
    future_horizon = 15  # Look 15 minutes ahead for TP/SL predictions 
    
    # Track active trades for loss detection (now a list to support multiple trades)
    active_trades = []
    simulated_trades = []  # Track simulated trades in TEST mode
    trade_counter = 0  # Counter for trade index
    
    # Track model testing state
    testing_model = False
    testing_start_time = None
    testing_duration = 3 * 60  # 3 minutes in seconds
    current_model_predictions = []
    test_model_predictions = []
    
    # Track model refinement timing
    last_refinement_time = None  # Will be set on first refinement

    # Load daily trading stats
    daily_stats = load_stats()
    
    while True:
        try:
            # Check for day change to reset stats
            today = datetime.now().strftime("%Y-%m-%d")
            if daily_stats.get('date') != today:
                logging.info(f"New day ({today}). Resetting daily stats.")
                save_stats(daily_stats) # Save previous day's stats one last time
                daily_stats = {
                    "date": today, "successful_trades": 0, "failed_trades": 0, "total_profit_usdt": 0.0
                }

            # Display TEST mode warning
            if TEST:
                logging.info("\n" + "⚠️ "*30)
                logging.info("🧪 TEST MODE ACTIVE - SIMULATING TRADES (NO REAL MONEY)")
                logging.info("⚠️ "*30)
            
            # Check if testing period is over
            if testing_model:
                elapsed_time = time.time() - testing_start_time
                if elapsed_time >= testing_duration:
                    # Testing period complete - compare models and decide
                    if should_adopt_new_model(current_model_predictions, test_model_predictions):
                        logging.info("\n✅ New model performs BETTER! Adopting as primary model.")
                        adopt_new_model()
                    else:
                        logging.info("\n❌ New model does NOT perform better. Keeping current model.")
                    
                    testing_model = False
                    current_model_predictions = []
                    test_model_predictions = []
            
            # Load the current production model
            model_path = os.path.join(MODEL_DIR, 'btc_predicter_model.pth')
            scaler_path = os.path.join(MODEL_DIR, 'scaler.gz')
            close_scaler_path = os.path.join(MODEL_DIR, 'close_scaler.gz')
            
            model.load_model(model_path)
            model.eval()
            scaler = joblib.load(scaler_path)
            
            # Load close_scaler for TP/SL inverse transform
            if os.path.exists(close_scaler_path):
                close_scaler = joblib.load(close_scaler_path)
            else:
                close_scaler = None
                logging.warning(f"{close_scaler_path} not found. TP/SL predictions may be inaccurate.")
            
            if testing_model:
                # Also load the test model for comparison
                test_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
                test_model.load_model(test_model_path)
                test_model.eval()
                elapsed_time = time.time() - testing_start_time
                remaining_min = int((testing_duration - elapsed_time) / 60)
                remaining_sec = int((testing_duration - elapsed_time) % 60)
                logging.info(f"\n🧪 TESTING MODE - Comparing both models ({remaining_min}m {remaining_sec}s remaining)")
                logging.info(f"   Current: {model_path} | Test: {test_model_path}")
            else:
                logging.info(f"\nModel loaded: {model_path}")

            # Fetch data for prediction - need extra for technical indicators
            # Futures API returns up to 2000 records, so we can get all we need in one request
            df = client.get_kline_data(symbol='BTC_USDT', interval='Min1')  # Gets recent data
            
            if df is None or df.empty:
                logging.error("Failed to fetch market data. Skipping this cycle.")
                time.sleep(60)
                continue
            
            # Immediately append new data to training_data.csv
            append_data_to_training_file(df)
            
            # Check if it's time to trigger hourly model refinement
            current_time = time.time()
            should_refine = False
            if last_refinement_time is None:
                # First time - trigger refinement to initialize
                should_refine = True
                logging.info(f"\n⏰ First refinement cycle - initializing hourly refinement schedule")
            else:
                time_since_last_refinement = current_time - last_refinement_time
                if time_since_last_refinement >= REFINEMENT_INTERVAL_SECONDS:
                    should_refine = True
                    hours_elapsed = time_since_last_refinement / 3600
                    logging.info(f"\n⏰ Hourly refinement trigger: {hours_elapsed:.2f} hours since last refinement")
            
            # Trigger refinement if needed and not already testing
            if should_refine and not testing_model:
                logging.info("🔄 Triggering scheduled model refinement with recent data...")
                retrain_with_recent_data(client)
                # Start testing the new model
                testing_model = True
                testing_start_time = time.time()
                last_refinement_time = current_time  # Update last refinement time
                current_model_predictions = []
                test_model_predictions = []
                logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
            elif should_refine and testing_model:
                # Still in testing phase, reschedule for after testing completes
                logging.info(f"⏸️  Refinement scheduled but currently testing model. Will refine after testing completes.")
                last_refinement_time = current_time - (REFINEMENT_INTERVAL_SECONDS - testing_duration)  # Adjust to refine after testing
                
            current_price = df['close'].iloc[-1]
            
            # Check all active trades (real or simulated) for completion or loss
            trades_list = simulated_trades if TEST else active_trades
            trades_to_remove = []
            
            for i, trade in enumerate(trades_list):
                trade_loss   = check_trade_loss(trade, current_price)
                trade_profit = is_trade_complete(trade, current_price)
                
                if trade_loss:
                                # Use stored trade index - should always exist since we assign it when creating trades
                    trade_index = trade.get('index', None)
                    if trade_index is None:
                        # Fallback: assign index if somehow missing (shouldn't happen)
                        trade_counter += 1
                        trade_index = trade_counter
                        logging.warning(f"⚠️ Trade missing index, assigned new index: {trade_index}")
                    pnl = print_trade_result(trade, current_price, result='LOSS', simulated=TEST, trade_index=trade_index)
                    if not TEST:
                        update_stats(daily_stats, 'LOSS', pnl)
                    trades_to_remove.append(i)
                    # Trigger retraining on first loss (don't retrain for every loss)
                    if not testing_model:
                        logging.info(f"\n⚠️ Trade resulted in a LOSS. Triggering model retraining with recent 1 hour data.")
                        retrain_with_recent_data(client)
                        # Start testing the new model
                        testing_model = True
                        testing_start_time = time.time()
                        last_refinement_time = time.time()  # Update last refinement time
                        current_model_predictions = []
                        test_model_predictions = []
                        logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
                elif trade_profit:
                    # Use stored trade index - should always exist since we assign it when creating trades
                    trade_index = trade.get('index', None)
                    if trade_index is None:
                        # Fallback: assign index if somehow missing (shouldn't happen)
                        trade_counter += 1
                        trade_index = trade_counter
                        logging.warning(f"⚠️ Trade missing index, assigned new index: {trade_index}")
                    pnl = print_trade_result(trade, current_price, result='PROFIT', simulated=TEST, trade_index=trade_index)
                    if not TEST:
                        update_stats(daily_stats, 'PROFIT', pnl)
                    trades_to_remove.append(i)
            
            # Remove completed trades (iterate in reverse to avoid index issues)
            for i in reversed(trades_to_remove):
                trades_list.pop(i)
            
            # Display active trades summary
            if trades_list:
                mode_label = "🧪 SIMULATED Trades" if TEST else "📊 Active Trades"
                logging.info(f"\n{mode_label}: {len(trades_list)}")
                for trade in trades_list: 
                    pnl_pct, pnl_usdt = calculate_unrealized_pnl(trade, current_price)
                    status_emoji = "🟢" if pnl_pct > 0 else "🔴"
                    trade_index = trade.get('index', '?')  # Use stored trade index
                    logging.info(f"   {status_emoji} Trade #{trade_index}: {trade['side']} | Entry: ${trade['entry_price']:.2f} | "
                               f"Unrealized P&L: {pnl_pct:+.2f}% (${pnl_usdt:+.2f})")
                
                                                                                                                                                                                                                                                                                                                                                # Note: Active trades are displayed above but not logged to JSON
                                                                                                                                                                                                                                                                                                                                                # Only closed trades are logged to trades_log.json with simplified structure
            
            # Prediction logic for current model
            # Calculate technical indicators from raw OHLCV data
            from data_processor import DataProcessor
            temp_processor = DataProcessor(look_back=look_back, future_horizon=future_horizon)
            
            try:
                df_with_indicators = temp_processor.calculate_technical_indicators(df)
            except Exception as e:
                logging.error(f"Failed to calculate indicators: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            # Select the same features used in training
            feature_columns = [
                'close', 'price_change', 'high_low_range', 'close_open_diff',
                'sma_5', 'sma_10', 'sma_20', 'sma_100', 'ema_5', 'ema_10',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
                'volume_change', 'volume_ratio',
                'momentum', 'rate_of_change', 'volatility'
            ]
            
            # Ensure we have enough data after indicator calculation
            if len(df_with_indicators) < look_back:
                logging.warning(f"Not enough data after indicators: {len(df_with_indicators)} < {look_back}. Skipping cycle.")
                time.sleep(60)
                continue
            
            # Take the last look_back rows
            features = df_with_indicators[feature_columns].tail(look_back).values
            
            # Scale features using the saved scaler
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                logging.error(f"Failed to scale features: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            # Create tensor with shape (1, look_back, num_features)
            try:
                X_pred = torch.from_numpy(features_scaled).float().unsqueeze(0)
                signal, signal_probs, tp_scaled, sl_scaled = model.predict(X_pred)
            except Exception as e:
                logging.error(f"Failed to make prediction: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            confidence       = signal_probs.max().item()
            predicted_signal = signal.item()
            
            # Check for early stops on active trades (before processing new predictions)
            # This allows us to use the predicted_signal to check for opposite signals
            if trades_list:
                for i, trade in enumerate(trades_list):
                    if check_early_stop(trade, current_price, predicted_signal):
                        # Use stored trade index - should always exist since we assign it when creating trades
                        # This ensures indices persist even when trades are stopped early (e.g., 1, 2, 3, 4, 5 -> if 3 stopped, still shows 1, 2, 4, 5)
                        trade_index = trade.get('index', None)
                        if trade_index is None:
                            # Fallback: assign index if somehow missing (shouldn't happen)
                            trade_counter += 1
                            trade_index = trade_counter
                            logging.warning(f"⚠️ Trade missing index, assigned new index: {trade_index}")
                        pnl = print_trade_result(trade, current_price, result='LOSS', simulated=TEST, trade_index=trade_index, early_stop=True)
                        if not TEST:
                            update_stats(daily_stats, 'LOSS', pnl)
                        trades_list.pop(i)
                        logging.info("🛑 Trade closed due to early stop condition")
                        break  # Only close one trade per cycle to avoid index issues
            
            # Use close_scaler for inverse transform of TP/SL
            signal_mismatch = False  # Default to no mismatch
            try:
                if close_scaler is not None:
                    tp = close_scaler.inverse_transform(tp_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                    sl = close_scaler.inverse_transform(sl_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                else:
                    # Fallback: use current price with percentage
                    tp = current_price * 1.01
                    sl = current_price * 0.99
                
                signal_map = {0: 'SELL', 1: 'BUY'}
                model_predicted_signal = predicted_signal  # Keep original model prediction
                
                # Focus trading on predicted TP: TP > entry → BUY, TP < entry → SELL
                if tp > current_price:
                    # TP is above entry → BUY signal based on TP
                    tp_based_signal = 1  # BUY
                    tp_pct = ((tp - current_price) / current_price) * 100  # Calculate TP percentage
                else:
                    # TP is below entry → SELL signal based on TP
                    tp_based_signal = 0  # SELL
                    tp_pct = ((current_price - tp) / current_price) * 100  # Calculate TP percentage
                
                # Use TP-based signal for trading
                predicted_signal = tp_based_signal
                
                # Check if model signal differs from TP-based signal
                signal_mismatch = (model_predicted_signal != tp_based_signal)
                
                if signal_mismatch:
                    # Model signal differs from TP-based signal - decrease TP percentage
                    logging.warning(f"⚠️ Signal mismatch: Model predicts {signal_map[model_predicted_signal]}, "
                                  f"but TP-based signal is {signal_map[tp_based_signal]} (TP: ${tp:.2f}). "
                                  f"Using TP-based signal with reduced TP %")
                    # Reduce TP percentage by half when signals differ
                    reduced_tp_pct = tp_pct * 0.5
                    
                    if tp_based_signal == 1:  # BUY
                        tp = current_price * (1 + reduced_tp_pct / 100)
                        sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        tp = current_price * (1 - reduced_tp_pct / 100)
                        sl = current_price * 1.20  # 20% above
                    
                    logging.info(f"📊 Using TP-based signal ({signal_map[tp_based_signal]}) with reduced TP: "
                               f"${tp:.2f} ({reduced_tp_pct:.2f}% vs original {tp_pct:.2f}%)")
                else:
                    # Model signal matches TP-based signal - use normal TP/SL values
                    if tp_based_signal == 1:  # BUY
                        # Use model's TP or default to 10% above if too low
                        if tp <= current_price:
                            tp = current_price * 1.10  # 10% above
                        sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        # Use model's TP or default to 10% below if too high
                        if tp >= current_price:
                            tp = current_price * 0.90  # 10% below
                        sl = current_price * 1.20  # 20% above
                    
                    logging.info(f"✅ Model signal matches TP-based signal ({signal_map[tp_based_signal]}) | "
                               f"TP: ${tp:.2f} ({tp_pct:.2f}%) | SL: ${sl:.2f}")
            except Exception as e:
                logging.error(f"Failed to calculate TP/SL: {e}", exc_info=True)
                tp = current_price * 1.01
                sl = current_price * 0.99
                # Default to BUY if TP calculation fails
                predicted_signal = 1 if tp > current_price else 0
            
                                                                              # Calculate recommended trade amount based on confidence and balance
            try:
                usdt_balance = exchange.get_usdt_balance()
                # Pass signal alignment info: higher investment when model signal matches TP-based signal
                signal_aligned = not signal_mismatch  # True if signals match
                trade_percentage, trade_amount_usdt, trade_quantity_btc = calculate_trade_amount(
                    confidence, usdt_balance, current_price, sl, signal_aligned=signal_aligned
                )
            except Exception as e:
                logging.error(f"Failed to calculate trade amount: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            # Display current model prediction
            logging.info(f"\n📊 Signal: {signal_map[predicted_signal]} | Confidence: {confidence:.2f} | "
                        f"TP: ${tp:.2f} | SL: ${sl:.2f} | Trade: ${trade_amount_usdt:.2f} ({trade_percentage:.1f}%)")

            # Track current model predictions during testing
            if testing_model:
                current_model_predictions.append({
                    'signal': predicted_signal,
                    'confidence': confidence,
                    'entry_price': current_price,
                    'tp': tp,
                    'sl': sl,
                    'timestamp': time.time()
                })
                
                # Also get test model prediction (same input features)
                test_signal, test_signal_probs, test_tp_scaled, test_sl_scaled = test_model.predict(X_pred)
                test_confidence = test_signal_probs.max().item()
                test_predicted_signal = test_signal.item()
                
                if close_scaler is not None:
                    test_tp = close_scaler.inverse_transform(test_tp_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                    test_sl = close_scaler.inverse_transform(test_sl_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                else:
                    test_tp = current_price * 1.01
                    test_sl = current_price * 0.99
                
                # Focus trading on predicted TP: TP > entry → BUY, TP < entry → SELL
                test_model_predicted_signal = test_predicted_signal  # Keep original model prediction
                
                if test_tp > current_price:
                    # TP is above entry → BUY signal based on TP
                    test_tp_based_signal = 1  # BUY
                    test_tp_pct = ((test_tp - current_price) / current_price) * 100
                else:
                    # TP is below entry → SELL signal based on TP
                    test_tp_based_signal = 0  # SELL
                    test_tp_pct = ((current_price - test_tp) / current_price) * 100
                
                # Use TP-based signal for trading
                test_predicted_signal = test_tp_based_signal
                
                # Check if test model signal differs from TP-based signal
                test_signal_mismatch = (test_model_predicted_signal != test_tp_based_signal)
                
                if test_signal_mismatch:
                    # Model signal differs from TP-based signal - decrease TP percentage
                    # Reduce TP percentage by half when signals differ
                    test_reduced_tp_pct = test_tp_pct * 0.5
                    
                    if test_tp_based_signal == 1:  # BUY
                        test_tp = current_price * (1 + test_reduced_tp_pct / 100)
                        test_sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        test_tp = current_price * (1 - test_reduced_tp_pct / 100)
                        test_sl = current_price * 1.20  # 20% above
                else:
                    # Model signal matches TP-based signal - use normal TP/SL values
                    if test_tp_based_signal == 1:  # BUY
                        # Use model's TP or default to 10% above if too low
                        if test_tp <= current_price:
                            test_tp = current_price * 1.10  # 10% above
                        test_sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        # Use model's TP or default to 10% below if too high
                        if test_tp >= current_price:
                            test_tp = current_price * 0.90  # 10% below
                        test_sl = current_price * 1.20  # 20% above
                
                # Continue with test model comparison after adjustments
                # Pass signal alignment info for test model
                test_signal_aligned = not test_signal_mismatch  # True if signals match
                test_trade_percentage, test_trade_amount_usdt, test_trade_quantity_btc = calculate_trade_amount(
                    test_confidence, usdt_balance, current_price, test_sl, signal_aligned=test_signal_aligned
                )
                
                # Display test model prediction
                logging.info("\n" + "="*60)
                logging.info("🧪 TEST MODEL PREDICTION")
                logging.info("="*60)
                print_prediction(test_predicted_signal, test_confidence, test_tp, test_sl, 
                               test_trade_percentage, test_trade_amount_usdt, test_trade_quantity_btc, usdt_balance)
                
                test_model_predictions.append({
                    'signal': test_predicted_signal,
                    'confidence': test_confidence,
                    'entry_price': current_price,
                    'tp': test_tp,
                    'sl': test_sl,
                    'timestamp': time.time()
                })

            # Trading and Learning Logic - STRICT THRESHOLDS FOR FUTURES
            if testing_model:
                logging.info("\n🧪 Testing mode: Both models running in parallel. Using current model for trading.")
                # Still execute trades with current model during testing if VERY high confidence
                if confidence >= CONFIDENCE_THRESHOLD_TRADE:
                    # Check if we should skip this trade: only open new trade if entry is better OR confidence is higher (>=0.9)
                    should_skip = False
                    skip_reasons = []
                    override_allowed = False
                    
                    for trade in trades_list:
                        if trade['signal'] == predicted_signal:  # Same signal type
                            existing_confidence = trade.get('confidence', 0.0)
                            entry_worse = False
                            
                            if predicted_signal == 1:  # BUY - better entry = lower price
                                entry_worse = current_price >= trade['entry_price']
                            else:  # SELL - better entry = higher price
                                entry_worse = current_price <= trade['entry_price']
                            
                            if entry_worse:
                                # Entry is worse, but check if confidence is high enough to override
                                if confidence >= 0.9 and confidence > existing_confidence:
                                    # High confidence (>=0.9) and higher than existing - allow trade
                                    logging.info(f"✅ Overriding worse entry: Confidence {confidence:.2f} >= 0.9 and higher than existing trade at ${trade['entry_price']:.2f} (conf: {existing_confidence:.2f})")
                                    override_allowed = True
                                    # Continue checking other trades
                                else:
                                    # Entry worse and not enough confidence to override
                                    skip_reasons.append(f"Trade at ${trade['entry_price']:.2f} (conf: {existing_confidence:.2f})")
                                    # Continue checking other trades
                    
                    # Only skip if we have skip reasons AND no override was allowed
                    if skip_reasons and not override_allowed:
                        should_skip = True
                        logging.info(f"⏭️  Skipping {signal_map[predicted_signal]} trade: Entry ${current_price:.2f} worse than existing {', '.join(skip_reasons)}. "
                                   f"Confidence {confidence:.2f} not high enough to override (need >=0.9 and > all existing)")
                    
                    if not should_skip:
                        # Assign trade index when creating the trade
                        trade_counter += 1
                        trade_index = trade_counter
                        logging.info(f"✅ VERY HIGH CONFIDENCE ({confidence:.2f}) - Executing trade")
                        if TEST:
                            new_trade = simulate_trade(predicted_signal, trade_quantity_btc, current_price, tp, sl, confidence=confidence)
                            new_trade['index'] = trade_index  # Store index in trade
                            simulated_trades.append(new_trade)
                        else:
                            new_trade = execute_trade(predicted_signal, exchange, trade_quantity_btc, current_price, tp, sl, confidence=confidence)
                            new_trade['index'] = trade_index  # Store index in trade
                            active_trades.append(new_trade)
                else:
                    logging.info(f"⏸️  Confidence {confidence:.2f} below trading threshold ({CONFIDENCE_THRESHOLD_TRADE}). Waiting for better signal.")
            elif confidence >= CONFIDENCE_THRESHOLD_TRADE:
                # ONLY trade with high confidence (>70%) for futures
                # Check if we should skip this trade: only open new trade if entry is better OR confidence is higher (>=0.9)
                should_skip = False
                skip_reasons = []
                override_allowed = False
                
                for trade in trades_list:
                    if trade['signal'] == predicted_signal:  # Same signal type
                        existing_confidence = trade.get('confidence', 0.0)
                        entry_worse = False
                        
                        if predicted_signal == 1:  # BUY - better entry = lower price
                            entry_worse = current_price >= trade['entry_price']
                        else:  # SELL - better entry = higher price
                            entry_worse = current_price <= trade['entry_price']
                        
                        if entry_worse:
                            # Entry is worse, but check if confidence is high enough to override
                            if confidence >= 0.9 and confidence > existing_confidence:
                                # High confidence (>=0.9) and higher than existing - allow trade
                                logging.info(f"✅ Overriding worse entry: Confidence {confidence:.2f} >= 0.9 and higher than existing trade at ${trade['entry_price']:.2f} (conf: {existing_confidence:.2f})")
                                override_allowed = True
                                # Continue checking other trades
                            else:
                                # Entry worse and not enough confidence to override
                                skip_reasons.append(f"Trade at ${trade['entry_price']:.2f} (conf: {existing_confidence:.2f})")
                                # Continue checking other trades
                
                # Only skip if we have skip reasons AND no override was allowed
                if skip_reasons and not override_allowed:
                    should_skip = True
                    logging.info(f"⏭️  Skipping {signal_map[predicted_signal]} trade: Entry ${current_price:.2f} worse than existing {', '.join(skip_reasons)}. "
                               f"Confidence {confidence:.2f} not high enough to override (need >=0.9 and > all existing)")
                
                if not should_skip:
                    # Assign trade index when creating the trade
                    trade_counter += 1
                    trade_index = trade_counter
                    logging.info(f"✅ VERY HIGH CONFIDENCE ({confidence:.2f}) - Executing trade")
                    if TEST:
                        new_trade = simulate_trade(predicted_signal, trade_quantity_btc, current_price, tp, sl, confidence=confidence)
                        new_trade['index'] = trade_index  # Store index in trade
                        simulated_trades.append(new_trade)
                    else:
                        new_trade = execute_trade(predicted_signal, exchange, trade_quantity_btc, current_price, tp, sl, confidence=confidence)
                        new_trade['index'] = trade_index  # Store index in trade
                        active_trades.append(new_trade)
            elif confidence < CONFIDENCE_THRESHOLD_TEST:
                # Trigger model testing and fine-tuning when confidence is below threshold
                logging.info(f"\n⚠️ Confidence {confidence:.2f} below threshold ({CONFIDENCE_THRESHOLD_TEST})")
                logging.info("Triggering model fine-tuning with recent data...")
                if not testing_model:  # Don't start new test if already testing
                    retrain_with_recent_data(client)
                    # Start testing the new model
                    testing_model = True
                    testing_start_time = time.time()
                    last_refinement_time = time.time()  # Update last refinement time
                    current_model_predictions = []
                    test_model_predictions = []
                    logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
            else:
                # Confidence between thresholds - wait for better signal
                logging.info(f"⏸️  Confidence {confidence:.2f} is moderate. Waiting for higher confidence (>{CONFIDENCE_THRESHOLD_TRADE:.0%}) to trade.")

            # Save stats at the end of each cycle
            if not TEST:
                save_stats(daily_stats)

            logging.info("Waiting for the next trading interval...")
            time.sleep(60)

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            time.sleep(60)

def calculate_trade_amount(confidence, balance, current_price, stop_loss, signal_aligned=True):
    """
    Calculate recommended trade amount based on confidence level and signal alignment.
    Higher confidence with aligned signals (model signal matches TP-based signal) = larger position size.
    
    Base position sizing by confidence:
    - 1.0: 75% of balance
    - 0.95-0.999: 70% of balance
    - 0.90-0.95: 65% of balance
    - 0.85-0.90: 55% of balance
    - 0.80-0.85: 45% of balance
    - 0.75-0.80: 35% of balance
    - 0.70-0.75: 25% of balance
    
    When signals are aligned (model signal matches TP-based signal):
    - Position size is increased by up to 25% based on confidence
    - Higher confidence = larger multiplier
    
    When signals are mismatched:
    - Position size is reduced by 50%
    
    Args:
        confidence: Model confidence (0.0 to 1.0)
        balance: Available USDT balance
        current_price: Current BTC price
        stop_loss: Stop loss price (for reference, not used in calculation)
        signal_aligned: True if model signal matches TP-based signal, False otherwise
    
    Returns           : 
    trade_percentage  : Percentage of balance to trade
    trade_amount_usdt : Amount in USDT
    trade_quantity_btc: Quantity in BTC
    """
    # Base confidence-based position sizing (% of balance to trade)
    if confidence >= 1.0:
        base_percentage = 75.0  # 75% of balance for perfect confidence (1.0)
    elif confidence >= 0.95:
        base_percentage = 70.0  # 70% of balance for very high confidence (0.95-0.999)
    elif confidence >= 0.90:
        base_percentage = 65.0  # 65% of balance
    elif confidence >= 0.85:
        base_percentage = 55.0  # 55% of balance (0.85-0.90)
    elif confidence >= 0.80:
        base_percentage = 45.0  # 45% of balance
    elif confidence >= 0.75:
        base_percentage = 35.0  # 35% of balance
    else:  # 0.70 - 0.75
        base_percentage = 25.0  # 25% of balance for minimum confidence threshold
    
    # Adjust position size based on signal alignment
    if signal_aligned:
        # Signals match - increase position size based on confidence
        # Higher confidence = larger increase (up to 25% additional)
        alignment_multiplier = 1.0 + (confidence * 0.25)  # 1.0 to 1.25 multiplier
        trade_percentage = base_percentage * alignment_multiplier
        # Cap at 95% of balance for safety
        if trade_percentage > 95.0:
            trade_percentage = 95.0
    else:
        # Signals mismatch - reduce position size by 50%
        trade_percentage = base_percentage * 0.5
    
    # Calculate trade amount in USDT
    trade_amount_usdt = balance * (trade_percentage / 100)
    
    # Minimum trade size
    min_trade_amount = 10.0  # Minimum $10 USDT
    if trade_amount_usdt < min_trade_amount:
        trade_amount_usdt = min_trade_amount
        trade_percentage = (trade_amount_usdt / balance * 100) if balance > 0 else 0
    
    # Calculate quantity in BTC
    trade_quantity_btc = trade_amount_usdt / current_price
    
    # Calculate actual risk based on stop loss distance
    sl_distance_pct = abs(current_price - stop_loss) / current_price * 100
    actual_risk_usdt = trade_amount_usdt * (sl_distance_pct / 100)
    
    # Log risk management info
    alignment_status = "ALIGNED" if signal_aligned else "MISMATCHED"
    logging.debug(
        f"Position sizing: Conf={confidence:.2f}, Signals={alignment_status}, Position={trade_percentage:.1f}%, "
        f"Trade=${trade_amount_usdt:.2f}, SL_dist={sl_distance_pct:.2f}%, Risk=${actual_risk_usdt:.2f}"
    )
    
    return trade_percentage, trade_amount_usdt, trade_quantity_btc

def print_prediction(signal, confidence, tp, sl, trade_percentage, trade_amount_usdt, trade_quantity_btc, balance):
    signal_map = {0: 'SELL', 1: 'BUY'}
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PREDICTION")
    logging.info(f"{'='*60}")
    logging.info(f"Signal: {signal_map[signal]} | Confidence: {confidence:.2f}")
    logging.info(f"Take Profit: ${tp:.2f} | Stop Loss: ${sl:.2f}")
    logging.info(f"\n--- TRADE RECOMMENDATION ---")
    logging.info(f"Current USDT Balance: ${balance:.2f}")
    logging.info(f"Recommended Trade Size: {trade_percentage:.1f}% of balance")
    logging.info(f"Trade Amount: ${trade_amount_usdt:.2f} USDT ({trade_quantity_btc:.6f} BTC)")
    logging.info(f"{'='*60}\n")

def simulate_trade(signal, quantity, entry_price, tp, sl, confidence=None):
    """Simulate a trade without actually executing it (for TEST mode)"""
    side = {0: 'SELL', 1: 'BUY'}[signal]
    # Don't log simulated message - already shown at startup
    
    # Calculate trade amounts
    trade_amount_usdt = quantity * entry_price
    
    # Calculate expected profit/loss amounts
    if signal == 1:  # BUY
        expected_profit_usdt = quantity * (tp - entry_price)
        expected_loss_usdt = quantity * (entry_price - sl)
    else:  # SELL
        expected_profit_usdt = quantity * (entry_price - tp)
        expected_loss_usdt = quantity * (sl - entry_price)
    
    # Return simulated trade info for tracking
    trade_info = {
        'signal': signal,
        'entry_price': entry_price,
        'tp': tp,
        'sl': sl,
        'side': side,
        'entry_time': time.time(),
        'quantity': quantity,
        'trade_amount_usdt': trade_amount_usdt,
        'expected_profit_usdt': expected_profit_usdt,
        'expected_loss_usdt': expected_loss_usdt
    }
    # Store confidence if provided
    if confidence is not None:
        trade_info['confidence'] = confidence
    return trade_info

def log_trade_to_file(trade_index, trade, exit_price, result, actual_pnl_usdt, pnl_percentage, price_change_pct, duration_minutes, simulated=False):
    """Log trade details to a separate JSON file"""
    try:
        # Load existing trades if file exists and is valid
        trades_log = []
        if os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and content != 'w':  # Handle corrupted file
                        trades_log = json.loads(content)
                    else:
                        logging.warning(f"⚠️ {TRADES_LOG_FILE} appears corrupted. Starting fresh.")
                        trades_log = []
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"⚠️ Could not parse {TRADES_LOG_FILE}: {e}. Starting fresh.")
                trades_log = []
        
        # Ensure trades_log is a list
        if not isinstance(trades_log, list):
            logging.warning(f"⚠️ {TRADES_LOG_FILE} format invalid. Starting fresh.")
            trades_log = []
        
        # Remove any ACTIVE entry for this trade first (using old or new field names)
        trades_log = [t for t in trades_log if 
                     not (t.get('status') == 'ACTIVE' and 
                          (abs(t.get('entry_price', 0) - float(trade['entry_price'])) < 0.01 or
                           abs(t.get('entry price', 0) - float(trade['entry_price'])) < 0.01))]
        
        # Create trade log entry with simplified structure
        trade_entry = {
            'index': trade_index,
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Profit/loss': result,  # 'PROFIT' or 'LOSS'
            'PL percentage': float(pnl_percentage),
            'PL in $': float(actual_pnl_usdt),
            'entry price': float(trade['entry_price'])
        }
        
        # Append to log
        trades_log.append(trade_entry)
        
        # Save back to file with proper formatting
        with open(TRADES_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(trades_log, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📝 Trade #{trade_index} logged to {TRADES_LOG_FILE}")
    except Exception as e:
        logging.error(f"Error logging trade to file: {e}", exc_info=True)

def log_active_trades_to_file(trades_list, current_price, simulated=False):
    """Log active trades with unrealized P&L to JSON file"""
    try:
        # Load existing trades if file exists and is valid
        trades_log = []
        if os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and content != 'w':
                        trades_log = json.loads(content)
                    else:
                        trades_log = []
            except (json.JSONDecodeError, ValueError):
                trades_log = []
        
        if not isinstance(trades_log, list):
            trades_log = []
        
        # Update or add active trades
        for i, trade in enumerate(trades_list, 1):
            pnl_pct, unrealized_pnl_usdt = calculate_unrealized_pnl(trade, current_price)
            entry_time_str = datetime.fromtimestamp(trade['entry_time']).strftime('%Y-%m-%d %H:%M:%S')
            duration_minutes = int((time.time() - trade['entry_time']) / 60)
            
            # Check if this trade already exists in log (by timestamp and entry_price)
            trade_exists = False
            for existing_trade in trades_log:
                if (existing_trade.get('entry_price') == float(trade['entry_price']) and
                    existing_trade.get('timestamp') == entry_time_str and
                    existing_trade.get('status') == 'ACTIVE'):
                    # Update existing active trade
                    existing_trade['unrealized_pnl_usdt'] = float(unrealized_pnl_usdt)
                    existing_trade['unrealized_pnl_percentage'] = float(pnl_pct)
                    existing_trade['current_price'] = float(current_price)
                    existing_trade['duration_minutes'] = duration_minutes
                    existing_trade['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    trade_exists = True
                    break
            
            if not trade_exists:
                # Add new active trade
                active_trade_entry = {
                    'trade_index': f"ACTIVE_{i}",
                    'timestamp': entry_time_str,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'ACTIVE',
                    'simulated': simulated,
                    'side': trade['side'],
                    'entry_price': float(trade['entry_price']),
                    'current_price': float(current_price),
                    'take_profit': float(trade['tp']),
                    'stop_loss': float(trade['sl']),
                    'quantity_btc': float(trade['quantity']),
                    'trade_amount_usdt': float(trade['trade_amount_usdt']),
                    'unrealized_pnl_usdt': float(unrealized_pnl_usdt),
                    'unrealized_pnl_percentage': float(pnl_pct),
                    'duration_minutes': duration_minutes,
                    'expected_profit_usdt': float(trade.get('expected_profit_usdt', 0)),
                    'expected_loss_usdt': float(trade.get('expected_loss_usdt', 0))
                }
                trades_log.append(active_trade_entry)
        
        # Remove old ACTIVE entries that are no longer in trades_list
        # (they were closed but we keep them as completed trades)
        active_entry_prices = {float(t['entry_price']) for t in trades_list}
        trades_log = [t for t in trades_log if 
                     not (t.get('status') == 'ACTIVE' and 
                          t.get('entry_price') not in active_entry_prices)]
        
        # Save back to file
        with open(TRADES_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(trades_log, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Error logging active trades to file: {e}", exc_info=True)

def print_trade_result(trade, exit_price, result, simulated=False, trade_index=None, early_stop=False):
    """Print detailed trade result summary"""
    duration = time.time() - trade['entry_time']
    duration_minutes = int(duration / 60)
    
    # Calculate actual profit/loss
    if trade['signal'] == 1:  # BUY trade
        actual_pnl_usdt = trade['quantity'] * (exit_price - trade['entry_price'])
        price_change_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
    else:  # SELL trade
        actual_pnl_usdt = trade['quantity'] * (trade['entry_price'] - exit_price)
        price_change_pct = ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
    
    pnl_percentage = (actual_pnl_usdt / trade['trade_amount_usdt']) * 100
    
    # Log to separate file if trade_index is provided
    if trade_index is not None:
        log_trade_to_file(trade_index, trade, exit_price, result, actual_pnl_usdt, pnl_percentage, price_change_pct, duration_minutes, simulated)
    
    # Result emoji and color
    if result == 'PROFIT':
       emoji        = "✅"
       result_text  = "PROFIT (TP REACHED)"
    elif early_stop:
        emoji       = "🛑"
        result_text = "LOSS (EARLY STOP)"
    else:
        emoji       = "❌"
        result_text = "LOSS (SL HIT)"
    
    # Simplified logging - show only essential info (no "simulated" marker - shown at startup)
    if actual_pnl_usdt > 0:
        pnl_sign = "+"
        pnl_emoji = "💰"
    else:
        pnl_sign = ""
        pnl_emoji = "📉"
    
    logging.info(f"\n{emoji} Trade #{trade_index if trade_index else '?'} {result_text} | "
                 f"{trade['side']} | Entry: ${trade['entry_price']:.2f} → Exit: ${exit_price:.2f} | "
                 f"Expected: ${trade['expected_profit_usdt']:.2f} | "
                 f"Actual: {pnl_sign}${actual_pnl_usdt:.2f} ({pnl_percentage:+.2f}%) | "
                 f"Duration: {duration_minutes}m")
    
    return actual_pnl_usdt

def execute_trade(signal, exchange, quantity, entry_price, tp, sl, confidence=None):
    """Execute trade and return trade info for tracking"""
    side = {0: 'SELL', 1: 'BUY'}[signal]
    logging.info(f"\n💰 Confidence >= 0.65. Placing {side} order.")
    # Note: For futures, you may need to use a different order placement method
    # This will depend on your Exchange class implementation for futures
    exchange.place_order('BTC_USDT', 'MARKET', side, quantity)

    # Calculate trade amounts
    trade_amount_usdt = quantity * entry_price
    
    # Calculate expected profit/loss amounts
    if signal               == 1:                             # BUY
       expected_profit_usdt  = quantity * (tp - entry_price)
       expected_loss_usdt    = quantity * (entry_price - sl)
    else:  # SELL
        expected_profit_usdt = quantity * (entry_price - tp)
        expected_loss_usdt = quantity * (sl - entry_price)
    
    # Return trade info for tracking
    trade_info = {
        'signal'              : signal,
        'entry_price'         : entry_price,
        'tp'                  : tp,
        'sl'                  : sl,
        'side'                : side,
        'entry_time'          : time.time(),
        'quantity'            : quantity,
        'trade_amount_usdt'   : trade_amount_usdt,
        'expected_profit_usdt': expected_profit_usdt,
        'expected_loss_usdt'  : expected_loss_usdt
    }
    # Store confidence if provided
    if confidence is not None:
        trade_info['confidence'] = confidence
    return trade_info

def calculate_unrealized_pnl(trade, current_price):
    """Calculate unrealized P&L percentage and dollar amount for an active trade"""
    if trade['signal'] == 1:  # BUY trade
        pnl_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
        pnl_usdt = trade['quantity'] * (current_price - trade['entry_price'])
    else:  # SELL trade
        pnl_pct  = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
        pnl_usdt = trade['quantity'] * (trade['entry_price'] - current_price)
    return pnl_pct, pnl_usdt

def check_trade_loss(trade, current_price):
    """Check if the active trade hit stop loss (resulted in a loss)"""
    if trade['signal'] == 1:  # BUY trade
        # Loss if price went below stop loss
        if current_price <= trade['sl']:
            logging.info(f"BUY trade hit SL: Entry={trade['entry_price']:.2f}, SL={trade['sl']:.2f}, Current={current_price:.2f}")
            return True
    else:  # SELL trade
        # Loss if price went above stop loss
        if current_price >= trade['sl']:
            logging.info(f"SELL trade hit SL: Entry={trade['entry_price']:.2f}, SL={trade['sl']:.2f}, Current={current_price:.2f}")
            return True
    return False

def is_trade_complete(trade, current_price):
    """Check if the trade reached take profit (successful completion)"""
    if trade['signal'] == 1:  # BUY trade
        return current_price >= trade['tp']
    else:   # SELL trade
        return current_price <= trade['tp']

def check_early_stop(trade, current_price, predicted_signal=None):
    """
    Check if trade should be early stopped based on:
    1. Trade has been red (losing) for too long
    3. Model is predicting opposite signal (if enabled)
    
    Early stop triggers when BOTH conditions 1 AND 3 are True.
    """
    pnl_pct, _ = calculate_unrealized_pnl(trade, current_price)
    duration_minutes = (time.time() - trade['entry_time']) / 60
    
    # Check if trade is losing (red)
    if pnl_pct >= 0:
        return False  # Trade is green, no early stop
    
    # Condition 1: Trade has been red for too long
    time_condition = duration_minutes >= EARLY_STOP_MAX_TIME_MINUTES
    
    # Condition 3: Model is predicting opposite signal (if enabled)
    opposite_signal_condition = False
    if EARLY_STOP_OPPOSITE_SIGNAL and predicted_signal is not None:
        # Check if predicted signal is opposite to current trade
        if trade['signal'] == 1 and predicted_signal == 0:  # BUY trade, model predicts SELL
            opposite_signal_condition = True
        elif trade['signal'] == 0 and predicted_signal == 1:  # SELL trade, model predicts BUY
            opposite_signal_condition = True
    
    # Early stop only if BOTH conditions are True
    if time_condition and opposite_signal_condition:
        logging.warning(f"🛑 Early stop: Trade has been red for {duration_minutes:.1f} minutes AND model predicts opposite signal (loss: {pnl_pct:.2f}%)")
        return True
    
    return False

def should_adopt_new_model(current_predictions, test_predictions):
    """
    Compare performance of current model vs test model to decide adoption.
    
    Compares:
    - Average confidence levels
    - Number of high-confidence predictions
    - Consistency of predictions
    
    Args:
        current_predictions: List of predictions from current model
        test_predictions: List of predictions from test model
    
    Returns:
        bool: True if test model performs better, False otherwise
    """
    if len(test_predictions) < 2 or len(current_predictions) < 2:
        logging.info(f"❌ Insufficient predictions during test (Current: {len(current_predictions)}, Test: {len(test_predictions)}). Need at least 2 each.")
        return False
    
    # Calculate metrics for current model
    current_avg_confidence = sum(p['confidence'] for p in current_predictions) / len(current_predictions)
    current_high_conf_count = sum(1 for p in current_predictions if p['confidence'] >= 0.65)
    current_high_conf_ratio = current_high_conf_count / len(current_predictions)
    
    # Calculate metrics for test model
    test_avg_confidence = sum(p['confidence'] for p in test_predictions) / len(test_predictions)
    test_high_conf_count = sum(1 for p in test_predictions if p['confidence'] >= 0.65)
    test_high_conf_ratio = test_high_conf_count / len(test_predictions)
    
    # Calculate confidence variance (lower is better - more consistent)
    current_conf_variance = np.var([p['confidence'] for p in current_predictions])
    test_conf_variance = np.var([p['confidence'] for p in test_predictions])
    
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 MODEL COMPARISON RESULTS")
    logging.info(f"{'='*60}")
    logging.info(f"\n🔵 CURRENT MODEL:")
    logging.info(f"   Total Predictions: {len(current_predictions)}")
    logging.info(f"   Average Confidence: {current_avg_confidence:.3f}")
    logging.info(f"   High Confidence (≥0.65): {current_high_conf_count} ({current_high_conf_ratio*100:.1f}%)")
    logging.info(f"   Confidence Variance: {current_conf_variance:.4f}")
    
    logging.info(f"\n🟢 TEST MODEL:")
    logging.info(f"   Total Predictions: {len(test_predictions)}")
    logging.info(f"   Average Confidence: {test_avg_confidence:.3f}")
    logging.info(f"   High Confidence (≥0.65): {test_high_conf_count} ({test_high_conf_ratio*100:.1f}%)")
    logging.info(f"   Confidence Variance: {test_conf_variance:.4f}")
    
    logging.info(f"\n📈 COMPARISON:")
    
    # Score each model (higher is better)
    current_score = 0
    test_score = 0
    
    # 1. Average confidence (40% weight)
    if test_avg_confidence > current_avg_confidence:
        improvement = ((test_avg_confidence - current_avg_confidence) / current_avg_confidence) * 100
        logging.info(f"   ✅ Avg Confidence: Test model is {improvement:.1f}% higher")
        test_score += 40
    else:
        decline = ((current_avg_confidence - test_avg_confidence) / current_avg_confidence) * 100
        logging.info(f"   ❌ Avg Confidence: Test model is {decline:.1f}% lower")
        current_score += 40
    
    # 2. High confidence ratio (35% weight)
    if test_high_conf_ratio > current_high_conf_ratio:
        logging.info(f"   ✅ High Conf Ratio: Test model has more ({test_high_conf_ratio*100:.1f}% vs {current_high_conf_ratio*100:.1f}%)")
        test_score += 35
    else:
        logging.info(f"   ❌ High Conf Ratio: Current model has more ({current_high_conf_ratio*100:.1f}% vs {test_high_conf_ratio*100:.1f}%)")
        current_score += 35
    
    # 3. Consistency - lower variance is better (25% weight)
    if test_conf_variance < current_conf_variance:
        improvement = ((current_conf_variance - test_conf_variance) / current_conf_variance) * 100
        logging.info(f"   ✅ Consistency: Test model is {improvement:.1f}% more consistent")
        test_score += 25
    else:
        decline = ((test_conf_variance - current_conf_variance) / test_conf_variance) * 100
        logging.info(f"   ❌ Consistency: Current model is {decline:.1f}% more consistent")
        current_score += 25
    
    logging.info(f"\n🎯 FINAL SCORES:")
    logging.info(f"   Current Model: {current_score}/100")
    logging.info(f"   Test Model: {test_score}/100")
    
    # Test model must score higher to be adopted
    if test_score > current_score:
        logging.info(f"\n✅ VERDICT: Test model performs BETTER (+{test_score - current_score} points)")
        return True
    else:
        logging.info(f"\n❌ VERDICT: Current model performs BETTER (+{current_score - test_score} points)")
        return False

def adopt_new_model():
    """Replace the current production model with the tested model"""
    import os
    import shutil
    
    current_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model.pth')
    test_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
    backup_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_backup.pth')
    
    # Backup current model
    if os.path.exists(current_model_path):
        shutil.copy(current_model_path, backup_model_path)
        logging.info(f"📦 Backed up current model to {backup_model_path}")
    
    # Replace with new model
    if os.path.exists(test_model_path):
        shutil.copy(test_model_path, current_model_path)
        logging.info("✅ New model adopted as primary model")
    else:
        logging.info("❌ Error: Test model file not found!")

def append_data_to_training_file(df):
    """Append new data to training_data.csv immediately when fetched"""
    if df is None or df.empty:
        return
    
    try:
        # Ensure open_time is datetime and set as index
        df_copy = df.copy()
        if 'open_time' in df_copy.columns:
            # open_time is a column - convert to datetime and set as index
            df_copy['open_time'] = pd.to_datetime(df_copy['open_time'])
            df_copy.set_index('open_time', inplace=True)
        elif df_copy.index.name == 'open_time' or isinstance(df_copy.index, pd.DatetimeIndex):
            # open_time is already the index - just ensure it's datetime
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
        else:
            # No open_time found - this shouldn't happen, but handle gracefully
            logging.warning("No 'open_time' column or index found in fetched data. Skipping append.")
            return
        
        # Load existing training data if it exists
        if os.path.exists('training_data.csv'):
            try:
                existing_df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
                
                # Find new rows that don't exist in training_data.csv
                new_rows = df_copy[~df_copy.index.isin(existing_df.index)]
                
                if len(new_rows) > 0:
                    # Append new rows
                    updated_df = pd.concat([existing_df, new_rows])
                    # Sort by index (time) and remove duplicates
                    updated_df = updated_df.sort_index()
                    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                    # Save back to file
                    updated_df.to_csv('training_data.csv')
                    logging.info(f"✅ Appended {len(new_rows)} new records to training_data.csv (total: {len(updated_df)} rows)")
                else:
                    logging.debug("No new data to append to training_data.csv")
            except Exception as e:
                logging.warning(f"Could not append to training_data.csv: {e}. Creating new file.")
                # If append fails, create new file with current data
                df_copy.to_csv('training_data.csv')
                logging.info(f"✅ Created new training_data.csv with {len(df_copy)} records")
        else:
            # If file doesn't exist, create it with current data
            df_copy.to_csv('training_data.csv')
            logging.info(f"✅ Created training_data.csv with {len(df_copy)} records")
    except Exception as e:
        logging.error(f"Error appending data to training_data.csv: {e}", exc_info=True)

def retrain_with_recent_data(client): 
    """Fine-tune the model with the latest portion of historical training data"""
    logging.info("Loading latest historical data for fine-tuning...")
    
    # Load the existing training dataset
    if not os.path.exists('training_data.csv'):
        logging.error("No training_data.csv found. Cannot fine-tune. Run train.py first.")
        return
    
    try:
        # Load the full training dataset
        full_df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
        
        if full_df.empty or len(full_df) < 100:
            logging.error(f"Training data is too small ({len(full_df)} rows). Cannot fine-tune.")
            return
        
        # Use the latest 500 rows (or all if less than 500) for fine-tuning
        # This ensures we have enough data for signal generation
        num_rows_to_use = min(500, len(full_df))
        recent_df       = full_df.tail(num_rows_to_use)
        
        logging.info(f"Using latest {len(recent_df)} rows from training dataset (out of {len(full_df)} total) for fine-tuning")
        
        # Save as temporary training data for fine-tuning
        recent_df.to_csv('recent_training_data.csv')
        
    except Exception as e:
        logging.error(f"Failed to load training data for fine-tuning: {e}")
        return
    
    logging.info("Starting model fine-tuning with recent data (time-weighted)...")
    # Run training script with test mode flag using the conda environment
    # The presence of 'recent_training_data.csv' automatically triggers fine-tuning mode
    result             = subprocess.run(['python', 'train.py', '--test-mode'], capture_output=True, text=True)
    # python_path        = '/usr/local/anaconda3/envs/crypto/bin/python'
    # result             = subprocess.run([python_path, 'train.py', '--test-mode'], capture_output=True, text=True)
    if result.returncode == 0:
        logging.info("Fine-tuning complete successfully.")
        test_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
        logging.info(f"New model saved as {test_model_path}")
    else:
        logging.warning(f"Fine-tuning encountered issues: {result.stderr[:500]}")
        logging.info("Continuing with current model without fine-tuning.")
        # Don't fail completely, just continue
        return
    
    logging.info("Fine-tuned model will enter testing phase.")

def load_stats():
    """Load daily stats from JSON file."""
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            try:
                stats = json.load(f)
                # If the date in file is not today, reset
                if stats.get('date') == today:
                    return stats
            except json.JSONDecodeError:
                logging.warning(f"Could not decode {STATS_FILE}. Starting with fresh stats.")
    # Default stats for a new day or new file
    return {"date": today, "successful_trades": 0, "failed_trades": 0, "total_profit_usdt": 0.0}

def save_stats(stats): 
    """Save daily stats to JSON file."""
    # Calculate win rate
    total_trades = stats['successful_trades'] + stats['failed_trades']
    if total_trades > 0:
        stats['win_rate_pct'] = (stats['successful_trades'] / total_trades) * 100
    else:
        stats['win_rate_pct'] = 0.0
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)
    logging.info(f"📊 Daily stats saved: Wins={stats['successful_trades']}, Losses={stats['failed_trades']}, P&L=${stats['total_profit_usdt']:.2f}, Win Rate={stats['win_rate_pct']:.2f}%")

def update_stats(stats, result, pnl):
    """Update daily stats after a trade closes."""
    if result == 'PROFIT':
        stats['successful_trades'] += 1
    else: # LOSS
        stats['failed_trades'] += 1
    
    stats['total_profit_usdt'] += pnl

if __name__ == '__main__':
    main()
