#!/usr/bin/env python3
# ============================================
# REAL-TIME TRAINING MONITOR
# ============================================
# Run in a separate terminal to monitor training progress

import os
import time
import json
import argparse
from datetime import datetime


def monitor_training(log_dir: str, refresh_rate: int = 5):
    """Monitor training progress in real-time"""
    
    print("=" * 60)
    print("📊 TRAINING MONITOR")
    print("=" * 60)
    print(f"Watching: {log_dir}")
    print(f"Refresh rate: {refresh_rate}s")
    print("Press Ctrl+C to stop\n")
    
    metrics_file = os.path.join(log_dir, '../checkpoints/training_metrics.json')
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("=" * 60)
            print(f"📊 TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            # Check for metrics file
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                train_losses = metrics.get('train_losses', [])
                val_losses = metrics.get('val_losses', [])
                
                if train_losses:
                    print(f"\n📈 Training Progress:")
                    print(f"   Epochs completed: {len(train_losses)}")
                    print(f"   Latest train loss: {train_losses[-1]:.4f}")
                    
                    if val_losses:
                        print(f"   Latest val loss: {val_losses[-1]:.4f}")
                        print(f"   Best val loss: {min(val_losses):.4f}")
                        
                        # Simple ASCII progress bar
                        print(f"\n   Loss History (last 10 epochs):")
                        for i, (tl, vl) in enumerate(zip(train_losses[-10:], val_losses[-10:])):
                            epoch = len(train_losses) - min(10, len(train_losses)) + i + 1
                            bar_t = "█" * int(tl * 10)
                            bar_v = "▓" * int(vl * 10)
                            print(f"   E{epoch:02d}: T={tl:.3f} {bar_t}")
                            print(f"        V={vl:.3f} {bar_v}")
            
            # Check GPU usage
            print(f"\n💾 GPU Status:")
            os.system('nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -4')
            
            # Check for latest log entries
            log_files = [f for f in os.listdir(log_dir) if f. endswith('.log')]
            if log_files:
                latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                log_path = os.path.join(log_dir, latest_log)
                
                print(f"\n📝 Latest Log Entries ({latest_log}):")
                os.system(f'tail -5 {log_path}')
            
            print(f"\n⏳ Refreshing in {refresh_rate}s...  (Ctrl+C to stop)")
            time.sleep(refresh_rate)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--refresh', type=int, default=5, help='Refresh rate in seconds')
    args = parser.parse_args()
    
    monitor_training(args.log_dir, args.refresh)


if __name__ == '__main__':
    main()