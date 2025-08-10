"""
GRPO V3 Metrics - Real-time training metrics visualization
"""

import time
import threading
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .config import *


class GRPOMetricsTracker:
    def __init__(self, save_path="grpo_v3_training_metrics.png", update_interval=2.0):
        self.save_path = save_path
        self.update_interval = update_interval
        self.metrics = {
            'train_loss': deque(maxlen=1000),
            'val_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=1000),
            'steps': deque(maxlen=1000),
            'epochs': deque(maxlen=1000),
            'batch_loss': deque(maxlen=100),
            'gradient_norm': deque(maxlen=1000),
            'avg_reward': deque(maxlen=1000),
            'normalized_reward': deque(maxlen=1000),
            'speaker_sim': deque(maxlen=1000),
            'length_penalty': deque(maxlen=1000),
            'kl_divergence': deque(maxlen=1000),
            'baseline_reward': deque(maxlen=1000),
        }
        self.start_time = time.time()
        self.last_update = 0
        self.running = True
        self.lock = threading.Lock()
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(26, 16))
        self.fig.suptitle('Chatterbox TTS GRPO V3 Professional Training Metrics', fontsize=16, fontweight='bold')
        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self._create_initial_plot()
    
    def _create_initial_plot(self):
        """Create the initial plot layout"""
        self.fig.clf()
        
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_reward = self.fig.add_subplot(gs[0, 2:])
        self.ax_speaker = self.fig.add_subplot(gs[1, 0])
        self.ax_length = self.fig.add_subplot(gs[1, 1])
        self.ax_kl = self.fig.add_subplot(gs[1, 2])
        self.ax_lr = self.fig.add_subplot(gs[2, 0])
        self.ax_grad = self.fig.add_subplot(gs[2, 1])
        self.ax_baseline = self.fig.add_subplot(gs[2, 2:])
        self.ax_info = self.fig.add_subplot(gs[3, :2])
        self.ax_epoch = self.fig.add_subplot(gs[3, 2:])
        
        self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight')
    
    def add_metrics(self, **kwargs):
        """Thread-safe metric addition"""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.metrics and value is not None:
                    self.metrics[key].append(float(value))
    
    def _update_loop(self):
        """Background thread for updating plots"""
        while self.running:
            time.sleep(self.update_interval)
            if self.metrics['train_loss']:  # Only update if we have data
                self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current metrics"""
        try:
            with self.lock:
                if not self.metrics['train_loss']:
                    return
                
                self.fig.clf()
                gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
                
                # Training Loss
                ax_loss = self.fig.add_subplot(gs[0, :2])
                if self.metrics['train_loss']:
                    ax_loss.plot(list(self.metrics['train_loss']), 'cyan', linewidth=2, alpha=0.8)
                    ax_loss.set_title('Training Loss', color='white')
                    ax_loss.grid(True, alpha=0.3)
                
                # Rewards
                ax_reward = self.fig.add_subplot(gs[0, 2:])
                if self.metrics['avg_reward']:
                    ax_reward.plot(list(self.metrics['avg_reward']), 'lime', linewidth=2, label='Raw Reward')
                    if self.metrics['normalized_reward'] and REWARD_NORMALIZATION:
                        ax_reward.plot(list(self.metrics['normalized_reward']), 'gold', linewidth=2, label='Norm Reward')
                    ax_reward.set_title('GRPO Rewards', color='white')
                    ax_reward.legend()
                    ax_reward.grid(True, alpha=0.3)
                
                # Speaker Similarity
                ax_speaker = self.fig.add_subplot(gs[1, 0])
                if self.metrics['speaker_sim']:
                    ax_speaker.plot(list(self.metrics['speaker_sim']), 'magenta', linewidth=2)
                    ax_speaker.set_title('Speaker Similarity', color='white')
                    ax_speaker.grid(True, alpha=0.3)
                
                # Length Penalty
                ax_length = self.fig.add_subplot(gs[1, 1])
                if self.metrics['length_penalty']:
                    ax_length.plot(list(self.metrics['length_penalty']), 'orange', linewidth=2)
                    ax_length.set_title('Length Penalty', color='white')
                    ax_length.grid(True, alpha=0.3)
                
                # KL Divergence
                ax_kl = self.fig.add_subplot(gs[1, 2])
                if self.metrics['kl_divergence']:
                    ax_kl.plot(list(self.metrics['kl_divergence']), 'red', linewidth=2)
                    ax_kl.set_title('KL Divergence', color='white')
                    ax_kl.grid(True, alpha=0.3)
                
                # Learning Rate
                ax_lr = self.fig.add_subplot(gs[2, 0])
                if self.metrics['learning_rate']:
                    ax_lr.plot(list(self.metrics['learning_rate']), 'yellow', linewidth=2)
                    ax_lr.set_title('Learning Rate', color='white')
                    ax_lr.grid(True, alpha=0.3)
                
                # Gradient Norm
                ax_grad = self.fig.add_subplot(gs[2, 1])
                if self.metrics['gradient_norm']:
                    ax_grad.plot(list(self.metrics['gradient_norm']), 'lightblue', linewidth=2)
                    ax_grad.set_title('Gradient Norm', color='white')
                    ax_grad.grid(True, alpha=0.3)
                
                # Baseline Reward
                ax_baseline = self.fig.add_subplot(gs[2, 2:])
                if self.metrics['baseline_reward']:
                    ax_baseline.plot(list(self.metrics['baseline_reward']), 'lightgreen', linewidth=2)
                    ax_baseline.set_title('Baseline Reward (Value Function)', color='white')
                    ax_baseline.grid(True, alpha=0.3)
                
                # Training Info
                ax_info = self.fig.add_subplot(gs[3, :2])
                ax_info.axis('off')
                
                # Current stats
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                recent_loss = list(self.metrics['train_loss'])[-1] if self.metrics['train_loss'] else 0
                recent_reward = list(self.metrics['avg_reward'])[-1] if self.metrics['avg_reward'] else 0
                recent_lr = list(self.metrics['learning_rate'])[-1] if self.metrics['learning_rate'] else 0
                
                info_text = f"""Training Status:
Time Elapsed: {elapsed/3600:.1f}h
Current Loss: {recent_loss:.4f}
Current Reward: {recent_reward:.3f}
Learning Rate: {recent_lr:.2e}
Enhanced GRPO: KL={KL_COEFF}, Reward Norm={REWARD_NORMALIZATION}"""
                
                ax_info.text(0.05, 0.9, info_text, transform=ax_info.transAxes, 
                           color='white', fontsize=12, verticalalignment='top')
                
                # Epoch Progress
                ax_epoch = self.fig.add_subplot(gs[3, 2:])
                if self.metrics['epochs']:
                    epochs = list(self.metrics['epochs'])
                    ax_epoch.plot(epochs, 'white', marker='o', linewidth=2, markersize=4)
                    ax_epoch.set_title('Epoch Progress', color='white')
                    ax_epoch.grid(True, alpha=0.3)
                
                self.fig.suptitle('Chatterbox TTS GRPO V3 Professional Training Metrics', 
                                fontsize=16, fontweight='bold', color='white')
                
                self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', 
                               facecolor='black', edgecolor='none')
                
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def stop(self):
        """Stop the background update thread"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)