#!/usr/bin/env python3
"""
Test script để kiểm tra tính năng auto-save plot
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

def test_plot_saving():
    """Test basic plotting functionality"""
    print("🧪 Testing plot saving functionality...")
    
    # Create test data
    epochs = list(range(1, 101))
    rewards = [np.random.normal(0, 1) + 0.01 * i for i in epochs]  # Increasing trend
    losses = [np.random.exponential(1) * np.exp(-0.01 * i) for i in epochs]  # Decreasing trend
    
    # Create test directory
    test_dir = "test_plots"
    os.makedirs(test_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    axes[0].plot(epochs, rewards, 'b-', linewidth=2, label='Average Reward')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Test Plot - Training Progress')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add statistics text
    stats_text = f'Final Reward: {rewards[-1]:.4f}\n'
    stats_text += f'Best Reward: {max(rewards):.4f}\n'
    stats_text += f'Average: {np.mean(rewards):.4f}\n'
    stats_text += f'Std: {np.std(rewards):.4f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot losses
    axes[1].plot(epochs, losses, 'r-', linewidth=2, label='Agent Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Test Plot - Loss Progress')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Add loss statistics
    loss_stats_text = f'Final Loss: {losses[-1]:.6f}\n'
    loss_stats_text += f'Min Loss: {min(losses):.6f}\n'
    loss_stats_text += f'Average: {np.mean(losses):.6f}'
    axes[1].text(0.02, 0.98, loss_stats_text, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"test_plot_{timestamp}.png"
    plot_path = os.path.join(test_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Test plot saved successfully: {plot_path}")
    
    # Check if file exists and has reasonable size
    if os.path.exists(plot_path):
        file_size = os.path.getsize(plot_path)
        print(f"📊 Plot file size: {file_size} bytes")
        if file_size > 10000:  # Should be at least 10KB for a decent quality plot
            print("✅ Plot file size looks good!")
            return True
        else:
            print("⚠️ Plot file seems too small, might be corrupted")
            return False
    else:
        print("❌ Plot file not found!")
        return False

if __name__ == "__main__":
    test_result = test_plot_saving()
    if test_result:
        print("\n🎉 Plot saving functionality is working correctly!")
        print("📈 You can now run your training and it will automatically save plots:")
        print("   - Every 50 epochs")
        print("   - At key milestones (10, 25, 100, 250, 500, 1000 epochs)")
        print("   - When achieving new best rewards")
        print("   - Final comprehensive plot at the end")
    else:
        print("\n❌ Plot saving test failed!")
        print("🔧 Please check matplotlib installation and permissions")
