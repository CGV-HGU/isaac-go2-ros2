import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_scalars_to_png(log_dir, output_png="learning_report.png"):
    """
    Parses the latest TensorBoard event file in log_dir and exports
    Reward and Loss curves as a high-resolution PNG image.
    """
    # 1. Search for tfevents files recursively
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        # Fallback to local unitree_go2 folders if logs/ doesn't exist
        event_files = glob.glob(os.path.join("unitree_go2_*", "**", "events.out.tfevents.*"), recursive=True)
        
    if not event_files:
        print(f"[Error] No tfevents logs found in '{log_dir}' or local directories.")
        return
    
    # Sort files by modification time and pick the most recent one
    latest_event_file = max(event_files, key=os.path.getmtime)
    print(f"[INFO] Parsing latest event file: {latest_event_file}")
    
    # 2. Initialize EventAccumulator (only load scalars for fast parsing)
    ea = EventAccumulator(latest_event_file, size_guidance={'scalars': 0})
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    if not tags:
        print("[Warning] No scalar metrics found in this log file.")
        return
        
    # 3. Identify reward and loss tags (flexible match for both SKRL and RSL-RL)
    reward_tag = None
    loss_tag = None
    
    # Prioritized keywords for matching
    for tag in tags:
        # Match reward/return/episode reward
        if any(kw in tag.lower() for kw in ["reward/reward", "reward/total", "reward/return", "episode/reward", "rewards/incoming"]):
            reward_tag = tag
        elif reward_tag is None and "reward" in tag.lower():
            reward_tag = tag
            
        # Match loss
        if any(kw in tag.lower() for kw in ["loss/policy", "loss/total", "losses/policy", "loss/surrogate"]):
            loss_tag = tag
        elif loss_tag is None and "loss" in tag.lower():
            loss_tag = tag
            
    # Default fallback tags if priority tags not found
    if reward_tag is None and tags:
        reward_tag = tags[0]
    if loss_tag is None and len(tags) > 1:
        loss_tag = tags[1]

    # 4. Generate Matplotlib Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Report - {os.path.basename(os.path.dirname(latest_event_file))}", fontsize=14, fontweight='bold')
    
    # Plot Reward
    if reward_tag:
        events = ea.Scalars(reward_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        axes[0].plot(steps, values, label=f"Metric: {reward_tag}", color="#1f77b4", linewidth=1.5)
        axes[0].set_title("Reward Convergence", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Steps", fontsize=10)
        axes[0].set_ylabel("Value", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="best", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No Reward Data Found", ha="center", va="center", fontsize=12, color="gray")
        axes[0].set_title("Reward Curve", fontsize=12)
        
    # Plot Loss
    if loss_tag:
        events = ea.Scalars(loss_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        axes[1].plot(steps, values, label=f"Metric: {loss_tag}", color="#d62728", linewidth=1.5)
        axes[1].set_title("Loss Optimization", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Steps", fontsize=10)
        axes[1].set_ylabel("Value", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="best", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No Loss Data Found", ha="center", va="center", fontsize=12, color="gray")
        axes[1].set_title("Loss Curve", fontsize=12)
        
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"[SUCCESS] Final training report successfully generated & saved to: {os.path.abspath(output_png)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training report PNG from latest tfevents.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where logs are stored (default: logs)")
    parser.add_argument("--output", type=str, default="learning_report.png", help="Output PNG file path")
    args = parser.parse_args()
    
    export_scalars_to_png(args.log_dir, args.output)
