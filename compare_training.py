#!/usr/bin/env python3
"""
Quick comparison: Self-play vs Training against best checkpoint
Run this to see which approach is more stable
"""

from pathlib import Path

# Check what you have
ckpt_dir = Path("checkpoints")
if not ckpt_dir.exists():
    print("❌ No checkpoints directory found")
    print("📁 Creating checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)
    print()

print("=" * 60)
print("TRAINING SETUP CHECK")
print("=" * 60)

# List available checkpoints
checkpoints = sorted(ckpt_dir.glob("*.pt"))
if checkpoints:
    print(f"✅ Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints[-5:]:  # Show last 5
        size_mb = ckpt.stat().st_size / 1024 / 1024
        print(f"   - {ckpt.name} ({size_mb:.2f} MB)")
    print()
else:
    print("❌ No checkpoints found yet")
    print("   Run training first to create checkpoints")
    print()

# Check which training script to use
best_ckpt = ckpt_dir / "best.pt"
epoch_50k = ckpt_dir / "epoch_50000.pt"

print("=" * 60)
print("RECOMMENDED NEXT STEPS")
print("=" * 60)

if epoch_50k.exists() and not best_ckpt.exists():
    print("✅ Found epoch_50000.pt (your 54% checkpoint)")
    print("📝 Action: Copy it to best.pt")
    print()
    print("   Run this command:")
    print(f"   cp {epoch_50k} {best_ckpt}")
    print()

if best_ckpt.exists():
    print(f"✅ Best checkpoint exists: {best_ckpt}")
    print()
    print("🚀 READY TO TRAIN AGAINST BEST!")
    print()
    print("   Option 1 (RECOMMENDED): Train against frozen best")
    print("   ────────────────────────────────────────────────")
    print("   python train_vs_best.py")
    print()
    print("   This trains against your 54% checkpoint (frozen)")
    print("   - More stable learning")
    print("   - Prevents regression")
    print("   - Keeps improving incrementally")
    print()
    print("   Option 2 (OLD METHOD): Traditional self-play")
    print("   ────────────────────────────────────────────────")
    print("   python train.py")
    print()
    print("   This trains agent vs itself (current method)")
    print("   - Unstable (you saw 54% → 34% drops)")
    print("   - Circular learning issues")
    print("   - Not recommended")
else:
    print("⚠️  No best checkpoint yet")
    print()
    print("   Option A: Start fresh training")
    print("   ──────────────────────────────────")
    print("   python train.py")
    print("   (Will create initial checkpoints)")
    print()
    print("   Option B: Copy your 50k checkpoint")
    print("   ──────────────────────────────────")
    print("   If you have epoch_50000.pt (54% win rate):")
    print(f"   cp checkpoints/epoch_50000.pt {best_ckpt}")
    print("   python train_vs_best.py")

print()
print("=" * 60)
