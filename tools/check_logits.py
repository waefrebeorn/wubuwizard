"""Check logits dump from gen_text_cpu."""
import numpy as np

logits = np.fromfile('/tmp/our_logits.bin', dtype=np.float32)
print(f"shape={logits.shape}")
print(f"range=[{logits.min():.4f},{logits.max():.4f}]")
print(f"mean={logits.mean():.6f}")
print(f"std={logits.std():.6f}")
print(f"nan={np.isnan(logits).sum()}")
print(f"inf={np.isinf(logits).sum()}")
print(f"var={logits.var():.6f}")

top10 = np.argsort(-logits)[:10]
print(f"top10_indices={top10}")
print(f"top10_values={logits[top10]}")

# Check if top tokens are all similar (uniform distribution = broken)
diff = logits[top10[0]] - logits[top10[9]]
print(f"top1-top10_diff={diff:.4f}")

# Check for "hello" token (the prompt word repeated)
# In Qwen tokenizer, "hello" might be a single token or multi-byte
# Let's see what tokens have high values
for i, idx in enumerate(top10):
    print(f"  [{idx}] = {logits[idx]:.4f}")
