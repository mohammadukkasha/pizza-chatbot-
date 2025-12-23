import sys
print("Starting verification...")
try:
    import torch
    print(f"Torch imported: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Torch import failed: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Transformers imported")
except Exception as e:
    print(f"Transformers import failed: {e}")

print("Verification complete.")
