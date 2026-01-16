import torch
from traiNNer.archs import paragonsr2_arch


def main() -> None:
    print("Instantiating ParagonSR2 Ultimate (scale=2)...")
    model = paragonsr2_arch.paragonsr2_ultimate(scale=2)
    model.eval()

    # Tiny input to fit on any GPU/CPU
    dummy_input = torch.randn(1, 3, 32, 32)

    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")

    target_shape = (1, 3, 64, 64)
    if output.shape == target_shape:
        print("SUCCESS: Inference sanity check passed!")
    else:
        print(f"FAILURE: Expected {target_shape}, got {output.shape}")


if __name__ == "__main__":
    main()
