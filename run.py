import os
import sys
import time
import numpy as np
import tensorflow as tf


# ============================================================
# Model location
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "best_model.keras"
)

# Keep this moderate so the script also works on weaker GPUs.
BATCH_SIZE = 8


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check command-line arguments
    # --------------------------------------------------------

    if len(sys.argv) != 3:
        print(
            "\nUsage:\n"
            "    python run.py <test_images_directory> <output_directory>\n\n"
            "Example:\n"
            "    python run.py \"D:\\Hackathon\\NoisyLR\" "
            "\"D:\\Hackathon\\outputs\"\n"
        )
        sys.exit(1)

    test_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory does not exist:\n{test_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found:\n{MODEL_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("AI-BASED IMAGE RESTORATION")
    print("=" * 60)

    print("\nModel:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")
    print("Input shape :", model.input_shape)
    print("Output shape:", model.output_shape)

    # --------------------------------------------------------
    # Find test images
    # --------------------------------------------------------

    test_files = sorted(
        filename
        for filename in os.listdir(test_dir)
        if filename.lower().endswith(".npy")
    )

    if len(test_files) == 0:
        print(f"\nERROR: No .npy files found in:\n{test_dir}")
        sys.exit(1)

    print(f"\nTest images found: {len(test_files)}")

    # --------------------------------------------------------
    # Load all test images
    # --------------------------------------------------------

    images = []

    for filename in test_files:

        path = os.path.join(
            test_dir,
            filename
        )

        image = np.load(path).astype(np.float32)

        if image.shape != (128, 128):
            raise ValueError(
                f"{filename}: expected shape (128, 128), "
                f"but got {image.shape}"
            )

        # IMPORTANT:
        # Do NOT clip or normalize the noisy LR input.
        #
        # The supplied test images may contain values
        # outside [0, 1] because of speckle noise.

        images.append(image)

    # Shape:
    # (N, 128, 128)
    images = np.stack(images, axis=0)

    # Add channel dimension:
    # (N, 128, 128, 1)
    images = images[..., np.newaxis]

    print("Input tensor shape:", images.shape)

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print("\nRunning inference...")

    start_time = time.time()

    predictions = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    inference_time = time.time() - start_time

    # Model output:
    # (N, 256, 256, 1)
    #
    # Remove channel dimension:
    # (N, 256, 256)

    predictions = predictions[..., 0]

    # --------------------------------------------------------
    # Output post-processing
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # Input is NOT clipped.
    #
    # Only the final restored output is clipped to [0, 1].

    predictions = np.clip(
        predictions,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    print("\nSaving restored images...")

    for filename, prediction in zip(
        test_files,
        predictions
    ):

        output_path = os.path.join(
            output_dir,
            filename
        )

        np.save(
            output_path,
            prediction.astype(np.float32)
        )

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Verify output count
    # --------------------------------------------------------

    output_files = [
        f for f in os.listdir(output_dir)
        if f.lower().endswith(".npy")
    ]

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)

    print(f"Input images       : {len(test_files)}")
    print(f"Output images      : {len(output_files)}")
    print(f"Output directory   : {output_dir}")
    print(f"Output shape       : {predictions[0].shape}")
    print(f"Output dtype       : {predictions.dtype}")
    print(
        f"Output range      : "
        f"{predictions.min():.6f} - {predictions.max():.6f}"
    )
    print(f"Inference time     : {inference_time:.3f} sec")
    print(f"Total time         : {total_time:.3f} sec")

    if len(output_files) != len(test_files):
        print("\nWARNING: Number of outputs does not match inputs.")
        sys.exit(1)

    print("\nAll outputs saved successfully.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()