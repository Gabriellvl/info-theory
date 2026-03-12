import sys

from AudioCD import AudioCD
import numpy as np
import pytest


def test_C3_enc_8_parity_basic():
    """Test C3_enc_8_parity with single frame"""
    audio_cd = AudioCD(Fs=44100, configuration=3, max_interpolation=8)
    n_frames = 2
    input_data = np.random.randint(0, 256, 24 * n_frames, dtype=np.uint8)

    output, output_frames = audio_cd.C3_enc_8_parity(input_data, n_frames)
    print("Output:", output)
    print("Output Frames:", output_frames)

    assert output.shape == (32 * n_frames,)
    assert output_frames == n_frames
    assert output.dtype == np.uint8


def introduce_bit_errors(data, n_frames, num_bit_errors, frame_size=32):
    """Introduce random bit errors into data"""
    corrupted_data = data.copy()
    for frame in range(n_frames):
        error_indices = np.random.choice(frame_size * 8, num_bit_errors, replace=False)
        for idx in error_indices:
            byte_idx = idx // 8
            bit_idx = idx % 8
            corrupted_data[frame * frame_size + byte_idx] ^= 1 << bit_idx
    return corrupted_data


def test_C3_enc_dec_8_parity_with_bit_errors():
    """Test encoding and decoding with different amounts of bit errors"""
    audio_cd = AudioCD(Fs=44100, configuration=3, max_interpolation=8)
    n_frames = 2
    input_data = np.random.randint(0, 256, 24 * n_frames, dtype=np.uint8)

    encoded_data, encoded_frames = audio_cd.C3_enc_8_parity(input_data, n_frames)

    for num_bit_errors in range(5):  # Test with 0 to 11 bit errors
        corrupted_data = introduce_bit_errors(encoded_data, n_frames, num_bit_errors)
        # Decode the corrupted data
        decoded_data, erasure_flags_out, decoded_frames = audio_cd.C3_dec_8_parity(
            corrupted_data, encoded_frames
        )

        print(f"Input Data: {input_data}")
        print(f"Encoded Data: {encoded_data}")
        print(f"Corrupted Data: {corrupted_data}")
        print(f"Decoded Data: {decoded_data}")
        print(f"Erasure Flags Out: {erasure_flags_out}")
        print(f"Encoded Frames: {encoded_frames}")
        print(f"Decoded Frames: {decoded_frames}")
        print(f"Number of Bit Errors: {num_bit_errors}")

        # Assertions
        assert encoded_data.shape == (32 * n_frames,)
        assert encoded_frames == n_frames
        assert decoded_data.shape == (24 * n_frames,)
        assert decoded_frames == n_frames
        assert erasure_flags_out.shape == (24 * n_frames,)
        assert erasure_flags_out.dtype == np.float64

        if num_bit_errors == 5:
            pass
        elif num_bit_errors >= 5:
            # Decoding should fail for more than 5 bit errors
            assert 0 not in erasure_flags_out
        else:
            assert 1 not in erasure_flags_out
            assert np.array_equal(input_data, decoded_data)


def test_CIRC_enc_C2_basic():
    """Test CIRC_enc_C2 with single frame"""
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    input_data = np.random.randint(0, 256, 24, dtype=np.uint8)
    n_frames = 1

    output, output_frames = audio_cd.CIRC_enc_C2(input_data, n_frames)
    print("Input Data:", input_data)
    print("Output Data:", output)
    print("Output Frames:", output_frames)

    assert output.shape == (28,)
    assert output_frames == 1
    assert output.dtype == np.uint8


def test_CIRC_enc_dec_C2():
    """Test CIRC encoding and decoding together"""
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    n_frames = 2
    input_data = np.random.randint(0, 256, 24 * n_frames, dtype=np.uint8)

    # Encode the data
    encoded_data, encoded_frames = audio_cd.CIRC_enc_C2(input_data, n_frames)

    for num_bit_errors in range(3):  # Test with 0 to 11 bit errors
        corrupted_data = encoded_data.copy()
        error_indices = np.random.choice(
            len(corrupted_data) * 8, num_bit_errors, replace=False
        )
        for idx in error_indices:
            byte_idx = idx // 8
            bit_idx = idx % 8
            corrupted_data[byte_idx] ^= 1 << bit_idx  # Flip the bit

        erasure_flags_in = np.zeros(28)  # No erasures for this test
        # erasure_flags_in = None
        # Decode the corrupted data
        decoded_data, erasure_flags_out, decoded_frames = audio_cd.CIRC_dec_C2(
            corrupted_data,
            erasure_flags_in,
            encoded_frames,
        )

        print(f"Input Data: {input_data}")
        print(f"Encoded Data: {encoded_data}")
        print(f"Corrupted Data: {corrupted_data}")
        print(f"Decoded Data: {decoded_data}")
        print(f"Erasure Flags Out: {erasure_flags_out}")
        print(f"Encoded Frames: {encoded_frames}")
        print(f"Decoded Frames: {decoded_frames}")
        print(f"Number of Bit Errors: {num_bit_errors}")

        # Assertions
        assert encoded_data.shape == (28 * n_frames,)
        assert encoded_frames == n_frames
        assert decoded_data.shape == (24 * n_frames,)
        assert decoded_frames == n_frames
        assert erasure_flags_out.shape == (24 * n_frames,)
        assert erasure_flags_out.dtype == np.float64

        if num_bit_errors == -1:
            pass
        elif num_bit_errors >= 3:
            # Decoding should fail for more than 5 bit errors
            assert 0 not in erasure_flags_out
        else:
            assert 1 not in erasure_flags_out
            assert np.array_equal(input_data, decoded_data)


def test_CIRC_enc_dec_C1():
    """Test C1 encoding and decoding together"""
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    n_frames = 2
    input_data = np.random.randint(0, 256, 28 * n_frames, dtype=np.uint8)

    # Encode
    encoded_data, encoded_frames = audio_cd.CIRC_enc_C1(input_data, n_frames)

    assert encoded_data.shape == (
        32 * n_frames,
    ), f"Expected (32,), got {encoded_data.shape}"
    assert encoded_frames == n_frames

    for num_byte_errors in range(3):  # RS(32,28) can correct up to 2 symbol errors
        corrupted_data = encoded_data.copy()
        error_indices = np.random.choice(
            len(corrupted_data), num_byte_errors, replace=False
        )
        for idx in error_indices:
            corrupted_data[idx] ^= np.random.randint(1, 256)  # Flip random bits in byte

        decoded_data, erasure_flags_out, decoded_frames = audio_cd.CIRC_dec_C1(
            corrupted_data,
            n_frames,
        )

        print(f"\n--- num_byte_errors={num_byte_errors} ---")
        print(f"Input Data:      {input_data}")
        print(f"Encoded Data:    {encoded_data}")
        print(f"Corrupted Data:  {corrupted_data}")
        print(f"Decoded Data:    {decoded_data}")
        print(f"Erasure Flags:   {erasure_flags_out}")

        assert decoded_data.shape == (
            28 * n_frames,
        ), f"Expected (28,), got {decoded_data.shape}"
        assert decoded_frames == n_frames
        assert erasure_flags_out.shape == (28 * n_frames,)
        assert erasure_flags_out.dtype == np.float64

        if num_byte_errors <= 2:
            assert np.array_equal(input_data, decoded_data)
            assert 1 not in erasure_flags_out
        elif num_byte_errors >= 3:
            assert 0 not in erasure_flags_out


def test_delay_interleave_visual():
    """Test C1 encoding and decoding together"""
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    SYMBOLS_PER_FRAME = 24

    # 3 frames: first frame all ones, rest zeros
    frame0 = np.ones(SYMBOLS_PER_FRAME, dtype=float)
    frame1 = np.zeros(SYMBOLS_PER_FRAME, dtype=float)
    frame2 = np.zeros(SYMBOLS_PER_FRAME, dtype=float)
    input_data = np.concatenate([frame0, frame1, frame2])
    n_frames = 3

    # Label each symbol with its original index for clarity
    # (overwrite with symbol index so we can track where things go)
    input_data = np.tile(np.arange(SYMBOLS_PER_FRAME, dtype=float), n_frames)
    # Only frame 0 has values, rest are zero — mark frame with 100*frame_idx + sym_idx
    input_data = np.zeros(n_frames * SYMBOLS_PER_FRAME, dtype=float)
    input_data[:SYMBOLS_PER_FRAME] = (
        np.arange(SYMBOLS_PER_FRAME, dtype=float) + 1
    )  # 1..24 in frame 0

    print("=== INPUT (3 frames x 24 symbols) ===")
    input_2d = input_data.reshape(n_frames, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames):
        print(f"  {f:<6}", input_2d[f].astype(int).tolist())

    output, n_frames_out = audio_cd.CIRC_enc_delay_interleave(input_data, n_frames)

    print(f"\n=== OUTPUT ({n_frames_out} frames x 24 symbols) ===")
    output_2d = output.reshape(n_frames_out, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_out):
        print(f"  {f:<6}", output_2d[f].astype(int).tolist())


if __name__ == "__main__":
    # off_test_CIRC_enc_dec_C2()
    # visual_test_delay_interleave_visual()
    pytest.main([__file__] + sys.argv[1:])
