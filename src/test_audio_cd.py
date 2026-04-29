import sys

from AudioCD import AudioCD
import numpy as np
import pytest


def test_C3_enc_8_parity_basic():
    """Tests given function"""
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
    """Helper function
    Introduces random bit errors into data"""
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

    for num_bit_errors in range(2):  # Test with 0 to 11 bit errors
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

        assert encoded_data.shape == (32 * n_frames,)
        assert encoded_frames == n_frames
        assert decoded_data.shape == (24 * n_frames,)
        assert decoded_frames == n_frames
        assert erasure_flags_out.shape == (24 * n_frames,)

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

    for num_bit_errors in range(3):
        corrupted_data = encoded_data.copy()
        error_indices = np.random.choice(
            len(corrupted_data) * 8, num_bit_errors, replace=False
        )
        for idx in error_indices:
            byte_idx = idx // 8
            bit_idx = idx % 8
            corrupted_data[byte_idx] ^= 1 << bit_idx  # Flip the bit

        erasure_flags_in = np.zeros(28)
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

        assert encoded_data.shape == (28 * n_frames,)
        assert encoded_frames == n_frames
        assert decoded_data.shape == (24 * n_frames,)
        assert decoded_frames == n_frames
        assert erasure_flags_out.shape == (24 * n_frames,)

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

    for num_byte_errors in range(2):
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

        if num_byte_errors <= 2:
            assert np.array_equal(input_data, decoded_data)
            assert 1 not in erasure_flags_out
        elif num_byte_errors >= 3:
            assert 0 not in erasure_flags_out


def test_enc_dec_deinterlave_delay():
    """Test that enc_delay_interleave and dec_deinterleave_delay are inverses"""
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    SYMBOLS_PER_FRAME = 24

    # Only frame 0 has values (1..24), rest are zeros
    n_frames = 3
    input_data = np.zeros(n_frames * SYMBOLS_PER_FRAME)
    input_data[:SYMBOLS_PER_FRAME] = (
        np.arange(SYMBOLS_PER_FRAME) + 1
    )  # 1..24 in frame 0

    print("=== ORIGINAL INPUT (3 frames x 24 symbols) ===")
    input_2d = input_data.reshape(n_frames, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames):
        print(f"  {f:<6}", input_2d[f].astype(int).tolist())

    # --- ENCODE ---
    encoded, n_frames_enc = audio_cd.CIRC_enc_delay_interleave(input_data, n_frames)

    print(f"\n=== ENCODED ({n_frames_enc} frames x 24 symbols) ===")
    encoded_2d = encoded.reshape(n_frames_enc, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_enc):
        print(f"  {f:<6}", encoded_2d[f].astype(int).tolist())

    # --- DECODE --- use encoded output as erasure flags too
    erasure_flags = encoded.copy()
    decoded, erasure_flags_out, n_frames_dec = audio_cd.CIRC_dec_deinterleave_delay(
        encoded, erasure_flags, n_frames_enc
    )

    print(f"\n=== DECODED ({n_frames_dec} frames x 24 symbols) ===")
    decoded_2d = decoded.reshape(n_frames_dec, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_dec):
        print(f"  {f:<6}", decoded_2d[f].astype(int).tolist())

    print(f"\n=== ERASURE FLAGS OUT ({n_frames_dec} frames x 24 symbols) ===")
    erasure_2d = erasure_flags_out.reshape(n_frames_dec, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_dec):
        print(f"  {f:<6}", erasure_2d[f].astype(int).tolist())

    expected = input_data[: n_frames_dec * SYMBOLS_PER_FRAME]
    assert np.array_equal(decoded, expected)


def test_enc_dec_delay_unequal():
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    SYMBOLS_PER_FRAME = 28

    # Only frame 0 has values (1..24), rest are zeros
    n_frames = 2
    input_data = np.zeros(n_frames * SYMBOLS_PER_FRAME)
    input_data[:SYMBOLS_PER_FRAME] = (
        np.arange(SYMBOLS_PER_FRAME) + 1
    )  # 1..24 in frame 0

    print("=== ORIGINAL INPUT (3 frames x 28 symbols) ===")
    input_2d = input_data.reshape(n_frames, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames):
        print(f"  {f:<6}", input_2d[f].astype(int).tolist())

    # --- ENCODE ---
    encoded, n_frames_enc = audio_cd.CIRC_enc_delay_unequal(input_data, n_frames)

    print(f"\n=== ENCODED ({n_frames_enc} frames x 24 symbols) ===")
    encoded_2d = encoded.reshape(n_frames_enc, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_enc):
        print(f"  {f:<6}", encoded_2d[f].astype(int).tolist())

    # assert encoded_2d

    # --- DECODE --- use encoded output as erasure flags too
    erasure_flags = encoded.copy()
    decoded, erasure_flags_out, n_frames_dec = audio_cd.CIRC_dec_delay_unequal(
        encoded, erasure_flags, n_frames_enc
    )

    print(f"\n=== DECODED ({n_frames_dec} frames x 24 symbols) ===")
    decoded_2d = decoded.reshape(n_frames_dec, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_dec):
        print(f"  {f:<6}", decoded_2d[f].astype(int).tolist())

    print(f"\n=== ERASURE FLAGS OUT ({n_frames_dec} frames x 24 symbols) ===")
    erasure_2d = erasure_flags_out.reshape(n_frames_dec, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_dec):
        print(f"  {f:<6}", erasure_2d[f].astype(int).tolist())

    expected = input_data[: n_frames_dec * SYMBOLS_PER_FRAME]
    assert np.array_equal(decoded, expected)


def test_enc_dec_delay_inv():
    audio_cd = AudioCD(Fs=44100, configuration=1, max_interpolation=8)
    SYMBOLS_PER_FRAME = 32

    # Only frame 0 has values (1..24), rest are zeros
    n_frames = 2
    input_data = np.zeros(n_frames * SYMBOLS_PER_FRAME)
    input_data[:SYMBOLS_PER_FRAME] = (
        np.arange(SYMBOLS_PER_FRAME) + 1
    )  # 1..24 in frame 0

    print("=== ORIGINAL INPUT (3 frames x 32 symbols) ===")
    input_2d = input_data.reshape(n_frames, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames):
        print(f"  {f:<6}", input_2d[f].astype(int).tolist())

    # --- ENCODE ---
    encoded, n_frames_enc = audio_cd.CIRC_enc_delay_inv(input_data, n_frames)

    print(f"\n=== ENCODED ({n_frames_enc} frames x 32 symbols) ===")
    encoded_2d = encoded.reshape(n_frames_enc, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_enc):
        print(f"  {f:<6}", encoded_2d[f].astype(int).tolist())

    # assert encoded_2d

    # --- DECODE --- use encoded output as erasure flags too
    decoded, n_frames_dec = audio_cd.CIRC_dec_delay_inv(encoded, n_frames_enc)

    print(f"\n=== DECODED ({n_frames_dec} frames x 32 symbols) ===")
    decoded_2d = decoded.reshape(n_frames_dec, SYMBOLS_PER_FRAME)
    print(f"{'Frame':<8}", [f"s{i:<3}" for i in range(SYMBOLS_PER_FRAME)])
    for f in range(n_frames_dec):
        print(f"  {f:<6}", decoded_2d[f].astype(int).tolist())
    expected = input_data[: n_frames_dec * SYMBOLS_PER_FRAME]
    assert np.array_equal(decoded, expected)


if __name__ == "__main__":
    # If this python file is run, all functions starting with 'test' is run using pytest
    pytest.main([__file__] + sys.argv[1:])
