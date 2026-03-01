from AudioCD import AudioCD
import numpy as np
import pytest


def test_C3_enc_8_parity_basic():
    """Test C3_enc_8_parity with single frame"""
    audio_cd = AudioCD(Fs=44100, configuration=3, max_interpolation=8)
    input_data = np.random.randint(0, 256, 24, dtype=np.uint8)
    n_frames = 1

    output, output_frames = audio_cd.C3_enc_8_parity(input_data, n_frames)
    print("Output:", output)
    print("Output Frames:", output_frames)

    assert output.shape == (32,)
    assert output_frames == 1
    assert output.dtype == np.uint8


def test_C3_dec_8_parity_basic():
    """Test C3_dec_8_parity with single frame"""
    audio_cd = AudioCD(Fs=44100, configuration=3, max_interpolation=8)
    input_data = np.random.randint(0, 256, 32, dtype=np.uint8)
    n_frames = 1

    output, erasure_flags_out, output_frames = audio_cd.C3_dec_8_parity(
        input_data, n_frames
    )
    print("Input:", input_data)
    print("Output:", output)
    print("Erasure Flags Out:", erasure_flags_out)
    print("Output Frames:", output_frames)

    assert output.shape == (24,)
    assert erasure_flags_out.shape == (24,)
    assert output_frames == 1
    assert output.dtype == np.uint8
    assert erasure_flags_out.dtype == np.float64


def test_C3_enc_dec_8_parity_with_bit_errors():
    """Test encoding and decoding with different amounts of bit errors"""
    audio_cd = AudioCD(Fs=44100, configuration=3, max_interpolation=8)
    input_data = np.random.randint(0, 256, 24, dtype=np.uint8)
    n_frames = 1

    encoded_data, encoded_frames = audio_cd.C3_enc_8_parity(input_data, n_frames)

    for num_bit_errors in range(12):  # Test with 0 to 11 bit errors
        corrupted_data = encoded_data.copy()
        error_indices = np.random.choice(
            len(corrupted_data) * 8, num_bit_errors, replace=False
        )
        for idx in error_indices:
            byte_idx = idx // 8
            bit_idx = idx % 8
            corrupted_data[byte_idx] ^= 1 << bit_idx  # Flip the bit

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
        assert encoded_data.shape == (32,)
        assert encoded_frames == 1
        assert decoded_data.shape == (24,)
        assert decoded_frames == 1
        assert erasure_flags_out.shape == (24,)
        assert erasure_flags_out.dtype == np.float64

        if num_bit_errors == 5:
            pass
        elif num_bit_errors >= 5:
            # Decoding should fail for more than 5 bit errors
            assert 0 not in erasure_flags_out
        else:
            assert 1 not in erasure_flags_out
            assert np.array_equal(input_data, decoded_data)


if __name__ == "__main__":
    pytest.main([__file__])
