import numpy as np
import librosa
import soundfile as sf
import pywt
import argparse

def load_audio(file_path, target_sr=44100, mono=False):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    if not mono:
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
    return audio, sr

def save_audio(file_path, audio, sr, subtype='PCM_16'):
    if audio.ndim > 1:
        audio = audio.T
    sf.write(file_path, audio, sr, subtype=subtype)

def text_to_data(text):
    binary_str = ''.join(f'{ord(char):08b}' for char in text + '\0')
    return np.array([int(bit) for bit in binary_str])

def data_to_text(data):
    binary_str = ''.join(str(int(bit)) for bit in data)
    text = ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))
    return text.split('\0')[0]

def embed_dwt(cover, text, wavelet='haar', level=1, alpha=0.1):
    binary_text = ''.join(format(ord(char), '08b') for char in text) + '00000000'
    secret_data = np.array([int(bit) for bit in binary_text], dtype=np.float32)

    if cover.ndim == 1:
        cover = np.expand_dims(cover, axis=0)
    channels = cover.shape[0]

    stego = np.copy(cover)

    for ch in range(channels):
        coeffs_cover = pywt.wavedec(cover[ch], wavelet, level=level)

        stego_coeffs = [coeffs_cover[0]]
        for i in range(1, len(coeffs_cover)):
            coeffs_len = len(coeffs_cover[i])
            secret_data_segment = np.pad(secret_data[:coeffs_len], (0, max(0, coeffs_len - len(secret_data))), 'constant')
            stego_coeffs.append(coeffs_cover[i] + alpha * secret_data_segment)
            secret_data = secret_data[coeffs_len:]

        stego_channel = pywt.waverec(stego_coeffs, wavelet)
        stego_channel = stego_channel[:len(cover[ch])]
        stego[ch] = np.clip(stego_channel, -1.0, 1.0)

    if channels == 1:
        stego = stego.flatten()

    return stego

def extract_dwt(stego, wavelet='haar', level=1, alpha=0.1):
    if stego.ndim == 1:
        stego = np.expand_dims(stego, axis=0)
    channels = stego.shape[0]

    extracted_data = []

    for ch in range(channels):
        coeffs_stego = pywt.wavedec(stego[ch], wavelet, level=level)

        for i in range(1, len(coeffs_stego)):
            extracted_data.extend(np.round(coeffs_stego[i] / alpha).astype(int).clip(0, 1))

    binary_str = ''.join(str(int(bit)) for bit in extracted_data)
    extracted_text = ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))

    return extracted_text.split('\0')[0]

parser = argparse.ArgumentParser(description="Audio steganography")
parser.add_argument('-s', '--secret', type=str, help="Specify secret text")
parser.add_argument('-i', '--input', type=str, help="Specify input filename")
parser.add_argument('-o', '--output', type=str, help="Specify output filename")
parser.add_argument('-e', '--extract', type=str, help="Specify embedded filename to extract secret")
args = parser.parse_args()
# Cấu hình
TARGET_SAMPLE_RATE = 44100
WAVELET = 'haar'
LEVEL = 1
ALPHA = 0.1

# Đường dẫn file
text_to_hide = ""
if args.secret:
    text_to_hide = args.secret
if args.input:
    cover_file = args.input
    # Load audio
    cover, sr_cover = load_audio(cover_file, target_sr=TARGET_SAMPLE_RATE, mono=False)
    if args.output:
        stego_file = args.output
        # Nhúng văn bản vào âm thanh
        stego = embed_dwt(cover, text_to_hide, wavelet=WAVELET, level=LEVEL, alpha=ALPHA)
        save_audio(stego_file, stego, sr_cover)
        print("Secret was embedded successfully")

if args.extract:
    stego_file = args.extract
    stego_loaded, _ = load_audio(stego_file, target_sr=TARGET_SAMPLE_RATE, mono=False)
    extracted_text = extract_dwt(stego_loaded, wavelet=WAVELET, level=LEVEL, alpha=ALPHA)
    print(f"Secret: {extracted_text}")