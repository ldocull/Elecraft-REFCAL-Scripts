#
# This script uses WWV to adjust the internal clock reference to perfect
# the reference at 10Mhz..
#
# WR9R 11/11/2027
#

import time
import numpy as np
import serial
import sounddevice as sd


# --- Configuration ---
COM_PORT = "COM4"                  # Update if KX3 uses a different COM port
BAUD_RATE = 38400
AUDIO_DEVICE_NAME = "Microphone (USB PnP Sound Devic"  # Automatically matches input device by name

REF_RF_HZ = 10_000_000.0           # GPSDO reference frequency (10.000000 MHz)
DEFAULT_AUDIO_HZ = 700.0           # Fallback CW pitch if PI; read fails
SAMPLE_RATE = 48000                # Soundcard sample rate (Hz)
CAPTURE_SECONDS = 4.0              # 4s capture = 0.25 Hz bin width (~0.01 Hz interpolated)

# On the KX3, each step in REF CAL is 10 Hz (~0.875 Hz RF shift at 10 MHz)
KX3_REFCAL_STEP_HZ = 10
DEADBAND_STEPS = 1                 # Skip EEPROM write if correction is <= 1 step (~0.88 Hz at 10 MHz)
MAX_STEP_CLAMP = 100               # Safety cap on single-run step count (100 steps = 1,000 Hz)


def find_input_device_index(name_substring: str) -> int:
    """Finds the sounddevice input index matching the given device name."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and name_substring.lower() in dev["name"].lower():
            print(f"Selected Audio Input [{idx}]: {dev['name']}")
            return idx

    available = [f"[{i}] {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    raise RuntimeError(
        f"Audio input device matching '{name_substring}' not found.\n"
        f"Available input devices:\n  " + "\n  ".join(available)
    )


def read_cw_pitch_hz(ser: serial.Serial) -> float:
    """Queries the KX3's current CW sidetone pitch via 'PI;' (e.g., PI070; -> 700.0 Hz)."""
    ser.reset_input_buffer()
    ser.write(b"PI;")
    ser.flush()
    raw = ser.read_until(b";").strip()
    if raw.startswith(b"PI") and raw.endswith(b";") and len(raw) >= 6:
        return float(int(raw[2:5]) * 10)
    return DEFAULT_AUDIO_HZ


def measure_audio_peak_hz(
    device_index: int,
    target_hz: float,
    sample_rate: int = SAMPLE_RATE,
    duration: float = CAPTURE_SECONDS,
    search_window_hz: float = 150.0,
) -> tuple[float, float]:
    """
    Records audio from the KX3 audio output, runs a Hann-windowed FFT with
    parabolic interpolation, and returns (detected_freq_hz, snr_db).
    """
    num_samples = int(sample_rate * duration)
    print(f"Sampling audio for {duration:.1f}s...")
    recording = sd.rec(
        num_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float64",
        device=device_index,
    )
    sd.wait()

    samples = recording[:, 0]
    samples = samples - np.mean(samples)  # Remove DC offset

    window = np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(samples * window))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)

    mask = (freqs >= target_hz - search_window_hz) & (freqs <= target_hz + search_window_hz)
    window_spec = spectrum[mask]

    peak_idx_local = int(np.argmax(window_spec))
    peak_idx_global = int(np.flatnonzero(mask)[peak_idx_local])

    noise_floor = float(np.median(window_spec))
    peak_mag = float(spectrum[peak_idx_global])
    snr_db = 20.0 * np.log10(peak_mag / max(noise_floor, 1e-12))

    if 0 < peak_idx_global < len(spectrum) - 1:
        y0 = np.log(max(spectrum[peak_idx_global - 1], 1e-12))
        y1 = np.log(max(spectrum[peak_idx_global], 1e-12))
        y2 = np.log(max(spectrum[peak_idx_global + 1], 1e-12))
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0

    bin_width = sample_rate / len(samples)
    exact_freq = (peak_idx_global + delta) * bin_width
    return float(exact_freq), float(snr_db)


def decode_kx3_refcal_hz(raw: bytes) -> int:
    """
    Decodes KX3 'DS;' response inside MN062 into an integer Hz value.
    KX3 displays 8 digits as '114.nnn.nn' (10 Hz resolution), so we multiply by 10.
    """
    raw = raw.strip()
    if not (raw.startswith(b"DS") and raw.endswith(b";")) or len(raw) < 13:
        raise ValueError(f"Unexpected DS response: {raw!r}")

    digits = [chr(b & 0x7F) for b in raw[2:10] if chr(b & 0x7F).isdigit()]
    if len(digits) != 8:
        raise ValueError(f"Expected 8 digits in KX3 REF CAL display, got: {raw!r}")
    return int("".join(digits)) * KX3_REFCAL_STEP_HZ


def read_kx3_refcal(ser: serial.Serial) -> int:
    """Opens MN062 on KX3 (tech mode auto-unlocks via CAT), reads REF CAL in Hz, and exits."""
    try:
        ser.write(b"MN062;")
        ser.flush()
        time.sleep(0.15)

        ser.reset_input_buffer()
        ser.write(b"DS;")
        ser.flush()
        raw = ser.read_until(b";")
        return decode_kx3_refcal_hz(raw)
    finally:
        ser.write(b"MN255;")
        ser.flush()
        time.sleep(0.05)


def write_kx3_refcal(ser: serial.Serial, target_hz: int) -> int:
    """
    Opens MN062 on KX3, steps UP; or DN; (10 Hz per step) until REF CAL matches target_hz,
    verifies via DS;, and exits with MN255; (commits once to EEPROM).
    """
    try:
        ser.write(b"MN062;")
        ser.flush()
        time.sleep(0.15)

        ser.reset_input_buffer()
        ser.write(b"DS;")
        ser.flush()
        current_hz = decode_kx3_refcal_hz(ser.read_until(b";"))

        steps = round((target_hz - current_hz) / KX3_REFCAL_STEP_HZ)
        if steps == 0:
            return current_hz

        if abs(steps) > MAX_STEP_CLAMP:
            raise RuntimeError(f"Refusing large jump of {steps} steps (exceeds {MAX_STEP_CLAMP} limit).")

        cmd = b"UP;" if steps > 0 else b"DN;"
        for _ in range(abs(steps)):
            ser.write(cmd)
            ser.flush()
            time.sleep(0.04)

        time.sleep(0.12)
        ser.reset_input_buffer()
        ser.write(b"DS;")
        ser.flush()
        verified_hz = decode_kx3_refcal_hz(ser.read_until(b";"))
        return verified_hz
    finally:
        ser.write(b"MN255;")
        ser.flush()
        time.sleep(0.05)


def run_kx3_autocal():
    device_idx = find_input_device_index(AUDIO_DEVICE_NAME)

    with serial.Serial(COM_PORT, BAUD_RATE, timeout=1.0) as ser:
        # 1. Tune KX3 to 10.000000 MHz in CW mode (MD3) and read live CW pitch
        freq_cmd = f"FA{int(REF_RF_HZ):011d};MD3;".encode("ascii")
        ser.write(freq_cmd)
        ser.flush()
        time.sleep(0.25)

        target_pitch_hz = read_cw_pitch_hz(ser)
        print(f"KX3 CW Pitch Setting: {target_pitch_hz:.1f} Hz")

        # 2. Pre-calibration beat-note measurement
        print("\n--- Pre-Calibration Measurement ---")
        measured_audio_hz, snr_db = measure_audio_peak_hz(
            device_index=device_idx,
            target_hz=target_pitch_hz,
        )
        print(f"Detected Audio Tone: {measured_audio_hz:.3f} Hz (SNR: {snr_db:.1f} dB)")

        if snr_db < 20.0:
            print("Aborting: Audio carrier SNR too low (< 20 dB). Check GPSDO signal & audio level.")
            return

        # 3. Calculate RF error and corresponding KX3 REF CAL steps (10 Hz/step)
        dial_error_hz = measured_audio_hz - target_pitch_hz
        current_refcal_hz = read_kx3_refcal(ser)

        # Note: Verify sign on first run via the post-calibration retest below
        raw_correction_hz = +dial_error_hz * (current_refcal_hz / REF_RF_HZ)
        step_correction = round(raw_correction_hz / KX3_REFCAL_STEP_HZ)
        new_refcal_hz = current_refcal_hz + (step_correction * KX3_REFCAL_STEP_HZ)

        print(f"Dial Error @ {REF_RF_HZ/1e6:.3f} MHz: {dial_error_hz:+.3f} Hz")
        print(f"Current REF CAL:        {current_refcal_hz:,} Hz ({current_refcal_hz/1e6:.5f} MHz)")
        print(f"Calculated Adjustment:  {step_correction:+d} steps ({step_correction * KX3_REFCAL_STEP_HZ:+d} Hz)")

        # 4. EEPROM Protection Deadband Check
        if abs(step_correction) <= DEADBAND_STEPS:
            print(f"\nWithin deadband (±{DEADBAND_STEPS} step / ~0.88 Hz at 10 MHz). Skipping EEPROM write.")
            return

        # 5. Apply update
        print("\n--- Applying Calibration ---")
        verified_hz = write_kx3_refcal(ser, new_refcal_hz)
        print(f"Updated & Verified REF CAL: {verified_hz:,} Hz ({verified_hz/1e6:.5f} MHz)")

        # 6. Post-calibration retest
        print("\n--- Post-Calibration Verification ---")
        time.sleep(0.35)
        post_audio_hz, post_snr_db = measure_audio_peak_hz(
            device_index=device_idx,
            target_hz=target_pitch_hz,
        )
        post_error_hz = post_audio_hz - target_pitch_hz
        actual_rf_hz = REF_RF_HZ + post_error_hz

        print(f"New Audio Tone:         {post_audio_hz:.3f} Hz (Target: {target_pitch_hz:.1f} Hz, SNR: {post_snr_db:.1f} dB)")
        print(f"Residual Dial Error:    {post_error_hz:+.3f} Hz")
        print(f"Effective Calibrated RF:{actual_rf_hz:14.3f} Hz")


if __name__ == "__main__":
    run_kx3_autocal()
