# Elecraft-REFCAL-Scripts
Python Scripts to tune the REFCAL parameter automatically to WWV or a 10MHz GPSDO

Be sure to adjust the configuration for your setup:

'''
#--- Configuration ---
COM_PORT = "COM4"
BAUD_RATE = 38400
AUDIO_DEVICE_NAME = "Microphone (4- USB Audio Device"  # Automatically matches input device by name

REF_RF_HZ = 10_000_000.0           # GPSDO reference frequency (10.000000 MHz)
DEFAULT_AUDIO_HZ = 700.0           # Fallback CW pitch if PI; read fails
SAMPLE_RATE = 48000                # Soundcard sample rate (Hz)
CAPTURE_SECONDS = 4.0              # 4s capture = 0.25 Hz bin width (~0.01 Hz interpolated)

DEADBAND_REFCAL_HZ = 0.1  # Skip EEPROM write if correction is <= 1 Hz (~0.4 Hz at 10 MHz)
MAX_STEP_CLAMP = 200               # Safety cap on single-run adjustment (Hz)
'''

The scripts samples the CW audio output at 10.000.000 Mhz for 4 seconds.
Sidetone is then measured and the difference computed.
The REFCAL register is then accessed and set.

You should let the radio be on and warm for 10 minutes or more prior to tuning.
The input signal should be S7 or greater for a solid measurement.
Outcome should land within a couple of Hertz.

Hope this helps,
73
WR9R
