# Chat2M v0.1.0

First working hardware release for the local voice assistant and Waveshare display stack.

## Highlights

- Voice agent runs with wake word detection, offline ASR, local Piper TTS, and the local Ollama gateway.
- Waveshare ESP32-S3-Touch-LCD-3.5 display firmware is fixed for the ST7796 LCD and FT6336 touch controller.
- Display state now follows the voice flow: idle, listening, thinking, speaking, and error.
- Startup and runtime were verified on the connected ESP32-S3 board and `/dev/ttyACM0` serial link.

## Firmware Assets

- `chat2m_display-v0.1.0.bin`
- `bootloader-v0.1.0.bin`
- `partition-table-v0.1.0.bin`
- `flash_args-v0.1.0`

## Verified

- `idf.py build`
- `idf.py -p /dev/ttyACM0 flash`
- ESP32 boot log reaches `chat2m_display: ui ready`
- `chat2m-voice-agent` successfully detects wake word, handles a conversation turn, speaks the response, and drives the display state over serial
