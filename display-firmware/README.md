# Chat2M Display Firmware

ESP-IDF firmware for the Waveshare ESP32-S3-Touch-LCD-3.5B status display.

The firmware initializes the board PMU, LCD, touch controller, backlight, and LVGL UI. It reads newline-delimited JSON status updates from the USB Serial/JTAG console:

```json
{"state":"speaking","text":"playing response"}
```

Supported states:

- `idle`
- `listening`
- `thinking`
- `speaking`
- `error`

## Build And Flash

```bash
. /opt/esp-idf-v5.3.5/export.sh
cd /opt/chat2m/display-firmware
idf.py build
idf.py -p /dev/ttyACM0 flash
```

The boot sequence shows red, green, blue, and white bars for a few seconds before the Chat2M status UI starts. That makes it easier to distinguish a panel data-path issue from a voice-agent status issue.
