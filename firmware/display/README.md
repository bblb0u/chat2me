# Chat2Me Display Firmware

ESP-IDF firmware for the Waveshare ESP32-S3 LCD status display.

The firmware initializes the board PMU, LCD, backlight, and LVGL UI. It reads newline-delimited JSON status updates from the USB Serial/JTAG console:

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
. /opt/esp-idf-v5.5.4/export.sh
cd firmware/display
idf.py build
idf.py -p /dev/ttyACM0 flash
```

Use ESP-IDF 5.5.x for this board. The Waveshare examples require ESP-IDF 5.4.0 or newer, and this firmware has been verified with ESP-IDF 5.5.4. This board uses an ST7796 LCD controller.

The boot sequence forces the backlight off before PMU/LCD initialization, clears the panel to black, renders the first Chat2Me status frame, and only then enables the backlight. Status updates are normally forwarded by the optional `chat2me-relay` container.
