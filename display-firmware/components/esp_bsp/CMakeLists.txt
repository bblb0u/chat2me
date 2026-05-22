set(SRC_FILES
    "bsp_axp2101.cpp"
    "bsp_display.c"
    "bsp_i2c.c"
    "bsp_touch.c"
)

idf_component_register(SRCS ${SRC_FILES}
                    INCLUDE_DIRS "."
                    REQUIRES "esp_lcd" "driver" "espressif__esp_lcd_axs15231b" "XPowersLib")
