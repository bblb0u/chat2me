set(SRC_FILES
    "bsp_axp2101.cpp"
    "bsp_display.c"
    "bsp_i2c.c"
)

idf_component_register(SRCS ${SRC_FILES}
                    INCLUDE_DIRS "."
                    REQUIRES "esp_lcd" "esp_driver_gpio" "esp_driver_i2c" "esp_driver_ledc" "esp_driver_spi" "esp_lcd_st7796" "XPowersLib")
