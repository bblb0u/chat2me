idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES "esp_bsp" "esp_lv_port" "lvgl" "espressif__esp_io_expander_tca9554" "esp_lcd"
)
