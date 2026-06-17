#include "bsp_display.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/spi_master.h"
#include "esp_check.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_st7796.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "bsp_display";

static uint8_t g_brightness = 0;

void bsp_display_backlight_force_off(void)
{
    gpio_config_t bl_gpio = {};
    bl_gpio.pin_bit_mask = 1ULL << EXAMPLE_PIN_LCD_BL;
    bl_gpio.mode = GPIO_MODE_OUTPUT;
    bl_gpio.pull_up_en = GPIO_PULLUP_DISABLE;
    bl_gpio.pull_down_en = GPIO_PULLDOWN_ENABLE;
    bl_gpio.intr_type = GPIO_INTR_DISABLE;
    ESP_ERROR_CHECK(gpio_config(&bl_gpio));
    ESP_ERROR_CHECK(gpio_set_level(EXAMPLE_PIN_LCD_BL, 0));
    g_brightness = 0;
}

void bsp_display_init(esp_lcd_panel_io_handle_t *io_handle, esp_lcd_panel_handle_t *panel_handle, size_t max_transfer_sz)
{
    ESP_LOGI(TAG, "SPI bus init for ST7796");

    spi_bus_config_t buscfg = {};
    buscfg.sclk_io_num = EXAMPLE_PIN_LCD_SCLK;
    buscfg.mosi_io_num = EXAMPLE_PIN_LCD_MOSI;
    buscfg.miso_io_num = EXAMPLE_PIN_LCD_MISO;
    buscfg.quadwp_io_num = GPIO_NUM_NC;
    buscfg.quadhd_io_num = GPIO_NUM_NC;
    buscfg.max_transfer_sz = max_transfer_sz;
    ESP_ERROR_CHECK(spi_bus_initialize(EXAMPLE_SPI_HOST, &buscfg, SPI_DMA_CH_AUTO));

    ESP_LOGI(TAG, "Install ST7796 panel IO");
    esp_lcd_panel_io_spi_config_t io_config = {};
    io_config.cs_gpio_num = EXAMPLE_PIN_LCD_CS;
    io_config.dc_gpio_num = EXAMPLE_PIN_LCD_DC;
    io_config.spi_mode = 0;
    io_config.pclk_hz = EXAMPLE_LCD_PIXEL_CLOCK_HZ;
    io_config.trans_queue_depth = 10;
    io_config.lcd_cmd_bits = 8;
    io_config.lcd_param_bits = 8;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)EXAMPLE_SPI_HOST, &io_config, io_handle));

    esp_lcd_panel_dev_config_t panel_config = {};
    panel_config.reset_gpio_num = EXAMPLE_PIN_LCD_RST;
    panel_config.rgb_ele_order = LCD_RGB_ELEMENT_ORDER_BGR;
    panel_config.bits_per_pixel = 16;
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7796(*io_handle, &panel_config, panel_handle));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(*panel_handle));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(*panel_handle));
    vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(*panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(*panel_handle, true, false));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(*panel_handle, true));
    vTaskDelay(pdMS_TO_TICKS(80));
}

void bsp_display_brightness_init(void)
{
    bsp_display_backlight_force_off();

    ledc_timer_config_t ledc_timer = {};
    ledc_timer.speed_mode = LCD_BL_LEDC_MODE;
    ledc_timer.timer_num = LCD_BL_LEDC_TIMER;
    ledc_timer.duty_resolution = LCD_BL_LEDC_DUTY_RES;
    ledc_timer.freq_hz = LCD_BL_LEDC_FREQUENCY;
    ledc_timer.clk_cfg = LEDC_AUTO_CLK;
    ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));

    ledc_channel_config_t ledc_channel = {};
    ledc_channel.speed_mode = LCD_BL_LEDC_MODE;
    ledc_channel.channel = LCD_BL_LEDC_CHANNEL;
    ledc_channel.timer_sel = LCD_BL_LEDC_TIMER;
    ledc_channel.intr_type = LEDC_INTR_DISABLE;
    ledc_channel.gpio_num = EXAMPLE_PIN_LCD_BL;
    ledc_channel.duty = 0;
    ledc_channel.hpoint = 0;
    ESP_ERROR_CHECK(ledc_channel_config(&ledc_channel));
}

void bsp_display_set_brightness(uint8_t brightness)
{
    if (brightness > 100) {
        brightness = 100;
        ESP_LOGE(TAG, "Brightness value out of range");
    }

    g_brightness = brightness;
    uint32_t duty = (brightness * (LCD_BL_LEDC_DUTY - 1)) / 100;

    ESP_ERROR_CHECK(ledc_set_duty(LCD_BL_LEDC_MODE, LCD_BL_LEDC_CHANNEL, duty));
    ESP_ERROR_CHECK(ledc_update_duty(LCD_BL_LEDC_MODE, LCD_BL_LEDC_CHANNEL));

    ESP_LOGI(TAG, "LCD brightness set to %d%%", brightness);
}

uint8_t bsp_display_get_brightness(void)
{
    return g_brightness;
}
