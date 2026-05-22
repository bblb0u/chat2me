#include <stdio.h>
#include <string.h>

#include "bsp_axp2101.h"
#include "bsp_display.h"
#include "bsp_i2c.h"
#include "bsp_touch.h"
#include "esp_heap_caps.h"
#include "esp_io_expander_tca9554.h"
#include "esp_lcd_panel_ops.h"
#include "esp_log.h"
#include "lv_port.h"

#define EXAMPLE_DISPLAY_ROTATION LV_DISP_ROT_NONE
#define EXAMPLE_LCD_H_RES 320
#define EXAMPLE_LCD_V_RES 480
#define LCD_BUFFER_PIXELS (EXAMPLE_LCD_H_RES * EXAMPLE_LCD_V_RES)
#define LCD_BUFFER_BYTES (LCD_BUFFER_PIXELS * sizeof(uint16_t))

#define BOOT_PATTERN_MS 3000
#define BOOT_PATTERN_LINES 32
#define DISPLAY_LINE_SIZE 512

static const char *TAG = "chat2m_display";

static esp_io_expander_handle_t expander_handle = NULL;
static esp_lcd_panel_io_handle_t io_handle = NULL;
static esp_lcd_panel_handle_t panel_handle = NULL;
static lv_disp_t *lvgl_disp = NULL;
static lv_indev_t *lvgl_touch_indev = NULL;

static lv_obj_t *root = NULL;
static lv_obj_t *title_label = NULL;
static lv_obj_t *subtitle_label = NULL;
static lv_obj_t *status_label = NULL;
static lv_obj_t *orb = NULL;
static lv_obj_t *ring = NULL;
static lv_obj_t *bar_left = NULL;
static lv_obj_t *bar_mid = NULL;
static lv_obj_t *bar_right = NULL;

static char current_state[24] = "idle";
static char current_text[96] = "";

static lv_color_t color_bg = lv_color_hex(0x081014);
static lv_color_t color_panel = lv_color_hex(0x0e1d22);
static lv_color_t color_primary = lv_color_hex(0x32d9c8);
static lv_color_t color_accent = lv_color_hex(0xffc857);
static lv_color_t color_error = lv_color_hex(0xff5d73);
static lv_color_t color_text = lv_color_hex(0xe6f5f3);
static lv_color_t color_muted = lv_color_hex(0x7d9794);

extern "C" void app_main(void);
void lv_port_init(void);

static void draw_boot_pattern(void)
{
    uint16_t *band = (uint16_t *)heap_caps_malloc(
        EXAMPLE_LCD_H_RES * BOOT_PATTERN_LINES * sizeof(uint16_t), MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!band) {
        ESP_LOGW(TAG, "boot pattern band allocation failed");
        return;
    }

    const uint16_t colors[] = {
        0xf800,
        0x07e0,
        0x001f,
        0xffff,
    };

    ESP_LOGI(TAG, "drawing boot color bars");
    for (int y = 0; y < EXAMPLE_LCD_V_RES; y += BOOT_PATTERN_LINES) {
        int h = BOOT_PATTERN_LINES;
        if (y + h > EXAMPLE_LCD_V_RES) {
            h = EXAMPLE_LCD_V_RES - y;
        }

        for (int row = 0; row < h; ++row) {
            uint16_t color = colors[((y + row) * 4) / EXAMPLE_LCD_V_RES];
            for (int x = 0; x < EXAMPLE_LCD_H_RES; ++x) {
                band[row * EXAMPLE_LCD_H_RES + x] = color;
            }
        }

        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel_handle, 0, y, EXAMPLE_LCD_H_RES, y + h, band));
    }

    vTaskDelay(pdMS_TO_TICKS(BOOT_PATTERN_MS));
    heap_caps_free(band);
}

static void io_expander_init(i2c_master_bus_handle_t bus_handle)
{
    ESP_ERROR_CHECK(esp_io_expander_new_i2c_tca9554(
        bus_handle, ESP_IO_EXPANDER_I2C_TCA9554_ADDRESS_000, &expander_handle));
    ESP_ERROR_CHECK(esp_io_expander_set_dir(expander_handle, IO_EXPANDER_PIN_NUM_1, IO_EXPANDER_OUTPUT));
    ESP_ERROR_CHECK(esp_io_expander_set_level(expander_handle, IO_EXPANDER_PIN_NUM_1, 0));
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_ERROR_CHECK(esp_io_expander_set_level(expander_handle, IO_EXPANDER_PIN_NUM_1, 1));
    vTaskDelay(pdMS_TO_TICKS(200));
}

static void touchpad_read(lv_indev_drv_t *indev_drv, lv_indev_data_t *data)
{
    static lv_coord_t last_x = 0;
    static lv_coord_t last_y = 0;
    touch_data_t touch_data;

    bsp_touch_read();
    if (bsp_touch_get_coordinates(&touch_data)) {
        last_x = touch_data.coords[0].x;
        last_y = touch_data.coords[0].y;
        data->state = LV_INDEV_STATE_PR;
    } else {
        data->state = LV_INDEV_STATE_REL;
    }

    data->point.x = last_x;
    data->point.y = last_y;
}

void lv_port_init(void)
{
    lvgl_port_cfg_t port_cfg = {};
    port_cfg.task_priority = 4;
    port_cfg.task_stack = 1024 * 5;
    port_cfg.task_affinity = 1;
    port_cfg.task_max_sleep_ms = 500;
    port_cfg.timer_period_ms = 5;
    lvgl_port_init(&port_cfg);

    lvgl_port_display_cfg_t disp_cfg = {};
    disp_cfg.io_handle = io_handle;
    disp_cfg.panel_handle = panel_handle;
    disp_cfg.buffer_size = LCD_BUFFER_PIXELS;
    disp_cfg.sw_rotate = EXAMPLE_DISPLAY_ROTATION;
    disp_cfg.hres = EXAMPLE_LCD_H_RES;
    disp_cfg.vres = EXAMPLE_LCD_V_RES;
    disp_cfg.trans_size = LCD_BUFFER_PIXELS / 10;
    disp_cfg.draw_wait_cb = NULL;
    disp_cfg.flags.buff_dma = false;
    disp_cfg.flags.buff_spiram = true;

    if (disp_cfg.sw_rotate == LV_DISP_ROT_180 || disp_cfg.sw_rotate == LV_DISP_ROT_NONE) {
        disp_cfg.hres = EXAMPLE_LCD_H_RES;
        disp_cfg.vres = EXAMPLE_LCD_V_RES;
    } else {
        disp_cfg.hres = EXAMPLE_LCD_V_RES;
        disp_cfg.vres = EXAMPLE_LCD_H_RES;
    }
    lvgl_disp = lvgl_port_add_disp(&disp_cfg);

    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.read_cb = touchpad_read;
    lvgl_touch_indev = lv_indev_drv_register(&indev_drv);
}

static void set_label_style(lv_obj_t *label, const lv_font_t *font, lv_color_t color)
{
    lv_obj_set_style_text_font(label, font, 0);
    lv_obj_set_style_text_color(label, color, 0);
    lv_obj_set_style_text_letter_space(label, 0, 0);
}

static lv_obj_t *make_bar(lv_obj_t *parent, int x)
{
    lv_obj_t *bar = lv_obj_create(parent);
    lv_obj_remove_style_all(bar);
    lv_obj_set_size(bar, 14, 54);
    lv_obj_set_pos(bar, x, 350);
    lv_obj_set_style_radius(bar, 7, 0);
    lv_obj_set_style_bg_opa(bar, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_color(bar, color_primary, 0);
    return bar;
}

static void build_ui(void)
{
    root = lv_scr_act();
    lv_obj_clear_flag(root, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_opa(root, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_color(root, color_bg, 0);

    title_label = lv_label_create(root);
    lv_label_set_text(title_label, "Chat2M");
    set_label_style(title_label, &lv_font_montserrat_20, color_text);
    lv_obj_align(title_label, LV_ALIGN_TOP_MID, 0, 28);

    subtitle_label = lv_label_create(root);
    lv_label_set_text(subtitle_label, "IDLE");
    set_label_style(subtitle_label, &lv_font_montserrat_16, color_muted);
    lv_obj_align(subtitle_label, LV_ALIGN_TOP_MID, 0, 70);

    ring = lv_obj_create(root);
    lv_obj_remove_style_all(ring);
    lv_obj_set_size(ring, 210, 210);
    lv_obj_align(ring, LV_ALIGN_CENTER, 0, -18);
    lv_obj_set_style_radius(ring, 105, 0);
    lv_obj_set_style_border_width(ring, 4, 0);
    lv_obj_set_style_border_color(ring, color_primary, 0);
    lv_obj_set_style_bg_opa(ring, LV_OPA_10, 0);
    lv_obj_set_style_bg_color(ring, color_panel, 0);

    orb = lv_obj_create(root);
    lv_obj_remove_style_all(orb);
    lv_obj_set_size(orb, 86, 86);
    lv_obj_align(orb, LV_ALIGN_CENTER, 0, -18);
    lv_obj_set_style_radius(orb, 43, 0);
    lv_obj_set_style_bg_opa(orb, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_color(orb, color_primary, 0);

    bar_left = make_bar(root, 122);
    bar_mid = make_bar(root, 153);
    bar_right = make_bar(root, 184);

    status_label = lv_label_create(root);
    lv_label_set_long_mode(status_label, LV_LABEL_LONG_DOT);
    lv_obj_set_width(status_label, 270);
    lv_label_set_text(status_label, "");
    set_label_style(status_label, &lv_font_montserrat_16, color_muted);
    lv_obj_align(status_label, LV_ALIGN_BOTTOM_MID, 0, -34);
}

static const char *state_label(const char *state)
{
    if (strcmp(state, "listening") == 0) {
        return "LISTENING";
    }
    if (strcmp(state, "thinking") == 0) {
        return "THINKING";
    }
    if (strcmp(state, "speaking") == 0) {
        return "SPEAKING";
    }
    if (strcmp(state, "error") == 0) {
        return "ERROR";
    }
    return "IDLE";
}

static lv_color_t state_color(const char *state)
{
    if (strcmp(state, "thinking") == 0) {
        return color_accent;
    }
    if (strcmp(state, "error") == 0) {
        return color_error;
    }
    return color_primary;
}

static void apply_state_ui(void)
{
    lv_color_t c = state_color(current_state);
    lv_label_set_text(subtitle_label, state_label(current_state));
    lv_label_set_text(status_label, current_text);
    lv_obj_set_style_text_color(subtitle_label, c, 0);
    lv_obj_set_style_border_color(ring, c, 0);
    lv_obj_set_style_bg_color(orb, c, 0);
    lv_obj_set_style_bg_color(bar_left, c, 0);
    lv_obj_set_style_bg_color(bar_mid, c, 0);
    lv_obj_set_style_bg_color(bar_right, c, 0);

    bool speaking = strcmp(current_state, "speaking") == 0;
    if (speaking) {
        lv_obj_clear_flag(bar_left, LV_OBJ_FLAG_HIDDEN);
        lv_obj_clear_flag(bar_mid, LV_OBJ_FLAG_HIDDEN);
        lv_obj_clear_flag(bar_right, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(bar_left, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(bar_mid, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(bar_right, LV_OBJ_FLAG_HIDDEN);
    }
}

static void animate_cb(lv_timer_t *timer)
{
    uint32_t now = lv_tick_get();
    int phase = (now / 90) % 24;
    int ring_size = 188 + ((phase <= 12) ? phase : 24 - phase) * 3;
    int orb_size = 74 + ((phase <= 12) ? phase : 24 - phase);

    if (strcmp(current_state, "idle") == 0) {
        ring_size = 188;
        orb_size = 74;
    } else if (strcmp(current_state, "thinking") == 0) {
        ring_size = 178 + (phase * 2);
        orb_size = 64 + (phase % 6) * 3;
    } else if (strcmp(current_state, "error") == 0) {
        ring_size = (phase % 2) ? 202 : 188;
        orb_size = (phase % 2) ? 86 : 70;
    }

    lv_obj_set_size(ring, ring_size, ring_size);
    lv_obj_set_style_radius(ring, ring_size / 2, 0);
    lv_obj_align(ring, LV_ALIGN_CENTER, 0, -18);
    lv_obj_set_size(orb, orb_size, orb_size);
    lv_obj_set_style_radius(orb, orb_size / 2, 0);
    lv_obj_align(orb, LV_ALIGN_CENTER, 0, -18);

    if (strcmp(current_state, "speaking") == 0) {
        int h1 = 26 + ((phase * 5) % 60);
        int h2 = 36 + ((phase * 7) % 74);
        int h3 = 24 + ((phase * 11) % 58);
        lv_obj_set_height(bar_left, h1);
        lv_obj_set_height(bar_mid, h2);
        lv_obj_set_height(bar_right, h3);
        lv_obj_set_y(bar_left, 374 - h1 / 2);
        lv_obj_set_y(bar_mid, 374 - h2 / 2);
        lv_obj_set_y(bar_right, 374 - h3 / 2);
    }
}

static bool extract_json_value(const char *line, const char *key, char *out, size_t out_size)
{
    char needle[32];
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    const char *start = strstr(line, needle);
    if (!start) {
        return false;
    }
    start += strlen(needle);
    const char *end = strchr(start, '"');
    if (!end) {
        return false;
    }
    size_t len = end - start;
    if (len >= out_size) {
        len = out_size - 1;
    }
    memcpy(out, start, len);
    out[len] = '\0';
    return true;
}

static void handle_line(const char *line)
{
    char state[24] = "";
    char text[96] = "";
    if (!extract_json_value(line, "state", state, sizeof(state))) {
        return;
    }
    extract_json_value(line, "text", text, sizeof(text));

    strncpy(current_state, state, sizeof(current_state) - 1);
    current_state[sizeof(current_state) - 1] = '\0';
    strncpy(current_text, text, sizeof(current_text) - 1);
    current_text[sizeof(current_text) - 1] = '\0';

    if (lvgl_port_lock(pdMS_TO_TICKS(100))) {
        apply_state_ui();
        lvgl_port_unlock();
    }
}

static void uart_task(void *arg)
{
    while (true) {
        char line[DISPLAY_LINE_SIZE] = {};
        if (fgets(line, sizeof(line), stdin) != NULL) {
            handle_line(line);
        } else {
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}

extern "C" void app_main(void)
{
    i2c_master_bus_handle_t i2c_bus_handle = bsp_i2c_init();

    ESP_ERROR_CHECK(bsp_axp2101_init(i2c_bus_handle));
    io_expander_init(i2c_bus_handle);
    bsp_display_init(&io_handle, &panel_handle, LCD_BUFFER_BYTES);
    bsp_touch_init(i2c_bus_handle, EXAMPLE_LCD_H_RES, EXAMPLE_LCD_V_RES, 0);
    bsp_display_brightness_init();
    bsp_display_set_brightness(100);
    draw_boot_pattern();

    lv_port_init();
    ESP_LOGI(TAG, "serial status input ready on console stdin");

    if (lvgl_port_lock(0)) {
        build_ui();
        apply_state_ui();
        lv_timer_create(animate_cb, 80, NULL);
        lv_obj_invalidate(root);
        lvgl_port_unlock();
        ESP_LOGI(TAG, "ui ready");
    }

    xTaskCreate(uart_task, "uart_status", 4096, NULL, 8, NULL);
}
