#include "bsp_touch.h"

#include "bsp_i2c.h"
#include "freertos/FreeRTOS.h"

static uint16_t g_rotation = 0;
static uint16_t g_width = 0;
static uint16_t g_height = 0;

static i2c_master_dev_handle_t dev_handle;
static touch_data_t g_touch_data;

void bsp_touch_read(void)
{
    uint8_t data[14] = {0};
    const uint8_t read_cmd[11] = {0xb5, 0xab, 0xa5, 0x5a, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00};

    g_touch_data.touch_num = 0;

    if (!bsp_i2c_lock(0)) {
        return;
    }

    esp_err_t err = i2c_master_transmit_receive(
        dev_handle, read_cmd, sizeof(read_cmd), data, sizeof(data), pdMS_TO_TICKS(1000));
    bsp_i2c_unlock();
    if (err != ESP_OK) {
        return;
    }

    if (data[0] == 0xff || data[1] == 0 || data[1] > MAX_TOUCH_POINTS ||
        data[2] == 0 || data[3] < 2 || data[5] < 2) {
        return;
    }

    g_touch_data.touch_num = data[1];
    for (int i = 0; i < g_touch_data.touch_num; i++) {
        g_touch_data.coords[i].x = ((data[6 * i + 2] & 0x0f) << 8) | data[6 * i + 3];
        g_touch_data.coords[i].y = ((data[6 * i + 4] & 0x0f) << 8) | data[6 * i + 5];
    }
}

bool bsp_touch_get_coordinates(touch_data_t *touch_data)
{
    if ((touch_data == NULL) || (g_touch_data.touch_num == 0)) {
        return false;
    }

    touch_data->touch_num = g_touch_data.touch_num;
    for (int i = 0; i < g_touch_data.touch_num; i++) {
        switch (g_rotation) {
        case 1:
            touch_data->coords[i].y = g_height - 1 - g_touch_data.coords[i].x;
            touch_data->coords[i].x = g_touch_data.coords[i].y;
            break;
        case 2:
            touch_data->coords[i].x = g_width - 1 - g_touch_data.coords[i].x;
            touch_data->coords[i].y = g_height - 1 - g_touch_data.coords[i].y;
            break;
        case 3:
            touch_data->coords[i].y = g_touch_data.coords[i].x;
            touch_data->coords[i].x = g_width - 1 - g_touch_data.coords[i].y;
            break;
        default:
            touch_data->coords[i].x = g_touch_data.coords[i].x;
            touch_data->coords[i].y = g_touch_data.coords[i].y;
            break;
        }
    }
    return true;
}

void bsp_touch_init(i2c_master_bus_handle_t bus_handle, uint16_t width, uint16_t height, uint16_t rotation)
{
    g_rotation = rotation;
    g_width = width;
    g_height = height;
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = I2C_AXS15231B_ADDRESS,
        .scl_speed_hz = 400000,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &dev_cfg, &dev_handle));
}
