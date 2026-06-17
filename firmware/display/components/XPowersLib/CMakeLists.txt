set(src_dirs ./src)
set(include_dirs ./src
                ./src/REG)
idf_component_register( SRC_DIRS ${src_dirs}
                        INCLUDE_DIRS ${include_dirs}
                        REQUIRES esp_driver_i2c)
