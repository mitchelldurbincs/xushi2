add_executable(xushi2_viewer
    src/main.cpp
)

target_link_libraries(xushi2_viewer
    PRIVATE
        xushi2::sim
        xushi2::bots
        raylib
)

# raylib on Windows needs winmm. On Linux, -lpthread -ldl -lrt -lX11. raylib
# adds those automatically when built via FetchContent.

set_target_properties(xushi2_viewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
