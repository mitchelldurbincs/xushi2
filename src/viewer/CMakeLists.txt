add_executable(xushi2_viewer
    src/main.cpp
    src/replay_loader.cpp
    src/render_arena.cpp
    src/render_debug.cpp
    src/panel.cpp
    src/viewer_labels.cpp
    src/viewer_bench_output.cpp
    src/playback_controller.cpp
    src/benchmark_writer.cpp
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

add_executable(xushi2_viewer_bench
    src/bench_main.cpp
    src/replay_loader.cpp
    src/render_arena.cpp
    src/render_debug.cpp
    src/panel.cpp
    src/viewer_labels.cpp
    src/viewer_bench_output.cpp
    src/playback_controller.cpp
    src/benchmark_writer.cpp
)

target_link_libraries(xushi2_viewer_bench
    PRIVATE
        xushi2::sim
        xushi2::bots
        raylib
)

set_target_properties(xushi2_viewer_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
