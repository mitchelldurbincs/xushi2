add_library(xushi2_bots STATIC
    src/bot.cpp
    src/runner.cpp
)
add_library(xushi2::bots ALIAS xushi2_bots)

target_include_directories(xushi2_bots PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(xushi2_bots
    PUBLIC
        xushi2::common
        xushi2::sim
)
