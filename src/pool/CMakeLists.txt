add_library(xushi2_pool STATIC
    src/sim_pool.cpp
)
add_library(xushi2::pool ALIAS xushi2_pool)

target_include_directories(xushi2_pool PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(xushi2_pool PUBLIC xushi2::sim xushi2::bots xushi2::common)
