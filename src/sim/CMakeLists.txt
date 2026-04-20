add_library(xushi2_sim STATIC
    src/sim.cpp
)
add_library(xushi2::sim ALIAS xushi2_sim)

target_include_directories(xushi2_sim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(xushi2_sim PUBLIC xushi2::common)

# Determinism: sim is float-precise and has no wall-clock / no entropy source
# other than the seeded PRNG stored in Sim state.
