add_library(xushi2_common INTERFACE)
add_library(xushi2::common ALIAS xushi2_common)

target_include_directories(xushi2_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(xushi2_common INTERFACE cxx_std_20)
