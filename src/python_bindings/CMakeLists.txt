pybind11_add_module(xushi2_cpp MODULE
    module.cpp
)

target_link_libraries(xushi2_cpp
    PRIVATE
        xushi2::sim
        xushi2::bots
)

# Output next to the Python package so `pip install -e ../python` finds it.
# This can be overridden from the CMake command line.
set(XUSHI2_PY_EXT_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/python/xushi2
    CACHE PATH "Output directory for the xushi2_cpp extension module")

set_target_properties(xushi2_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${XUSHI2_PY_EXT_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${XUSHI2_PY_EXT_OUTPUT_DIR}
)
