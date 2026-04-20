add_subdirectory(common)
add_subdirectory(sim)
add_subdirectory(bots)

if(XUSHI2_BUILD_VIEWER)
    add_subdirectory(viewer)
endif()

if(XUSHI2_BUILD_PYTHON_MODULE)
    add_subdirectory(python_bindings)
endif()

# tools/ is intentionally empty for Phase 0. When offline tools land
# (replay inspector, obs dumper, golden-replay recorder) they go here.
# add_subdirectory(tools)
