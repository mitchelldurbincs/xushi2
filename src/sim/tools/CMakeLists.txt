add_executable(dump_golden dump_golden.cpp)
target_link_libraries(dump_golden PRIVATE xushi2::sim xushi2::bots)

if(XUSHI2_BUILD_BENCHMARKS)
    add_executable(bench_sim bench_sim.cpp)
    target_link_libraries(bench_sim PRIVATE xushi2::sim xushi2::common)
endif()
