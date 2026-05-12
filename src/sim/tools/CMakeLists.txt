add_executable(dump_golden dump_golden.cpp)
target_link_libraries(dump_golden PRIVATE xushi2::sim xushi2::bots)

add_executable(benchmark_sim benchmark_sim.cpp)
target_link_libraries(benchmark_sim PRIVATE xushi2::sim)
