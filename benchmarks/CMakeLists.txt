function(xushi2_add_benchmark name src)
    add_executable(${name} ${src})
    target_include_directories(${name}
        PRIVATE ${CMAKE_SOURCE_DIR}/tests/common)
    target_link_libraries(${name}
        PRIVATE
            benchmark::benchmark
            benchmark::benchmark_main
            xushi2::sim
            xushi2::bots
            xushi2::common
    )
endfunction()

xushi2_add_benchmark(bench_sim_tick      bench_sim_tick.cpp)
xushi2_add_benchmark(bench_obs_build     bench_obs_build.cpp)
xushi2_add_benchmark(bench_bot_decision  bench_bot_decision.cpp)
