include(GoogleTest)

function(xushi2_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name}
        PRIVATE
            GTest::gtest_main
            xushi2::sim
            xushi2::bots
            xushi2::common
    )
    gtest_discover_tests(${name})
endfunction()

xushi2_add_test(test_sim_smoke       sim/test_smoke.cpp)
xushi2_add_test(test_determinism     sim/test_determinism.cpp)
xushi2_add_test(test_golden_replay   replay/test_golden_replay.cpp)
xushi2_add_test(test_actor_leak      observations/test_actor_leak.cpp)
xushi2_add_test(test_integration     integration/test_smoke.cpp)
