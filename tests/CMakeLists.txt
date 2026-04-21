include(GoogleTest)

function(xushi2_add_test name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_link_libraries(${name}
        PRIVATE
            GTest::gtest_main
            xushi2::sim
            xushi2::bots
            xushi2::common
    )
    gtest_discover_tests(${name})
endfunction()

xushi2_add_test(test_sim_smoke            sim/test_smoke.cpp)
xushi2_add_test(test_determinism          sim/test_determinism.cpp)
xushi2_add_test(test_movement             sim/test_movement.cpp)
xushi2_add_test(test_combat               sim/test_combat.cpp)
xushi2_add_test(test_reload               sim/test_reload.cpp)
xushi2_add_test(test_objective            sim/test_objective.cpp)
xushi2_add_test(test_respawn              sim/test_respawn.cpp)
xushi2_add_test(test_config_validation    sim/test_config_validation.cpp)
xushi2_add_test(test_bots_runner          bots/test_runner.cpp)
xushi2_add_test(test_golden_replay        replay/test_golden_replay.cpp)
xushi2_add_test(test_actor_leak           observations/test_actor_leak.cpp)
xushi2_add_test(test_obs_dims             observations/test_obs_dims.cpp)
xushi2_add_test(test_obs_utils            observations/test_obs_utils.cpp)
xushi2_add_test(test_actor_obs            observations/test_actor_obs.cpp)
xushi2_add_test(test_critic_obs           observations/test_critic_obs.cpp)
xushi2_add_test(test_integration          integration/test_smoke.cpp)

# The golden-replay test reads data/replays/golden_phase0_basic.txt from the
# source tree. Pass the repo root in as a macro.
target_compile_definitions(test_golden_replay
    PRIVATE XUSHI2_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
