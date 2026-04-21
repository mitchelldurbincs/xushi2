#include <xushi2/bots/runner.h>

#include <array>
#include <memory>

#include <xushi2/bots/bot.h>
#include <xushi2/common/assert.hpp>

namespace xushi2::bots {

namespace {

std::unique_ptr<IBot> make_bot_by_name(std::string_view name) {
    if (name == "walk_to_objective") {
        return make_walk_to_objective_bot();
    }
    if (name == "hold_and_shoot") {
        return make_hold_and_shoot_bot();
    }
    if (name == "basic") {
        return make_basic_bot();
    }
    if (name == "noop") {
        return make_noop_bot();
    }
    X2_REQUIRE(false, common::ErrorCode::InvalidAction);
    return nullptr;
}

}  // namespace

ScriptedEpisodeResult run_scripted_episode(const sim::MatchConfig& config,
                                           std::string_view bot_a_name,
                                           std::string_view bot_b_name) {
    sim::Sim sim(config);

    auto bot_a = make_bot_by_name(bot_a_name);
    auto bot_b = make_bot_by_name(bot_b_name);

    // Decisions per episode are driven by episode_over(). Upper bound: round
    // length in ticks divided by action_repeat. Reserve to avoid reallocs.
    const std::uint32_t max_ticks =
        static_cast<std::uint32_t>(config.round_length_seconds) *
        static_cast<std::uint32_t>(sim::kTickHz);
    const std::uint32_t max_decisions = max_ticks / config.action_repeat;

    ScriptedEpisodeResult result;
    result.decision_hashes.reserve(max_decisions);

    while (!sim.episode_over()) {
        std::array<common::Action, sim::kAgentsPerMatch> actions{};
        const auto& state = sim.state();
        for (std::uint32_t i = 0; i < sim::kAgentsPerMatch; ++i) {
            const auto& h = state.heroes[i];
            if (!h.present) {
                continue;  // absent slot: leave Action{} (zeros)
            }
            IBot& bot = (h.team == common::Team::A) ? *bot_a : *bot_b;
            actions[i] = bot.decide(state, static_cast<int>(i));
        }
        sim.step_decision(actions);
        result.decision_hashes.push_back(sim.state_hash());
    }
    result.final_tick = sim.state().tick;
    result.team_a_kills = sim.team_a_kills();
    result.team_b_kills = sim.team_b_kills();
    result.winner = sim.winner();
    return result;
}

}  // namespace xushi2::bots
