#include <xushi2/bots/bot.h>

#include "internal/behavior_primitives.h"

#include <string>

// Phase-0 scripted bots. Deterministic (no RNG, no wall clock) and
// observation-blind (they read the full MatchState — Tier 1 is allowed to).
// "Objective" is the arena center; Phase 0 has no real capture mechanics.

namespace xushi2::bots {

namespace {

// All scripted bots share this gate so invalid/dead slots deterministically
// produce a no-op action before running any movement/combat policy.
const sim::HeroState* get_active_hero_or_null(const sim::MatchState& state,
                                              int agent_index) {
    if (agent_index < 0) {
        return nullptr;
    }
    const std::size_t idx = static_cast<std::size_t>(agent_index);
    if (idx >= state.heroes.size()) {
        return nullptr;
    }
    const sim::HeroState& hero = state.heroes[idx];
    if (!hero.present || !hero.alive) {
        return nullptr;
    }
    return &hero;
}


class WalkToObjectiveBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state, int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        return internal::walk_to_objective(*self);
    }
    std::string name() const override { return "walk_to_objective"; }
};

class HoldAndShootBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state, int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        return internal::hold_and_shoot(state, *self);
    }
    std::string name() const override { return "hold_and_shoot"; }
};

class BasicBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state, int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        common::Action walk = internal::walk_to_objective(*self);
        common::Action shoot = internal::hold_and_shoot(state, *self);
        // Combine: walk's movement, shoot's aim + fire.
        walk.aim_delta = shoot.aim_delta;
        walk.primary_fire = shoot.primary_fire;
        return walk;
    }
    std::string name() const override { return "basic"; }
};

class NoopBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& /*state*/, int /*agent_index*/) override {
        return common::Action{};
    }
    std::string name() const override { return "noop"; }
};

}  // namespace

std::unique_ptr<IBot> make_walk_to_objective_bot() {
    return std::make_unique<WalkToObjectiveBot>();
}

std::unique_ptr<IBot> make_hold_and_shoot_bot() {
    return std::make_unique<HoldAndShootBot>();
}

std::unique_ptr<IBot> make_basic_bot() {
    return std::make_unique<BasicBot>();
}

std::unique_ptr<IBot> make_noop_bot() {
    return std::make_unique<NoopBot>();
}

}  // namespace xushi2::bots
