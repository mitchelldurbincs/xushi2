#include <xushi2/bots/bot.h>

#include "internal/behavior_primitives.h"

#include <cstdint>
#include <string>

// Phase-0 scripted bots. Deterministic (no RNG, no wall clock) and
// observation-blind (they read the full MatchState — Tier 1 is allowed to).
// "Objective" is the arena center; Phase 0 has no real capture mechanics.

namespace xushi2::bots {

namespace {

// Integer hash (splitmix64). Cross-platform deterministic — unlike a
// std::sin-based GLSL hash, the result is bit-identical on any compiler
// because it uses only integer arithmetic.
constexpr std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Returns a deterministic float in [-1, 1] from (tick, slot). Per-slot
// per-decision variation; no positional aliasing.
float deterministic_unit_noise(std::uint32_t tick, int agent_index) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(tick) << 32) ^
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(agent_index));
    const std::uint64_t h = splitmix64(key);
    // Use top 24 bits → exact float mantissa, no rounding double-round.
    const float u = static_cast<float>(h >> 40) * (1.0F / 16777216.0F);  // [0,1)
    return u * 2.0F - 1.0F;
}

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
    common::Action decide(const sim::MatchState& state,
                          const sim::MatchConfig& config,
                          int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        return internal::walk_to_objective(*self, config.map);
    }
    std::string name() const override { return "walk_to_objective"; }
};

class HoldAndShootBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state,
                          const sim::MatchConfig& /*config*/,
                          int agent_index) override {
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
    common::Action decide(const sim::MatchState& state,
                          const sim::MatchConfig& config,
                          int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        common::Action walk = internal::walk_to_objective(*self, config.map);
        common::Action shoot = internal::hold_and_shoot(state, *self);
        // Combine: walk's movement, shoot's aim + fire.
        walk.aim_delta = shoot.aim_delta;
        walk.primary_fire = shoot.primary_fire;
        return walk;
    }
    std::string name() const override { return "basic"; }
};

class WeakBasicBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state,
                          const sim::MatchConfig& config,
                          int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        // Deterministic ±0.5 rad noise applied pre-clamp inside
        // aim_delta_toward so the final aim_delta stays within
        // kAimDeltaMax — noise reduces accuracy without saturating
        // the action.
        constexpr float kAimNoiseScale = 0.5F;  // ±0.5 rad (~±28.6°)
        const float noise =
            deterministic_unit_noise(state.tick, agent_index) * kAimNoiseScale;
        common::Action walk = internal::walk_to_objective(*self, config.map);
        common::Action shoot = internal::hold_and_shoot(state, *self, noise);
        walk.aim_delta = shoot.aim_delta;
        walk.primary_fire = shoot.primary_fire;
        return walk;
    }
    std::string name() const override { return "weak_basic"; }
};

class WeakBasicV2Bot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state,
                          const sim::MatchConfig& config,
                          int agent_index) override {
        const sim::HeroState* self = get_active_hero_or_null(state, agent_index);
        if (self == nullptr) {
            return common::Action{};
        }
        constexpr float kAimNoiseScale = 1.5F;  // ±1.5 rad (~±86°)
        constexpr std::uint32_t kFireCadenceTicks = 60;
        const float noise =
            deterministic_unit_noise(state.tick, agent_index) * kAimNoiseScale;
        common::Action walk = internal::walk_to_objective(*self, config.map);
        common::Action shoot = internal::hold_and_shoot(state, *self, noise);
        walk.aim_delta = shoot.aim_delta;
        walk.primary_fire = shoot.primary_fire &&
                            (state.tick % kFireCadenceTicks == 0U);
        return walk;
    }
    std::string name() const override { return "weak_basic_v2"; }
};

class NoopBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& /*state*/,
                          const sim::MatchConfig& /*config*/,
                          int /*agent_index*/) override {
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

std::unique_ptr<IBot> make_weak_basic_bot() {
    return std::make_unique<WeakBasicBot>();
}

std::unique_ptr<IBot> make_weak_basic_v2_bot() {
    return std::make_unique<WeakBasicV2Bot>();
}

std::unique_ptr<IBot> make_noop_bot() {
    return std::make_unique<NoopBot>();
}

}  // namespace xushi2::bots
