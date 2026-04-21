#include <xushi2/bots/bot.h>

#include <cmath>
#include <string>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/math.hpp>

// Phase-0 scripted bots. Deterministic (no RNG, no wall clock) and
// observation-blind (they read the full MatchState — Tier 1 is allowed to).
// "Objective" is the arena center; Phase 0 has no real capture mechanics.

namespace xushi2::bots {

namespace {

constexpr float kObjectiveX = 25.0F;
constexpr float kObjectiveY = 25.0F;
constexpr float kArriveRadius = 0.25F;

// Deterministic scan for the first living, present hero on the opposite team.
// Returns nullptr if none exists.
const sim::HeroState* find_opponent(const sim::MatchState& state,
                                    const sim::HeroState& self) {
    for (const auto& h : state.heroes) {
        if (!h.present || !h.alive) {
            continue;
        }
        if (h.team == self.team) {
            continue;
        }
        return &h;
    }
    return nullptr;
}

// Aim-delta needed to rotate `self.aim_angle` toward the target point,
// clamped to the per-decision ±π/4 cap.
float aim_delta_toward(const sim::HeroState& self, float tx, float ty) {
    const float dx = tx - self.position.x;
    const float dy = ty - self.position.y;
    const float desired = std::atan2(dy, dx);
    const float raw = common::wrap_angle(desired - self.aim_angle);
    return common::clampf(raw, -common::kAimDeltaMax, common::kAimDeltaMax);
}

common::Action walk_to_objective(const sim::HeroState& self) {
    common::Action a{};
    const float dx = kObjectiveX - self.position.x;
    const float dy = kObjectiveY - self.position.y;
    const float dist2 = dx * dx + dy * dy;
    if (dist2 > kArriveRadius * kArriveRadius) {
        // Raw direction; sim's normalize_move_input caps |v| <= 1.
        const float inv = 1.0F / std::sqrt(dist2);
        a.move_x = dx * inv;
        a.move_y = dy * inv;
    }
    return a;
}

common::Action hold_and_shoot(const sim::MatchState& state,
                              const sim::HeroState& self) {
    common::Action a{};
    const sim::HeroState* opp = find_opponent(state, self);
    if (opp == nullptr) {
        return a;
    }
    a.aim_delta = aim_delta_toward(self, opp->position.x, opp->position.y);
    a.primary_fire = true;
    return a;
}

class WalkToObjectiveBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state, int agent_index) override {
        const sim::HeroState& self = state.heroes[static_cast<std::size_t>(agent_index)];
        if (!self.present || !self.alive) {
            return common::Action{};
        }
        return walk_to_objective(self);
    }
    std::string name() const override { return "walk_to_objective"; }
};

class HoldAndShootBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state, int agent_index) override {
        const sim::HeroState& self = state.heroes[static_cast<std::size_t>(agent_index)];
        if (!self.present || !self.alive) {
            return common::Action{};
        }
        return hold_and_shoot(state, self);
    }
    std::string name() const override { return "hold_and_shoot"; }
};

class BasicBot final : public IBot {
   public:
    common::Action decide(const sim::MatchState& state, int agent_index) override {
        const sim::HeroState& self = state.heroes[static_cast<std::size_t>(agent_index)];
        if (!self.present || !self.alive) {
            return common::Action{};
        }
        common::Action walk = walk_to_objective(self);
        common::Action shoot = hold_and_shoot(state, self);
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
