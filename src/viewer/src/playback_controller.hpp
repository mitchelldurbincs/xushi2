#pragma once

#include <array>
#include <memory>
#include <optional>

#include <xushi2/bots/bot.h>
#include <xushi2/sim/sim.h>

#include "replay_loader.hpp"
#include "render_debug.hpp"

struct PlaybackState {
    std::size_t replay_idx = 0;
    float decision_accum = 0.0F;
    bool paused = false;
    float playback_speed = 1.0F;
};

struct PlaybackContext {
    xushi2::sim::Sim* sim = nullptr;
    const std::optional<Replay>* replay = nullptr;
    std::unique_ptr<xushi2::bots::IBot>* bot_a = nullptr;
    std::unique_ptr<xushi2::bots::IBot>* bot_b = nullptr;
    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>* actions = nullptr;
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>* shots = nullptr;
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>* tethers = nullptr;
    std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>* prev_heroes = nullptr;
};

void reset_playback(PlaybackContext& ctx, PlaybackState& state);
void step_once(PlaybackContext& ctx, PlaybackState& state);
void handle_input(PlaybackContext& ctx, PlaybackState& state);
void advance_playback(PlaybackContext& ctx, PlaybackState& state, float delta_seconds, float decision_seconds);
