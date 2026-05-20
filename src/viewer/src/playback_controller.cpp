#include "playback_controller.hpp"

#include <raylib.h>

namespace {

void apply_playback_speed_hotkeys(PlaybackState& state) {
    if (IsKeyPressed(KEY_ONE)) state.playback_speed = 0.5F;
    if (IsKeyPressed(KEY_TWO)) state.playback_speed = 1.0F;
    if (IsKeyPressed(KEY_THREE)) state.playback_speed = 2.0F;
}

}  // namespace

void reset_playback(PlaybackContext& ctx, PlaybackState& state) {
    ctx.sim->reset();
    state.replay_idx = 0;
    *ctx.actions = {};
    *ctx.shots = {};
    *ctx.tethers = {};
    *ctx.prev_heroes = ctx.sim->state().heroes;
    state.decision_accum = 0.0F;
}

void step_once(PlaybackContext& ctx, PlaybackState& state) {
    *ctx.actions = {};
    if (ctx.replay->has_value() && state.replay_idx < (*ctx.replay)->decisions.size()) {
        *ctx.actions = (*ctx.replay)->decisions[state.replay_idx].actions;
        ++state.replay_idx;
    } else if (!ctx.replay->has_value()) {
        (*ctx.actions)[0] = (*ctx.bot_a)->decide(ctx.sim->state(), ctx.sim->config(), 0);
        (*ctx.actions)[3] = (*ctx.bot_b)->decide(ctx.sim->state(), ctx.sim->config(), 3);
    }

    ctx.sim->step_decision(*ctx.actions);
    update_shot_tracers(*ctx.shots, *ctx.prev_heroes, ctx.sim->state().heroes, ctx.sim->state().tick);
    update_tether_trails(*ctx.tethers, *ctx.prev_heroes, ctx.sim->state().heroes, ctx.sim->state().tick);
    *ctx.prev_heroes = ctx.sim->state().heroes;
}

void handle_input(PlaybackContext& ctx, PlaybackState& state) {
    if (IsKeyPressed(KEY_SPACE)) state.paused = !state.paused;
    if (IsKeyPressed(KEY_R)) reset_playback(ctx, state);
    apply_playback_speed_hotkeys(state);
    if (IsKeyPressed(KEY_RIGHT) && !ctx.sim->episode_over()) {
        step_once(ctx, state);
        state.decision_accum = 0.0F;
    }
}

void advance_playback(PlaybackContext& ctx,
                      PlaybackState& state,
                      float delta_seconds,
                      float decision_seconds) {
    if (state.paused) return;

    state.decision_accum += delta_seconds * state.playback_speed;
    while (state.decision_accum >= decision_seconds && !ctx.sim->episode_over()) {
        step_once(ctx, state);
        state.decision_accum -= decision_seconds;
    }
}
