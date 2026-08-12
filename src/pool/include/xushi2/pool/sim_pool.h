#pragma once

// SimPool — the batched sim boundary. Owns N independent 3v3 sims plus, per
// env, an ObservationEngine (native entity obs / fog / last-seen) and
// persistent scripted-bot instances. One step() call advances every env and
// writes entity obs, critic obs, and the reward feature block into
// caller-owned buffers, collapsing the ~110–160 per-env FFI crossings of the
// legacy per-env path into a single crossing per vector step (the binding
// releases the GIL for the duration).
//
// Design notes:
//  - No auto-reset. step() reports terminated/truncated; the caller resets
//    finished envs with reset_env() (a per-episode, not per-step, cost).
//    This keeps per-episode concerns that live in Python — map
//    randomization, seed bookkeeping, reward-calculator resets — in Python.
//  - Envs are fully independent (disjoint state, no shared RNG), so the
//    per-env step body can later run on a thread pool without changing
//    results. Phase 4 is deliberately serial.
//  - Scripted slots are driven inside the pool by persistent bots (fixing
//    the per-call heap allocation of the scripted_bot_action binding) with
//    the same deterministic handicap transform the legacy env applied.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <xushi2/bots/bot.h>
#include <xushi2/sim/entity_obs.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_config.h>
#include <xushi2/sim/reward_features.h>
#include <xushi2/sim/sim.h>

namespace xushi2::pool {

enum class SlotMode : std::uint8_t {
    Policy = 0,    // action taken from the step() input tensor
    Scripted = 1,  // action produced by the slot's persistent bot
};

class SimPool {
   public:
    // All envs start from `base_cfg` (must have team_size == 3) and build
    // observations under `obs_cfg`. Every slot starts in Policy mode.
    SimPool(std::uint32_t num_envs,
            const sim::MatchConfig& base_cfg,
            const sim::ObsConfig& obs_cfg);

    [[nodiscard]] std::uint32_t num_envs() const noexcept {
        return static_cast<std::uint32_t>(envs_.size());
    }

    // --- Slot wiring -------------------------------------------------------
    // `bot_name` must be a valid bots::make_bot_by_name name (the binding
    // pre-validates; the bots factory aborts on unknown names).
    void set_slot_scripted(std::uint32_t env, std::uint32_t slot,
                           const std::string& bot_name);
    void set_slot_policy(std::uint32_t env, std::uint32_t slot);
    // Entity obs are built only for enabled slots (all enabled by default);
    // disabled slots' rows are zeroed. Skipping opponent rows halves obs
    // cost for scripted-opponent training.
    void set_obs_slot(std::uint32_t env, std::uint32_t slot, bool enabled);

    // --- Curriculum (mirrors the legacy env setters) -------------------------
    // Softens the named scripted bot with deterministic aim noise and a
    // fire-cadence gate — bit-identical to Phase4MappoEnv's
    // _apply_opponent_handicap. aim_noise 0 and cadence 1 mean full strength.
    void set_opponent_handicap(const std::string& bot,
                               float aim_noise_radians,
                               std::uint32_t fire_cadence_ticks);
    void clear_opponent_handicap() noexcept { handicap_.reset(); }
    // Applies to live sims immediately and to the stored config for future
    // resets, like the legacy setter.
    void set_objective_timing_ticks(std::uint32_t unlock_ticks,
                                    std::uint32_t capture_ticks);
    // Stored config only; takes effect from each env's next reset.
    void set_respawn_ticks(std::uint32_t respawn_ticks);
    // Replace one env's config (per-episode map randomization). Takes
    // effect at that env's next reset_env().
    void set_env_config(std::uint32_t env, const sim::MatchConfig& cfg);

    // --- Episode control ----------------------------------------------------
    // Rebuild env's sim from its stored config with `seed`, clear its
    // observation memory. The caller refreshes outputs via env_outputs().
    void reset_env(std::uint32_t env, std::uint64_t seed);

    // Write the env's current-state outputs (post-reset refill or reads
    // outside the step loop). Buffers may be null to skip that output.
    void env_outputs(std::uint32_t env,
                     float* entity_obs, std::uint32_t entity_capacity,
                     float* critic_obs, std::uint32_t critic_capacity,
                     float* features, std::uint32_t features_capacity);

    // --- The hot path -------------------------------------------------------
    // actions:            [num_envs][kAgentsPerMatch][kActionDim] float32,
    //                     team-relative policy controls in [-1,1]/[0,1]
    //                     (move_x, move_y, aim_delta, fire, ability_1,
    //                     ability_2); ignored for Scripted slots.
    // out_entity_obs:     [num_envs][kAgentsPerMatch][kEntityGridObsDim]
    // out_critic_obs:     [num_envs][2][kCriticObsDim] (Team A row, Team B row)
    // out_features:       [num_envs][kRewardFeatureDim]
    // out_terminated:     [num_envs] (score threshold reached)
    // out_truncated:      [num_envs] (round timer expired without threshold)
    //
    // Every env must be non-terminal (reset_env any env whose previous step
    // reported terminated or truncated before stepping again).
    static constexpr std::uint32_t kActionDim = 6;
    void step(const float* actions,
              float* out_entity_obs,
              float* out_critic_obs,
              float* out_features,
              std::uint8_t* out_terminated,
              std::uint8_t* out_truncated);

    [[nodiscard]] bool env_episode_over(std::uint32_t env) const;
    [[nodiscard]] std::uint64_t env_state_hash(std::uint32_t env) const;
    [[nodiscard]] std::uint64_t env_obs_state_hash(std::uint32_t env) const;

   private:
    struct Handicap {
        std::string bot;
        float aim_noise_radians = 0.0F;
        std::uint32_t fire_cadence_ticks = 1;
    };

    struct EnvState {
        sim::MatchConfig cfg{};
        std::unique_ptr<sim::Sim> sim;
        sim::ObservationEngine obs_engine;
        std::array<SlotMode, sim::kAgentsPerMatch> modes{};
        std::array<std::unique_ptr<bots::IBot>, sim::kAgentsPerMatch> bots{};
        std::array<bool, sim::kAgentsPerMatch> obs_enabled{};

        explicit EnvState(const sim::ObsConfig& obs_cfg)
            : obs_engine(obs_cfg) {}
    };

    void step_env(std::uint32_t env_idx, const float* actions,
                  float* out_entity_obs, float* out_critic_obs,
                  float* out_features, std::uint8_t* out_terminated,
                  std::uint8_t* out_truncated);
    void write_outputs(EnvState& env, float* entity_obs, float* critic_obs,
                       float* features);
    [[nodiscard]] common::Action scripted_action(EnvState& env,
                                                 std::uint32_t slot) const;

    std::vector<EnvState> envs_;
    std::optional<Handicap> handicap_;
};

}  // namespace xushi2::pool
