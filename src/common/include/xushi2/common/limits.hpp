#pragma once

// Project-wide fixed capacity constants. See docs/coding_philosophy.md §5.
// No dynamic allocation after initialization — all Tier 0 buffers size from
// these.

#include <cstdint>

namespace xushi2::common {

inline constexpr std::uint32_t kTickHz = 30;
inline constexpr std::uint32_t kTeamSize = 3;
inline constexpr std::uint32_t kMaxHeroes = 6;
inline constexpr std::uint32_t kMaxWalls = 128;
inline constexpr std::uint32_t kMaxBarriers = 6;
inline constexpr std::uint32_t kMaxEventsPerTick = 128;
inline constexpr std::uint32_t kMaxRaysPerTick = 32;

// At most 240 s @ 15 Hz policy rate; see docs/replay_format.md.
inline constexpr std::uint32_t kMaxReplayDecisionRecords = 240U * 15U;

// Default action-repeat window (sim ticks per policy decision).
// See docs/action_spec.md. Valid values: 2 or 3.
inline constexpr std::uint32_t kDefaultActionRepeat = 3;

// --- Phase 1 doc-pinned constants ---
// Values fixed by docs/game_design.md §3, §6. Not user-config.

// Ranger kit (game_design.md §6).
inline constexpr std::int32_t  kRangerMaxHpCentiHp          = 15000;      // 150.0 HP × 100
inline constexpr std::uint8_t  kRangerMaxMagazine           = 6;
inline constexpr float         kRangerRevolverRange         = 22.0F;      // u
inline constexpr std::uint32_t kRangerCombatRollCooldownTicks = 150;      // 5 s × 30 Hz
inline constexpr float         kRangerCombatRollDistance    = 3.0F;       // u
inline constexpr std::uint32_t kRangerMarkTargetCooldownTicks = 180;      // 6 s × 30 Hz
inline constexpr std::uint32_t kRangerMarkTargetDurationTicks = 90;       // 3 s × 30 Hz
inline constexpr std::uint32_t kAutoReloadDelayTicks        = 60;         // 2 s × 30 Hz
inline constexpr std::uint32_t kReloadDurationTicks         = 45;         // 1.5 s × 30 Hz

// Phase-10 roster smoke constants. Full ability kits are still deferred, but
// kind/role/HP must be real state once compositions vary.
inline constexpr std::int32_t  kVanguardMaxHpCentiHp        = 25000;      // 250.0 HP × 100
inline constexpr std::int32_t  kVanguardBarrierHpCenti      = 25000;      // 250.0 HP × 100
inline constexpr float         kVanguardBarrierRadius       = 1.5F;       // diagnostic circular shield
inline constexpr std::uint32_t kVanguardBarrierBrokenCooldownTicks = 90;  // 3 s × 30 Hz
inline constexpr float         kVanguardGuardStepDistance   = 2.0F;       // u
inline constexpr std::uint32_t kVanguardGuardStepCooldownTicks = 150;     // 5 s × 30 Hz
inline constexpr std::int32_t  kVanguardWarhammerDamageCentiHp = 5500;    // 55.0 HP × 100
inline constexpr float         kVanguardWarhammerRange      = 3.0F;       // u
inline constexpr float         kVanguardWarhammerHalfAngleCos = 0.8660254F; // cos(30 deg)
inline constexpr std::uint32_t kVanguardWarhammerCooldownTicks = 15;      // 2 hits/sec
inline constexpr std::int32_t  kMenderMaxHpCentiHp          = 16000;      // 160.0 HP × 100
inline constexpr std::uint32_t kMenderWeaponSwapCooldownTicks = 15;       // 0.5 s × 30 Hz
inline constexpr std::int32_t  kMenderSidearmDamageCentiHp  = 2500;       // 25.0 HP × 100
inline constexpr float         kMenderSidearmRange          = 15.0F;      // u
inline constexpr std::uint32_t kMenderSidearmCooldownTicks  = 10;         // 3 shots/sec
inline constexpr std::int32_t  kMenderBeamHealCentiHpPerTick = 167;       // ~50 HP/s at 30 Hz
inline constexpr float         kMenderBeamRange             = 12.0F;      // u
inline constexpr float         kMenderBeamHalfAngleCos      = 0.9238795F; // cos(22.5 deg)
inline constexpr float         kMenderTetherRange           = 18.0F;      // u
inline constexpr float         kMenderTetherStopDistance    = 1.5F;       // u from target ally
inline constexpr std::uint32_t kMenderTetherCooldownTicks   = 240;        // 8 s × 30 Hz

// Objective state machine (game_design.md §3).
inline constexpr std::uint32_t kCaptureTicks                = 240;        // 8 s × 30 Hz
inline constexpr std::uint32_t kWinTicks                    = 3000;       // 100 s × 30 Hz
inline constexpr std::uint32_t kObjectiveLockTicks          = 450;        // 15 s × 30 Hz
inline constexpr float         kObjectiveRadius             = 3.0F;       // u

}  // namespace xushi2::common
