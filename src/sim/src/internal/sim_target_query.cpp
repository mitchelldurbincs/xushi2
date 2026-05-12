#include "sim_target_query.h"

#include <cmath>
#include <limits>

#include "sim_combat.h"

namespace xushi2::sim::internal {
namespace {

float ray_circle_hit_t(common::Vec2 origin, common::Vec2 d, common::Vec2 center, float radius) {
    const common::Vec2 oc{origin.x - center.x, origin.y - center.y};
    const float b = oc.x * d.x + oc.y * d.y;
    const float c_term = oc.x * oc.x + oc.y * oc.y - radius * radius;
    const float disc = b * b - c_term;
    if (disc < 0.0F) return -1.0F;
    const float sqrt_disc = std::sqrt(disc);
    const float t_near = -b - sqrt_disc;
    if (t_near >= 0.0F) return t_near;
    const float t_far = -b + sqrt_disc;
    if (t_far >= 0.0F) return 0.0F;
    return -1.0F;
}

bool better_candidate(float t, std::uint32_t slot, float best_t, int best_slot) {
    constexpr float kTieEps = 1e-6F;
    if (t + kTieEps < best_t) return true;
    if (std::abs(t - best_t) <= kTieEps && (best_slot < 0 || slot < static_cast<std::uint32_t>(best_slot))) return true;
    return false;
}

}  // namespace

int nearest_enemy_in_range_with_los(const MatchState& state, std::uint32_t actor_slot, float range,
                                    const MatchConfig& config) {
    const HeroState& actor = state.heroes[actor_slot];
    float best_dist_sq = range * range;
    int best_slot = -1;
    for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
        const HeroState& target = state.heroes[j];
        if (!target.present || !target.alive || target.team == actor.team) continue;
        const common::Vec2 to{target.position.x - actor.position.x, target.position.y - actor.position.y};
        const float dist_sq = to.x * to.x + to.y * to.y;
        if (dist_sq > best_dist_sq || segment_blocked_by_cover(actor.position, target.position, config)) continue;
        best_dist_sq = dist_sq;
        best_slot = static_cast<int>(j);
    }
    return best_slot;
}

int nearest_enemy_in_cone_with_los(const MatchState& state, std::uint32_t actor_slot, float range,
                                  float half_angle_cos, const MatchConfig& config) {
    const HeroState& actor = state.heroes[actor_slot];
    const common::Vec2 facing{std::cos(actor.aim_angle), std::sin(actor.aim_angle)};
    float best_dist_sq = range * range;
    int best_slot = -1;
    for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
        const HeroState& target = state.heroes[j];
        if (!target.present || !target.alive || target.team == actor.team) continue;
        const common::Vec2 to{target.position.x - actor.position.x, target.position.y - actor.position.y};
        const float dist_sq = to.x * to.x + to.y * to.y;
        if (dist_sq <= 1e-6F || dist_sq > best_dist_sq || segment_blocked_by_cover(actor.position, target.position, config)) continue;
        const float inv = 1.0F / std::sqrt(dist_sq);
        const float dot = (to.x * facing.x + to.y * facing.y) * inv;
        if (dot < half_angle_cos) continue;
        best_dist_sq = dist_sq;
        best_slot = static_cast<int>(j);
    }
    return best_slot;
}

int nearest_ally_in_cone_with_los(const MatchState& state, std::uint32_t actor_slot, float range,
                                  float half_angle_cos, const MatchConfig& config) {
    const HeroState& actor = state.heroes[actor_slot];
    const common::Vec2 facing{std::cos(actor.aim_angle), std::sin(actor.aim_angle)};
    float best_dist_sq = range * range;
    int best_slot = -1;
    for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
        if (j == actor_slot) continue;
        const HeroState& ally = state.heroes[j];
        if (!ally.present || !ally.alive || ally.team != actor.team) continue;
        const common::Vec2 to{ally.position.x - actor.position.x, ally.position.y - actor.position.y};
        const float dist_sq = to.x * to.x + to.y * to.y;
        if (dist_sq <= 1e-6F || dist_sq > best_dist_sq || segment_blocked_by_cover(actor.position, ally.position, config)) continue;
        const float inv = 1.0F / std::sqrt(dist_sq);
        const float dot = (to.x * facing.x + to.y * facing.y) * inv;
        if (dot < half_angle_cos) continue;
        best_dist_sq = dist_sq;
        best_slot = static_cast<int>(j);
    }
    return best_slot;
}

RayHitCandidate first_ray_hit_enemy_or_barrier(const MatchState& state, std::uint32_t actor_slot,
                                               common::Vec2 direction_unit, float max_range,
                                               float enemy_hitbox_radius, const MatchConfig& config) {
    const HeroState& actor = state.heroes[actor_slot];
    const float cover_t = first_cover_hit_t(actor.position, direction_unit, max_range, config);
    float best_t = std::numeric_limits<float>::infinity();
    int best_slot = -1;
    RayHitCandidate::Kind best_kind = RayHitCandidate::Kind::None;
    for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
        const HeroState& other = state.heroes[j];
        if (!other.present || !other.alive || other.team == actor.team) continue;

        if (other.vanguard_barrier_active && other.vanguard_barrier_hp_centi > 0) {
            const float t = ray_circle_hit_t(actor.position, direction_unit, other.position, common::kVanguardBarrierRadius);
            if (t >= 0.0F && t <= max_range && t < cover_t && better_candidate(t, j, best_t, best_slot)) {
                best_t = t;
                best_slot = static_cast<int>(j);
                best_kind = RayHitCandidate::Kind::Barrier;
            }
        }

        const float t = ray_circle_hit_t(actor.position, direction_unit, other.position, enemy_hitbox_radius);
        if (t >= 0.0F && t <= max_range && t < cover_t && better_candidate(t, j, best_t, best_slot)) {
            best_t = t;
            best_slot = static_cast<int>(j);
            best_kind = RayHitCandidate::Kind::Enemy;
        }
    }
    return RayHitCandidate{best_kind, best_slot, best_t};
}

}  // namespace xushi2::sim::internal
