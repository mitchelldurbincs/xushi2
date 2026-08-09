// pybind11 bindings. Exposes the sim to the Python trainer.
// Keep this layer *thin* — only adapt C++ types to Python, never put game
// logic here.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <xushi2/bots/bot.h>
#include <xushi2/bots/runner.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_utils.h>
#include <xushi2/sim/sim.h>

namespace py = pybind11;

namespace {

// --- Boundary validation -------------------------------------------------
//
// The sim validates its own preconditions with X2_REQUIRE, which aborts the
// process. Aborting is the right call *inside* the sim: a Tier-0 invariant
// violation means the state is no longer trustworthy. It is the wrong call
// at this boundary, where the caller is a Python program that can and should
// handle a bad config. A SIGABRT gives Python no traceback, no exception to
// catch, and no chance to run `finally` blocks -- and under the async vector
// env it strands the parent process waiting on a pipe that will never be
// written to.
//
// Everything below mirrors a check in the Sim constructor
// (src/sim/src/sim.cpp) or in a builder, and raises ValueError naming the
// offending field instead.
//
// INVARIANT: an X2_REQUIRE firing during a Python call is a bug in this file.

[[noreturn]] void bad_field(const std::string& field, const std::string& why) {
    throw std::invalid_argument("MatchConfig." + field + " " + why);
}

void validate_mechanics(const xushi2::sim::Phase1MechanicsConfig& m) {
    constexpr auto kU32Unset = std::numeric_limits<std::uint32_t>::max();

    if (m.revolver_damage_centi_hp == kU32Unset) {
        bad_field("mechanics.revolver_damage_centi_hp",
                  "is unset; it has no default and must be supplied explicitly");
    }
    if (m.revolver_damage_centi_hp == 0U) {
        bad_field("mechanics.revolver_damage_centi_hp", "must be > 0");
    }
    if (m.revolver_fire_cooldown_ticks == kU32Unset) {
        bad_field("mechanics.revolver_fire_cooldown_ticks",
                  "is unset; it has no default and must be supplied explicitly");
    }
    if (m.revolver_fire_cooldown_ticks < 1U) {
        bad_field("mechanics.revolver_fire_cooldown_ticks", "must be >= 1");
    }
    if (!std::isfinite(m.revolver_hitbox_radius)) {
        bad_field("mechanics.revolver_hitbox_radius",
                  "is unset or non-finite; it has no default and must be supplied "
                  "explicitly");
    }
    if (m.revolver_hitbox_radius <= 0.0F) {
        bad_field("mechanics.revolver_hitbox_radius", "must be > 0");
    }
    if (m.respawn_ticks == kU32Unset) {
        bad_field("mechanics.respawn_ticks",
                  "is unset; it has no default and must be supplied explicitly");
    }
    if (m.respawn_ticks == 0U) {
        bad_field("mechanics.respawn_ticks", "must be > 0");
    }
}

void validate_geometry(const xushi2::sim::MatchConfig& cfg) {
    if (cfg.num_cover_circles > xushi2::common::kMaxWalls) {
        bad_field("cover_circles", "exceeds kMaxWalls");
    }
    for (std::uint32_t i = 0; i < cfg.num_cover_circles; ++i) {
        const auto& c = cfg.cover_circles[i];
        const std::string at = "cover_circles[" + std::to_string(i) + "]";
        if (!std::isfinite(c.center.x) || !std::isfinite(c.center.y)) {
            bad_field(at + ".center", "must be finite");
        }
        if (!std::isfinite(c.radius) || c.radius <= 0.0F) {
            bad_field(at + ".radius", "must be finite and > 0");
        }
        if (c.center.x - c.radius < cfg.map.min_x || c.center.x + c.radius > cfg.map.max_x ||
            c.center.y - c.radius < cfg.map.min_y || c.center.y + c.radius > cfg.map.max_y) {
            bad_field(at, "must lie entirely within map bounds");
        }
    }

    if (cfg.num_wall_segments > xushi2::common::kMaxWalls) {
        bad_field("wall_segments", "exceeds kMaxWalls");
    }
    for (std::uint32_t i = 0; i < cfg.num_wall_segments; ++i) {
        const auto& w = cfg.wall_segments[i];
        const std::string at = "wall_segments[" + std::to_string(i) + "]";
        if (!std::isfinite(w.a.x) || !std::isfinite(w.a.y) || !std::isfinite(w.b.x) ||
            !std::isfinite(w.b.y)) {
            bad_field(at, "endpoints must be finite");
        }
        if (!std::isfinite(w.half_width) || w.half_width <= 0.0F) {
            bad_field(at + ".half_width", "must be finite and > 0");
        }
        const float dx = w.b.x - w.a.x;
        const float dy = w.b.y - w.a.y;
        if (dx * dx + dy * dy <= 1e-6F) {
            bad_field(at, "must have non-zero length");
        }
        if (w.a.x - w.half_width < cfg.map.min_x || w.a.x + w.half_width > cfg.map.max_x ||
            w.a.y - w.half_width < cfg.map.min_y || w.a.y + w.half_width > cfg.map.max_y ||
            w.b.x - w.half_width < cfg.map.min_x || w.b.x + w.half_width > cfg.map.max_x ||
            w.b.y - w.half_width < cfg.map.min_y || w.b.y + w.half_width > cfg.map.max_y) {
            bad_field(at, "must lie entirely within map bounds");
        }
    }
}

void validate_match_config(const xushi2::sim::MatchConfig& cfg) {
    if (cfg.action_repeat != 2U && cfg.action_repeat != 3U) {
        bad_field("action_repeat",
                  "must be 2 or 3, got " + std::to_string(cfg.action_repeat));
    }
    if (cfg.map.max_x <= cfg.map.min_x) {
        bad_field("map", "requires max_x > min_x");
    }
    if (cfg.map.max_y <= cfg.map.min_y) {
        bad_field("map", "requires max_y > min_y");
    }
    if (cfg.team_size != 1U && cfg.team_size != 3U) {
        bad_field("team_size", "must be 1 or 3, got " + std::to_string(cfg.team_size));
    }
    if (cfg.objective_unlock_ticks == 0U) {
        bad_field("objective_unlock_ticks", "must be > 0");
    }
    if (cfg.objective_capture_ticks == 0U) {
        bad_field("objective_capture_ticks", "must be > 0");
    }
    validate_mechanics(cfg.mechanics);
    validate_geometry(cfg);
}

// Canonical scripted-bot registry, mirrored from
// src/bots/src/runner.cpp::kBotFactories. make_bot_by_name aborts on an
// unknown name, so every binding that accepts one validates first.
constexpr std::array<std::string_view, 6> kValidBotNames{
    {"walk_to_objective", "hold_and_shoot", "basic", "weak_basic", "weak_basic_v2", "noop"}};

void validate_bot_name(const std::string& name, const char* arg_name) {
    for (const auto& valid : kValidBotNames) {
        if (name == valid) {
            return;
        }
    }
    std::string msg = "unknown ";
    msg += arg_name;
    msg += " '" + name + "'; valid:";
    for (const auto& valid : kValidBotNames) {
        msg += " ";
        msg += valid;
    }
    throw std::invalid_argument(msg);
}

}  // namespace

PYBIND11_MODULE(xushi2_cpp, m) {
    m.doc() = "xushi2 C++ extension — deterministic 3v3 hero-shooter simulation";

    py::enum_<xushi2::common::Team>(m, "Team")
        .value("Neutral", xushi2::common::Team::Neutral)
        .value("A", xushi2::common::Team::A)
        .value("B", xushi2::common::Team::B);

    py::enum_<xushi2::common::Role>(m, "Role")
        .value("Tank", xushi2::common::Role::Tank)
        .value("Damage", xushi2::common::Role::Damage)
        .value("Support", xushi2::common::Role::Support);

    py::enum_<xushi2::common::HeroKind>(m, "HeroKind")
        .value("Vanguard", xushi2::common::HeroKind::Vanguard)
        .value("Ranger", xushi2::common::HeroKind::Ranger)
        .value("Mender", xushi2::common::HeroKind::Mender);

    py::enum_<xushi2::common::MenderWeapon>(m, "MenderWeapon")
        .value("Staff", xushi2::common::MenderWeapon::Staff)
        .value("Sidearm", xushi2::common::MenderWeapon::Sidearm);

    py::class_<xushi2::common::Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("move_x", &xushi2::common::Action::move_x)
        .def_readwrite("move_y", &xushi2::common::Action::move_y)
        .def_readwrite("aim_delta", &xushi2::common::Action::aim_delta)
        .def_readwrite("primary_fire", &xushi2::common::Action::primary_fire)
        .def_readwrite("ability_1", &xushi2::common::Action::ability_1)
        .def_readwrite("ability_2", &xushi2::common::Action::ability_2)
        .def_readwrite("target_slot", &xushi2::common::Action::target_slot);

    py::class_<xushi2::common::Vec2>(m, "Vec2")
        .def(py::init<>())
        .def_readwrite("x", &xushi2::common::Vec2::x)
        .def_readwrite("y", &xushi2::common::Vec2::y);

    py::class_<xushi2::sim::Phase1MechanicsConfig>(m, "Phase1MechanicsConfig")
        .def(py::init<>())
        .def_readwrite("revolver_damage_centi_hp",
                       &xushi2::sim::Phase1MechanicsConfig::revolver_damage_centi_hp)
        .def_readwrite("revolver_fire_cooldown_ticks",
                       &xushi2::sim::Phase1MechanicsConfig::revolver_fire_cooldown_ticks)
        .def_readwrite("revolver_hitbox_radius",
                       &xushi2::sim::Phase1MechanicsConfig::revolver_hitbox_radius)
        .def_readwrite("respawn_ticks",
                       &xushi2::sim::Phase1MechanicsConfig::respawn_ticks);

    py::class_<xushi2::sim::MapBounds>(m, "MapBounds")
        .def(py::init<>())
        .def_readwrite("min_x", &xushi2::sim::MapBounds::min_x)
        .def_readwrite("min_y", &xushi2::sim::MapBounds::min_y)
        .def_readwrite("max_x", &xushi2::sim::MapBounds::max_x)
        .def_readwrite("max_y", &xushi2::sim::MapBounds::max_y);

    py::class_<xushi2::sim::CoverCircle>(m, "CoverCircle")
        .def(py::init<>())
        .def_readwrite("center", &xushi2::sim::CoverCircle::center)
        .def_readwrite("radius", &xushi2::sim::CoverCircle::radius);

    py::class_<xushi2::sim::WallSegment>(m, "WallSegment")
        .def(py::init<>())
        .def_readwrite("a", &xushi2::sim::WallSegment::a)
        .def_readwrite("b", &xushi2::sim::WallSegment::b)
        .def_readwrite("half_width", &xushi2::sim::WallSegment::half_width);

    py::class_<xushi2::sim::MatchConfig>(m, "MatchConfig")
        .def(py::init<>())
        .def_readwrite("seed", &xushi2::sim::MatchConfig::seed)
        .def_readwrite("round_length_seconds",
                       &xushi2::sim::MatchConfig::round_length_seconds)
        .def_readwrite("fog_of_war_enabled",
                       &xushi2::sim::MatchConfig::fog_of_war_enabled)
        .def_readwrite("randomize_map", &xushi2::sim::MatchConfig::randomize_map)
        .def_readwrite("action_repeat", &xushi2::sim::MatchConfig::action_repeat)
        .def_readwrite("objective_unlock_ticks",
                       &xushi2::sim::MatchConfig::objective_unlock_ticks)
        .def_readwrite("objective_capture_ticks",
                       &xushi2::sim::MatchConfig::objective_capture_ticks)
        .def_readwrite("map", &xushi2::sim::MatchConfig::map)
        .def_property(
            "cover_circles",
            [](const xushi2::sim::MatchConfig& cfg) {
                std::vector<xushi2::sim::CoverCircle> out;
                out.reserve(cfg.num_cover_circles);
                for (std::uint32_t i = 0; i < cfg.num_cover_circles; ++i) {
                    out.push_back(cfg.cover_circles[i]);
                }
                return out;
            },
            [](xushi2::sim::MatchConfig& cfg,
               const std::vector<xushi2::sim::CoverCircle>& covers) {
                if (covers.size() > xushi2::common::kMaxWalls) {
                    throw std::invalid_argument("cover_circles exceeds kMaxWalls");
                }
                cfg.num_cover_circles = static_cast<std::uint32_t>(covers.size());
                for (std::size_t i = 0; i < covers.size(); ++i) {
                    cfg.cover_circles[i] = covers[i];
                }
            })
        .def_property(
            "wall_segments",
            [](const xushi2::sim::MatchConfig& cfg) {
                std::vector<xushi2::sim::WallSegment> out;
                out.reserve(cfg.num_wall_segments);
                for (std::uint32_t i = 0; i < cfg.num_wall_segments; ++i) {
                    out.push_back(cfg.wall_segments[i]);
                }
                return out;
            },
            [](xushi2::sim::MatchConfig& cfg,
               const std::vector<xushi2::sim::WallSegment>& walls) {
                if (walls.size() > xushi2::common::kMaxWalls) {
                    throw std::invalid_argument("wall_segments exceeds kMaxWalls");
                }
                cfg.num_wall_segments = static_cast<std::uint32_t>(walls.size());
                for (std::size_t i = 0; i < walls.size(); ++i) {
                    cfg.wall_segments[i] = walls[i];
                }
            })
        .def_readwrite("mechanics", &xushi2::sim::MatchConfig::mechanics)
        .def_readwrite("team_size", &xushi2::sim::MatchConfig::team_size)
        .def_property(
            "hero_kinds",
            [](const xushi2::sim::MatchConfig& cfg) {
                return std::vector<xushi2::common::HeroKind>(
                    cfg.hero_kinds.begin(), cfg.hero_kinds.end());
            },
            [](xushi2::sim::MatchConfig& cfg,
               const std::vector<xushi2::common::HeroKind>& kinds) {
                if (kinds.size() != xushi2::sim::kAgentsPerMatch) {
                    throw std::invalid_argument("hero_kinds length must be 6");
                }
                for (std::size_t i = 0; i < cfg.hero_kinds.size(); ++i) {
                    cfg.hero_kinds[i] = kinds[i];
                }
            });

    py::class_<xushi2::sim::Sim>(m, "Sim")
        // Validate before constructing: the Sim ctor's X2_REQUIRE checks abort
        // the process, which Python cannot catch or diagnose.
        .def(py::init([](const xushi2::sim::MatchConfig& config) {
            validate_match_config(config);
            return std::make_unique<xushi2::sim::Sim>(config);
        }))
        .def("reset", py::overload_cast<>(&xushi2::sim::Sim::reset))
        .def("reset", py::overload_cast<std::uint64_t>(&xushi2::sim::Sim::reset),
             py::arg("seed"))
        .def("set_objective_timing_ticks",
             &xushi2::sim::Sim::set_objective_timing_ticks,
             py::arg("unlock_ticks"), py::arg("capture_ticks"))
        .def("step",
             [](xushi2::sim::Sim& self, std::vector<xushi2::common::Action> actions) {
                 if (actions.size() != xushi2::sim::kAgentsPerMatch) {
                     throw std::invalid_argument(
                         "actions length must equal kAgentsPerMatch (= 6)");
                 }
                 std::array<xushi2::common::Action, xushi2::sim::kAgentsPerMatch> arr{};
                 for (std::size_t i = 0; i < arr.size(); ++i) {
                     arr[i] = actions[i];
                 }
                 self.step(arr);
             },
             py::arg("actions"))
        .def("step_decision",
             [](xushi2::sim::Sim& self, std::vector<xushi2::common::Action> actions) {
                 if (actions.size() != xushi2::sim::kAgentsPerMatch) {
                     throw std::invalid_argument(
                         "actions length must equal kAgentsPerMatch (= 6)");
                 }
                 std::array<xushi2::common::Action, xushi2::sim::kAgentsPerMatch> arr{};
                 for (std::size_t i = 0; i < arr.size(); ++i) {
                     arr[i] = actions[i];
                 }
                 self.step_decision(arr);
             },
             py::arg("actions"))
        .def_property_readonly("tick",
                               [](const xushi2::sim::Sim& s) { return s.state().tick; })
        .def_property_readonly("team_a_score_ticks",
                               [](const xushi2::sim::Sim& s) {
                                   return s.state().objective.team_a_score_ticks;
                               })
        .def_property_readonly("team_b_score_ticks",
                               [](const xushi2::sim::Sim& s) {
                                   return s.state().objective.team_b_score_ticks;
                               })
        .def_property_readonly("objective_unlock_ticks",
                               [](const xushi2::sim::Sim& s) {
                                   return s.config().objective_unlock_ticks;
                               })
        .def_property_readonly("objective_capture_ticks",
                               [](const xushi2::sim::Sim& s) {
                                   return s.config().objective_capture_ticks;
                               })
        .def_property_readonly("team_a_score",
                               [](const xushi2::sim::Sim& s) {
                                   return static_cast<double>(
                                              s.state().objective.team_a_score_ticks) /
                                          static_cast<double>(xushi2::sim::kTickHz);
                               })
        .def_property_readonly("team_b_score",
                               [](const xushi2::sim::Sim& s) {
                                   return static_cast<double>(
                                              s.state().objective.team_b_score_ticks) /
                                          static_cast<double>(xushi2::sim::kTickHz);
                               })
        .def_property_readonly("episode_over", &xushi2::sim::Sim::episode_over)
        // Gymnasium's terminated/truncated split needs the *reason* the episode
        // ended, not just that it did. Deriving it from `winner` is wrong: a
        // timeout with one team ahead has a winner but is still a time limit.
        .def_property_readonly("score_threshold_reached",
                               &xushi2::sim::Sim::score_threshold_reached)
        .def_property_readonly("round_timer_expired",
                               &xushi2::sim::Sim::round_timer_expired)
        .def_property_readonly("winner", &xushi2::sim::Sim::winner)
        .def_property_readonly("team_a_kills", &xushi2::sim::Sim::team_a_kills)
        .def_property_readonly("team_b_kills", &xushi2::sim::Sim::team_b_kills)
        .def_property_readonly("kills_by_slot",
                               [](const xushi2::sim::Sim& s) {
                                   const auto a = s.kills_by_slot();
                                   return std::vector<std::uint32_t>(a.begin(),
                                                                     a.end());
                               })
        .def_property_readonly("deaths_by_slot",
                               [](const xushi2::sim::Sim& s) {
                                   const auto a = s.deaths_by_slot();
                                   return std::vector<std::uint32_t>(a.begin(),
                                                                     a.end());
                               })
        .def_property_readonly("damage_dealt_by_slot",
                               [](const xushi2::sim::Sim& s) {
                                   const auto a = s.damage_dealt_by_slot();
                                   return std::vector<std::uint64_t>(a.begin(),
                                                                     a.end());
                               })
        .def_property_readonly("state_hash", &xushi2::sim::Sim::state_hash);

    m.def("line_of_sight",
          [](const xushi2::sim::Sim& sim, std::uint32_t from_slot,
             std::uint32_t to_slot) {
              return sim.line_of_sight(from_slot, to_slot);
          },
          py::arg("sim"), py::arg("from_slot"), py::arg("to_slot"),
          "Return true when cover geometry does not block the segment.");

    m.def("observable_enemy_slots",
          [](const xushi2::sim::Sim& sim, std::uint32_t viewer_slot) {
              const auto mask =
                  xushi2::sim::obs_utils::observable_enemy_slots(sim, viewer_slot);
              std::vector<bool> out;
              out.reserve(mask.size());
              for (bool v : mask) {
                  out.push_back(v);
              }
              return out;
          },
          py::arg("sim"), py::arg("viewer_slot"),
          "Return a six-slot mask of opposite-team heroes observable from "
          "viewer_slot under native fog/LoS rules.");

    // Return the Action a named scripted bot would produce for a given
    // agent slot, given the current sim state. Used by the Gymnasium env
    // wrapper to drive the opponent without exposing MatchState to Python.
    // Unknown bot names raise ValueError.
    m.def(
        "scripted_bot_action",
        [](const xushi2::sim::Sim& sim, int agent_slot,
           const std::string& bot_name) {
            validate_bot_name(bot_name, "bot_name");
            auto bot = xushi2::bots::make_bot_by_name(bot_name);
            return bot->decide(sim.state(), sim.config(), agent_slot);
        },
        py::arg("sim"), py::arg("agent_slot"), py::arg("bot_name"),
        "Return the Action the named scripted bot would emit for "
        "agent_slot on the current sim state.");

    m.def("run_scripted_episode",
          [](const xushi2::sim::MatchConfig& config, const std::string& bot_a,
             const std::string& bot_b) {
              // Both the config and the bot names reach abort-on-invalid code
              // (Sim ctor, make_bot_by_name) inside run_scripted_episode.
              validate_match_config(config);
              validate_bot_name(bot_a, "bot_a");
              validate_bot_name(bot_b, "bot_b");
              auto result = xushi2::bots::run_scripted_episode(config, bot_a, bot_b);
              return py::make_tuple(std::move(result.decision_hashes), result.final_tick,
                                    result.team_a_kills, result.team_b_kills,
                                    static_cast<int>(result.winner));
          },
          py::arg("config"), py::arg("bot_a"), py::arg("bot_b"),
          "Run one scripted-vs-scripted episode. Returns "
          "(decision_hashes, final_tick, team_a_kills, team_b_kills, winner_int).");

    m.attr("TICK_HZ") = xushi2::sim::kTickHz;
    m.attr("AGENTS_PER_MATCH") = xushi2::sim::kAgentsPerMatch;
    m.attr("TEAM_SIZE") = xushi2::sim::kTeamSize;
    m.attr("ACTOR_OBS_PHASE1_DIM") = xushi2::sim::kActorObsPhase1Dim;
    m.attr("CRITIC_OBS_DIM") = xushi2::sim::kCriticObsDim;

    // Write the Phase-1 actor observation for `agent_slot` into the
    // caller-provided float32 numpy buffer. Zero-copy: Python owns the
    // buffer; C++ writes directly. Any buffer smaller than
    // ACTOR_OBS_PHASE1_DIM raises ValueError.
    m.def(
        "build_actor_obs",
        [](const xushi2::sim::Sim& sim, std::uint32_t agent_slot,
           py::array_t<float, py::array::c_style | py::array::forcecast> out) {
            if (out.ndim() != 1) {
                throw std::invalid_argument(
                    "out buffer must be 1-D float32");
            }
            if (static_cast<std::uint32_t>(out.shape(0)) <
                xushi2::sim::kActorObsPhase1Dim) {
                throw std::invalid_argument(
                    "out buffer length must be >= ACTOR_OBS_PHASE1_DIM");
            }
            xushi2::sim::build_actor_obs_phase1(
                sim, agent_slot, out.mutable_data(0),
                static_cast<std::uint32_t>(out.shape(0)));
        },
        py::arg("sim"), py::arg("agent_slot"), py::arg("out"),
        "Write the Phase-1 actor observation for agent_slot into `out`.");

    // Write the Phase-4 critic observation for `team_perspective` into the
    // caller-provided float32 numpy buffer. Team must be Team.A or Team.B.
    // Requires the Sim was constructed with MatchConfig::team_size == 3.
    m.def(
        "build_critic_obs",
        [](const xushi2::sim::Sim& sim, xushi2::common::Team team_perspective,
           py::array_t<float, py::array::c_style | py::array::forcecast> out) {
            if (out.ndim() != 1) {
                throw std::invalid_argument(
                    "out buffer must be 1-D float32");
            }
            if (static_cast<std::uint32_t>(out.shape(0)) <
                xushi2::sim::kCriticObsDim) {
                throw std::invalid_argument(
                    "out buffer length must be >= CRITIC_OBS_DIM");
            }
            // Pre-validate team so Python gets a real exception instead of a
            // process-abort from X2_REQUIRE inside the builder.
            if (team_perspective != xushi2::common::Team::A &&
                team_perspective != xushi2::common::Team::B) {
                throw std::invalid_argument(
                    "team_perspective must be Team.A or Team.B "
                    "(Team.Neutral is not a valid critic side)");
            }
            // The builder also asserts team_size == 3 (it needs three present
            // Rangers per side); that assert aborts, so check it here too.
            if (sim.config().team_size != 3U) {
                throw std::invalid_argument(
                    "build_critic_obs requires a Sim built with team_size == 3, got " +
                    std::to_string(sim.config().team_size));
            }
            xushi2::sim::build_critic_obs(
                sim, team_perspective, out.mutable_data(0),
                static_cast<std::uint32_t>(out.shape(0)));
        },
        py::arg("sim"), py::arg("team_perspective"), py::arg("out"),
        "Write the Phase-4 critic observation (kCriticObsDim floats) for a "
        "team perspective into `out`. Requires team_size == 3.");
}
