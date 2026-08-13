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
#include <xushi2/pool/sim_pool.h>
#include <xushi2/sim/entity_obs.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_config.h>
#include <xushi2/sim/obs_utils.h>
#include <xushi2/sim/reward_features.h>
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

    // Output buffers must be exact, caller-owned NumPy storage. forcecast is
    // intentionally excluded: converting a writable argument would direct
    // C++ writes into a temporary and silently discard the result.
    using FloatOutputArray = py::array_t<float, py::array::c_style>;
    using U8OutputArray = py::array_t<std::uint8_t, py::array::c_style>;
    auto require_writable = [](const char* name, const py::array& array) {
        if (!array.writeable()) {
            throw std::invalid_argument(std::string(name) +
                                        " must be writable");
        }
    };

    // Write the Phase-1 actor observation for `agent_slot` into the
    // caller-provided float32 numpy buffer. Zero-copy: Python owns the
    // buffer; C++ writes directly. Any buffer smaller than
    // ACTOR_OBS_PHASE1_DIM raises ValueError.
    m.def(
        "build_actor_obs",
        [require_writable](const xushi2::sim::Sim& sim,
                           std::uint32_t agent_slot, FloatOutputArray out) {
            require_writable("out", out);
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
        py::arg("sim"), py::arg("agent_slot"), py::arg("out").noconvert(),
        "Write the Phase-1 actor observation for agent_slot into `out`.");

    // Write the Phase-4 critic observation for `team_perspective` into the
    // caller-provided float32 numpy buffer. Team must be Team.A or Team.B.
    // Requires the Sim was constructed with MatchConfig::team_size == 3.
    m.def(
        "build_critic_obs",
        [require_writable](const xushi2::sim::Sim& sim,
                           xushi2::common::Team team_perspective,
                           FloatOutputArray out) {
            require_writable("out", out);
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
        py::arg("sim"), py::arg("team_perspective"),
        py::arg("out").noconvert(),
        "Write the Phase-4 critic observation (kCriticObsDim floats) for a "
        "team perspective into `out`. Requires team_size == 3.");

    // --- Native entity observation (ObservationEngine) ---------------------

    m.attr("ENTITY_TOKEN_DIM") = xushi2::sim::kEntityTokenDim;
    m.attr("ENTITY_TOKEN_COUNT") = xushi2::sim::kEntityTokenCount;
    m.attr("ENTITY_GRID_CHANNELS") = xushi2::sim::kEntityGridChannels;
    m.attr("ENTITY_GRID_SIZE") = xushi2::sim::kEntityGridSize;
    m.attr("ENTITY_GRID_OBS_DIM") = xushi2::sim::kEntityGridObsDim;

    py::enum_<xushi2::sim::FogMode>(m, "FogMode")
        .value("NoFog", xushi2::sim::FogMode::None)
        .value("TeamShared", xushi2::sim::FogMode::TeamShared)
        .value("PerAgent", xushi2::sim::FogMode::PerAgent);

    py::class_<xushi2::sim::ObsConfig>(m, "ObsConfig")
        .def(py::init<>())
        .def_readwrite("fog_mode", &xushi2::sim::ObsConfig::fog_mode)
        .def_readwrite("visible_radius",
                       &xushi2::sim::ObsConfig::visible_radius)
        .def_readwrite("last_seen_enabled",
                       &xushi2::sim::ObsConfig::last_seen_enabled)
        .def_readwrite("zero_hidden_token_markers",
                       &xushi2::sim::ObsConfig::zero_hidden_token_markers);

    // Shared preconditions for the entity-obs entry points. The engine's
    // internals assert with X2_REQUIRE (which aborts), so everything a
    // Python caller could get wrong is checked here and raised instead.
    auto validate_entity_obs_call = [](const xushi2::sim::Sim& sim,
                                       std::uint32_t viewer_slot,
                                       bool check_slot) {
        if (sim.config().team_size != 3U) {
            throw std::invalid_argument(
                "entity obs requires a Sim built with team_size == 3, got " +
                std::to_string(sim.config().team_size));
        }
        if (check_slot &&
            viewer_slot >= static_cast<std::uint32_t>(
                               xushi2::sim::kAgentsPerMatch)) {
            throw std::invalid_argument(
                "viewer_slot must be < AGENTS_PER_MATCH (= 6)");
        }
    };

    py::class_<xushi2::sim::ObservationEngine>(m, "ObservationEngine")
        .def(py::init([](const xushi2::sim::ObsConfig& cfg) {
                 // The engine ctor X2_REQUIREs a positive radius; raise
                 // instead of aborting.
                 if (xushi2::sim::has_visible_radius(cfg) &&
                     (cfg.visible_radius <= 0.0F ||
                      !std::isfinite(cfg.visible_radius))) {
                     throw std::invalid_argument(
                         "ObsConfig.visible_radius must be finite and > 0 "
                         "when set (NaN means unset)");
                 }
                 return std::make_unique<xushi2::sim::ObservationEngine>(cfg);
             }),
             py::arg("cfg"))
        .def("reset", &xushi2::sim::ObservationEngine::reset,
             "Clear last-seen memory. Call whenever the paired Sim resets.")
        .def(
            "build_entity_obs",
            [validate_entity_obs_call,
             require_writable](xushi2::sim::ObservationEngine& engine,
                               const xushi2::sim::Sim& sim,
                               std::uint32_t viewer_slot,
                               FloatOutputArray out) {
                validate_entity_obs_call(sim, viewer_slot, true);
                require_writable("out", out);
                if (out.ndim() != 1) {
                    throw std::invalid_argument("out buffer must be 1-D float32");
                }
                if (static_cast<std::uint32_t>(out.shape(0)) <
                    xushi2::sim::kEntityGridObsDim) {
                    throw std::invalid_argument(
                        "out buffer length must be >= ENTITY_GRID_OBS_DIM");
                }
                engine.build_entity_obs(
                    sim, viewer_slot, out.mutable_data(0),
                    static_cast<std::uint32_t>(out.shape(0)));
            },
            py::arg("sim"), py::arg("viewer_slot"),
            py::arg("out").noconvert(),
            "Write the entity-token/grid observation (ENTITY_GRID_OBS_DIM "
            "floats) for viewer_slot into `out`, updating last-seen memory.")
        .def(
            "build_entity_obs_all",
            [validate_entity_obs_call,
             require_writable](xushi2::sim::ObservationEngine& engine,
                               const xushi2::sim::Sim& sim,
                               FloatOutputArray out) {
                validate_entity_obs_call(sim, 0, false);
                require_writable("out", out);
                const std::uint32_t total =
                    static_cast<std::uint32_t>(xushi2::sim::kAgentsPerMatch) *
                    xushi2::sim::kEntityGridObsDim;
                if (static_cast<std::uint32_t>(out.size()) < total) {
                    throw std::invalid_argument(
                        "out buffer must hold AGENTS_PER_MATCH * "
                        "ENTITY_GRID_OBS_DIM floats");
                }
                engine.build_entity_obs_all(
                    sim, out.mutable_data(),
                    static_cast<std::uint32_t>(out.size()));
            },
            py::arg("sim"), py::arg("out").noconvert(),
            "Write entity observations for all six viewer slots (ascending) "
            "into `out`, updating last-seen memory.")
        .def(
            "visible_enemies",
            [validate_entity_obs_call](
                const xushi2::sim::ObservationEngine& engine,
                const xushi2::sim::Sim& sim, std::uint32_t viewer_slot) {
                validate_entity_obs_call(sim, viewer_slot, true);
                return engine.visible_enemies(sim, viewer_slot);
            },
            py::arg("sim"), py::arg("viewer_slot"),
            "Read-only visibility of the viewer's three enemies (ascending "
            "enemy slot order); does not update last-seen memory.")
        .def_property_readonly(
            "obs_state_hash",
            &xushi2::sim::ObservationEngine::obs_state_hash,
            "Deterministic hash of last-seen memory only. Not part of "
            "Sim.state_hash.");

    // --- Batched sim boundary (SimPool) -------------------------------------

    m.attr("REWARD_FEATURE_DIM") = xushi2::sim::kRewardFeatureDim;
    m.attr("POOL_ACTION_DIM") = xushi2::pool::SimPool::kActionDim;

    using FloatInputArray =
        py::array_t<float, py::array::c_style | py::array::forcecast>;

    auto require_size = [](const char* name, py::ssize_t actual,
                           std::size_t expected) {
        if (actual < static_cast<py::ssize_t>(expected)) {
            throw std::invalid_argument(
                std::string(name) + " must hold at least " +
                std::to_string(expected) + " elements, got " +
                std::to_string(actual));
        }
    };

    py::class_<xushi2::pool::SimPool>(m, "SimPool")
        .def(py::init([](std::uint32_t num_envs,
                         const xushi2::sim::MatchConfig& cfg,
                         const xushi2::sim::ObsConfig& obs_cfg) {
                 if (num_envs == 0) {
                     throw std::invalid_argument("num_envs must be > 0");
                 }
                 validate_match_config(cfg);
                 if (cfg.team_size != 3U) {
                     throw std::invalid_argument(
                         "SimPool requires team_size == 3");
                 }
                 if (xushi2::sim::has_visible_radius(obs_cfg) &&
                     (obs_cfg.visible_radius <= 0.0F ||
                      !std::isfinite(obs_cfg.visible_radius))) {
                     throw std::invalid_argument(
                         "ObsConfig.visible_radius must be finite and > 0 "
                         "when set (NaN means unset)");
                 }
                 return std::make_unique<xushi2::pool::SimPool>(num_envs, cfg,
                                                                obs_cfg);
             }),
             py::arg("num_envs"), py::arg("config"), py::arg("obs_config"))
        .def_property_readonly("num_envs", &xushi2::pool::SimPool::num_envs)
        .def(
            "set_slot_scripted",
            [](xushi2::pool::SimPool& pool, std::uint32_t env,
               std::uint32_t slot, const std::string& bot_name) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                if (slot >= xushi2::sim::kAgentsPerMatch) {
                    throw std::invalid_argument("slot index out of range");
                }
                validate_bot_name(bot_name, "bot_name");
                pool.set_slot_scripted(env, slot, bot_name);
            },
            py::arg("env"), py::arg("slot"), py::arg("bot_name"))
        .def(
            "set_slot_policy",
            [](xushi2::pool::SimPool& pool, std::uint32_t env,
               std::uint32_t slot) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                if (slot >= xushi2::sim::kAgentsPerMatch) {
                    throw std::invalid_argument("slot index out of range");
                }
                pool.set_slot_policy(env, slot);
            },
            py::arg("env"), py::arg("slot"))
        .def(
            "set_obs_slot",
            [](xushi2::pool::SimPool& pool, std::uint32_t env,
               std::uint32_t slot, bool enabled) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                if (slot >= xushi2::sim::kAgentsPerMatch) {
                    throw std::invalid_argument("slot index out of range");
                }
                pool.set_obs_slot(env, slot, enabled);
            },
            py::arg("env"), py::arg("slot"), py::arg("enabled"))
        .def(
            "set_opponent_handicap",
            [](xushi2::pool::SimPool& pool, const std::string& bot,
               float aim_noise_radians, std::uint32_t fire_cadence_ticks) {
                validate_bot_name(bot, "bot");
                if (aim_noise_radians < 0.0F ||
                    !std::isfinite(aim_noise_radians)) {
                    throw std::invalid_argument(
                        "aim_noise_radians must be finite and >= 0");
                }
                if (fire_cadence_ticks < 1U) {
                    throw std::invalid_argument(
                        "fire_cadence_ticks must be >= 1");
                }
                pool.set_opponent_handicap(bot, aim_noise_radians,
                                           fire_cadence_ticks);
            },
            py::arg("bot"), py::arg("aim_noise_radians"),
            py::arg("fire_cadence_ticks"))
        .def("clear_opponent_handicap",
             &xushi2::pool::SimPool::clear_opponent_handicap)
        .def(
            "set_objective_timing_ticks",
            [](xushi2::pool::SimPool& pool, std::uint32_t unlock,
               std::uint32_t capture) {
                if (unlock == 0U || capture == 0U) {
                    throw std::invalid_argument(
                        "objective timing ticks must be > 0");
                }
                pool.set_objective_timing_ticks(unlock, capture);
            },
            py::arg("unlock_ticks"), py::arg("capture_ticks"))
        .def(
            "set_respawn_ticks",
            [](xushi2::pool::SimPool& pool, std::uint32_t ticks) {
                if (ticks == 0U) {
                    throw std::invalid_argument("respawn_ticks must be > 0");
                }
                pool.set_respawn_ticks(ticks);
            },
            py::arg("respawn_ticks"))
        .def(
            "set_env_config",
            [](xushi2::pool::SimPool& pool, std::uint32_t env,
               const xushi2::sim::MatchConfig& cfg) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                validate_match_config(cfg);
                if (cfg.team_size != 3U) {
                    throw std::invalid_argument(
                        "SimPool requires team_size == 3");
                }
                pool.set_env_config(env, cfg);
            },
            py::arg("env"), py::arg("config"),
            "Replace one env's MatchConfig; applies at its next reset_env().")
        .def(
            "reset_env",
            [](xushi2::pool::SimPool& pool, std::uint32_t env,
               std::uint64_t seed) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                pool.reset_env(env, seed);
            },
            py::arg("env"), py::arg("seed"))
        .def(
            "env_outputs",
            [require_size, require_writable](
                xushi2::pool::SimPool& pool, std::uint32_t env,
                FloatOutputArray entity_obs, FloatOutputArray critic_obs,
                FloatOutputArray features) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                require_writable("entity_obs", entity_obs);
                require_writable("critic_obs", critic_obs);
                require_writable("features", features);
                constexpr std::size_t kAgents = xushi2::sim::kAgentsPerMatch;
                require_size("entity_obs", entity_obs.size(),
                             kAgents * xushi2::sim::kEntityGridObsDim);
                require_size("critic_obs", critic_obs.size(),
                             2U * xushi2::sim::kCriticObsDim);
                require_size("features", features.size(),
                             xushi2::sim::kRewardFeatureDim);
                pool.env_outputs(
                    env, entity_obs.mutable_data(),
                    static_cast<std::uint32_t>(entity_obs.size()),
                    critic_obs.mutable_data(),
                    static_cast<std::uint32_t>(critic_obs.size()),
                    features.mutable_data(),
                    static_cast<std::uint32_t>(features.size()));
            },
            py::arg("env"), py::arg("entity_obs").noconvert(),
            py::arg("critic_obs").noconvert(),
            py::arg("features").noconvert(),
            "Write env's current-state entity obs [6*ENTITY_GRID_OBS_DIM], "
            "critic obs [2*CRITIC_OBS_DIM] (Team A row then Team B), and "
            "reward features [REWARD_FEATURE_DIM].")
        .def(
            "step",
            [require_size,
             require_writable](xushi2::pool::SimPool& pool,
                               FloatInputArray actions,
                               FloatOutputArray entity_obs,
                               FloatOutputArray critic_obs,
                               FloatOutputArray features,
                               U8OutputArray terminated,
                               U8OutputArray truncated) {
                const std::size_t n = pool.num_envs();
                constexpr std::size_t kAgents = xushi2::sim::kAgentsPerMatch;
                require_writable("entity_obs", entity_obs);
                require_writable("critic_obs", critic_obs);
                require_writable("features", features);
                require_writable("terminated", terminated);
                require_writable("truncated", truncated);
                require_size("actions", actions.size(),
                             n * kAgents * xushi2::pool::SimPool::kActionDim);
                require_size("entity_obs", entity_obs.size(),
                             n * kAgents * xushi2::sim::kEntityGridObsDim);
                require_size("critic_obs", critic_obs.size(),
                             n * 2U * xushi2::sim::kCriticObsDim);
                require_size("features", features.size(),
                             n * xushi2::sim::kRewardFeatureDim);
                require_size("terminated", terminated.size(), n);
                require_size("truncated", truncated.size(), n);
                for (std::uint32_t i = 0;
                     i < static_cast<std::uint32_t>(n); ++i) {
                    if (pool.env_episode_over(i)) {
                        throw std::invalid_argument(
                            "env " + std::to_string(i) +
                            " is terminal; call reset_env before stepping");
                    }
                }
                const float* actions_ptr = actions.data();
                float* entity_ptr = entity_obs.mutable_data();
                float* critic_ptr = critic_obs.mutable_data();
                float* features_ptr = features.mutable_data();
                std::uint8_t* term_ptr = terminated.mutable_data();
                std::uint8_t* trunc_ptr = truncated.mutable_data();
                {
                    // The pool touches no Python objects: all buffers are
                    // caller-owned numpy memory validated above.
                    py::gil_scoped_release release;
                    pool.step(actions_ptr, entity_ptr, critic_ptr,
                              features_ptr, term_ptr, trunc_ptr);
                }
            },
            py::arg("actions"), py::arg("entity_obs").noconvert(),
            py::arg("critic_obs").noconvert(),
            py::arg("features").noconvert(),
            py::arg("terminated").noconvert(),
            py::arg("truncated").noconvert(),
            "Advance every env one decision step. actions is "
            "[num_envs*6*POOL_ACTION_DIM] float32 team-relative controls; "
            "outputs are written into the caller-owned buffers. The GIL is "
            "released for the duration.")
        .def(
            "env_episode_over",
            [](const xushi2::pool::SimPool& pool, std::uint32_t env) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                return pool.env_episode_over(env);
            },
            py::arg("env"))
        .def(
            "env_state_hash",
            [](const xushi2::pool::SimPool& pool, std::uint32_t env) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                return pool.env_state_hash(env);
            },
            py::arg("env"))
        .def(
            "env_obs_state_hash",
            [](const xushi2::pool::SimPool& pool, std::uint32_t env) {
                if (env >= pool.num_envs()) {
                    throw std::invalid_argument("env index out of range");
                }
                return pool.env_obs_state_hash(env);
            },
            py::arg("env"));
}
