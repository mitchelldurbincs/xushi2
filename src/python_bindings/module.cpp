// pybind11 bindings. Exposes the sim to the Python trainer.
// Keep this layer *thin* — only adapt C++ types to Python, never put game
// logic here.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <xushi2/bots/bot.h>
#include <xushi2/bots/runner.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_utils.h>
#include <xushi2/sim/sim.h>

namespace py = pybind11;

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
        .def(py::init<const xushi2::sim::MatchConfig&>())
        .def("reset", py::overload_cast<>(&xushi2::sim::Sim::reset))
        .def("reset", py::overload_cast<std::uint64_t>(&xushi2::sim::Sim::reset),
             py::arg("seed"))
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
            static const std::array<std::string, 4> kValidNames{
                {"walk_to_objective", "hold_and_shoot", "basic", "noop"}};
            bool valid = false;
            for (const auto& n : kValidNames) {
                if (bot_name == n) { valid = true; break; }
            }
            if (!valid) {
                throw std::invalid_argument(
                    "unknown bot_name; valid: walk_to_objective, "
                    "hold_and_shoot, basic, noop");
            }
            auto bot = xushi2::bots::make_bot_by_name(bot_name);
            return bot->decide(sim.state(), agent_slot);
        },
        py::arg("sim"), py::arg("agent_slot"), py::arg("bot_name"),
        "Return the Action the named scripted bot would emit for "
        "agent_slot on the current sim state.");

    m.def("run_scripted_episode",
          [](const xushi2::sim::MatchConfig& config, const std::string& bot_a,
             const std::string& bot_b) {
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
            xushi2::sim::build_critic_obs(
                sim, team_perspective, out.mutable_data(0),
                static_cast<std::uint32_t>(out.shape(0)));
        },
        py::arg("sim"), py::arg("team_perspective"), py::arg("out"),
        "Write the Phase-4 critic observation (kCriticObsDim floats) for a "
        "team perspective into `out`. Requires team_size == 3.");
}
