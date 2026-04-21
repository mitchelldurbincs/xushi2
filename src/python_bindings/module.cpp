// pybind11 bindings. Exposes the sim to the Python trainer.
// Keep this layer *thin* — only adapt C++ types to Python, never put game
// logic here.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xushi2/bots/runner.h>
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

    py::class_<xushi2::common::Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("move_x", &xushi2::common::Action::move_x)
        .def_readwrite("move_y", &xushi2::common::Action::move_y)
        .def_readwrite("aim_delta", &xushi2::common::Action::aim_delta)
        .def_readwrite("primary_fire", &xushi2::common::Action::primary_fire)
        .def_readwrite("ability_1", &xushi2::common::Action::ability_1)
        .def_readwrite("ability_2", &xushi2::common::Action::ability_2)
        .def_readwrite("target_slot", &xushi2::common::Action::target_slot);

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

    py::class_<xushi2::sim::MatchConfig>(m, "MatchConfig")
        .def(py::init<>())
        .def_readwrite("seed", &xushi2::sim::MatchConfig::seed)
        .def_readwrite("round_length_seconds",
                       &xushi2::sim::MatchConfig::round_length_seconds)
        .def_readwrite("fog_of_war_enabled",
                       &xushi2::sim::MatchConfig::fog_of_war_enabled)
        .def_readwrite("randomize_map", &xushi2::sim::MatchConfig::randomize_map)
        .def_readwrite("action_repeat", &xushi2::sim::MatchConfig::action_repeat)
        .def_readwrite("mechanics", &xushi2::sim::MatchConfig::mechanics);

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
        .def_property_readonly("state_hash", &xushi2::sim::Sim::state_hash);

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
}
