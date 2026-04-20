#include <xushi2/bots/bot.h>

namespace xushi2::bots {

namespace {

class NoopBot final : public IBot {
   public:
    explicit NoopBot(std::string n) : name_(std::move(n)) {}

    Action decide(const sim::MatchState& /*state*/, int /*agent_index*/) override {
        return Action{};  // all zeros — no movement, no fire
    }

    std::string name() const override { return name_; }

   private:
    std::string name_;
};

}  // namespace

// Phase-0 stubs: all three return a no-op bot. Real scripted logic lands
// once the sim tick pipeline is implemented.

std::unique_ptr<IBot> make_walk_to_objective_bot() {
    return std::make_unique<NoopBot>("walk_to_objective (stub)");
}

std::unique_ptr<IBot> make_hold_and_shoot_bot() {
    return std::make_unique<NoopBot>("hold_and_shoot (stub)");
}

std::unique_ptr<IBot> make_basic_bot() {
    return std::make_unique<NoopBot>("basic (stub)");
}

}  // namespace xushi2::bots
