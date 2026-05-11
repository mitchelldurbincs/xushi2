#pragma once

namespace viewer_layout {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;

// Layout constants shared by arena transform and right-hand panel rendering.
// Keep these in sync so world->screen mapping and panel origin stay aligned.
constexpr int kArenaPx = 720;
constexpr int kArenaMarginPx = 12;
constexpr int kPanelX = kArenaPx;

}  // namespace viewer_layout
