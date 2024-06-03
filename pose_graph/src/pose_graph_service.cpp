/**
 * @file pose_graph_service.cpp
 * @brief
 * @date 04-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include "pose_graph/pose_graph_service.hpp"

namespace pose_graph {
PoseGraphService::PoseGraphService(const PoseGraphConfig &config)
    : pose_graph_(std::make_unique<PoseGraph>(config)) {}

void PoseGraphService::LoadPoseGraph() {
  pose_graph_->Load();
  OnPoseGraphLoaded();
}

void PoseGraphService::SavePoseGraph() {
  pose_graph_->Save();
  OnPoseGraphSaved();
}

void PoseGraphService::AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe) {
  pose_graph_->AddKeyFrame(current_keyframe);
  OnKeyFrameAdded(current_keyframe);
}

int PoseGraphService::GetCurrentPoseGraphSequenceCount() const {
  return pose_graph_->GetCurrentSequenceCount();
}

void PoseGraphService::UpdateKeyFrameLoop(
    const int &index, const Eigen::Matrix<double, 8, 1> &loop_info) {
  pose_graph_->UpdateKeyFrameLoop(index, loop_info);
}

void PoseGraphService::StartOptimizationThread() {
  // Start optimization thread
  optimization_thread_ = std::thread([this]() {
    while (true) {
      // Perform optimization
      pose_graph_->Optimize4DoF();
      OnPoseGraphOptimization();
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
  });
}
}  // namespace pose_graph
