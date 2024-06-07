/**
 * @file pose_graph_service.cpp
 * @brief
 * @date 04-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include "pose_graph/pose_graph_service.hpp"

namespace pose_graph {
PoseGraphService::PoseGraphService(PoseGraphConfig &config)
    : pose_graph_(std::make_unique<PoseGraph>(config)), config_(config) {}

PoseGraphService::~PoseGraphService() {
  keep_running_ = false;
  if (optimization_thread_.joinable()) {
    optimization_thread_.join();
  }
}

bool PoseGraphService::LoadPoseGraph() {
  // Load previously saved pose graph from file
  TicToc clock;
  FILE *pFile;
  std::string file_path = config_.saved_pose_graph_dir + "pose_graph.txt";
  printf("loading pose graph from: %s \n", file_path.c_str());
  printf("pose graph loading...\n");
  pFile = fopen(file_path.c_str(), "r");
  if (pFile == NULL) {
    printf(
        "load previous pose graph error: wrong previous pose graph path or no "
        "previous pose graph \n the system will start with new pose graph \n");
    return false;
  }

  KeyFrame::Attributes old_kf_attribute;
  KeyFrame::Attributes current_kf_attribute;
  std::vector<cv::Point2f> matched_2d_old_norm;
  std::vector<double> matched_id;
  while (pose_graph_->LoadSingleConfigEntry(pFile, old_kf_attribute,
                                            current_kf_attribute,
                                            matched_2d_old_norm, matched_id)) {
    if (old_kf_attribute.time_stamp >= 0.0) {
      OnKeyFrameConnectionFound(current_kf_attribute, old_kf_attribute,
                                matched_2d_old_norm, matched_id);
    }
  }

  fclose(pFile);
  printf("pose graph loaded, time cost: %f s\n", clock.toc() / 1000);

  // Generic callback
  OnPoseGraphLoaded();

  return true;
}

void PoseGraphService::SavePoseGraph() {
  pose_graph_->Save();
  OnPoseGraphSaved();
}

void PoseGraphService::AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe) {
  KeyFrame *old_keyframe = nullptr;
  std::vector<cv::Point2f> matched_2d_old_norm;
  std::vector<double> matched_id;
  pose_graph_->AddKeyFrame(current_keyframe, old_keyframe, matched_2d_old_norm,
                           matched_id);

  if (old_keyframe != nullptr) {
    OnKeyFrameConnectionFound(current_keyframe->getAttributes(),
                              old_keyframe->getAttributes(),
                              matched_2d_old_norm, matched_id);
  }

  // Show sequential edge
  auto attributes = pose_graph_->GetKeyFrameAttributes();
  std::vector<KeyFrame::Attributes>::reverse_iterator rit = attributes.rbegin();
  Vector3d P;
  Matrix3d R;
  current_keyframe->getPose(P, R);
  for (int i = 0; i < 4; i++) {
    if (rit == attributes.rend()) break;
    Vector3d connected_P;
    Matrix3d connected_R;
    if ((*rit).sequence == current_keyframe->sequence) {
      OnNewSequentialEdge(P, (*rit).position);
    }
    rit++;
  }

  // Show loop edge
  if (current_keyframe->has_loop) {
    // printf("has loop \n");
    KeyFrame::Attributes connected_kf_attributes =
        pose_graph_->GetKeyFrameAttribute(current_keyframe->loop_index);
    Vector3d P0;
    Matrix3d R0;
    // current_keyframe->getVioPose(P0, R0);
    current_keyframe->getPose(P0, R0);
    if (current_keyframe->sequence > 0) {
      // printf("add loop into visual \n");
      OnNewLoopEdge(
          P0, connected_kf_attributes.position +
                  Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));

      // posegraph_visualization->add_loopedge(P0, connected_P +
      // Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
    }
  }

  // Generic callback
  OnKeyFrameAdded(current_keyframe->getAttributes());
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
    while (keep_running_) {
      // Perform optimization
      pose_graph_->Optimize4DoF();
      auto kf_attributes = pose_graph_->GetKeyFrameAttributes();
      OnPoseGraphOptimization(kf_attributes);
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
  });
}
}  // namespace pose_graph
