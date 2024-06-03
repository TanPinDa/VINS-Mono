/**
 * @file pose_graph_service.hpp
 * @brief
 * @date 02-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef POSE_GRAPH_POSE_GRAPH_SERVICE_HPP
#define POSE_GRAPH_POSE_GRAPH_SERVICE_HPP

#include <memory>
#include <thread>

#include <Eigen/Core>

#include "pose_graph/details/pose_graph.hpp"

namespace pose_graph {
class PoseGraphService {
 public:
  PoseGraphService(const PoseGraphConfig& config);
  ~PoseGraphService() = default;

  void LoadPoseGraph();
  void SavePoseGraph();
  void AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe);
  int GetCurrentPoseGraphSequenceCount() const;

  void UpdateKeyFrameLoop(const int& index,
                          const Eigen::Matrix<double, 8, 1>& loop_info);

 private:
  void StartOptimizationThread();
  virtual void OnPoseGraphLoaded() = 0;
  virtual void OnPoseGraphSaved() = 0;
  virtual void OnKeyFrameAdded(std::shared_ptr<KeyFrame> current_keyframe) = 0;
  virtual void OnPoseGraphOptimization() = 0;
  std::unique_ptr<PoseGraph> pose_graph_;
  std::thread optimization_thread_;
};
}  // namespace pose_graph
#endif /* POSE_GRAPH_POSE_GRAPH_SERVICE_HPP */
