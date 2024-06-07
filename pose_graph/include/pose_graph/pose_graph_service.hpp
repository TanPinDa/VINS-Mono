/**
 * @file pose_graph_service.hpp
 * @brief
 * @date 02-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef POSE_GRAPH_POSE_GRAPH_SERVICE_HPP
#define POSE_GRAPH_POSE_GRAPH_SERVICE_HPP

#include <atomic>
#include <memory>
#include <thread>

#include <Eigen/Core>

#include "pose_graph/details/pose_graph.hpp"

namespace pose_graph {
class PoseGraphService {
 public:
  PoseGraphService(PoseGraphConfig& config);
  ~PoseGraphService();

  bool LoadPoseGraph();
  void SavePoseGraph();
  void AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe);
  int GetCurrentPoseGraphSequenceCount() const;

  void UpdateKeyFrameLoop(const int& index,
                          const Eigen::Matrix<double, 8, 1>& loop_info);

 private:
  void StartOptimizationThread();
  virtual void OnPoseGraphLoaded() = 0;
  virtual void OnPoseGraphSaved() = 0;
  virtual void OnKeyFrameAdded(KeyFrame::Attributes kf_attribute) = 0;
  virtual void OnKeyFrameConnectionFound(
      KeyFrame::Attributes current_kf_attribute,
      KeyFrame::Attributes old_kf_attribute,
      std::vector<cv::Point2f> matched_2d_old_norm,
      std::vector<double> matched_id) = 0;
  virtual void OnPoseGraphOptimization(
      std::vector<KeyFrame::Attributes> kf_attributes) = 0;
  virtual void OnNewSequentialEdge(Vector3d p1, Vector3d p2) = 0;
  virtual void OnNewLoopEdge(Vector3d p1, Vector3d p2) = 0;

  PoseGraphConfig& config_;
  std::unique_ptr<PoseGraph> pose_graph_;
  std::atomic<bool> keep_running_{true};
  std::thread optimization_thread_;
};
}  // namespace pose_graph
#endif /* POSE_GRAPH_POSE_GRAPH_SERVICE_HPP */
