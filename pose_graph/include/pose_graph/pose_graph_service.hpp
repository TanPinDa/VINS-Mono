/**
 * @file pose_graph_service.hpp
 * @brief
 * @date 02-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef POSE_GRAPH_POSE_GRAPH_SERVICE_HPP
#define POSE_GRAPH_POSE_GRAPH_SERVICE_HPP

#include "pose_graph/details/pose_graph.hpp"

namespace pose_graph {
class PoseGraphService {
 public:
  PoseGraphService(const PoseGraphConfig& config);
  ~PoseGraphService() = default;

  void LoadPoseGraph();
  void SavePoseGraph();

  void UpdateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1>& loop_info);

 private:
  std::unique_ptr<PoseGraph> pose_graph_;
};
}  // namespace pose_graph
#endif /* POSE_GRAPH_POSE_GRAPH_SERVICE_HPP */
