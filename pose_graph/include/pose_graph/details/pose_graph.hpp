/**
 * @file pose_graph.hpp
 * @brief
 * @date 30-05-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef DETAILS_POSE_GRAPH_HPP
#define DETAILS_POSE_GRAPH_HPP

#include <atomic>
#include <memory>
#include <string>

#include "pose_graph/keyframe.h"

namespace pose_graph {
class PoseGraph {
 public:
  struct PoseGraphConfig {
    bool detect_loop_closure = true;
    std::string vocabulary_path = "";
  };
  struct Pose {
    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
  };
  struct Drift {
    Eigen::Vector3d translation;
    double yaw;
    Eigen::Matrix3d rotation;
  };

 public:
  PoseGraph(const PoseGraphConfig& config);
  ~PoseGraph();

  void Load();
  void Save();
  void AddKeyFrame(std::shared_ptr<KeyFrame> current_kf);
  void LoadKeyFrame(std::shared_ptr<KeyFrame> current_kf);
  int GetCurrentSequenceCount() const;
  Drift GetDrift() const;

 private:
  int DetectLoopClosure(std::shared_ptr<KeyFrame> current_kf);
  void LoadVocabulary();
  void AddKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe, int frame_index);
	void Optimize4DoF();

 private:
  PoseGraphConfig config_;
  Drift drift_;
  std::list<std::shared_ptr<KeyFrame>> keyframes_;
  Pose world_vio_;
  std::atomic<int> current_sequence_count_;
};
}  // namespace pose_graph

#endif /* DETAILS_POSE_GRAPH_HPP */
