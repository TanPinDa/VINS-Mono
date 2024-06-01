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
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>

#include "DBoW/DBoW2.h"
#include "pose_graph/keyframe.h"

namespace pose_graph {
struct PoseGraphConfig {
  bool detect_loop_closure = true;
  std::string vocabulary_path = "";
  std::string saved_pose_graph_dir = "";
  bool save_debug_image = false;
};

class PoseGraph {
 public:
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
  ~PoseGraph() = default;

  std::vector<std::shared_ptr<KeyFrame>> Load();
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
  std::mutex keyframes_mutex_;
  Pose world_vio_;
  std::atomic<int> current_sequence_count_;
  std::map<int, cv::Mat> image_pool_;
  int earliest_loop_index = -1;
  int base_sequence = 1;

  BriefDatabase db_;
	std::unique_ptr<BriefVocabulary> vocabulary_;
};
}  // namespace pose_graph

#endif /* DETAILS_POSE_GRAPH_HPP */
