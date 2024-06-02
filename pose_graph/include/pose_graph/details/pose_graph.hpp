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
#include <vector>

#include <opencv2/opencv.hpp>

#include "DBoW/DBoW2.h"
#include "pose_graph/keyframe.h"

namespace pose_graph {
struct PoseGraphConfig {
  bool detect_loop_closure = true;
  std::string vocabulary_path = "";
  std::string saved_pose_graph_dir = "";
  bool save_debug_image = false;
  bool fast_relocalization = false;
};

class PoseGraph {
 public:
  struct Pose {
    Eigen::Vector3d translation = Eigen::Vector3d(0, 0, 0);
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  };
  struct Drift {
    Eigen::Vector3d translation = Eigen::Vector3d(0, 0, 0);
    double yaw = 0.0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  };

 public:
  PoseGraph(const PoseGraphConfig& config);
  ~PoseGraph() = default;

  std::vector<std::shared_ptr<KeyFrame>> Load();
  void Save();
  void AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe);
  void LoadKeyFrame(std::shared_ptr<KeyFrame> current_keyframe);
  void UpdateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1>& loop_info);
  int GetCurrentSequenceCount() const;
  Drift GetDrift() const;

 private:
  int DetectLoopClosure(std::shared_ptr<KeyFrame> current_keyframe);
  std::shared_ptr<KeyFrame> GetKeyFrame(int index);
  void LoadVocabulary();
  void AddKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe);
  void Optimize4DoF();

 private:
  PoseGraphConfig config_;
  Drift drift_;
  std::mutex drift_mutex_;
  std::list<std::shared_ptr<KeyFrame>> keyframes_;
  std::mutex keyframes_mutex_;
  Pose world_vio_;
  std::atomic<int> current_sequence_count_ = 0;
  std::map<int, cv::Mat> image_pool_;
  int earliest_loop_index = -1;
  int base_sequence = 1;
  std::vector<bool> sequence_loop_flags_ = {false}; // TODO: check what this means
  int global_keyframe_index_counter_ = 0;

  // TODO: try to replace queue buffer with a single int
  std::queue<int> optimize_buf_;
  std::mutex optimize_buf_mutex_;

  BriefDatabase db_;
  std::unique_ptr<BriefVocabulary> vocabulary_;
};
}  // namespace pose_graph

#endif /* DETAILS_POSE_GRAPH_HPP */
