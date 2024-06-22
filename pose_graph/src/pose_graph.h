/**
 * @file pose_graph.hpp
 * @brief
 * @date 30-05-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef POSE_GRAPH_POSE_GRAPH_HPP
#define POSE_GRAPH_POSE_GRAPH_HPP

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include "camodocal/camera_models/CameraFactory.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "DBoW/DBoW2.h"
#include "pose_graph/details/keyframe.h"
#include "pose_graph/details/pose_graph_event_observer.hpp"

namespace pose_graph {
struct PoseGraphConfig {
  bool detect_loop_closure = true;
  std::string vocabulary_path = "";
  std::string saved_pose_graph_dir = "";
  std::string brief_pattern_file_path = "";
  bool save_debug_image = false;
  bool fast_relocalization = false;
  int image_height;
  int image_width;
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
  ~PoseGraph();

  void Initialize(const PoseGraphConfig& config, camodocal::CameraPtr camera);
  void RegisterEventObserver(
      std::shared_ptr<PoseGraphEventObserver> event_observer);
  bool Load();
  void Save();
  void AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe);
  void UpdateKeyFrameLoop(int index,
                          const Eigen::Matrix<double, 8, 1>& loop_info);
  void UpdateImuCameraPose(const Pose& imu_camera_pose);
  int GetCurrentSequenceCount() const;
  Drift GetDrift() const;
  Pose GetWorldVio() const;

 private:
  KeyFrame::Attributes GetKeyFrameAttribute(int index) const;
  std::vector<KeyFrame::Attributes> GetKeyFrameAttributes() const;
  int DetectLoopClosure(std::shared_ptr<KeyFrame> current_keyframe);
  std::shared_ptr<KeyFrame> GetKeyFrame(int index);
  void LoadVocabulary();
  void AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe,
                   int& old_keyframe_loop_index,
                   std::vector<cv::Point2f>& matched_2d_old_norm,
                   std::vector<double>& matched_id);
  void AddKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe);
  void LoadKeyFrame(std::shared_ptr<KeyFrame> current_keyframe,
                    KeyFrame* old_keyframe,
                    std::vector<cv::Point2f>& matched_2d_old_norm,
                    std::vector<double>& matched_id);
  bool LoadSingleConfigEntry(FILE* pFile,
                             KeyFrame::Attributes& old_kf_attribute,
                             KeyFrame::Attributes& current_kf_attribute,
                             std::vector<cv::Point2f>& matched_2d_old_norm,
                             std::vector<double>& matched_id,
                             cv::Mat& current_kf_thumb_image);
  Pose GetImuCameraPose() const;
  void Optimize4DoF();
  void StartOptimizationThread();

 private:
  std::shared_ptr<PoseGraphEventObserver> event_observer_;
  PoseGraphConfig config_;
  camodocal::CameraPtr camera_;  // Note: internally it uses a shared pointer.
  Drift drift_;
  std::mutex drift_mutex_;
  std::list<std::shared_ptr<KeyFrame>> keyframes_;
  mutable std::mutex keyframes_mutex_;
  Pose world_vio_;
  Pose imu_camera_pose_;
  mutable std::mutex imu_camera_pose_mutex_;
  std::atomic<int> current_sequence_count_ = 0;
  std::map<int, cv::Mat> image_pool_;
  int earliest_loop_index = -1;
  int base_sequence = 1;
  std::vector<bool> sequence_loop_flags_ = {
      false};  // TODO: check what this means
  int global_keyframe_index_counter_ = 0;

  // TODO: try to replace queue buffer with a single int
  std::queue<int> optimize_buf_;
  std::mutex optimize_buf_mutex_;

  BriefDatabase db_;
  std::unique_ptr<BriefVocabulary> vocabulary_;
  std::atomic<bool> keep_running_{true};
  std::thread optimization_thread_;
};
}  // namespace pose_graph

#endif /* POSE_GRAPH_POSE_GRAPH_HPP */
