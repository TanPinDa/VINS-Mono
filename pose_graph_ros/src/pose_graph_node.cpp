/**
 * @file pose_graph_node.cpp
 * @brief
 * @date 06-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include <memory>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "pose_graph/pose_graph_service.hpp"
#include "pose_graph_ros/utility/CameraPoseVisualization.h"

namespace pose_graph {
class PoseGraphNode : public PoseGraphService {
 public:
  PoseGraphNode();
  ~PoseGraphNode();

  void OnPoseGraphLoaded() override;
  void OnPoseGraphSaved() override;
  void OnKeyFrameAdded(KeyFrame::Attributes kf_attribute) override;
  void OnKeyFrameConnectionFound(KeyFrame::Attributes current_kf_attribute,
                                 KeyFrame::Attributes old_kf_attribute,
                                 std::vector<cv::Point2f> matched_2d_old_norm,
                                 std::vector<double> matched_id) override;
  void OnPoseGraphOptimization(
      std::vector<KeyFrame::Attributes> kf_attributes) override;
  void OnNewSequentialEdge(Vector3d p1, Vector3d p2) override;
  void OnNewLoopEdge(Vector3d p1, Vector3d p2) override;

 private:
  void Initialize();
  bool ReadParameters();
  void PublishAndSubscribe();
  void StartBackgroundThreads();

 private:
  ros::NodeHandle nh_("~");
  PoseGraphConfig config_;
  std::unique_ptr<PoseGraphService> pose_graph_service_;
  CameraPoseVisualization camera_pose_vis_;
  int loop_closure_;  // TODO: check if this should be a bool
  camodocal::CameraPtr camera_;
  bool visualise_imu_forward_;

  // ros parameters
  std::string config_file_path_;
  int visualization_shift_x_ = 0;
  int visualization_shift_y_ = 0;
  int skip_cnt_ = 0;
  double skip_distance_ = 0.0;
};

PoseGraphNode::PoseGraphNode() {
  if (Initialize()) {
    ROS_INFO("PoseGraphNode initialized");
  }
}

PoseGraphNode::~PoseGraphNode() { ros::shutdown(); }

void PoseGraphNode::Initialize() {
  if (!ReadParameters()) {
    ROS_ERROR("Failed to read parameters");
    ros::shutdown();
    return;
  }

  // Load config file
  cv::FileStorage fs(config_file_path_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    ROS_ERROR("Failed to open config file: %s", config_file_path_.c_str());
    ros::shutdown();
    return;
  }

  double camera_visual_size = fs["visualize_camera_size"];
  camera_pose_vis_.setScale(camera_visual_size);
  camera_pose_vis_.setLineWidth(camera_visual_size / 10.0);

  loop_closure_ = fs["loop_closure"];
  std::string image_topic;
  bool load_previous_pose_graph;
  if (loop_closure_) {
    config_.image_rows = fs["image_height"];
    config_.image_cols = fs["image_width"];
    std::string pkg_path = ros::package::getPath("pose_graph_ros");
    config_.vocabulary_path = pkg_path + "/../support_files/brief_k10L6.bin";
    config_.brief_pattern_file_path =
        pkg_path + "/../support_files/brief_pattern.yml";

    camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
        config_file_path_.c_str());
    image_topic = fs["image_topic"];
    config_.saved_pose_graph_dir = fs["pose_graph_save_path"];
    if (fs["save_image"] == 0) {
      config_.save_debug_image = false;
    } else {
      config_.save_debug_image = true;
    }

    std::string vins_result_path = fs["output_path"];
    FileSystemHelper::createDirectoryIfNotExists(config_.saved_pose_graph_dir);
    FileSystemHelper::createDirectoryIfNotExists(vins_result_path);

    visualise_imu_forward_ = fs["visualise_imu_forward"];
    load_previous_pose_graph = fs["load_previous_pose_graph"];
    config_.fast_relocalization = fs["fast_relocalization"];
    vins_result_path = vins_result_path + "/vins_result_loop.csv";
    std::ofstream fout(vins_result_path, std::ios::out);
    fout.close();
  }
  fs.release();

  pose_graph_service_ = std::make_unique<PoseGraphService>(config_);

  if (loop_closure_) {
    if (load_previous_pose_graph) {
      // TODO
      //   printf("load pose graph\n");
      //   m_process.lock();
      //   posegraph.loadPoseGraph();
      //   m_process.unlock();
      //   printf("load pose graph finish\n");
      //   load_flag = 1;
    } else {
      //   printf("no previous pose graph\n");
      //   load_flag = 1;
    }
  }

  pose_graph_service_->LoadPoseGraph();
  PublishAndSubscribe();
  StartBackgroundThreads();
}

bool PoseGraphNode::ReadParameters() {
  nh_.param<std::string>("config_file", config_file_path_,
                         "/config/config.yaml");
  nh_.param<int>("visualization_shift_x", visualization_shift_x_, 0);
  nh_.param<int>("visualization_shift_y", visualization_shift_y_, 0);
  nh_.param<int>("skip_cnt", skip_cnt_, 0);
  nh_.param<double>("skip_dis", skip_distance_, 0.0);

  ROS_INFO(
      "Loaded parameters: config_file: %s, visualization_shift_x: %d, "
      "visualization_shift_y: %d, skip_cnt: %d, skip_dis: %f",
      config_file_path_.c_str(), visualization_shift_x_, visualization_shift_y_,
      skip_cnt_, skip_distance_);

  return true;
}

}  // namespace pose_graph

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_graph_node");

  pose_graph::PoseGraphNode pose_graph_node;
  ros::spin();
  return 0;
}