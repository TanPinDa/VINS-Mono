/**
 * @file pose_graph_node.cpp
 * @brief
 * @date 06-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include "pose_graph_ros/pose_graph_node.hpp"

#include <fstream>

#include <geometry_msgs/PointStamped.h>
#include <opencv2/opencv.hpp>
#include <ros/package.h>

namespace pose_graph {
PoseGraphNode::PoseGraphNode() {
  if (Initialize()) {
    ROS_INFO("PoseGraphNode initialized");
  }
}

PoseGraphNode::~PoseGraphNode() {
  ros::shutdown();
  if (measurement_thread_.joinable()) {
    measurement_thread_.join();
  }
  if (keyboard_command_thread_.joinable()) {
    keyboard_command_thread_.join();
  }
}

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
      ROS_INFO("Loading previous pose graph");
      std::lock_guard<std::mutex> lock(process_mutex_);
      pose_graph_service_->LoadPoseGraph();
    } else {
      ROS_INFO("Not loading any previous pose graph");
    }
  }

  posegraph_visualization_ =
      std::make_unique<CameraPoseVisualization>(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization_->setScale(0.1);
  posegraph_visualization_->setLineWidth(0.01);

  StartPublishersAndSubscribers();
  StartBackgroundThreads();
}

bool PoseGraphNode::ReadParameters() {
  nh_.param<std::string>("config_file", config_file_path_,
                         "/config/config.yaml");
  nh_.param<int>("visualization_shift_x", visualization_shift_x_, 0);
  nh_.param<int>("visualization_shift_y", visualization_shift_y_, 0);
  nh_.param<int>("skip_cnt", skip_cnt_threshold_, 0);
  nh_.param<double>("skip_dis", skip_distance_, 0.0);

  ROS_INFO(
      "Loaded parameters: config_file: %s, visualization_shift_x: %d, "
      "visualization_shift_y: %d, skip_cnt: %d, skip_dis: %f",
      config_file_path_.c_str(), visualization_shift_x_, visualization_shift_y_,
      skip_cnt_threshold_, skip_distance_);

  return true;
}

void PoseGraphNode::StartPublishersAndSubscribers() {
  sub_imu_forward_ = nh_.subscribe("/vins_estimator/imu_propagate", 2000,
                                   std::bind(&PoseGraphNode::ImuForwardCallback,
                                             this, std::placeholders::_1));
  sub_vio_ = nh_.subscribe(
      "/vins_estimator/odometry", 2000,
      std::bind(&PoseGraphNode::VioCallback, this, std::placeholders::_1));
  sub_image_ = nh_.subscribe(
      IMAGE_TOPIC, 2000,
      std::bind(&PoseGraphNode::ImageCallback, this, std::placeholders::_1));
  sub_pose_ = nh_.subscribe(
      "/vins_estimator/keyframe_pose", 2000,
      std::bind(&PoseGraphNode::PoseCallback, this, std::placeholders::_1));
  sub_extrinsic_ = nh_.subscribe("/vins_estimator/extrinsic", 2000,
                                 std::bind(&PoseGraphNode::ExtrinsicCallback,
                                           this, std::placeholders::_1));
  sub_point_ = nh_.subscribe(
      "/vins_estimator/keyframe_point", 2000,
      std::bind(&PoseGraphNode::PointCallback, this, std::placeholders::_1));
  sub_relo_relative_pose_ =
      nh_.subscribe("/vins_estimator/relo_relative_pose", 2000,
                    std::bind(&PoseGraphNode::ReloRelativePoseCallback, this,
                              std::placeholders::_1));

  pub_match_img_ = nh_.advertise<sensor_msgs::Image>("match_image", 1000);
  pub_camera_pose_visual_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "camera_pose_visual", 1000);
  pub_key_odometrys_ =
      nh_.advertise<visualization_msgs::Marker>("key_odometrys", 1000);
  pub_vio_path_ = nh_.advertise<nav_msgs::Path>("no_loop_path", 1000);
  pub_match_points_ =
      nh_.advertise<sensor_msgs::PointCloud>("match_points", 100);
  pub_pg_path_ = nh_.advertise<nav_msgs::Path>("pose_graph_path", 1000);
  pub_base_path_ = nh_.advertise<nav_msgs::Path>("base_path", 1000);
  pub_pose_graph_ =
      nh_.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
  for (int i = 1; i < 10; i++)
    pub_path_[i] = nh_.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
}

void PoseGraphNode::StartBackgroundThreads() {
  measurement_thread_ = std::thread(std::bind(&PoseGraphNode::Process, this));
  keyboard_command_thread_ =
      std::thread(std::bind(&PoseGraphNode::Command, this));
}

void PoseGraphNode::Process() {
  if (!loop_closure_) {
    return;
  }

  while (ros::ok()) {
    sensor_msgs::ImageConstPtr image_msg = NULL;
    sensor_msgs::PointCloudConstPtr point_msg = NULL;
    nav_msgs::Odometry::ConstPtr pose_msg = NULL;

    // find out the messages with same time stamp
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      if (!image_buffer_.empty() && !point_buffer_.empty() &&
          !pose_buffer_.empty()) {
        if (image_buffer_.front()->header.stamp.toSec() >
            pose_buffer_.front()->header.stamp.toSec()) {
          pose_buffer_.pop();
          printf("throw pose at beginning\n");
        } else if (image_buffer_.front()->header.stamp.toSec() >
                   point_buffer_.front()->header.stamp.toSec()) {
          point_buffer_.pop();
          printf("throw point at beginning\n");
        } else if (image_buffer_.back()->header.stamp.toSec() >=
                       pose_buffer_.front()->header.stamp.toSec() &&
                   point_buffer_.back()->header.stamp.toSec() >=
                       pose_buffer_.front()->header.stamp.toSec()) {
          pose_msg = pose_buffer_.front();
          pose_buffer_.pop();
          while (!pose_buffer_.empty()) pose_buffer_.pop();
          while (image_buffer_.front()->header.stamp.toSec() <
                 pose_msg->header.stamp.toSec())
            image_buffer_.pop();
          image_msg = image_buffer_.front();
          image_buffer_.pop();

          while (point_buffer_.front()->header.stamp.toSec() <
                 pose_msg->header.stamp.toSec())
            point_buffer_.pop();
          point_msg = point_buffer_.front();
          point_buffer_.pop();
        }
      }
    }

    if (pose_msg != NULL) {
      // printf(" pose time %f \n", pose_msg->header.stamp.toSec());
      // printf(" point time %f \n", point_msg->header.stamp.toSec());
      // printf(" image time %f \n", image_msg->header.stamp.toSec());
      //  skip first few messages
      if (skip_first_cnt_ < skip_first_cnt_threshold) {
        skip_first_cnt_++;
        continue;
      }

      if (skip_cnt_ < skip_cnt_threshold_) {
        skip_cnt_++;
        continue;
      } else {
        skip_cnt_ = 0;
      }

      cv_bridge::CvImageConstPtr ptr;
      if (image_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = image_msg->header;
        img.height = image_msg->height;
        img.width = image_msg->width;
        img.is_bigendian = image_msg->is_bigendian;
        img.step = image_msg->step;
        img.data = image_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      } else
        ptr =
            cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

      cv::Mat image = ptr->image;
      // build keyframe
      Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                            pose_msg->pose.pose.position.y,
                            pose_msg->pose.pose.position.z);
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                               pose_msg->pose.pose.orientation.x,
                               pose_msg->pose.pose.orientation.y,
                               pose_msg->pose.pose.orientation.z)
                       .toRotationMatrix();
      if ((T - last_t_).norm() > skip_distance_) {
        std::vector<cv::Point3f> point_3d;
        std::vector<cv::Point2f> point_2d_uv;
        std::vector<cv::Point2f> point_2d_normal;
        std::vector<double> point_id;

        for (unsigned int i = 0; i < point_msg->points.size(); i++) {
          cv::Point3f p_3d;
          p_3d.x = point_msg->points[i].x;
          p_3d.y = point_msg->points[i].y;
          p_3d.z = point_msg->points[i].z;
          point_3d.push_back(p_3d);

          cv::Point2f p_2d_uv, p_2d_normal;
          double p_id;
          p_2d_normal.x = point_msg->channels[i].values[0];
          p_2d_normal.y = point_msg->channels[i].values[1];
          p_2d_uv.x = point_msg->channels[i].values[2];
          p_2d_uv.y = point_msg->channels[i].values[3];
          p_id = point_msg->channels[i].values[4];
          point_2d_normal.push_back(p_2d_normal);
          point_2d_uv.push_back(p_2d_uv);
          point_id.push_back(p_id);

          // printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
        }

        std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(
            pose_msg->header.stamp.toSec(), frame_index_, T, R, image, point_3d,
            point_2d_uv, point_2d_normal, point_id, sequence_index_,
            config_.image_rows, config_.image_cols,
            config_.brief_pattern_file_path);
        {
          std::lock_guard<std::mutex> lock(process_mutex_);
          pose_graph_service_->AddKeyFrame(keyframe);
          {
            int sequence_cnt = posegraph.getCurrentSequenceCount();
            auto kf_attribute = keyframe->getAttributes();
            Eigen::Quaterniond quarternion{kf_attribute.rotation};
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time(keyframe->time_stamp);
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose.position.x =
                kf_attribute.position.x() + visualization_shift_x_;
            pose_stamped.pose.position.y =
                kf_attribute.position.y() + visualization_shift_y_;
            pose_stamped.pose.position.z = kf_attribute.position.z();
            pose_stamped.pose.orientation.x = quarternion.x();
            pose_stamped.pose.orientation.y = quarternion.y();
            pose_stamped.pose.orientation.z = quarternion.z();
            pose_stamped.pose.orientation.w = quarternion.w();
            path_[sequence_cnt].poses.push_back(pose_stamped);
            path_[sequence_cnt].header = pose_stamped.header;

            if (save_loop_path) {
              std::ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
              loop_path_file.setf(ios::fixed, ios::floatfield);
              loop_path_file.precision(0);
              loop_path_file << keyframe->time_stamp * 1e9 << ",";
              loop_path_file.precision(5);
              loop_path_file << P.x() << "," << P.y() << "," << P.z() << ","
                             << Q.w() << "," << Q.x() << "," << Q.y() << ","
                             << Q.z() << "," << endl;
              loop_path_file.close();
            }
          }
          Publish();
        }
        frame_index_++;
        last_t_ = T;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void PoseGraphNode::Command() {
  if (!loop_closure_) return;
  while (ros::ok()) {
    char c = getchar();
    if (c == 's') {
      {
        std::lock_guard<std::mutex> lock(process_mutex_);
        pose_graph_service_->SavePoseGraph();
      }
      // printf(
      //     "save pose graph finish\nyou can set 'load_previous_pose_graph' to
      //     1 " "in the config file to reuse it next time\n");
      ROS_INFO(
          "save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 "
          "in the config file to reuse it next time");
      // printf("program shutting down...\n");
      // ros::shutdown();
    }
    if (c == 'n') new_sequence();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void PoseGraphNode::Publish() {
  int sequence_cnt = posegraph.getCurrentSequenceCount();
  for (int i = 1; i <= sequence_cnt; i++) {
    pub_pg_path_.publish(path[i]);
    pub_path_[i].publish(path[i]);
    posegraph_visualization_->publish_by(pub_pose_graph_,
                                         path_[sequence_cnt].header);
  }
  base_path_.header.frame_id = "world";
  pub_base_path_.publish(base_path_);
}

void PoseGraphNode::OnPoseGraphLoaded() {
  // TODO: Implement this
}

void PoseGraphNode::OnPoseGraphSaved() {
  // TODO: Implement this
}
void PoseGraphNode::OnKeyFrameAdded(KeyFrame::Attributes kf_attribute) {
  // TODO: Implement this
}
void PoseGraphNode::OnKeyFrameConnectionFound(
    KeyFrame::Attributes current_kf_attribute,
    KeyFrame::Attributes old_kf_attribute,
    std::vector<cv::Point2f> matched_2d_old_norm,
    std::vector<double> matched_id) {
  // TODO: Implement this
}
void PoseGraphNode::OnPoseGraphOptimization(
    std::vector<KeyFrame::Attributes> kf_attributes) {
  // TODO: Implement this
}
void PoseGraphNode::OnNewSequentialEdge(Vector3d p1, Vector3d p2) {
  // TODO: Implement this
}
void PoseGraphNode::OnNewLoopEdge(Vector3d p1, Vector3d p2) {
  // TODO: Implement this
}

void PoseGraphNode::ImuForwardCallback(const nav_msgs::OdometryConstPtr& msg) {
  // TODO: Implement this
}

void PoseGraphNode::VioCallback(const nav_msgs::OdometryConstPtr& msg) {
  // TODO: Implement this
}

void PoseGraphNode::ImageCallback(const sensor_msgs::ImageConstPtr& msg) {
  // TODO: Implement this
}

void PoseGraphNode::PoseCallback(const nav_msgs::OdometryConstPtr& msg) {
  // TODO: Implement this
}

void PoseGraphNode::ExtrinsicCallback(const nav_msgs::OdometryConstPtr& msg) {
  // TODO: Implement this
}

void PoseGraphNode::PointCallback(const sensor_msgs::PointCloudConstPtr& msg) {
  // TODO: Implement this
}

void PoseGraphNode::ReloRelativePoseCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  // TODO: Implement this
}
}  // namespace pose_graph

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_graph_node");

  pose_graph::PoseGraphNode pose_graph_node;
  ros::spin();
  return 0;
}