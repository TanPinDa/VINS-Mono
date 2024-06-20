/**
 * @file pose_graph_node.cpp
 * @brief
 * @date 06-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include "pose_graph_ros/pose_graph_node.hpp"

#include <fstream>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <ros/package.h>

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true

namespace pose_graph {

PoseGraphNode::~PoseGraphNode() {
  ros::shutdown();
  if (measurement_thread_.joinable()) {
    measurement_thread_.join();
  }
  if (keyboard_command_thread_.joinable()) {
    keyboard_command_thread_.join();
  }
}

bool PoseGraphNode::Start() {
  if (!ReadParameters()) {
    ROS_ERROR("Failed to read parameters");
    ros::shutdown();
    return false;
  }

  // Load config file
  cv::FileStorage fs(config_file_path_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    ROS_ERROR("Failed to open config file: %s", config_file_path_.c_str());
    ros::shutdown();
    return false;
  }

  double camera_visual_size =
      fs["visualize_camera_size"];  // TODO: Consider splitting the visualiser
                                    // into a ros and non-ros segment. And the
                                    // non ros configs go in the "CV config
                                    // file"
  camera_pose_vis_ =
      std::make_unique<CameraPoseVisualization>(1.0, 0.0, 0.0, 1.0);
  camera_pose_vis_->setScale(camera_visual_size);
  camera_pose_vis_->setLineWidth(camera_visual_size / 10.0);

  posegraph_visualization_ =
      std::make_unique<CameraPoseVisualization>(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization_->setScale(0.1);
  posegraph_visualization_->setLineWidth(0.01);

  loop_closure_ = (int)fs["loop_closure"];

  bool load_previous_pose_graph = false;
  if (loop_closure_) {
    config_.image_height = fs["image_height"];
    config_.image_width = fs["image_width"];
    std::string pkg_path = ros::package::getPath("pose_graph_ros");
    // TODO: expose BoW vocabulary path as a ros parameter
    config_.vocabulary_path = pkg_path + "/../support_files/brief_k10L6.bin";
    config_.brief_pattern_file_path =
        pkg_path + "/../support_files/brief_pattern.yml";

    camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
        config_file_path_.c_str());
    image_topic_ = std::string(fs["image_topic"]);
    config_.saved_pose_graph_dir = std::string(fs["pose_graph_save_path"]);
    config_.save_debug_image = (int)fs["save_image"];

    vins_result_path_ = std::string(fs["output_path"]);
    FileSystemHelper::createDirectoryIfNotExists(
        config_.saved_pose_graph_dir.c_str());
    FileSystemHelper::createDirectoryIfNotExists(vins_result_path_.c_str());
    vins_result_path_ = vins_result_path_ + "/vins_result_loop.csv";

    visualise_imu_forward_ = (int)fs["visualise_imu_forward"];
    load_previous_pose_graph = (int)fs["load_previous_pose_graph"];
    config_.fast_relocalization = (int)fs["fast_relocalization"];

    std::ofstream fout(vins_result_path_, std::ios::out);

    if (load_previous_pose_graph) {
      ROS_INFO("Loading previous pose graph");
      std::lock_guard<std::mutex> lock(process_mutex_);
      pose_graph_.Load();
    } else {
      ROS_INFO("Not loading any previous pose graph");
    }
  }

  fs.release();

  StartPublishersAndSubscribers();

  pose_graph_.Initialize(config_, camera_);
  pose_graph_.RegisterEventObserver(PoseGraphEventObserver::shared_from_this());

  StartCommandAndProcessingThreads();
  return true;
}

bool PoseGraphNode::ReadParameters() {
  nh_.param<std::string>("config_file", config_file_path_);
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
  sub_imu_forward_ = nh_.subscribe<nav_msgs::Odometry>(
      "/vins_estimator/imu_propagate", 2000, &PoseGraphNode::ImuForwardCallback,
      this);
  sub_vio_ = nh_.subscribe<nav_msgs::Odometry>(
      "/vins_estimator/odometry", 2000, &PoseGraphNode::VioCallback, this);
  sub_image_ = nh_.subscribe<sensor_msgs::Image>(
      image_topic_, 2000, &PoseGraphNode::ImageCallback, this);
  sub_pose_ =
      nh_.subscribe<nav_msgs::Odometry>("/vins_estimator/keyframe_pose", 2000,
                                        &PoseGraphNode::PoseCallback, this);
  sub_extrinsic_ = nh_.subscribe<nav_msgs::Odometry>(
      "/vins_estimator/extrinsic", 2000, &PoseGraphNode::ExtrinsicCallback,
      this);
  sub_point_ = nh_.subscribe<sensor_msgs::PointCloud>(
      "/vins_estimator/keyframe_point", 2000, &PoseGraphNode::PointCallback,
      this);
  sub_relo_relative_pose_ = nh_.subscribe<nav_msgs::Odometry>(
      "/vins_estimator/relo_relative_pose", 2000,
      &PoseGraphNode::ReloRelativePoseCallback, this);

  pub_match_img_ = nh_.advertise<sensor_msgs::Image>("match_image", 2000);
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

void PoseGraphNode::StartCommandAndProcessingThreads() {
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
            config_.image_height, config_.image_width,
            config_.brief_pattern_file_path, config_.save_debug_image, camera_);
        {
          std::lock_guard<std::mutex> lock(process_mutex_);
          pose_graph_.AddKeyFrame(keyframe);
          {
            int sequence_cnt = pose_graph_.GetCurrentSequenceCount();
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
              std::ofstream loop_path_file(vins_result_path_, ios::app);
              loop_path_file.setf(ios::fixed, ios::floatfield);
              loop_path_file.precision(0);
              loop_path_file << kf_attribute.time_stamp * 1e9 << ",";
              loop_path_file.precision(5);
              loop_path_file << kf_attribute.position.x() << ","
                             << kf_attribute.position.y() << ","
                             << kf_attribute.position.z() << ","
                             << quarternion.w() << "," << quarternion.x() << ","
                             << quarternion.y() << "," << quarternion.z() << ","
                             << endl;
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
        pose_graph_.Save();
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
    if (c == 'n') NewSequence();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void PoseGraphNode::Publish() {
  int sequence_cnt = pose_graph_.GetCurrentSequenceCount();
  for (int i = 1; i <= sequence_cnt; i++) {
    pub_pg_path_.publish(path_[i]);
    pub_path_[i].publish(path_[i]);
    posegraph_visualization_->publish_by(pub_pose_graph_,
                                         path_[sequence_cnt].header);
  }
  base_path_.header.frame_id = "world";
  pub_base_path_.publish(base_path_);
}

void PoseGraphNode::NewSequence() {
  ROS_INFO("new sequence");
  sequence_index_++;
  ROS_INFO("sequence cnt %d", sequence_index_);
  if (sequence_index_ > 5) {
    ROS_WARN(
        "only support 5 sequences since it's boring to copy code for more "
        "sequences.");
    ROS_BREAK();
  }
  posegraph_visualization_->reset();
  Publish();
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    while (!image_buffer_.empty()) image_buffer_.pop();
    while (!point_buffer_.empty()) point_buffer_.pop();
    while (!pose_buffer_.empty()) pose_buffer_.pop();
    while (!odometry_buffer_.empty()) odometry_buffer_.pop();
  }
}

void PoseGraphNode::OnPoseGraphLoaded() { ROS_DEBUG("Pose graph loaded"); }

void PoseGraphNode::OnPoseGraphSaved() { ROS_DEBUG("Pose graph saved"); }

void PoseGraphNode::OnKeyFrameAdded(KeyFrame::Attributes kf_attribute) {
  ROS_DEBUG("On Keyframe added");
}

void PoseGraphNode::OnKeyFrameLoaded(KeyFrame::Attributes kf_attribute,
                                     int count) {
  ROS_INFO("On Keyframe loaded");
  Eigen::Quaterniond Q{kf_attribute.rotation};
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(kf_attribute.time_stamp);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x =
      kf_attribute.position.x() + visualization_shift_x_;
  pose_stamped.pose.position.y =
      kf_attribute.position.y() + visualization_shift_y_;
  pose_stamped.pose.position.z = kf_attribute.position.z();
  pose_stamped.pose.orientation.x = Q.x();
  pose_stamped.pose.orientation.y = Q.y();
  pose_stamped.pose.orientation.z = Q.z();
  pose_stamped.pose.orientation.w = Q.w();
  base_path_.poses.push_back(pose_stamped);
  base_path_.header = pose_stamped.header;

  if (count % 20 == 0) {
    Publish();
  }
}

void PoseGraphNode::OnKeyFrameConnectionFound(
    KeyFrame::Attributes current_kf_attribute,
    KeyFrame::Attributes old_kf_attribute,
    std::vector<cv::Point2f> matched_2d_old_norm,
    std::vector<double> matched_id, cv::Mat& thumb_image) {
  ROS_DEBUG("On Keyframe connection found");
  {
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumb_image)
            .toImageMsg();
    msg->header.stamp = ros::Time(current_kf_attribute.time_stamp);
    pub_match_img_.publish(msg);
  }
  if (config_.fast_relocalization) {
    sensor_msgs::PointCloud msg_match_points;
    msg_match_points.header.stamp = ros::Time(current_kf_attribute.time_stamp);
    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) {
      geometry_msgs::Point32 p;
      p.x = matched_2d_old_norm[i].x;
      p.y = matched_2d_old_norm[i].y;
      p.z = matched_id[i];
      msg_match_points.points.push_back(p);
    }
    Eigen::Vector3d T = old_kf_attribute.position;
    Eigen::Matrix3d R = old_kf_attribute.rotation;
    Quaterniond Q(R);
    sensor_msgs::ChannelFloat32 t_q_index;
    t_q_index.values.push_back(T.x());
    t_q_index.values.push_back(T.y());
    t_q_index.values.push_back(T.z());
    t_q_index.values.push_back(Q.w());
    t_q_index.values.push_back(Q.x());
    t_q_index.values.push_back(Q.y());
    t_q_index.values.push_back(Q.z());
    t_q_index.values.push_back(current_kf_attribute.index);
    msg_match_points.channels.push_back(t_q_index);
    pub_match_points_.publish(msg_match_points);
  }
}

void PoseGraphNode::OnPoseGraphOptimization(
    std::vector<KeyFrame::Attributes> kf_attributes) {
  // ROS_INFO("On Pose graph optimization");
  std::vector<KeyFrame::Attributes>::iterator it;
  int sequence_cnt = pose_graph_.GetCurrentSequenceCount();
  for (int i = 1; i <= sequence_cnt; i++) {
    path_[i].poses.clear();
  }
  base_path_.poses.clear();
  posegraph_visualization_->reset();

  if (save_loop_path) {
    std::ofstream loop_path_file_tmp(vins_result_path_, ios::out);
    loop_path_file_tmp.close();
  }

  for (it = kf_attributes.begin(); it != kf_attributes.end(); it++) {
    Eigen::Quaterniond Q;
    Q = it->rotation;
    //        printf("path p: %f, %f, %f\n",  P.x(),  P.z(),  P.y() );

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(it->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = it->position.x() + visualization_shift_x_;
    pose_stamped.pose.position.y = it->position.y() + visualization_shift_y_;
    pose_stamped.pose.position.z = it->position.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    if (it->sequence == 0) {
      base_path_.poses.push_back(pose_stamped);
      base_path_.header = pose_stamped.header;
    } else {
      path_[it->sequence].poses.push_back(pose_stamped);
      path_[it->sequence].header = pose_stamped.header;
    }

    if (save_loop_path && !vins_result_path_.empty()) {
      std::ofstream loop_path_file(vins_result_path_, ios::app);
      loop_path_file.setf(ios::fixed, ios::floatfield);
      loop_path_file.precision(0);
      loop_path_file << it->time_stamp * 1e9 << ",";
      loop_path_file.precision(5);
      loop_path_file << it->position.x() << "," << it->position.y() << ","
                     << it->position.z() << "," << Q.w() << "," << Q.x() << ","
                     << Q.y() << "," << Q.z() << "," << endl;
      loop_path_file.close();
    }

    if (SHOW_S_EDGE) {
      std::vector<KeyFrame::Attributes>::reverse_iterator rit =
          kf_attributes.rbegin();
      std::vector<KeyFrame::Attributes>::reverse_iterator lrit;
      for (; rit != kf_attributes.rend(); rit++) {
        if (rit->index == it->index) {
          lrit = rit;
          lrit++;
          for (int i = 0; i < 4; i++) {
            if (lrit == kf_attributes.rend()) break;
            if (lrit->sequence == it->sequence) {
              posegraph_visualization_->add_edge(it->position, lrit->position);
            }
            lrit++;
          }
          break;
        }
      }
    }
    if (SHOW_L_EDGE) {
      if (it->has_loop && it->sequence == sequence_cnt) {
        std::find_if(
            kf_attributes.begin(), kf_attributes.end(),
            [&](KeyFrame::Attributes& attr) {
              if (attr.index == it->loop_index && it->sequence > 0) {
                posegraph_visualization_->add_loopedge(
                    it->position,
                    attr.position + Vector3d(visualization_shift_x_,
                                             visualization_shift_y_, 0));
                return true;
              }
              return false;
            });
      }
    }
  }

  Publish();
}

void PoseGraphNode::OnNewSequentialEdge(Vector3d p1, Vector3d p2) {
  if (!SHOW_S_EDGE) return;
  posegraph_visualization_->add_edge(p1, p2);
}

void PoseGraphNode::OnNewLoopEdge(Vector3d p1, Vector3d p2) {
  if (!SHOW_L_EDGE) return;
  p2 += Vector3d(visualization_shift_x_, visualization_shift_y_, 0);
  posegraph_visualization_->add_loopedge(p1, p2);
}

void PoseGraphNode::ImuForwardCallback(
    const nav_msgs::OdometryConstPtr& forward_msg) {
  if (!visualise_imu_forward_) return;

  Vector3d vio_t(forward_msg->pose.pose.position.x,
                 forward_msg->pose.pose.position.y,
                 forward_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = forward_msg->pose.pose.orientation.w;
  vio_q.x() = forward_msg->pose.pose.orientation.x;
  vio_q.y() = forward_msg->pose.pose.orientation.y;
  vio_q.z() = forward_msg->pose.pose.orientation.z;

  auto pose_graph_world_vio = pose_graph_.GetWorldVio();
  vio_t =
      pose_graph_world_vio.rotation * vio_t + pose_graph_world_vio.translation;
  vio_q = pose_graph_world_vio.rotation * vio_q;

  auto pose_graph_drift = pose_graph_.GetDrift();
  vio_t = pose_graph_drift.rotation * vio_t + pose_graph_drift.translation;
  vio_q = pose_graph_drift.rotation * vio_q;

  Vector3d vio_t_cam;
  Quaterniond vio_q_cam;
  vio_t_cam = vio_t + vio_q * imu_camera_pose_.translation;
  vio_q_cam = vio_q * imu_camera_pose_.rotation;

  camera_pose_vis_->reset();
  camera_pose_vis_->add_pose(vio_t_cam, vio_q_cam);
  camera_pose_vis_->publish_by(pub_camera_pose_visual_, forward_msg->header);
}

void PoseGraphNode::VioCallback(const nav_msgs::OdometryConstPtr& pose_msg) {
  // ROS_INFO("vio_callback!");
  Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = pose_msg->pose.pose.orientation.w;
  vio_q.x() = pose_msg->pose.pose.orientation.x;
  vio_q.y() = pose_msg->pose.pose.orientation.y;
  vio_q.z() = pose_msg->pose.pose.orientation.z;

  auto pose_graph_world_vio = pose_graph_.GetWorldVio();
  vio_t =
      pose_graph_world_vio.rotation * vio_t + pose_graph_world_vio.translation;
  vio_q = pose_graph_world_vio.rotation * vio_q;

  auto pose_graph_drift = pose_graph_.GetDrift();
  vio_t = pose_graph_drift.rotation * vio_t + pose_graph_drift.translation;
  vio_q = pose_graph_drift.rotation * vio_q;

  Vector3d vio_t_cam;
  Quaterniond vio_q_cam;
  vio_t_cam = vio_t + vio_q * imu_camera_pose_.translation;
  vio_q_cam = vio_q * imu_camera_pose_.rotation;

  if (!visualise_imu_forward_) {
    camera_pose_vis_->reset();
    camera_pose_vis_->add_pose(vio_t_cam, vio_q_cam);
    camera_pose_vis_->publish_by(pub_camera_pose_visual_, pose_msg->header);
  }

  odometry_buffer_.push(vio_t_cam);
  if (odometry_buffer_.size() > 10) {
    odometry_buffer_.pop();
  }

  visualization_msgs::Marker key_odometrys;
  key_odometrys.header = pose_msg->header;
  key_odometrys.header.frame_id = "world";
  key_odometrys.ns = "key_odometrys";
  key_odometrys.type = visualization_msgs::Marker::SPHERE_LIST;
  key_odometrys.action = visualization_msgs::Marker::ADD;
  key_odometrys.pose.orientation.w = 1.0;
  key_odometrys.lifetime = ros::Duration();

  // static int key_odometrys_id = 0;
  key_odometrys.id = 0;  // key_odometrys_id++;
  key_odometrys.scale.x = 0.1;
  key_odometrys.scale.y = 0.1;
  key_odometrys.scale.z = 0.1;
  key_odometrys.color.r = 1.0;
  key_odometrys.color.a = 1.0;

  for (unsigned int i = 0; i < odometry_buffer_.size(); i++) {
    geometry_msgs::Point pose_marker;
    Vector3d vio_t;
    vio_t = odometry_buffer_.front();
    odometry_buffer_.pop();
    pose_marker.x = vio_t.x();
    pose_marker.y = vio_t.y();
    pose_marker.z = vio_t.z();
    key_odometrys.points.push_back(pose_marker);
    odometry_buffer_.push(vio_t);
  }
  pub_key_odometrys_.publish(key_odometrys);

  if (!loop_closure_) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = pose_msg->header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = vio_t.x();
    pose_stamped.pose.position.y = vio_t.y();
    pose_stamped.pose.position.z = vio_t.z();
    no_loop_path_.header = pose_msg->header;
    no_loop_path_.header.frame_id = "world";
    no_loop_path_.poses.push_back(pose_stamped);
    pub_vio_path_.publish(no_loop_path_);
  }
}

void PoseGraphNode::ImageCallback(const sensor_msgs::ImageConstPtr& image_msg) {
  // ROS_INFO("image_callback!");
  if (!loop_closure_) return;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    image_buffer_.push(image_msg);
  }
  // printf(" image time %f \n", image_msg->header.stamp.toSec());

  // detect unstable camera stream
  // If we use gstreamer next time, this should be caught through as EOF or
  // something similar using the gstreamer API
  if (last_image_time_ == -1.0) {
    last_image_time_ = image_msg->header.stamp.toSec();
  } else if (image_msg->header.stamp.toSec() - last_image_time_ > 1.0 ||
             image_msg->header.stamp.toSec() < last_image_time_) {
    ROS_WARN("image discontinued! New sequence detected!");
    NewSequence();
  }
  last_image_time_ = image_msg->header.stamp.toSec();
}

void PoseGraphNode::PoseCallback(const nav_msgs::OdometryConstPtr& pose_msg) {
  // ROS_INFO("pose_callback!");
  if (!loop_closure_) return;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    pose_buffer_.push(pose_msg);
  }
  /*
  printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n",
  pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                                                     pose_msg->pose.pose.position.z,
                                                     pose_msg->pose.pose.orientation.w,
                                                     pose_msg->pose.pose.orientation.x,
                                                     pose_msg->pose.pose.orientation.y,
                                                     pose_msg->pose.pose.orientation.z);
  */
}

void PoseGraphNode::ExtrinsicCallback(
    const nav_msgs::OdometryConstPtr& pose_msg) {
  std::lock_guard<std::mutex> lock(process_mutex_);
  imu_camera_pose_.translation =
      Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
               pose_msg->pose.pose.position.z);
  imu_camera_pose_.rotation = Quaterniond(pose_msg->pose.pose.orientation.w,
                                          pose_msg->pose.pose.orientation.x,
                                          pose_msg->pose.pose.orientation.y,
                                          pose_msg->pose.pose.orientation.z)
                                  .toRotationMatrix();
  pose_graph_.UpdateImuCameraPose(imu_camera_pose_);
}

void PoseGraphNode::PointCallback(
    const sensor_msgs::PointCloudConstPtr& point_msg) {
  // ROS_INFO("point_callback!");
  if (!loop_closure_) return;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    point_buffer_.push(point_msg);
  }
  /*
  for (unsigned int i = 0; i < point_msg->points.size(); i++)
  {
      printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i ,
  point_msg->points[i].x, point_msg->points[i].y, point_msg->points[i].z,
                                                   point_msg->channels[i].values[0],
                                                   point_msg->channels[i].values[1]);
  }
  */
}

void PoseGraphNode::ReloRelativePoseCallback(
    const nav_msgs::OdometryConstPtr& pose_msg) {
  Vector3d relative_t =
      Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
               pose_msg->pose.pose.position.z);
  Quaterniond relative_q;
  relative_q.w() = pose_msg->pose.pose.orientation.w;
  relative_q.x() = pose_msg->pose.pose.orientation.x;
  relative_q.y() = pose_msg->pose.pose.orientation.y;
  relative_q.z() = pose_msg->pose.pose.orientation.z;
  double relative_yaw = pose_msg->twist.twist.linear.x;
  int index = pose_msg->twist.twist.linear.y;
  // printf("receive index %d \n", index );
  Eigen::Matrix<double, 8, 1> loop_info;
  loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(),
      relative_q.x(), relative_q.y(), relative_q.z(), relative_yaw;
  pose_graph_.UpdateKeyFrameLoop(index, loop_info);
}
}  // namespace pose_graph

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_graph_node");
  std::shared_ptr<pose_graph::PoseGraphNode> pose_graph_node =
      std::make_shared<pose_graph::PoseGraphNode>();
  if (!pose_graph_node->Start()) {
    ROS_ERROR("Failed to start PoseGraphNode");
    ros::shutdown();
  }
  ROS_INFO("PoseGraphNode started.");
  ros::spin();
  return 0;
}