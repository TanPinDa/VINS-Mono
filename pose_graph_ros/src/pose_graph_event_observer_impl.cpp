/**
 * @file pose_graph_event_observer_impl.cpp
 * @brief
 * @date 09-03-2025
 *
 * @copyright Copyright (c) 2025 Cheo Kee Jin.
 */

#include "pose_graph_event_observer_impl.hpp"

#include <fstream>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true

namespace pose_graph {
PoseGraphNode::PoseGraphEventObserverImpl::PoseGraphEventObserverImpl(
    ros::NodeHandle& nh, const std::string config_file_path)
    : nh_(nh) {
  posegraph_visualization_ =
      std::make_unique<CameraPoseVisualization>(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization_->setScale(0.1);
  posegraph_visualization_->setLineWidth(0.01);

  // Load config file
  cv::FileStorage fs(config_file_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    ROS_ERROR("PoseGraphEventObserver::Failed to open config file: %s",
              config_file_path.c_str());
    ros::shutdown();
  }

  fast_relocalization_ = (int)fs["fast_relocalization"];

  vins_result_path_ = std::string(fs["output_path"]);
  FileSystemHelper::createDirectoryIfNotExists(vins_result_path_.c_str());
  vins_result_path_ = vins_result_path_ + "/vins_result_loop.csv";

  fs.release();

  if (!ReadParameters()) {
    ROS_ERROR("PoseGraphEventObserver::Failed to read parameters");
    ros::shutdown();
  }

  StartPublishers();
}

void PoseGraphNode::PoseGraphEventObserverImpl::Publish(int sequence_count) {
  for (int i = 1; i <= sequence_count; i++) {
    pub_pg_path_.publish(path_[i]);
    pub_path_[i].publish(path_[i]);
    posegraph_visualization_->publish_by(pub_pose_graph_,
                                         path_[sequence_count].header);
  }
  base_path_.header.frame_id = "world";
  pub_base_path_.publish(base_path_);
}

void PoseGraphNode::PoseGraphEventObserverImpl::ResetPoseGraphVisualisation() {
  posegraph_visualization_->reset();
}

bool PoseGraphNode::PoseGraphEventObserverImpl::ReadParameters() {
  nh_.getParam("visualization_shift_x", visualization_shift_x_);
  nh_.getParam("visualization_shift_y", visualization_shift_y_);

  ROS_INFO(
      "Loaded parameters: visualization_shift_x: %d, "
      "visualization_shift_y: %d",
      visualization_shift_x_, visualization_shift_y_);

  return true;
}

void PoseGraphNode::PoseGraphEventObserverImpl::StartPublishers() {
  pub_match_img_ = nh_.advertise<sensor_msgs::Image>("match_image", 2000);
  pub_match_points_ =
      nh_.advertise<sensor_msgs::PointCloud>("match_points", 100);
  pub_pg_path_ = nh_.advertise<nav_msgs::Path>("pose_graph_path", 1000);
  pub_base_path_ = nh_.advertise<nav_msgs::Path>("base_path", 1000);
  pub_pose_graph_ =
      nh_.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
  for (int i = 1; i < 10; i++)
    pub_path_[i] = nh_.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnPoseGraphLoaded() {
  ROS_DEBUG("Pose graph loaded");
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnPoseGraphSaved() {
  ROS_DEBUG("Pose graph saved");
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnKeyFrameAdded(
    KeyFrame::Attributes kf_attribute, int sequence_count) {
  ROS_DEBUG("On Keyframe added");

  Eigen::Quaterniond quarternion{kf_attribute.rotation};
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(kf_attribute.time_stamp);
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
  path_[sequence_count].poses.push_back(pose_stamped);
  path_[sequence_count].header = pose_stamped.header;

  if (save_loop_path) {
    std::ofstream loop_path_file(vins_result_path_, ios::app);
    loop_path_file.setf(ios::fixed, ios::floatfield);
    loop_path_file.precision(0);
    loop_path_file << kf_attribute.time_stamp * 1e9 << ",";
    loop_path_file.precision(5);
    loop_path_file << kf_attribute.position.x() << ","
                   << kf_attribute.position.y() << ","
                   << kf_attribute.position.z() << "," << quarternion.w() << ","
                   << quarternion.x() << "," << quarternion.y() << ","
                   << quarternion.z() << "," << endl;
    loop_path_file.close();
  }

  Publish(sequence_count);
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnKeyFrameLoaded(
    KeyFrame::Attributes kf_attribute, int count, int sequence_count) {
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
    Publish(sequence_count);
  }
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnKeyFrameConnectionFound(
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
  if (fast_relocalization_) {
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

void PoseGraphNode::PoseGraphEventObserverImpl::OnPoseGraphOptimization(
    std::vector<KeyFrame::Attributes> kf_attributes, int sequence_count) {
  // ROS_INFO("On Pose graph optimization");
  std::vector<KeyFrame::Attributes>::iterator it;
  for (int i = 1; i <= sequence_count; i++) {
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
      if (it->has_loop && it->sequence == sequence_count) {
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

  Publish(sequence_count);
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnNewSequentialEdge(
    Vector3d p1, Vector3d p2) {
  if (!SHOW_S_EDGE) return;
  posegraph_visualization_->add_edge(p1, p2);
}

void PoseGraphNode::PoseGraphEventObserverImpl::OnNewLoopEdge(Vector3d p1,
                                                              Vector3d p2) {
  if (!SHOW_L_EDGE) return;
  p2 += Vector3d(visualization_shift_x_, visualization_shift_y_, 0);
  posegraph_visualization_->add_loopedge(p1, p2);
}
}  // namespace pose_graph