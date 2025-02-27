#include "feature_tracker_ros/feature_tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/opencv.hpp>
#include <iomanip>

bool FeatureTrackerNode::Start() {
  restart_flag_.data = true;

  feature_points_msg_.channels.push_back(feature_id_channel_);
  feature_points_msg_.channels.push_back(feature_u_pixel_channel_);
  feature_points_msg_.channels.push_back(feature_v_pixel_channel_);
  feature_points_msg_.channels.push_back(feature_x_velocity_channel_);
  feature_points_msg_.channels.push_back(feature_y_velocity_channel_);

  feature_id_channel_.name = "Feature Ids";
  feature_u_pixel_channel_.name = "Feature U Pixel";
  feature_v_pixel_channel_.name = "Feature V Pixel";
  feature_x_velocity_channel_.name = "Feature X Velocity";
  feature_y_velocity_channel_.name = "Feature Y Velocity";

  // Set this upon first camera message received
  optical_flow_img_.header.frame_id = "TODO";
  optical_flow_img_.encoding = sensor_msgs::image_encodings::BGR8;
  return true;
}

void FeatureTrackerNode::OnRegistered() {
  ROS_INFO("FeatureTrackerNode has been registered by feature tracker");
}

void FeatureTrackerNode::OnRestart() {
  ROS_INFO("FeatureTracker has been restarted");
  pub_restart_.publish(restart_flag_);
}

void FeatureTrackerNode::OnDurationBetweenFrameTooLarge(
    double current_image_time_s, double previous_image_time_s) {
  ROS_INFO_STREAM("Too much time has elapsed between images."
                  << "\n\t Current image received had timestamp: "
                  << std::setprecision(2) << current_image_time_s
                  << "\n\t Previous image received had timestamp:  "
                  << std::setprecision(2) << previous_image_time_s);
}

void FeatureTrackerNode::OnImageTimeMovingBackwards(
    double current_image_time_s, double previous_image_time_s) {
  ROS_WARN_STREAM("Timestamp went backwards/"
                  << "\n\t Current image received had timestamp: "
                  << std::setprecision(2) << current_image_time_s
                  << "\n\t Previous image received had timestamp:  "
                  << std::setprecision(2) << previous_image_time_s);
}
void FeatureTrackerNode::OnProcessedImage(
    cv::Mat new_frame, double current_image_time_s,
    std::vector<cv::Point2f> features,
    std::vector<cv::Point2f> undistorted_features, std::vector<int> ids,
    std::vector<int> track_count, std::vector<cv::Point2f> points_velocity) {
  feature_points_msg_.points.clear();
  feature_id_channel_.values.clear();
  feature_u_pixel_channel_.values.clear();
  feature_v_pixel_channel_.values.clear();
  feature_x_velocity_channel_.values.clear();
  feature_y_velocity_channel_.values.clear();

  feature_points_msg_.header.stamp = current_image_time_;
  for (size_t i = 0; i < ids.size(); i++) {
    if (track_count[i] > min_track_count_to_publish_) {
      geometry_msgs::Point32 p;
      p.x = undistorted_features[i].x;
      p.y = undistorted_features[i].y;
      p.z = 1;

      feature_points_msg_.points.push_back(p);
      feature_id_channel_.values.push_back(ids[i]);
      feature_u_pixel_channel_.values.push_back(features[i].x);
      feature_v_pixel_channel_.values.push_back(features[i].y);
      feature_x_velocity_channel_.values.push_back(points_velocity[i].x);
      feature_y_velocity_channel_.values.push_back(points_velocity[i].y);
    }
  }
  optical_flow_img_.header.stamp = current_image_time_;
  optical_flow_img_.image = CreateOpticalFlowImage(
      new_frame, features, track_count, 20, points_velocity);
  optical_flow_img_.toImageMsg(optical_flow_img_msg_);
  optical_flow_image_publisher_.publish(optical_flow_img_msg_);
  feature_point_cloud_publisher_.publish(feature_points_msg_);
}

void FeatureTrackerNode::img_callback(
    const sensor_msgs::ImageConstPtr& img_msg) {
  current_image_time_ = img_msg->header.stamp;
  cv_bridge::CvImagePtr cv_ptr =
      cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
//   feature_tracker.ProcessNewFrame(cv_ptr->image);
}

int main(int argc, char** argv) {}