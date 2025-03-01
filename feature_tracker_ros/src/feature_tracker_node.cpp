#include "feature_tracker_ros/feature_tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>

#include <iomanip>
#include <opencv2/opencv.hpp>

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

  first_image_ = false;

  // Set this upon first camera message received
  optical_flow_img_.header.frame_id = "TODO";
  optical_flow_img_.encoding = sensor_msgs::image_encodings::BGR8;
  return true;
}

void FeatureTrackerNode::StartPublishersAndSubscribers() {
  image_subscriber_ = n.subscribe(IMAGE_TOPIC, 100, img_callback);
  feature_point_cloud_publisher_ =
      n.advertise<sensor_msgs::PointCloud>("feature", 1000);
  optical_flow_image_publisher_ =
      n.advertise<sensor_msgs::Image>("feature_img", 1000);
  pub_restart_ = n.advertise<std_msgs::Bool>("restart", 1000);
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

void FeatureTrackerNode::ImageCallback(
    const sensor_msgs::ImageConstPtr& img_msg) {
  current_image_time_ = img_msg->header.stamp;
  feature_tracker.ProcessNewFrame(
      cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8)->image,
      img_msg->header.stamp.toSec());
  //   feature_tracker.ProcessNewFrame(cv_ptr->image);
}

bool FeatureTrackerNode::ReadParameters() {
  if (!nh_.getParam("config_file", config_file_path_)) {
    ROS_ERROR("Failed to read \"config_file\" parameter");
    return false;
  }

  nh_.getParam("fisheye", fisheye_);
  nh_.getParam("max_cnt", max_number_of_features_);
  nh_.getParam("min_dist", minimum_distance_between_features_);
  nh_.getParam("image_height", image_height_);
  nh_.getParam("image_width", image_width_);
  nh_.getParam("freq", pruning_frequency_);
  nh_.getParam("F_threshold", ransac_threshold_);

  nh_.getParam("equalize", run_histogram_equalisation_);
  nh_.getParam("F_threshold", ransac_threshold_);
  nh_.getParam("F_threshold", ransac_threshold_);

  ROS_INFO(
      "Loaded parameters: config_file: %s, visualization_shift_x: %d, "
      "visualization_shift_y: %d, skip_cnt: %d, skip_dis: %f",
      config_file_path_.c_str(), visualization_shift_x_, visualization_shift_y_,
      skip_cnt_threshold_, skip_distance_);

  return true;
}
int main(int argc, char** argv) {}