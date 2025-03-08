#include "feature_tracker_ros/feature_tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>

#include <iomanip>
#include <opencv2/opencv.hpp>

FeatureTrackerNode::~FeatureTrackerNode() { ros::shutdown(); }
void FeatureTrackerNode::ImageCallback(
    const sensor_msgs::ImageConstPtr& img_msg) {
  if (first_image_) {
    optical_flow_img_msg_ = *img_msg;
    first_image_ = false;
  }
  current_image_time_ = img_msg->header.stamp;
  ROS_INFO_STREAM("" << current_image_time_);
  feature_tracker_->ProcessNewFrame(
      cv_bridge::toCvShare(img_msg, img_msg->encoding)->image,
      img_msg->header.stamp.toSec());
  //   feature_tracker.ProcessNewFrame(cv_ptr->image);
}

bool FeatureTrackerNode::Start() {
  restart_flag_.data = true;
  // TODO Make this from a param
  min_track_count_to_publish_ = 1;

  feature_id_channel_.name = "Feature Ids";
  feature_u_pixel_channel_.name = "Feature U Pixel";
  feature_v_pixel_channel_.name = "Feature V Pixel";
  feature_x_velocity_channel_.name = "Feature X Velocity";
  feature_y_velocity_channel_.name = "Feature Y Velocity";

  // TODO 1: Try to make this pass in pointer instead.
  feature_points_msg_.channels.push_back(feature_id_channel_);
  feature_points_msg_.channels.push_back(feature_u_pixel_channel_);
  feature_points_msg_.channels.push_back(feature_v_pixel_channel_);
  feature_points_msg_.channels.push_back(feature_x_velocity_channel_);
  feature_points_msg_.channels.push_back(feature_y_velocity_channel_);

  first_image_ = false;

  // TODO 2: Add frame for this during first image.
  // TODO 2: because no one consumes this.
  optical_flow_img_.header.frame_id = "TODO";
  optical_flow_img_.encoding = sensor_msgs::image_encodings::BGR8;

  // Read parameters
  std::string config_file_path;
  int max_feature_count, min_feature_distance, pruning_frequency;
  double ransac_threshold;
  bool enable_histogram_equalization, use_fisheye;

  if (!ReadParameters(config_file_path, max_feature_count, min_feature_distance,
                      pruning_frequency, ransac_threshold,
                      enable_histogram_equalization, use_fisheye)) {
    ROS_ERROR("Failed to read parameters in Start()");
    return false;
  }

  // Initialize FeatureTracker with parameters

  // TODO 3: Set the fx,fy from ros param.
  // Consider obtaining from camera params, however, depending on model there
  // might not be a fx,fy
  feature_tracker_ = std::make_unique<FeatureTracker>(
      config_file_path, use_fisheye, enable_histogram_equalization,
      max_feature_count, min_feature_distance, ransac_threshold,
      /*fx=*/460.0, /*fy=*/460.0, pruning_frequency,
      /*max_time_difference=*/1.0);
  feature_tracker_->RegisterEventObserver(
      FeatureTrackerNode::shared_from_this());
  StartPublishersAndSubscribers();
  return true;
}

bool FeatureTrackerNode::ReadParameters(
    std::string& config_file_path, int& max_feature_count,
    int& min_feature_distance, int& pruning_frequency, double& ransac_threshold,
    bool& enable_histogram_equalization, bool& use_fisheye) {
  if (!nh_.getParam("config_file", config_file_path)) {
    ROS_ERROR("Failed to read \"config_file\" parameter");
    return false;
  }

  nh_.getParam("/max_cnt", max_feature_count);
  nh_.getParam("/min_dist", min_feature_distance);
  nh_.getParam("/freq", pruning_frequency);
  nh_.getParam("/F_threshold", ransac_threshold);
  nh_.getParam("/equalize", enable_histogram_equalization);
  nh_.getParam("/fisheye", use_fisheye);

  ROS_INFO(
      "Loaded parameters:\n"
      "\tconfig_file: %s\n"
      "\tmax_cnt: %u\n"
      "\tmin_dist: %u\n"

      "\tfreq: %u\n"
      "\tF_threshold: %f\n"
      "\tequalize: %s\n"
      "\tfisheye: %s",
      config_file_path.c_str(), max_feature_count, min_feature_distance,
      pruning_frequency, ransac_threshold,
      enable_histogram_equalization ? "true" : "false",
      use_fisheye ? "true" : "false");

  return true;
}

void FeatureTrackerNode::StartPublishersAndSubscribers() {
  ROS_INFO("Creating Subscribers and Publishers");
  feature_point_cloud_publisher_ =
      nh_.advertise<sensor_msgs::PointCloud>("feature", 1000);
  optical_flow_image_publisher_ =
      nh_.advertise<sensor_msgs::Image>("feature_img", 1000);
  pub_restart_ = nh_.advertise<std_msgs::Bool>("restart", 1000);
  image_subscriber_ = nh_.subscribe("/cam0/image_raw", 100,
                                    &FeatureTrackerNode::ImageCallback, this);
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
    const cv::Mat& new_frame, double current_image_time_s,
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

  for (size_t i = 0; i < points_velocity.size(); i++) {
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
  // TODO 1: Try to make this pass in pointer instead. So that we dont need to
  // clear and pushback again
  feature_points_msg_.channels.clear();
  feature_points_msg_.channels.push_back(feature_id_channel_);
  feature_points_msg_.channels.push_back(feature_u_pixel_channel_);
  feature_points_msg_.channels.push_back(feature_v_pixel_channel_);
  feature_points_msg_.channels.push_back(feature_x_velocity_channel_);
  feature_points_msg_.channels.push_back(feature_y_velocity_channel_);

  optical_flow_img_.header.stamp = current_image_time_;
  optical_flow_img_.image = CreateOpticalFlowImage(
      new_frame, features, track_count, 20, points_velocity);
  optical_flow_img_.toImageMsg(optical_flow_img_msg_);
  optical_flow_image_publisher_.publish(optical_flow_img_msg_);
  feature_point_cloud_publisher_.publish(feature_points_msg_);
}

void FeatureTrackerNode::OnImageRecieved(const cv::Mat& new_frame,
                                         double current_image_time_s) {
  ROS_DEBUG_STREAM("Received Image with timestamp: " << current_image_time_s
                                                     << "s");
};
void FeatureTrackerNode::OnHistogramEqualisation(const cv::Mat& new_frame,
                                                 double current_image_time_s) {
  ROS_DEBUG_STREAM("Performed Histogram equalisation on image with timestamp: "
                   << current_image_time_s << "s");
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "feature_tracker_node");
  std::shared_ptr<FeatureTrackerNode> feature_tracker_node =
      std::make_shared<FeatureTrackerNode>();
  if (!feature_tracker_node->Start()) {
    ROS_ERROR("Failed to start FeatureTrackerNode");
    ros::shutdown();
  }
  ROS_INFO("FeatureTrackerNode started.");
  ros::spin();
  return 0;
}