#ifndef FEATURE_TRACKER_NODE_HPP
#define FEATURE_TRACKER_NODE_HPP
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Time.h>

#include "feature_tracker/feature_tracker.hpp"
#include "feature_tracker/feature_tracker_observer.hpp"

class FeatureTrackerNode : public FeatureTrackerObserver {
 public:
  ~FeatureTrackerNode();
  bool Start();

 private:
  void OnRegistered() final;

  void OnRestart() final;
  void OnDurationBetweenFrameTooLarge(double current_image_time_s,
                                      double previous_image_time_s) final;
  void OnImageTimeMovingBackwards(double current_image_time_s,
                                  double previous_image_time_s) final;
  void OnProcessedImage(const cv::Mat& new_frame, double current_image_time_s,
                        std::vector<cv::Point2f> features,
                        std::vector<cv::Point2f> undistorted_features,

                        std::vector<int> ids, std::vector<int> track_count,
                        std::vector<cv::Point2f> points_velocity) final;

  void OnImageRecieved(const cv::Mat& new_frame,
                       double current_image_time_s) final;
  void OnHistogramEqualisation(const cv::Mat& new_frame,
                               double current_image_time_s) final;
  bool ReadParameters(std::string& configFilePath, int& maxFeatureCount,
                      int& minFeatureDistance, int& pruningFrequency,
                      double& ransacThreshold,
                      bool& enableHistogramEqualization, bool& useFisheye);

  void ImageCallback(const sensor_msgs::ImageConstPtr& img_msg);

  void StartPublishersAndSubscribers();

  std::unique_ptr<FeatureTracker> feature_tracker_;
  cv_bridge::CvImage optical_flow_img_;

  int min_track_count_to_publish_;
  bool first_image_;
  ros::NodeHandle nh_{"~"};
  ros::Subscriber image_subscriber_;
  ros::Publisher optical_flow_image_publisher_;
  ros::Publisher feature_point_cloud_publisher_;
  ros::Publisher pub_restart_;

  sensor_msgs::PointCloud feature_points_msg_;
  sensor_msgs::ChannelFloat32 feature_id_channel_;
  sensor_msgs::ChannelFloat32 feature_u_pixel_channel_;
  sensor_msgs::ChannelFloat32 feature_v_pixel_channel_;
  sensor_msgs::ChannelFloat32 feature_x_velocity_channel_;
  sensor_msgs::ChannelFloat32 feature_y_velocity_channel_;
  ros::Time current_image_time_;
  std_msgs::Bool restart_flag_;
  sensor_msgs::Image optical_flow_img_msg_;
};

#endif /* FEATURE_TRACKER_NODE_HPP */
