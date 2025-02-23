#ifndef FEATURE_TRACKER_OBSERVER_HPP
#define FEATURE_TRACKER_OBSERVER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureTrackerObserver
    : public std::enable_shared_from_this<FeatureTrackerObserver> {
 public:
  FeatureTrackerObserver() = default;
  virtual ~FeatureTrackerObserver() = default;

  virtual void OnRegistered() = 0;
  virtual void OnRestart() = 0;
  virtual void OnDurationBetweenFrameTooLarge(double current_image_time_s,
                                              double previous_image_time_s) = 0;
  virtual void OnImageTimeMovingBackwards(double current_image_time_s,
                                          double previous_image_time_s)=0;
  virtual void OnProcessedImage(cv::Mat new_frame, double current_image_time_s,
                                std::vector<cv::Point2f> features,
                                std::vector<cv::Point2f> undistorted_features,

                                std::vector<int> ids,
                                std::vector<int> track_count,
                                std::vector<cv::Point2f> points_velocity) = 0;

   private:
    void CreateTrackedFeatureImage(cv::Mat image,
                                   std::vector<cv::Point2f> features,
                                   std::vector<int> track_cnt,
                                   uint max_track_count);
};
#endif /* FEATURE_TRACKER_OBSERVER_HPP */
