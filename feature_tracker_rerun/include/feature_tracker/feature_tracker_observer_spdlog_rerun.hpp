#ifndef FEATURE_TRACKER_OBSERVER_SPDLOG_RERUN_HPP
#define FEATURE_TRACKER_OBSERVER_SPDLOG_RERUN_HPP
#include <rerun.hpp>
#include "feature_tracker/feature_tracker_observer.hpp"
class FeatureTrackerObserverSPDRerun : public FeatureTrackerObserver {
 public:
  //   FeatureTrackerObserverSPDRerun() = default;
  ~FeatureTrackerObserverSPDRerun();
//   const auto rec = rerun::RecordingStream("rerun_example_image_formats");

 private:
  std::unique_ptr<rerun::RecordingStream> recorder_;
  void OnRegistered() final;

  void OnRestart() final;
  void OnDurationBetweenFrameTooLarge(double current_image_time_s,
                                      double previous_image_time_s) final;
  void OnImageTimeMovingBackwards(double current_image_time_s,
                                  double previous_image_time_s) final;
  void OnProcessedImage(cv::Mat new_frame, double current_image_time_s,
                        std::vector<cv::Point2f> features,
                        std::vector<cv::Point2f> undistorted_features,

                        std::vector<int> ids, std::vector<int> track_count,
                        std::vector<cv::Point2f> points_velocity) final;
};

#endif /* FEATURE_TRACKER_OBSERVER_SPDLOG_RERUN_HPP */
