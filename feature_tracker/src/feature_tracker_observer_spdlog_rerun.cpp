#include "feature_tracker/feature_tracker_observer_spdlog_rerun.hpp"

#include "spdlog/spdlog.h"
// FeatureTrackerObserverSPDRerun::FeatureTrackerObserverSPDRerun() {
//   spdlog::info("FeatureTracker observer created");
// }

FeatureTrackerObserverSPDRerun::~FeatureTrackerObserverSPDRerun() {
  spdlog::info("Destroying Observer");
}
void FeatureTrackerObserverSPDRerun::OnRegistered() {
  spdlog::set_level(spdlog::level::debug);
  spdlog::info("FeatureTracker observer Registered");
}

void FeatureTrackerObserverSPDRerun::OnRestart() {
  spdlog::info(
      "FeatureTracker Restarted. Likely due to discontinous image frames");
}

void FeatureTrackerObserverSPDRerun::OnDurationBetweenFrameTooLarge(
    double current_image_time_s, double previous_image_time_s) {
  spdlog::info("Large time diff between prev and curr image.");
}

void FeatureTrackerObserverSPDRerun::OnImageTimeMovingBackwards(
    double current_image_time_s, double previous_image_time_s) {
  spdlog::info("Timestamp for current image is before previous image");
}

void FeatureTrackerObserverSPDRerun::OnProcessedImage(
    cv::Mat new_frame, double current_image_time_s,
    std::vector<cv::Point2f> features,
    std::vector<cv::Point2f> undistorted_features,

    std::vector<int> ids, std::vector<int> track_count,
    std::vector<cv::Point2f> points_velocity) {
  spdlog::info("Features have been pruned and new features added");
  cv::Mat img = CreateOpticalFlowImage(new_frame, features, track_count, 20,
                                       points_velocity);
  cv::imshow("Image", img);

  // Wait for 100 milliseconds before moving to the next image
  cv::waitKey(100);
}