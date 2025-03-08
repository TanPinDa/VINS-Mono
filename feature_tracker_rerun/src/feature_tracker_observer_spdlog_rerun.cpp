#include "feature_tracker/feature_tracker_observer_spdlog_rerun.hpp"

// #include <rerun.hpp>
#include "spdlog/spdlog.h"
#include <opencv2/core.hpp>

// FeatureTrackerObserverSPDRerun::FeatureTrackerObserverSPDRerun() {
//   spdlog::info("FeatureTracker observer created");
// }

FeatureTrackerObserverSPDRerun::~FeatureTrackerObserverSPDRerun() {
  spdlog::info("Destroying Observer");
}
void FeatureTrackerObserverSPDRerun::OnRegistered() {
  spdlog::set_level(spdlog::level::debug);
  spdlog::info("FeatureTracker observer Registered");
  recorder_ = std::make_unique<rerun::RecordingStream>("TEST");
  recorder_->spawn().exit_on_failure();
  // rec = rerun::RecordingStream("rerun_example_image");
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
  recorder_->log("Optical Flow Image",
                 rerun::Image::from_rgb24(img, {img.cols, img.rows}));

  // Wait for 100 milliseconds before moving to the next image
}


// Adapters so we can borrow an OpenCV image easily into Rerun images without copying:
template <>
struct rerun::CollectionAdapter<uint8_t, cv::Mat> {
    /// Borrow for non-temporary.
    Collection<uint8_t> operator()(const cv::Mat& img) {
        assert(
            "OpenCV matrix was expected have bit depth CV_U8" && CV_MAT_DEPTH(img.type()) == CV_8U
        );

        return Collection<uint8_t>::borrow(img.data, img.total() * img.channels());
    }

    // Do a full copy for temporaries (otherwise the data might be deleted when the temporary is destroyed).
    Collection<uint8_t> operator()(cv::Mat&& img) {
        assert(
            "OpenCV matrix was expected have bit depth CV_U8" && CV_MAT_DEPTH(img.type()) == CV_8U
        );

        std::vector<uint8_t> img_vec(img.total() * img.channels());
        img_vec.assign(img.data, img.data + img.total() * img.channels());
        return Collection<uint8_t>::take_ownership(std::move(img_vec));
    }
};