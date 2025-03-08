#include "feature_tracker/feature_tracker_observer.hpp"

#include <opencv2/opencv.hpp>
cv::Mat FeatureTrackerObserver::CreateTrackedFeatureImage(
    cv::Mat image, std::vector<cv::Point2f> features,
    std::vector<int> track_cnt, uint max_track_count) {
  cv::Mat show_img;
  cv::cvtColor(image, show_img, cv::COLOR_GRAY2RGB);

  for (unsigned int i = 0; i < features.size(); i++) {
    double len = std::min(1.0, 1.0 * track_cnt[i] / max_track_count);
    cv::circle(show_img, features[i], 2,
               cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  return show_img;
}

cv::Mat FeatureTrackerObserver::CreateOpticalFlowImage(
    cv::Mat image, std::vector<cv::Point2f> features,
    std::vector<int> track_cnt, uint max_track_count,
    std::vector<cv::Point2f> points_velocity) {
  cv::Mat show_img =
      CreateTrackedFeatureImage(image, features, track_cnt, max_track_count);

  for (size_t i = 0; i < points_velocity.size(); i++) {
    cv::Point2f start = features[i];
    // std::cout << "velocty: " << points_velocity[i].x << ", "
    //           << points_velocity[i].y << std::endl;
    cv::Point2f end =
        start + (points_velocity[i])*50;  // Add velocity to get end point

    cv::arrowedLine(show_img, start, end, cv::Scalar(0, 255, 0), 1,
                    cv::LINE_AA);
  }
  return show_img;
}