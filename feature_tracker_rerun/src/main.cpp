#include <yaml-cpp/yaml.h>  // Ensure you have yaml-cpp installed and linked
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#include "feature_tracker/feature_tracker.hpp"
#include "feature_tracker/feature_tracker_observer_spdlog_rerun.hpp"
#include "spdlog/spdlog.h"

int main(int argc, char** argv) {
  // Default values for file paths
  std::string default_feature_tracker_config =
      "/home/rosdev/workspace/ros_ws/src/VINS-Mono/config/euroc/"
      "feature_tracker_configs.yaml";
  std::string default_camera_config =
      "/home/rosdev/workspace/ros_ws/src/VINS-Mono/config/euroc/"
      "camera_config.yaml";
  std::string default_test_data_set_images =
      "/home/rosdev/workspace/data/MH_01_easy/mav0/cam0/data";
  std::string default_test_data_set_csv =
      "/home/rosdev/workspace/data/MH_01_easy/mav0/cam0/data.csv";
  // Parse command-line arguments

  std::string feature_tracker_config_file =
      (argc > 1) ? argv[1] : default_feature_tracker_config;
  std::string camera_config_file = (argc > 2) ? argv[2] : default_camera_config;
  std::string image_folder =
      (argc > 3) ? argv[4] : default_test_data_set_images;
  std::string timestamp_file = (argc > 4) ? argv[4] : default_test_data_set_csv;

  std::cout << "Using the following paths:"
            << "\n\tFeature Tracker Config: " << feature_tracker_config_file
            << "\n\tCamera Config: " << camera_config_file
            << "\n\tImage Folder: " << image_folder
            << "\n\tTimestamp File: " << timestamp_file << std::endl;

  // Load parameters from YAML file
  YAML::Node config = YAML::LoadFile(default_feature_tracker_config);
  if (!config) {
    std::cerr << "Failed to load config file: " << camera_config_file
              << std::endl;
    return 1;
  }

  // Extract parameters
  bool use_equalize = config["equalize"].as<bool>();
  bool use_fisheye = config["fisheye"].as<bool>();
  int max_features = config["max_cnt"].as<int>();
  int min_distance = config["min_dist"].as<int>();
  double ransac_threshold = config["F_threshold"].as<double>();
  double freq = config["freq"].as<double>();

  FeatureTracker feat(camera_config_file, use_fisheye, use_equalize,
                      max_features, min_distance, ransac_threshold, 460, 460,
                      freq, ransac_threshold);

  std::shared_ptr<FeatureTrackerObserverSPDRerun> observer =
      std::make_shared<FeatureTrackerObserverSPDRerun>();
  feat.RegisterEventObserver(observer);

  // Open the timestamp file
  std::ifstream tsFile(timestamp_file);
  if (!tsFile.is_open()) {
    std::cerr << "Error opening timestamp file: " << timestamp_file
              << std::endl;
    return 1;
  }

  std::string line;
  double time_conversion = pow(10.0, 9);

  while (std::getline(tsFile, line)) {
    std::stringstream ss(line);
    double timestamp;
    std::string imageFilename;

    ss >> timestamp;
    ss.ignore(1, ',');
    std::getline(ss, imageFilename);
    imageFilename.erase(0, imageFilename.find_first_not_of(" \t\r\n"));
    imageFilename.erase(imageFilename.find_last_not_of(" \t\r\n") + 1);

    std::string imagePath = image_folder + "/" + imageFilename;
    if (!std::filesystem::exists(imagePath)) {
      std::cerr << "File does not exist: " << imagePath << std::endl;
      continue;
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      std::cerr << "Error loading image: " << imagePath << std::endl;
      continue;
    }

    feat.ProcessNewFrame(image, timestamp / time_conversion);
  }

  tsFile.close();
  return 0;
}
