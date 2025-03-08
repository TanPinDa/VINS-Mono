#include <cmath>
#include <filesystem>  // For file existence check
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#include "feature_tracker/feature_tracker.h"
#include "feature_tracker/feature_tracker_observer_spdlog_rerun.hpp"
#include "spdlog/spdlog.h"
int main(int argc, char** argv) {
  FeatureTracker feat(
      "/home/rosdev/workspace/ros_ws/src/VINS-Mono/config/euroc/"
      "camera_config.yaml",
      false, true, 150, 30, 1.0, 460, 460, 10, 1.0);
  std::shared_ptr<FeatureTrackerObserverSPDRerun> observer =
      std::make_shared<FeatureTrackerObserverSPDRerun>();

  // FeatureTrackerObserverSPDRerun observer = FeatureTrackerObserverSPDRerun();
  feat.RegisterEventObserver(observer);
  std::cout << "HI" << std::endl;

  // Path to your dataset
  std::string folderPath = "/home/rosdev/workspace/data/MH_01_easy";
  std::string imageFolder = folderPath + "/mav0/cam0/data";
  std::string timestampFile = folderPath + "/mav0/cam0/data.csv";

  // Open the timestamp file
  std::ifstream tsFile(timestampFile);
  if (!tsFile.is_open()) {
    std::cerr << "Error opening timestamp file: " << timestampFile << std::endl;
    return 1;
  }

  std::string line;
  double time_covnersion = pow(10.0, 9);
  // Read the CSV file line by line

  while (std::getline(tsFile, line)) {
    std::stringstream ss(line);
    double timestamp;
    std::string imageFilename;

    // Read the timestamp
    ss >> timestamp;

    // Skip the comma after the timestamp
    ss.ignore(1, ',');

    // Read the filename (after the comma)
    std::getline(ss, imageFilename);

    // Trim leading/trailing spaces from filename
    imageFilename.erase(0, imageFilename.find_first_not_of(" \t\r\n"));
    imageFilename.erase(imageFilename.find_last_not_of(" \t\r\n") + 1);

    // Strip out any potential newline or carriage return characters
    imageFilename.erase(
        std::remove(imageFilename.begin(), imageFilename.end(), '\n'),
        imageFilename.end());
    imageFilename.erase(
        std::remove(imageFilename.begin(), imageFilename.end(), '\r'),
        imageFilename.end());

    // Construct the full image path
    std::string imagePath = imageFolder + "/" + imageFilename;

    // Debug: Print the image path

    // Check if the file exists
    if (!std::filesystem::exists(imagePath)) {
      std::cerr << "File does not exist: " << imagePath << std::endl;
      continue;
    }

    // Read the image
    cv::Mat image =
        cv::imread(imagePath, cv::IMREAD_GRAYSCALE);  // Read as color image

    // Check if image was loaded successfully
    if (image.empty()) {
      std::cerr << "Error loading image: " << imagePath << std::endl;
      continue;
    }

    // Display the image

    feat.ProcessNewFrame(image, timestamp / time_covnersion);
    // cv::imshow("Image", image);

    // Wait for 100 milliseconds before moving to the next image
    // if (cv::waitKey(1) == 'q') {  // Press 'q' to quit
    //   break;
    // }

    // Optionally, print the timestamp to the console (or use it for
    // synchronization)
  }

  // Close the file and cleanup
  tsFile.close();
  cv::destroyAllWindows();

  return 0;
}
