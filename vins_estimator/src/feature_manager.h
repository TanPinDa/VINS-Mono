#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include "parameters.h"

/**
 * @struct FeatureBase
 * @brief Base struct for common feature attributes.
 */
struct FeatureBase
{
  Vector3d point;             /**< Position of the feature in 3D space. */
  Vector2d pixel_coordinates; /**< Pixel coordinates of the feature. */
  Vector2d pixel_velocity;    /**< Pixel velocity of the feature. */
  /**
     * @brief Default constructor for FeatureBase struct.
     */
    FeatureBase() = default;
  /**
   * @brief Constructor for FeatureBase struct.
   *
   * @param x X-coordinate of the feature in 3D space.
   * @param y Y-coordinate of the feature in 3D space.
   * @param z Z-coordinate of the feature in 3D space.
   * @param p_u U-coordinate (horizontal) of the feature in pixel space.
   * @param p_v V-coordinate (vertical) of the feature in pixel space.
   * @param velocity_x X-component of the pixel velocity.
   * @param velocity_y Y-component of the pixel velocity.
   */
  FeatureBase(double x, double y, double z,
              double p_u, double p_v,
              double velocity_x, double velocity_y)
      : point(x, y, z),
        pixel_coordinates(p_u, p_v),
        pixel_velocity(velocity_x, velocity_y) {}
};
/**
 * @class FeatureObservation
 * @brief Class representing a feature detected in a single frame.
 */
class FeatureObservation : public FeatureBase
{
public:
  /**
   * @brief Constructs a new FeatureObservation object.
   *
   * @param _point The 7-dimensional vector representing the feature point, pixel coordinates, and pixel velocity.
   * @param imu_camera_clock_offset The IMU-camera clock offset.
   */
  // FeatureObservation(const FeatureBase &_point, double imu_camera_clock_offset)
  //     : FeatureBase(_point(0), _point(1), _point(2), _point(3), _point(4), _point(5), _point(6)),
  //       imu_camera_clock_offset_current_(imu_camera_clock_offset) {}
  FeatureObservation(const FeatureBase &base, double imu_camera_clock_offset)
      : FeatureBase(base),
        imu_camera_clock_offset_current_(imu_camera_clock_offset) {}
  double imu_camera_clock_offset_current_; /**< Current IMU-camera clock offset. */
  double z;                                /**< Depth of the feature. */
  bool is_used;                            /**< Boolean indicating whether the feature is used. */
  double parallax;                         /**< Parallax value. */
  MatrixXd A;                              /**< Matrix A. */
  VectorXd b;                              /**< Vector b. */
  double dep_gradient;                     /**< Depth gradient. */
};

/**
 * @class FeatureOccurrencesAcrossFrames
 * @brief This class represents all occurrences of a specific feature across multiple frames.
 *
 * This class tracks a feature's occurrences in different frames, including details such as
 * the frame it first appears in, the number of times it has been used, and whether it is
 * considered an outlier or at the margin. It also includes an estimated depth and solve status.
 */
class FeatureOccurrencesAcrossFrames
{
public:
  const int feature_id;                                  /**< Unique identifier for the feature. */
  int start_frame;                                       /**< Frame in which the feature first appears. */
  vector<FeatureObservation> matched_features_in_frames; /**< List of occurrences of the feature in different frames. */

  int used_num;           /**< Number of times the feature has been used. */
  bool is_outlier;        /**< Flag indicating whether the feature is an outlier. */
  bool is_margin;         /**< Flag indicating whether the feature is at the margin. */
  double estimated_depth; /**< Estimated depth of the feature. */
  int solve_flag;         /**< Flag indicating the status of the solve process. 0: haven't solved yet, 1: solved successfully, 2: solve failed. */

  Vector3d gt_p; /**< UNUSED Ground truth position of the feature. */

  /**
   * @brief Constructs a new FeatureOccurrencesAcrossFrames object.
   *
   * @param _feature_id The unique identifier for the feature.
   * @param _start_frame The frame in which the feature first appears.
   */
  FeatureOccurrencesAcrossFrames(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
  }

  /**
   * @brief Calculates the end frame of the feature.
   *
   * @return The frame index where the feature ends.
   */
  int endFrame();
};
class FeatureManager
{
public:
  FeatureManager(Matrix3d _Rs[]);

  void SetRotationCameraToImu(Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();

  bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, FeatureBase>>> &image, double td);
  void debugShow();
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  // void updateDepth(const VectorXd &x);
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth(const VectorXd &x);
  VectorXd getDepthVector();
  void triangulate(const Vector3d translations_imu_to_world[], const Vector3d translations_camera_to_imu[], const Matrix3d rotations_camera_to_imu[]);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier();
  list<FeatureOccurrencesAcrossFrames> feature;
  int last_track_num;

private:
  double compensatedParallax2(const FeatureOccurrencesAcrossFrames &it_per_id, int frame_count);

  // This is a pointer to an array of Matrices. The array is updated outside this class
  const Matrix3d *imu_orientations_wrt_world_;
  Matrix3d rotation_of_cameras_to_imu_[NUM_OF_CAM];
};

#endif