#include "feature_tracker/feature_tracker.h"
#include "spdlog/spdlog.h"

int FeatureTracker::n_id = 0;

/**
 * Checks if a 2D point is within the border of an image.
 *
 * @param pt The point to be checked.
 * @param image_width The width of the image.
 * @param image_height The height of the image.
 * @param border_size The size of the border.
 * @return True if the point is within the border, false otherwise.
 */
bool inBorder(const cv::Point2f &pt, const int &image_width, const int &image_height, const int border_size)
{
    // Round the x and y coordinates of the point to integers
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);

    // Check if the rounded coordinates are within the border of the image
    return border_size <= img_x && img_x < image_width - border_size &&
           border_size <= img_y && img_y < image_height - border_size;
}
/**
 * Reduces the size of a vector of cv::Point2f based on a status vector.
 * Only elements with a corresponding true value in the status vector are kept.
 *
 * @param v The vector of cv::Point2f to be reduced.
 * @param status The status vector indicating which elements to keep.
 */
void FilterPoints(vector<cv::Point2f> &v, vector<uchar> status)
{
    // Initialize a counter for the new vector
    int j = 0;

    // Iterate through each element of the original vector
    for (int i = 0; i < int(v.size()); i++)
    {
        // Check if the status at index i is true
        if (status[i])
        {
            // If true, copy the element at index i to the new vector at index j
            v[j++] = v[i];
        }
    }

    // Resize the original vector to the size of the new vector (j)
    v.resize(j);
}

void FilterFeatureIds(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], feature_ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    forw_pts.clear();
    feature_ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            feature_ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time,const bool &detect_new_feature_points)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    forw_pts.clear();

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        spdlog::debug("CLAHE costs: {}ms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = img;
    }
    forw_img = img;

    if (current_points.size() > 0)
    {
        // Get optical flow and filter points outisde of the image dimensions
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, current_points, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i],COL,ROW,1))
                status[i] = 0;
        FilterPoints(previous_points, status);
        FilterPoints(current_points, status);
        FilterPoints(forw_pts, status);
        FilterFeatureIds(feature_ids, status);
        FilterPoints(current_undistorted_points, status);
        FilterFeatureIds(track_cnt, status);
        spdlog::debug("temporal optical flow costs: {}ms", t_o.toc());
    }
    for (auto &n : track_cnt)
        n++;
    if (detect_new_feature_points)
    {
        rejectWithF();
        spdlog::debug("set mask begins");
        TicToc t_m;
        setMask();
        spdlog::debug("set mask costs {}ms", t_m.toc());

        spdlog::debug("detect feature begins");
        TicToc t_t;

        int max_number_of_new_feature_points = MAX_CNT - static_cast<int>(forw_pts.size());
        if (max_number_of_new_feature_points > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, max_number_of_new_feature_points, 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        spdlog::debug("detect feature costs: {}ms", t_t.toc());

        spdlog::debug("add feature begins");
        TicToc t_a;
        for (auto &p : n_pts)
        {
            forw_pts.push_back(p);
            feature_ids.push_back(-1);
            track_cnt.push_back(1);
        }
        spdlog::debug("selectFeature costs: {}ms", t_a.toc());
    }
    prev_img = cur_img;
    previous_points = current_points;
    previous_undistorted_points = current_undistorted_points;
    cur_img = forw_img;
    current_points = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        spdlog::debug("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(current_points.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < current_points.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(current_points[i].x, current_points[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = current_points.size();
        FilterPoints(previous_points, status);
        FilterPoints(current_points, status);
        FilterPoints(forw_pts, status);
        FilterPoints(current_undistorted_points, status);
        FilterFeatureIds(feature_ids, status);
        FilterFeatureIds(track_cnt, status);
        spdlog::debug("FM ransac: {0} -> {1}: {2}", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        spdlog::debug("FM ransac costs: {}ms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < feature_ids.size())
    {
        if (feature_ids[i] == -1)
            feature_ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    spdlog::info("reading paramerter of camera {}", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // spdlog::error("({0} {1}) -> ({2} {3})", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    current_undistorted_points.clear();
    current_undistorted_points_by_id.clear();
    // cv::undistortPoints(current_points, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < current_points.size(); i++)
    {
        Eigen::Vector2d a(current_points[i].x, current_points[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        current_undistorted_points.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        current_undistorted_points_by_id.insert(make_pair(feature_ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", feature_ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!previous_undistorted_points_by_id.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < current_undistorted_points.size(); i++)
        {
            if (feature_ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = previous_undistorted_points_by_id.find(feature_ids[i]);
                if (it != previous_undistorted_points_by_id.end())
                {
                    double v_x = (current_undistorted_points[i].x - it->second.x) / dt;
                    double v_y = (current_undistorted_points[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < current_points.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    previous_undistorted_points_by_id = current_undistorted_points_by_id;
}
