/*
 * FAR Planner
 * Copyright (C) 2021 Fan Yang - All rights reserved
 * fanyang2@andrew.cmu.edu,
 */

#include "far_planner/contour_detector.h"

// const static int BLUR_SIZE = 10;

/***************************************************************************************/

void ContourDetector::Init(const ContourDetectParams& params) {
    cd_params_ = params;
    /* Allocate Pointcloud pointer memory */
    new_corners_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    // Init projection cv Mat
    MAT_SIZE = std::ceil(cd_params_.sensor_range * 2.0f / cd_params_.voxel_dim);
    if (MAT_SIZE % 2 == 0) MAT_SIZE++;
    MAT_RESIZE = MAT_SIZE * (int)cd_params_.kRatio;
    CMAT = MAT_SIZE / 2, CMAT_RESIZE = MAT_RESIZE / 2;
    img_mat_ = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
    img_counter_ = 0;
    odom_node_ptr_ = NULL;
    refined_contours_.clear(), refined_hierarchy_.clear();
    DIST_LIMIT = cd_params_.kRatio * 1.5f;
    ALIGN_ANGLE_COS = cos(FARUtil::kAcceptAlign / 2.0f);
    VOXEL_DIM_INV = 1.0f / cd_params_.voxel_dim;
}

void ContourDetector::BuildTerrainImgAndExtractContour(
    const NavNodePtr& odom_node_ptr, const PointCloudPtr& surround_cloud, std::vector<PointStack>& realworl_contour) {
    CVPointStack cv_corners;
    PointStack corner_vec;
    this->UpdateOdom(odom_node_ptr);
    this->ResetImgMat(img_mat_);
    this->UpdateImgMatWithCloud(surround_cloud, img_mat_);
    this->ExtractContourFromImg(img_mat_, refined_contours_, realworl_contour);
}

void ContourDetector::UpdateImgMatWithCloud(const PointCloudPtr& pc, cv::Mat& img_mat) {
    int row_idx, col_idx, inf_row, inf_col;
    const std::vector<int> inflate_vec{-1, 0, 1};
    for (const auto& pcl_p : pc->points) {
        this->PointToImgSub(pcl_p, odom_pos_, row_idx, col_idx, false, false);
        if (!this->IsIdxesInImg(row_idx, col_idx)) continue;
        for (const auto& dr : inflate_vec) {
            for (const auto& dc : inflate_vec) {
                inf_row = row_idx + dr, inf_col = col_idx + dc;
                if (this->IsIdxesInImg(inf_row, inf_col)) {
                    img_mat.at<float>(inf_row, inf_col) += 1.0;
                }
            }
        }
    }
    if (!FARUtil::IsStaticEnv) {
        cv::threshold(img_mat, img_mat, cd_params_.kThredValue, 1.0, cv::ThresholdTypes::THRESH_BINARY);
    }
    if (cd_params_.is_save_img) this->SaveCurrentImg(img_mat);
}

void ContourDetector::ResizeAndBlurImg(const cv::Mat& img, cv::Mat& Rimg) {
    img.convertTo(Rimg, CV_8UC1, 255);
    cv::resize(Rimg, Rimg, cv::Size(), cd_params_.kRatio, cd_params_.kRatio, cv::InterpolationFlags::INTER_LINEAR);
    // cv::morphologyEx(Rimg, Rimg, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::boxFilter(Rimg, Rimg, -1, cv::Size(cd_params_.kBlurSize, cd_params_.kBlurSize), cv::Point2i(-1, -1), false);
    // cv::morphologyEx(Rimg, Rimg, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT,
    // cv::Size(cd_params_.kBlurSize+2, cd_params_.kBlurSize+2)));
}

void ContourDetector::ExtractContourFromImg(
    const cv::Mat& img, std::vector<CVPointStack>& img_contours, std::vector<PointStack>& realworld_contour) {
    cv::Mat Rimg;
    this->ResizeAndBlurImg(img, Rimg);
    this->ExtractRefinedContours(Rimg, img_contours);
    this->ConvertContoursToRealWorld(img_contours, realworld_contour);
}

void ContourDetector::ConvertContoursToRealWorld(
    const std::vector<CVPointStack>& ori_contours, std::vector<PointStack>& realWorld_contours) {
    const std::size_t C_N = ori_contours.size();
    realWorld_contours.clear(), realWorld_contours.resize(C_N);
    for (std::size_t i = 0; i < C_N; i++) {
        const CVPointStack cv_contour = ori_contours[i];
        this->ConvertCVToPoint3DVector(cv_contour, realWorld_contours[i], true);
    }
}

void ContourDetector::ShowCornerImage(const cv::Mat& img_mat, const PointCloudPtr& pc) {
    cv::Mat dst = cv::Mat::zeros(MAT_RESIZE, MAT_RESIZE, CV_8UC3);
    const int circle_size = (int)(cd_params_.kRatio * 1.5);
    for (std::size_t i = 0; i < pc->size(); i++) {
        cv::Point2f cv_p = this->ConvertPoint3DToCVPoint(pc->points[i], odom_pos_, true);
        cv::circle(dst, cv_p, circle_size, cv::Scalar(128, 128, 128), -1);
    }
    // show free odom point
    cv::circle(dst, free_odom_resized_, circle_size, cv::Scalar(0, 0, 255), -1);
    std::vector<std::vector<cv::Point2i>> round_contours;
    this->RoundContours(refined_contours_, round_contours);
    for (std::size_t idx = 0; idx < round_contours.size(); idx++) {
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
        cv::drawContours(dst, round_contours, idx, color, cv::LineTypes::LINE_4);
    }
    cv::imshow("Obstacle Cloud Image", dst);
    cv::waitKey(30);
}

void ContourDetector::ExtractRefinedContours(const cv::Mat& imgIn, std::vector<CVPointStack>& refined_contours) {
    std::vector<std::vector<cv::Point2i>> raw_contours;
    refined_contours.clear(), refined_hierarchy_.clear();
    cv::findContours(imgIn, raw_contours, refined_hierarchy_, cv::RetrievalModes::RETR_TREE,
        cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1);

    refined_contours.resize(raw_contours.size());
    for (std::size_t i = 0; i < raw_contours.size(); i++) {
        // using Ramer–Douglas–Peucker algorithm url:
        // https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        cv::approxPolyDP(raw_contours[i], refined_contours[i], DIST_LIMIT, true);
    }
    this->TopoFilterContours(refined_contours);
    this->AdjecentDistanceFilter(refined_contours);
}

void ContourDetector::AdjecentDistanceFilter(std::vector<CVPointStack>& contoursInOut) {
    /* filter out vertices that are overlapped with neighbor */
    std::unordered_set<int> remove_idxs;
    for (std::size_t i = 0; i < contoursInOut.size(); i++) {
        const auto c = contoursInOut[i];
        const std::size_t c_size = c.size();
        std::size_t refined_idx = 0;
        for (std::size_t j = 0; j < c_size; j++) {
            cv::Point2f p = c[j];
            if (refined_idx < 1 || FARUtil::PixelDistance(contoursInOut[i][refined_idx - 1], p) > DIST_LIMIT) {
                /** Reduce wall nodes */
                RemoveWallConnection(contoursInOut[i], p, refined_idx);
                contoursInOut[i][refined_idx] = p;
                refined_idx++;
            }
        }
        /** Reduce wall nodes */
        RemoveWallConnection(contoursInOut[i], contoursInOut[i][0], refined_idx);
        contoursInOut[i].resize(refined_idx);
        if (refined_idx > 1 && FARUtil::PixelDistance(contoursInOut[i].front(), contoursInOut[i].back()) < DIST_LIMIT) {
            contoursInOut[i].pop_back();
        }
        if (contoursInOut[i].size() < 3) remove_idxs.insert(i);
    }
    if (!remove_idxs.empty()) {  // clear contour with vertices size less that 3
        std::vector<CVPointStack> temp_contours = contoursInOut;
        contoursInOut.clear();
        for (int i = 0; i < temp_contours.size(); i++) {
            if (remove_idxs.find(i) != remove_idxs.end()) continue;
            contoursInOut.push_back(temp_contours[i]);
        }
    }
}

void ContourDetector::TopoFilterContours(std::vector<CVPointStack>& contoursInOut) {
    std::unordered_set<int> remove_idxs;
    for (int i = 0; i < contoursInOut.size(); i++) {
        if (remove_idxs.find(i) != remove_idxs.end()) continue;
        const auto poly = contoursInOut[i];
        if (poly.size() < 3) {
            remove_idxs.insert(i);
        } else if (!FARUtil::PointInsideAPoly(poly, free_odom_resized_)) {
            InternalContoursIdxs(refined_hierarchy_, i, remove_idxs);
        }
    }
    if (!remove_idxs.empty()) {
        std::vector<CVPointStack> temp_contours = contoursInOut;
        contoursInOut.clear();
        for (int i = 0; i < temp_contours.size(); i++) {
            if (remove_idxs.find(i) != remove_idxs.end()) continue;
            contoursInOut.push_back(temp_contours[i]);
        }
    }
}

// 陡坡提取
void ContourDetector::ExtractSteepSlopePoints(const PointCloudPtr& steep_cloud, const NavNodePtr& odom_node_ptr,
    std::vector<PointStack>& boundary_clusters, std::vector<PointStack>& inner_clusters) {
    boundary_clusters.clear();
    inner_clusters.clear();

    if (steep_cloud->empty()) {
        ROS_WARN("CD: Input steep cloud is empty");
        return;
    }

    // 更新机器人位置
    this->UpdateOdom(odom_node_ptr);

    // ============================================================
    // 步骤 1: 裁剪点云(只保留xy平面距离车10米以内的点)
    // ============================================================
    PointCloudPtr cropped_cloud(new pcl::PointCloud<PCLPoint>());
    const float crop_radius = cd_params_.steep_crop_radius;

    for (const auto& p : steep_cloud->points) {
        float dist_xy = std::hypotf(p.x - odom_pos_.x, p.y - odom_pos_.y);
        if (dist_xy <= crop_radius) {
            cropped_cloud->push_back(p);
        }
    }

    if (cropped_cloud->empty()) {
        ROS_WARN("CD: No steep points within crop radius %.1fm", crop_radius);
        return;
    }

    ROS_INFO("CD: Cropped %lu steep points within %.1fm", cropped_cloud->size(), crop_radius);

    // ============================================================
    // 步骤 2: 欧式聚类
    // ============================================================
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<PCLPoint>::Ptr tree(new pcl::search::KdTree<PCLPoint>);
    tree->setInputCloud(cropped_cloud);

    pcl::EuclideanClusterExtraction<PCLPoint> ec;
    ec.setClusterTolerance(cd_params_.steep_cluster_tolerance);
    ec.setMinClusterSize(cd_params_.steep_min_cluster_size);
    ec.setMaxClusterSize(cd_params_.steep_max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cropped_cloud);
    ec.extract(cluster_indices);

    if (cluster_indices.empty()) {
        ROS_WARN("CD: No valid clusters found");
        return;
    }

    ROS_INFO("CD: Found %lu steep slope clusters", cluster_indices.size());

    // [改进] 预分配空间
    boundary_clusters.resize(cluster_indices.size());
    inner_clusters.resize(cluster_indices.size());

    // ============================================================
    // 步骤 3: 对每个聚类提取边界点和内部点
    // ============================================================
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        // 提取当前聚类
        PointCloudPtr cluster(new pcl::PointCloud<PCLPoint>());
        for (int idx : cluster_indices[i].indices) {
            cluster->push_back((*cropped_cloud)[idx]);
        }

        ROS_INFO("CD: Processing cluster %lu with %lu points", i, cluster->size());

        // [改进] 直接存储到对应的聚类索引中
        ExtractBoundaryPoints(cluster, boundary_clusters[i]);
        ExtractInnerPoints(cluster, inner_clusters[i]);

        ROS_INFO(
            "CD: Cluster %lu - boundary: %lu, inner: %lu", i, boundary_clusters[i].size(), inner_clusters[i].size());
    }

    ROS_INFO("CD: Total clusters: %lu", boundary_clusters.size());
}

// void ContourDetector::ExtractBoundaryPoints(const PointCloudPtr& cluster, PointStack& boundary_points) {
//     boundary_points.clear();
//     if (cluster->size() < 3) return;

//     // 投影到xy平面
//     PointCloudPtr cloud_2d(new pcl::PointCloud<PCLPoint>());
//     for (const auto& p : cluster->points) {
//         PCLPoint p2d = p;
//         p2d.z = 0.0f;
//         cloud_2d->push_back(p2d);
//     }

//     // 提取凹包
//     pcl::PointCloud<PCLPoint>::Ptr hull_points(new pcl::PointCloud<PCLPoint>());
//     pcl::ConcaveHull<PCLPoint> hull;
//     hull.setInputCloud(cloud_2d);
//     hull.setDimension(2);
//     hull.setAlpha(cd_params_.steep_concave_alpha);
//     hull.reconstruct(*hull_points);

//     if (hull_points->size() < 3) {
//         pcl::ConvexHull<PCLPoint> convex_hull;
//         convex_hull.setInputCloud(cloud_2d);
//         convex_hull.setDimension(2);
//         convex_hull.reconstruct(*hull_points);
//         if (hull_points->size() < 3) return;
//     }

//     // 平滑处理
//     std::vector<PCLPoint> smoothed_hull;
//     const int smooth_window = cd_params_.steep_smooth_window;
//     for (size_t i = 0; i < hull_points->size(); i++) {
//         PCLPoint avg_p;
//         avg_p.x = 0.0f;
//         avg_p.y = 0.0f;
//         avg_p.z = 0.0f;

//         int count = 0;
//         for (int offset = -smooth_window; offset <= smooth_window; offset++) {
//             int idx = (i + offset + hull_points->size()) % hull_points->size();
//             avg_p.x += hull_points->points[idx].x;
//             avg_p.y += hull_points->points[idx].y;
//             count++;
//         }
//         avg_p.x /= count;
//         avg_p.y /= count;
//         smoothed_hull.push_back(avg_p);
//     }

//     // [改进] 栅格约束的周长采样
//     const float grid_size = cd_params_.steep_boundary_sample_dist;
//     const float sample_dist = grid_size * 0.8f;  // 比栅格稍小,确保覆盖

//     // 计算总周长
//     float total_perimeter = 0.0f;
//     std::vector<float> edge_lengths;
//     for (size_t i = 0; i < smoothed_hull.size(); i++) {
//         PCLPoint p1 = smoothed_hull[i];
//         PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
//         float edge_len = std::hypotf(p2.x - p1.x, p2.y - p1.y);
//         edge_lengths.push_back(edge_len);
//         total_perimeter += edge_len;
//     }

//     int total_samples = std::max(3, (int)(total_perimeter / sample_dist));
//     float actual_sample_dist = total_perimeter / total_samples;

//     // [关键改进] 使用栅格去重,但采样点在轮廓线上
//     std::set<std::pair<int, int>> sampled_grids;

//     pcl::KdTreeFLANN<PCLPoint> kdtree;
//     kdtree.setInputCloud(cluster);

//     float accumulated_length = 0.0f;
//     float next_sample_dist = 0.0f;

//     for (size_t i = 0; i < smoothed_hull.size(); i++) {
//         PCLPoint p1 = smoothed_hull[i];
//         PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
//         float edge_len = edge_lengths[i];

//         float edge_start = accumulated_length;
//         float edge_end = accumulated_length + edge_len;

//         while (next_sample_dist < edge_end) {
//             float t = (next_sample_dist - edge_start) / edge_len;
//             PCLPoint sample_p;
//             sample_p.x = p1.x + t * (p2.x - p1.x);
//             sample_p.y = p1.y + t * (p2.y - p1.y);
//             sample_p.z = 0.0f;

//             // 计算所属栅格
//             int gx = (int)std::round(sample_p.x / grid_size);
//             int gy = (int)std::round(sample_p.y / grid_size);

//             // 跳过已采样的栅格
//             if (!sampled_grids.count({gx, gy})) {
//                 sampled_grids.insert({gx, gy});

//                 // [关键] 采样点就在轮廓线上,不对齐到栅格中心
//                 std::vector<int> idx(1);
//                 std::vector<float> dist(1);

//                 if (kdtree.nearestKSearch(sample_p, 1, idx, dist) > 0) {
//                     const PCLPoint& nearest_p = cluster->points[idx[0]];
//                     boundary_points.push_back(Point3D(nearest_p.x, nearest_p.y, nearest_p.z));
//                 }
//             }

//             next_sample_dist += actual_sample_dist;
//         }

//         accumulated_length += edge_len;
//     }

//     ROS_INFO("CD: Steep boundary sampling: %lu points (grid-constrained)", boundary_points.size());
// }
void ContourDetector::ExtractBoundaryPoints(const PointCloudPtr& cluster, PointStack& boundary_points) {
    boundary_points.clear();
    if (cluster->size() < 3) return;

    // 投影到xy平面
    PointCloudPtr cloud_2d(new pcl::PointCloud<PCLPoint>());
    for (const auto& p : cluster->points) {
        PCLPoint p2d = p;
        p2d.z = 0.0f;
        cloud_2d->push_back(p2d);
    }

    // [改进1] 先尝试凹包,失败则用凸包
    pcl::PointCloud<PCLPoint>::Ptr hull_points(new pcl::PointCloud<PCLPoint>());
    bool is_concave = true;

    pcl::ConcaveHull<PCLPoint> concave_hull;
    concave_hull.setInputCloud(cloud_2d);
    concave_hull.setDimension(2);
    concave_hull.setAlpha(cd_params_.steep_concave_alpha);
    concave_hull.reconstruct(*hull_points);

    // 凹包失败检测:点数太少或退化
    if (hull_points->size() < 3 || hull_points->size() < cloud_2d->size() * 0.3) {
        ROS_WARN("CD: Concave hull failed/degenerated, using convex hull");
        hull_points->clear();
        pcl::ConvexHull<PCLPoint> convex_hull;
        convex_hull.setInputCloud(cloud_2d);
        convex_hull.setDimension(2);
        convex_hull.reconstruct(*hull_points);
        is_concave = false;

        if (hull_points->size() < 3) return;
    }

    // 平滑处理
    std::vector<PCLPoint> smoothed_hull;
    const int smooth_window = cd_params_.steep_smooth_window;
    for (size_t i = 0; i < hull_points->size(); i++) {
        PCLPoint avg_p;
        avg_p.x = 0.0f;
        avg_p.y = 0.0f;
        avg_p.z = 0.0f;

        int count = 0;
        for (int offset = -smooth_window; offset <= smooth_window; offset++) {
            int idx = (i + offset + hull_points->size()) % hull_points->size();
            avg_p.x += hull_points->points[idx].x;
            avg_p.y += hull_points->points[idx].y;
            count++;
        }
        avg_p.x /= count;
        avg_p.y /= count;
        smoothed_hull.push_back(avg_p);
    }

    // ========================================================
    // [核心改进] 两阶段采样: 先固定拐点,再插值采样
    // ========================================================
    const float grid_size = cd_params_.steep_boundary_sample_dist;
    const float angle_threshold = cd_params_.steep_corner_angle_threshold;  // 拐点角度阈值(度)
    const float angle_threshold_rad = angle_threshold * M_PI / 180.0f;

    // 阶段1: 检测并固定拐点
    struct CornerPoint {
        size_t index;              // 在轮廓中的索引
        PCLPoint position;         // 原始位置
        std::pair<int, int> grid;  // 对齐的栅格坐标
    };

    std::vector<CornerPoint> corner_points;
    std::set<std::pair<int, int>> used_grids;

    for (size_t i = 0; i < smoothed_hull.size(); i++) {
        PCLPoint p_prev = smoothed_hull[(i - 1 + smoothed_hull.size()) % smoothed_hull.size()];
        PCLPoint p_curr = smoothed_hull[i];
        PCLPoint p_next = smoothed_hull[(i + 1) % smoothed_hull.size()];

        // 计算两条边的向量
        float v1_x = p_curr.x - p_prev.x;
        float v1_y = p_curr.y - p_prev.y;
        float v2_x = p_next.x - p_curr.x;
        float v2_y = p_next.y - p_curr.y;

        float len1 = std::hypotf(v1_x, v1_y);
        float len2 = std::hypotf(v2_x, v2_y);

        if (len1 < 1e-6 || len2 < 1e-6) continue;

        // 归一化并计算夹角
        v1_x /= len1;
        v1_y /= len1;
        v2_x /= len2;
        v2_y /= len2;

        float dot = v1_x * v2_x + v1_y * v2_y;
        float angle = std::acos(std::max(-1.0f, std::min(1.0f, dot)));

        // [关键判断] 是否为拐点
        if (angle > angle_threshold_rad) {
            int gx = (int)std::round(p_curr.x / grid_size);
            int gy = (int)std::round(p_curr.y / grid_size);

            // 避免拐点在同一栅格(优先保留角度更大的)
            if (!used_grids.count({gx, gy})) {
                corner_points.push_back({i, p_curr, {gx, gy}});
                used_grids.insert({gx, gy});
            }
        }
    }

    ROS_INFO("CD: Detected %lu corner points (angle > %.1f deg)", corner_points.size(), angle_threshold);

    // 阶段2: 在相邻拐点之间插值采样
    pcl::KdTreeFLANN<PCLPoint> kdtree;
    kdtree.setInputCloud(cluster);

    // 如果没有拐点,退化为全周长均匀采样
    if (corner_points.empty()) {
        ROS_WARN("CD: No corners detected, fallback to uniform sampling");

        float total_perimeter = 0.0f;
        for (size_t i = 0; i < smoothed_hull.size(); i++) {
            PCLPoint p1 = smoothed_hull[i];
            PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
            total_perimeter += std::hypotf(p2.x - p1.x, p2.y - p1.y);
        }

        int total_samples = std::max(3, (int)(total_perimeter / grid_size));
        float actual_sample_dist = total_perimeter / total_samples;

        float accumulated = 0.0f, next_sample = 0.0f;
        for (size_t i = 0; i < smoothed_hull.size(); i++) {
            PCLPoint p1 = smoothed_hull[i];
            PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
            float edge_len = std::hypotf(p2.x - p1.x, p2.y - p1.y);

            while (next_sample < accumulated + edge_len) {
                float t = (next_sample - accumulated) / edge_len;
                PCLPoint sample_p;
                sample_p.x = p1.x + t * (p2.x - p1.x);
                sample_p.y = p1.y + t * (p2.y - p1.y);
                sample_p.z = 0.0f;

                std::vector<int> idx(1);
                std::vector<float> dist(1);
                if (kdtree.nearestKSearch(sample_p, 1, idx, dist) > 0) {
                    boundary_points.push_back(
                        Point3D(cluster->points[idx[0]].x, cluster->points[idx[0]].y, cluster->points[idx[0]].z));
                }
                next_sample += actual_sample_dist;
            }
            accumulated += edge_len;
        }

        ROS_INFO("CD: Steep boundary: %lu points (uniform)", boundary_points.size());
        return;
    }

    // 有拐点: 先添加所有拐点,再插值
    for (const auto& corner : corner_points) {
        std::vector<int> idx(1);
        std::vector<float> dist(1);
        if (kdtree.nearestKSearch(corner.position, 1, idx, dist) > 0) {
            boundary_points.push_back(
                Point3D(cluster->points[idx[0]].x, cluster->points[idx[0]].y, cluster->points[idx[0]].z));
        }
    }

    // 在相邻拐点之间插值采样
    for (size_t ci = 0; ci < corner_points.size(); ci++) {
        size_t start_idx = corner_points[ci].index;
        size_t end_idx = corner_points[(ci + 1) % corner_points.size()].index;

        // 计算拐点之间的弧长
        float arc_length = 0.0f;
        size_t current = start_idx;
        while (current != end_idx) {
            size_t next = (current + 1) % smoothed_hull.size();
            PCLPoint p1 = smoothed_hull[current];
            PCLPoint p2 = smoothed_hull[next];
            arc_length += std::hypotf(p2.x - p1.x, p2.y - p1.y);
            current = next;
        }

        // 在拐点之间插值
        int num_samples = std::max(0, (int)(arc_length / grid_size) - 1);  // 减1避免重复拐点

        if (num_samples > 0) {
            float sample_interval = arc_length / (num_samples + 1);
            float accumulated = 0.0f;
            float next_sample = sample_interval;

            current = start_idx;
            while (current != end_idx) {
                size_t next = (current + 1) % smoothed_hull.size();
                PCLPoint p1 = smoothed_hull[current];
                PCLPoint p2 = smoothed_hull[next];
                float edge_len = std::hypotf(p2.x - p1.x, p2.y - p1.y);

                while (next_sample < accumulated + edge_len && next_sample < arc_length) {
                    float t = (next_sample - accumulated) / edge_len;
                    PCLPoint sample_p;
                    sample_p.x = p1.x + t * (p2.x - p1.x);
                    sample_p.y = p1.y + t * (p2.y - p1.y);
                    sample_p.z = 0.0f;

                    // 栅格去重
                    int gx = (int)std::round(sample_p.x / grid_size);
                    int gy = (int)std::round(sample_p.y / grid_size);

                    if (!used_grids.count({gx, gy})) {
                        used_grids.insert({gx, gy});

                        std::vector<int> idx(1);
                        std::vector<float> dist(1);
                        if (kdtree.nearestKSearch(sample_p, 1, idx, dist) > 0) {
                            boundary_points.push_back(Point3D(
                                cluster->points[idx[0]].x, cluster->points[idx[0]].y, cluster->points[idx[0]].z));
                        }
                    }

                    next_sample += sample_interval;
                }

                accumulated += edge_len;
                current = next;
            }
        }
    }

    ROS_INFO("CD: Steep boundary: %lu points (%lu corners + interpolation, %s hull)", boundary_points.size(),
        corner_points.size(), is_concave ? "concave" : "convex");
}
// ============================================================
// 辅助函数: 提取内部点(体素滤波)
// ============================================================
void ContourDetector::ExtractInnerPoints(PointCloudPtr& cluster, PointStack& inner_points) {
    // 函数体保持不变，只是参数类型从 std::vector<Point3D>& 改为 PointStack&
    inner_points.clear();

    if (cluster->size() < 5) {
        return;
    }

    // 使用体素滤波进行稀疏采样
    float voxel_size = cd_params_.steep_inner_voxel_size;
    FARUtil::FilterCloud(cluster, voxel_size);

    // 转换为 Point3D
    for (const auto& p : cluster->points) {
        inner_points.push_back(Point3D(p.x, p.y, p.z));
    }
}

// ============================================================
// [新增] 缓坡提取主函数
// ============================================================
void ContourDetector::ExtractModerateSlopePoints(const PointCloudPtr& moderate_cloud, const NavNodePtr& odom_node_ptr,
    std::vector<PointStack>& boundary_clusters, std::vector<PointStack>& inner_clusters) {
    boundary_clusters.clear();
    inner_clusters.clear();

    if (moderate_cloud->empty()) {
        ROS_WARN("CD: Input moderate cloud is empty");
        return;
    }

    // ============================================================
    // 步骤 1: 裁剪点云
    // ============================================================
    PointCloudPtr cropped_cloud(new pcl::PointCloud<PCLPoint>());
    const float crop_radius = cd_params_.moderate_crop_radius;

    for (const auto& p : moderate_cloud->points) {
        float dist_xy = std::hypotf(p.x - odom_pos_.x, p.y - odom_pos_.y);
        if (dist_xy <= crop_radius) {
            cropped_cloud->push_back(p);
        }
    }

    if (cropped_cloud->empty()) {
        ROS_WARN("CD: No moderate points within crop radius %.1fm", crop_radius);
        return;
    }

    ROS_INFO("CD: Cropped %lu moderate points within %.1fm", cropped_cloud->size(), crop_radius);

    // ============================================================
    // 步骤 2: 欧式聚类
    // ============================================================
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<PCLPoint>::Ptr tree(new pcl::search::KdTree<PCLPoint>);
    tree->setInputCloud(cropped_cloud);

    pcl::EuclideanClusterExtraction<PCLPoint> ec;
    ec.setClusterTolerance(cd_params_.moderate_cluster_tolerance);
    ec.setMinClusterSize(cd_params_.moderate_min_cluster_size);
    ec.setMaxClusterSize(cd_params_.moderate_max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cropped_cloud);
    ec.extract(cluster_indices);

    if (cluster_indices.empty()) {
        ROS_WARN("CD: No valid moderate clusters found");
        return;
    }

    ROS_INFO("CD: Found %lu moderate slope clusters", cluster_indices.size());

    // 预分配空间
    boundary_clusters.resize(cluster_indices.size());
    inner_clusters.resize(cluster_indices.size());

    // ============================================================
    // 步骤 3: 对每个聚类提取边界点和内部点
    // ============================================================
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        // 提取当前聚类
        PointCloudPtr cluster(new pcl::PointCloud<PCLPoint>());
        for (int idx : cluster_indices[i].indices) {
            cluster->push_back((*cropped_cloud)[idx]);
        }

        ROS_INFO("CD: Processing moderate cluster %lu with %lu points", i, cluster->size());

        // 提取边界点和内部点
        // ExtractModerateBoundaryPoints(cluster, boundary_clusters[i]);
        ExtractModerateInnerPoints(cluster, inner_clusters[i]);
        boundary_clusters.clear();

        ROS_INFO("CD: Moderate cluster %lu - boundary: %lu, inner: %lu", i, boundary_clusters[i].size(),
            inner_clusters[i].size());
    }

    ROS_INFO("CD: Total moderate clusters: %lu", boundary_clusters.size());
}

// ============================================================
// [新增] 辅助函数: 提取缓坡边界点(凹包采样)
// ============================================================
void ContourDetector::ExtractModerateBoundaryPoints(const PointCloudPtr& cluster, PointStack& boundary_points) {
    boundary_points.clear();

    if (cluster->size() < 3) {
        return;
    }

    // 投影到xy平面
    PointCloudPtr cloud_2d(new pcl::PointCloud<PCLPoint>());
    for (const auto& p : cluster->points) {
        PCLPoint p2d = p;
        p2d.z = 0.0f;
        cloud_2d->push_back(p2d);
    }

    // 使用凹包提取边界（alpha值更大，能提取圆环内圈）
    pcl::PointCloud<PCLPoint>::Ptr hull_points(new pcl::PointCloud<PCLPoint>());
    pcl::ConcaveHull<PCLPoint> hull;
    hull.setInputCloud(cloud_2d);
    hull.setDimension(2);
    hull.setAlpha(cd_params_.moderate_concave_alpha);
    hull.reconstruct(*hull_points);

    if (hull_points->size() < 3) {
        ROS_WARN("CD: Moderate concave hull failed, fallback to convex hull");
        pcl::ConvexHull<PCLPoint> convex_hull;
        convex_hull.setInputCloud(cloud_2d);
        convex_hull.setDimension(2);
        convex_hull.reconstruct(*hull_points);

        if (hull_points->size() < 3) {
            return;
        }
    }

    // 平滑处理
    std::vector<PCLPoint> smoothed_hull;
    const int smooth_window = cd_params_.moderate_smooth_window;
    for (size_t i = 0; i < hull_points->size(); i++) {
        PCLPoint avg_p;
        avg_p.x = 0.0f;
        avg_p.y = 0.0f;
        avg_p.z = 0.0f;

        int count = 0;
        for (int offset = -smooth_window; offset <= smooth_window; offset++) {
            int idx = (i + offset + hull_points->size()) % hull_points->size();
            avg_p.x += hull_points->points[idx].x;
            avg_p.y += hull_points->points[idx].y;
            count++;
        }

        avg_p.x /= count;
        avg_p.y /= count;
        smoothed_hull.push_back(avg_p);
    }

    // 按周长均匀采样（更稀疏）
    const float sample_dist = cd_params_.moderate_boundary_sample_dist;

    // 计算总周长
    float total_perimeter = 0.0f;
    std::vector<float> edge_lengths;
    for (size_t i = 0; i < smoothed_hull.size(); i++) {
        PCLPoint p1 = smoothed_hull[i];
        PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
        float edge_len = std::hypotf(p2.x - p1.x, p2.y - p1.y);
        edge_lengths.push_back(edge_len);
        total_perimeter += edge_len;
    }

    int total_samples = std::max(3, (int)(total_perimeter / sample_dist));
    float actual_sample_dist = total_perimeter / total_samples;

    ROS_INFO(
        "CD: Moderate perimeter=%.2fm, samples=%d, dist=%.2fm", total_perimeter, total_samples, actual_sample_dist);

    // 沿周长采样
    float accumulated_length = 0.0f;
    float next_sample_dist = 0.0f;

    pcl::KdTreeFLANN<PCLPoint> kdtree;
    kdtree.setInputCloud(cluster);

    for (size_t i = 0; i < smoothed_hull.size(); i++) {
        PCLPoint p1 = smoothed_hull[i];
        PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
        float edge_len = edge_lengths[i];

        float edge_start = accumulated_length;
        float edge_end = accumulated_length + edge_len;

        while (next_sample_dist < edge_end) {
            float t = (next_sample_dist - edge_start) / edge_len;
            PCLPoint sample_p;
            sample_p.x = p1.x + t * (p2.x - p1.x);
            sample_p.y = p1.y + t * (p2.y - p1.y);
            sample_p.z = 0.0f;

            std::vector<int> idx(1);
            std::vector<float> dist(1);

            if (kdtree.nearestKSearch(sample_p, 1, idx, dist) > 0) {
                const PCLPoint& nearest_p = cluster->points[idx[0]];
                boundary_points.push_back(Point3D(nearest_p.x, nearest_p.y, nearest_p.z));
            }

            next_sample_dist += actual_sample_dist;
        }

        accumulated_length += edge_len;
    }
}

// [新增] 辅助函数: 沿轮廓均匀采样
void ContourDetector::SampleContourUniformly(const std::vector<PCLPoint>& contour, pcl::KdTreeFLANN<PCLPoint>& kdtree,
    const PointCloudPtr& cluster, PointStack& sampled_points, float sample_dist) {
    // 计算周长
    float total_perimeter = 0.0f;
    std::vector<float> edge_lengths;
    for (size_t i = 0; i < contour.size(); i++) {
        PCLPoint p1 = contour[i];
        PCLPoint p2 = contour[(i + 1) % contour.size()];
        float edge_len = std::hypotf(p2.x - p1.x, p2.y - p1.y);
        edge_lengths.push_back(edge_len);
        total_perimeter += edge_len;
    }

    int total_samples = std::max(3, (int)(total_perimeter / sample_dist));
    float actual_sample_dist = total_perimeter / total_samples;

    // 沿周长采样
    float accumulated_length = 0.0f;
    float next_sample_dist = 0.0f;

    for (size_t i = 0; i < contour.size(); i++) {
        PCLPoint p1 = contour[i];
        PCLPoint p2 = contour[(i + 1) % contour.size()];
        float edge_len = edge_lengths[i];

        float edge_start = accumulated_length;
        float edge_end = accumulated_length + edge_len;

        while (next_sample_dist < edge_end) {
            float t = (next_sample_dist - edge_start) / edge_len;
            PCLPoint sample_p;
            sample_p.x = p1.x + t * (p2.x - p1.x);
            sample_p.y = p1.y + t * (p2.y - p1.y);
            sample_p.z = 0.0f;

            std::vector<int> idx(1);
            std::vector<float> dist(1);

            if (kdtree.nearestKSearch(sample_p, 1, idx, dist) > 0) {
                const PCLPoint& nearest_p = cluster->points[idx[0]];
                sampled_points.push_back(Point3D(nearest_p.x, nearest_p.y, nearest_p.z));
            }

            next_sample_dist += actual_sample_dist;
        }

        accumulated_length += edge_len;
    }
}

// ============================================================
// [新增] 辅助函数: 提取缓坡内部点(体素滤波，更稀疏)
// ============================================================

// void ContourDetector::ExtractModerateInnerPoints(PointCloudPtr& cluster, PointStack& inner_points) {
//     inner_points.clear();

//     if (cluster->size() < 5) {
//         return;
//     }
//     // 使用体素滤波进行稀疏采样
//     float voxel_size = cd_params_.moderate_inner_voxel_size;
//     FARUtil::FilterCloud(cluster, voxel_size);

//     // 转换为 Point3D
//     for (const auto& p : cluster->points) {
//         inner_points.push_back(Point3D(p.x, p.y, p.z));
//     }
// }

void ContourDetector::ExtractModerateInnerPoints(PointCloudPtr& cluster, PointStack& inner_points) {
    inner_points.clear();

    if (cluster->size() < 5) {
        return;
    }

    // [改进1] 对齐到整数坐标网格
    const float grid_size = cd_params_.moderate_inner_voxel_size;  // 例如 4.0m
    const float crop_radius = cd_params_.moderate_crop_radius;

    // 使用map去重,key是整数坐标
    std::map<std::pair<int, int>, std::vector<PCLPoint>> grid_map;

    for (const auto& p : cluster->points) {
        // 检查范围
        float dist_xy = std::hypotf(p.x - odom_pos_.x, p.y - odom_pos_.y);
        if (dist_xy > crop_radius) {
            continue;
        }

        // [关键] 对齐到最近的整数坐标(单位是grid_size)
        int gx = (int)std::round(p.x / grid_size);
        int gy = (int)std::round(p.y / grid_size);

        grid_map[{gx, gy}].push_back(p);
    }

    // [改进2] 对每个网格,找离整数坐标最近的点
    for (const auto& [grid_key, points] : grid_map) {
        if (points.empty()) continue;

        // 整数坐标位置(世界坐标系)
        float target_x = grid_key.first * grid_size;
        float target_y = grid_key.second * grid_size;

        // 找距离目标位置最近的点
        float min_dist = std::numeric_limits<float>::max();
        PCLPoint best_point;

        for (const auto& p : points) {
            float dx = p.x - target_x;
            float dy = p.y - target_y;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist) {
                min_dist = dist_sq;
                best_point = p;
            }
        }

        // 添加最优点(使用其真实xyz坐标)
        inner_points.push_back(Point3D(best_point.x, best_point.y, best_point.z));
    }

    // ROS_INFO("CD: Moderate inner points: %lu unique grids (aligned to integers)", inner_points.size());
}
