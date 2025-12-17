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

void ContourDetector::ExtractContoursFromMask(
    const cv::Mat& mask_img, const NavNodePtr& odom_node_ptr, std::vector<PointStack>& realworld_contour) {
    if (mask_img.empty()) {
        realworld_contour.clear();
        return;
    }

    // 更新机器人位置
    this->UpdateOdom(odom_node_ptr);

    // 将输入的 mask 转换为合适的格式
    cv::Mat img_for_contour;
    if (mask_img.type() != CV_8UC1) {
        mask_img.convertTo(img_for_contour, CV_8UC1);
    } else {
        img_for_contour = mask_img.clone();
    }

    // 缩放和模糊处理
    cv::Mat Rimg;
    this->ResizeAndBlurImg(img_for_contour, Rimg);

    // 提取轮廓
    std::vector<CVPointStack> img_contours;
    this->ExtractRefinedContours(Rimg, img_contours);

    // 转换为世界坐标
    this->ConvertContoursToRealWorld(img_contours, realworld_contour);
    ROS_INFO("poly size:%ld", realworld_contour.size());
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

// ============================================================
// 辅助函数: 提取边界点(凹包采样)
// ============================================================
void ContourDetector::ExtractBoundaryPoints(const PointCloudPtr& cluster, PointStack& boundary_points) {
    // 函数体保持不变，只是参数类型从 std::vector<Point3D>& 改为 PointStack&
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

    // [改进1] 使用凹包（ConcaveHull）替代凸包，能处理凹进去的边界
    pcl::PointCloud<PCLPoint>::Ptr hull_points(new pcl::PointCloud<PCLPoint>());
    pcl::ConcaveHull<PCLPoint> hull;
    hull.setInputCloud(cloud_2d);
    hull.setDimension(2);
    hull.setAlpha(cd_params_.steep_concave_alpha);  // alpha值控制凹包的"凹陷程度"，值越小越贴合原始形状
    hull.reconstruct(*hull_points);

    if (hull_points->size() < 3) {
        ROS_WARN("CD: Concave hull has less than 3 points, fallback to convex hull");
        // 如果凹包失败，回退到凸包
        pcl::ConvexHull<PCLPoint> convex_hull;
        convex_hull.setInputCloud(cloud_2d);
        convex_hull.setDimension(2);
        convex_hull.reconstruct(*hull_points);

        if (hull_points->size() < 3) {
            return;
        }
    }

    // [改进2] 对凹包边界进行平滑处理（移动平均滤波）
    std::vector<PCLPoint> smoothed_hull;
    const int smooth_window = cd_params_.steep_smooth_window;  // 平滑窗口大小
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

    // [改进] 按周长均匀采样，而不是按边采样
    const float sample_dist = cd_params_.steep_boundary_sample_dist;

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

    // 计算需要的总采样点数
    int total_samples = std::max(3, (int)(total_perimeter / sample_dist));
    float actual_sample_dist = total_perimeter / total_samples;

    ROS_INFO("CD: Perimeter=%.2fm, target samples=%d, actual dist=%.2fm", total_perimeter, total_samples,
        actual_sample_dist);

    // 沿周长均匀采样
    float accumulated_length = 0.0f;
    float next_sample_dist = 0.0f;

    // [修复] 把 KdTree 构建移到循环外面
    pcl::KdTreeFLANN<PCLPoint> kdtree;
    kdtree.setInputCloud(cluster);  // 只构建一次

    for (size_t i = 0; i < smoothed_hull.size(); i++) {
        PCLPoint p1 = smoothed_hull[i];
        PCLPoint p2 = smoothed_hull[(i + 1) % smoothed_hull.size()];
        float edge_len = edge_lengths[i];

        float edge_start = accumulated_length;
        float edge_end = accumulated_length + edge_len;

        // 在当前边上采样
        while (next_sample_dist < edge_end) {
            float t = (next_sample_dist - edge_start) / edge_len;
            PCLPoint sample_p;
            sample_p.x = p1.x + t * (p2.x - p1.x);
            sample_p.y = p1.y + t * (p2.y - p1.y);
            sample_p.z = 0.0f;

            // 找原始点云中最近的点(恢复z值)
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