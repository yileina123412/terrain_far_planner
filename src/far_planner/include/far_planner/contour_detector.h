#ifndef CONTOUR_DETECTOR_H
#define CONTOUR_DETECTOR_H

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>

#include "map_handler.h"
#include "utility.h"

struct ContourDetectParams {
    ContourDetectParams() = default;
    float sensor_range;
    float voxel_dim;
    float kRatio;
    int kThredValue;
    int kBlurSize;
    bool is_save_img;
    std::string img_path;

    // [新增] 陡坡处理参数
    float steep_crop_radius;             // 陡坡点云裁剪半径(米)
    float steep_cluster_tolerance;       // 聚类容差(米)
    int steep_min_cluster_size;          // 最小聚类点数
    int steep_max_cluster_size;          // 最大聚类点数
    float steep_boundary_sample_dist;    // 边界采样间距(米)
    float steep_inner_voxel_size;        // 内部体素滤波大小(米)
    float steep_concave_alpha;           // [新增] 凹包alpha值
    int steep_smooth_window;             // [新增] 边界平滑窗口大小
    float steep_corner_angle_threshold;  // [新增] 拐点角度阈值(度)

    // [新增] 缓坡处理参数
    float moderate_crop_radius;
    float moderate_cluster_tolerance;
    int moderate_min_cluster_size;
    int moderate_max_cluster_size;
    float moderate_boundary_sample_dist;
    float moderate_inner_voxel_size;
    float moderate_concave_alpha;
    int moderate_smooth_window;
};

class ContourDetector {
private:
    Point3D odom_pos_;
    cv::Point2f free_odom_resized_;
    ContourDetectParams cd_params_;
    PointCloudPtr new_corners_cloud_;
    cv::Mat img_mat_;
    std::size_t img_counter_;
    std::vector<CVPointStack> refined_contours_;
    std::vector<cv::Vec4i> refined_hierarchy_;
    NavNodePtr odom_node_ptr_;

    int MAT_SIZE, CMAT;
    int MAT_RESIZE, CMAT_RESIZE;
    float DIST_LIMIT;
    float ALIGN_ANGLE_COS;
    float VOXEL_DIM_INV;

    void UpdateImgMatWithCloud(const PointCloudPtr& pc, cv::Mat& img_mat);

    void ExtractContourFromImg(
        const cv::Mat& img, std::vector<CVPointStack>& img_contours, std::vector<PointStack>& realworld_contour);

    void ExtractRefinedContours(const cv::Mat& imgIn, std::vector<CVPointStack>& refined_contours);

    void ResizeAndBlurImg(const cv::Mat& img, cv::Mat& Rimg);

    void ConvertContoursToRealWorld(
        const std::vector<CVPointStack>& ori_contours, std::vector<PointStack>& realWorld_contours);

    void TopoFilterContours(std::vector<CVPointStack>& contoursInOut);

    void AdjecentDistanceFilter(std::vector<CVPointStack>& contoursInOut);

    /* inline functions */
    inline void UpdateOdom(const NavNodePtr& odom_node_ptr) {
        odom_pos_ = odom_node_ptr->position;
        odom_node_ptr_ = odom_node_ptr;
        free_odom_resized_ = ConvertPoint3DToCVPoint(FARUtil::free_odom_p, odom_pos_, true);
    }

    inline void ConvertCVToPoint3DVector(const CVPointStack& cv_vec, PointStack& p_vec, const bool& is_resized_img) {
        const std::size_t vec_size = cv_vec.size();
        p_vec.clear(), p_vec.resize(vec_size);
        for (std::size_t i = 0; i < vec_size; i++) {
            cv::Point2f cv_p = cv_vec[i];
            Point3D p = ConvertCVPointToPoint3D(cv_p, odom_pos_, is_resized_img);
            // [新增] 使用邻域平均高度替代机器人高度
            float neighbor_height = GetNeighborAverageHeight(p, 1.5f);  // 1.5米搜索半径
            if (neighbor_height != p.z) {                               // 找到了有效的邻域高度
                p.z = neighbor_height;
            }
            p_vec[i] = p;
        }
    }

    inline void RemoveWallConnection(const CVPointStack& contour, const cv::Point2f& add_p, std::size_t& refined_idx) {
        if (refined_idx < 2) return;
        if (!IsPrevWallVertex(contour[refined_idx - 2], contour[refined_idx - 1], add_p)) {
            return;
        } else {
            --refined_idx;
            RemoveWallConnection(contour, add_p, refined_idx);
        }
    }

    inline void InternalContoursIdxs(
        const std::vector<cv::Vec4i>& hierarchy, const std::size_t& high_idx, std::unordered_set<int>& internal_idxs) {
        if (hierarchy[high_idx][2] == -1) return;
        SameAndLowLevelIdxs(hierarchy, hierarchy[high_idx][2], internal_idxs);
    }

    inline void SameAndLowLevelIdxs(
        const std::vector<cv::Vec4i>& hierarchy, const std::size_t& cur_idx, std::unordered_set<int>& remove_idxs) {
        if (cur_idx == -1) return;
        int next_idx = cur_idx;
        while (next_idx != -1) {
            remove_idxs.insert(next_idx);
            SameAndLowLevelIdxs(hierarchy, hierarchy[next_idx][2], remove_idxs);
            next_idx = hierarchy[next_idx][0];
        }
    }

    template <typename Point>
    inline cv::Point2f ConvertPoint3DToCVPoint(
        const Point& p, const Point3D& c_pos, const bool& is_resized_img = false) {
        cv::Point2f cv_p;
        int row_idx, col_idx;
        this->PointToImgSub(p, c_pos, row_idx, col_idx, is_resized_img);
        cv_p.x = col_idx;
        cv_p.y = row_idx;
        return cv_p;
    }

    template <typename Point>
    inline void PointToImgSub(const Point& posIn, const Point3D& c_posIn, int& row_idx, int& col_idx,
        const bool& is_resized_img = false, const bool& is_crop_idx = true) {
        const float ratio = is_resized_img ? cd_params_.kRatio : 1.0f;
        const int c_idx = is_resized_img ? CMAT_RESIZE : CMAT;
        row_idx = c_idx + (int)std::round((posIn.x - c_posIn.x) * VOXEL_DIM_INV * ratio);
        col_idx = c_idx + (int)std::round((posIn.y - c_posIn.y) * VOXEL_DIM_INV * ratio);
        if (is_crop_idx) {
            CropIdxes(row_idx, col_idx, is_resized_img);
        }
    }

    inline void CropIdxes(int& row_idx, int& col_idx, const bool& is_resized_img = false) {
        const int max_size = is_resized_img ? MAT_RESIZE : MAT_SIZE;
        row_idx = (int)std::max(std::min(row_idx, max_size - 1), 0);
        col_idx = (int)std::max(std::min(col_idx, max_size - 1), 0);
    }

    inline bool IsIdxesInImg(int& row_idx, int& col_idx, const bool& is_resized_img = false) {
        const int max_size = is_resized_img ? MAT_RESIZE : MAT_SIZE;
        if (row_idx < 0 || row_idx > max_size - 1 || col_idx < 0 || col_idx > max_size - 1) {
            return false;
        }
        return true;
    }

    inline void ResetImgMat(cv::Mat& img_mat) {
        img_mat.release();
        img_mat = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
    }

    inline Point3D ConvertCVPointToPoint3D(
        const cv::Point2f& cv_p, const Point3D& c_pos, const bool& is_resized_img = false) {
        Point3D p;
        const int c_idx = is_resized_img ? CMAT_RESIZE : CMAT;
        const float ratio = is_resized_img ? cd_params_.kRatio : 1.0f;
        p.x = (cv_p.y - c_idx) * cd_params_.voxel_dim / ratio + c_pos.x;
        p.y = (cv_p.x - c_idx) * cd_params_.voxel_dim / ratio + c_pos.y;
        p.z = odom_pos_.z;
        return p;
    }

    inline void SaveCurrentImg(const cv::Mat& img) {
        if (img.empty()) return;
        cv::Mat img_save;
        img.convertTo(img_save, CV_8UC1, 255);
        std::string filename = std::to_string(img_counter_);
        std::string img_name = cd_params_.img_path + filename + ".tiff";
        cv::imwrite(img_name, img_save);
        if (FARUtil::IsDebug) ROS_WARN_THROTTLE(1.0, "CD: image save success!");
        img_counter_++;
    }

    inline bool IsPrevWallVertex(const cv::Point2f& first_p, const cv::Point2f& mid_p, const cv::Point2f& add_p) {
        cv::Point2f diff_p1 = first_p - mid_p;
        cv::Point2f diff_p2 = add_p - mid_p;
        diff_p1 /= std::hypotf(diff_p1.x, diff_p1.y);
        diff_p2 /= std::hypotf(diff_p2.x, diff_p2.y);
        if (abs(diff_p1.dot(diff_p2)) > ALIGN_ANGLE_COS) return true;
        return false;
    }

    inline void CopyContours(
        const std::vector<std::vector<cv::Point2i>>& raw_contours, std::vector<std::vector<cv::Point2f>>& contours) {
        const std::size_t N = raw_contours.size();
        contours.clear(), contours.resize(N);
        for (std::size_t i = 0; i < N; i++) {
            const auto c = raw_contours[i];
            const std::size_t c_size = c.size();
            contours[i].resize(c_size);
            for (std::size_t j = 0; j < c.size(); j++) {
                contours[i][j] = (cv::Point2f)c[j];
            }
        }
    }

    inline void RoundContours(const std::vector<std::vector<cv::Point2f>>& filtered_contours,
        std::vector<std::vector<cv::Point2i>>& round_contours) {
        const std::size_t N = filtered_contours.size();
        round_contours.clear(), round_contours.resize(N);
        for (std::size_t i = 0; i < N; i++) {
            const auto c = filtered_contours[i];
            const std::size_t c_size = c.size();
            round_contours[i].resize(c_size);
            for (std::size_t j = 0; j < c.size(); j++) {
                round_contours[i][j] = (cv::Point2i)c[j];
            }
        }
    }
    // [新增] 获取点的邻域平均高度（排除障碍物区域）
    float GetNeighborAverageHeight(const Point3D& point, float search_radius = 1.5f);

    // 陡坡处理辅助函数
    void ExtractBoundaryPoints(const PointCloudPtr& cluster, PointStack& boundary_points);
    void ExtractInnerPoints(PointCloudPtr& cluster, PointStack& inner_points);

    // 缓坡处理辅助函数
    void ExtractModerateBoundaryPoints(const PointCloudPtr& cluster, PointStack& boundary_points);
    void ExtractModerateInnerPoints(PointCloudPtr& cluster, PointStack& inner_points);

    // [新增] 轮廓均匀采样辅助函数
    void SampleContourUniformly(const std::vector<PCLPoint>& contour, pcl::KdTreeFLANN<PCLPoint>& kdtree,
        const PointCloudPtr& cluster, PointStack& sampled_points, float sample_dist);

public:
    ContourDetector() = default;
    ~ContourDetector() = default;

    void Init(const ContourDetectParams& params);

    /**
     * Build terrian occupancy image and extract current terrian contour
     * @param odom_node_ptr current odom node pointer
     * @param surround_cloud surround obstacle cloud used for updating corner image
     * @param real_world_contour [return] current contour in world frame
     */
    void BuildTerrainImgAndExtractContour(const NavNodePtr& odom_node_ptr, const PointCloudPtr& surround_cloud,
        std::vector<PointStack>& realworl_contour);

    /**
     * [新增] 从陡坡点云中提取边界点和内部点（按聚类分组）
     * @param steep_cloud 陡坡点云
     * @param odom_node_ptr 当前机器人位置节点
     * @param boundary_clusters [返回] 每个聚类的边界点集合
     * @param inner_clusters [返回] 每个聚类的内部点集合
     */
    void ExtractSteepSlopePoints(const PointCloudPtr& steep_cloud, const NavNodePtr& odom_node_ptr,
        std::vector<PointStack>& boundary_clusters, std::vector<PointStack>& inner_clusters);

    /**
     * [新增] 从缓坡点云中提取边界点和内部点（按聚类分组）
     * @param moderate_cloud 缓坡点云
     * @param odom_node_ptr 当前机器人位置节点
     * @param boundary_clusters [返回] 每个聚类的边界点集合
     * @param inner_clusters [返回] 每个聚类的内部点集合
     */
    void ExtractModerateSlopePoints(const PointCloudPtr& moderate_cloud, const NavNodePtr& odom_node_ptr,
        std::vector<PointStack>& boundary_clusters, std::vector<PointStack>& inner_clusters);

    /**
     * Show Corners on Pointcloud projection image
     * @param img_mat pointcloud projection image
     * @param point_vec corners vector detected from cv corner detector
     */
    void ShowCornerImage(const cv::Mat& img_mat, const PointCloudPtr& pc);
    /* Get Internal Values */
    const PointCloudPtr GetNewVertices() const {
        return new_corners_cloud_;
    };
    const cv::Mat GetCloudImgMat() const {
        return img_mat_;
    };
};

#endif
