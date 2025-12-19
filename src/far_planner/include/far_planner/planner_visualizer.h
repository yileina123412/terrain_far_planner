#ifndef PLANNER_VISUALIZER_H
#define PLANNER_VISUALIZER_H

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "contour_graph.h"
#include "utility.h"

typedef visualization_msgs::Marker Marker;
typedef visualization_msgs::MarkerArray MarkerArray;

enum VizColor {
    RED = 0,
    ORANGE = 1,
    BLACK = 2,
    YELLOW = 3,
    BLUE = 4,
    GREEN = 5,
    EMERALD = 6,
    WHITE = 7,
    MAGNA = 8,
    PURPLE = 9
};

class DPVisualizer {
private:
    ros::NodeHandle nh_;
    // Utility Cloud
    PointCloudPtr point_cloud_ptr_;
    // rviz publisher
    ros::Publisher viz_node_pub_, viz_path_pub_, viz_poly_pub_, viz_graph_pub_;
    ros::Publisher viz_contour_pub_, viz_map_pub_, viz_view_extend;
    ros::Publisher risk_debug_pub_;
    ros::Publisher viz_steep_clusters_pub_, viz_moderate_clusters_pub_;

public:
    DPVisualizer() = default;
    ~DPVisualizer() = default;

    void Init(const ros::NodeHandle& nh);

    void VizNodes(const NodePtrStack& node_stack, const std::string& ns, const VizColor& color,
        const float scale = 0.75f, const float alpha = 0.75f);

    void VizGlobalPolygons(const std::vector<PointPair>& contour_pairs, const std::vector<PointPair>& unmatched_pairs);

    void VizViewpointExtend(const NavNodePtr& ori_nav_ptr, const Point3D& extend_point);

    // True for non-attempts path
    void VizPath(const NodePtrStack& global_path, const bool& is_free_nav = false);

    void VizMapGrids(const PointStack& neighbor_centers, const PointStack& occupancy_centers, const float& ceil_length,
        const float& ceil_height);

    void VizContourGraph(const CTNodeStack& contour_graph);

    void VizPoint3D(const Point3D& point, const std::string& ns, const VizColor& color, const float scale = 1.0f,
        const float alpha = 0.9f);

    void VizGraph(const NodePtrStack& graph);
    void VizPointCloud(const ros::Publisher& viz_pub, const PointCloudPtr& pc);
    void VizRGBPointCloud(const ros::Publisher& viz_pub, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc);
    // [新增] 可视化陡坡聚类结果（边界点和内部点）
    void VizSteepSlopeClusters(
        const std::vector<PointStack>& boundary_clusters, const std::vector<PointStack>& inner_clusters);
    // [新增] 可视化缓坡聚类结果（边界点和内部点）
    void VizModerateSlopeClusters(
        const std::vector<PointStack>& boundary_clusters, const std::vector<PointStack>& inner_clusters);

    static void SetMarker(const VizColor& color, const std::string& ns, const float& scale, const float& alpha,
        Marker& scan_marker, const float& scale_ratio = FARUtil::kVizRatio);

    static void SetColor(const VizColor& color, const float& alpha, Marker& scan_marker);
};

#endif