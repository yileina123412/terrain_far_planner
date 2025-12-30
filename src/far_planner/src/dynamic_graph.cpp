/*
 * FAR Planner
 * Copyright (C) 2021 Fan Yang - All rights reserved
 * fanyang2@andrew.cmu.edu,
 */

#include "far_planner/dynamic_graph.h"

/***************************************************************************************/

void DynamicGraph::Init(const ros::NodeHandle& nh, const DynamicGraphParams& params) {
    dg_params_ = params;
    CONNECT_ANGLE_COS = cos(dg_params_.kConnectAngleThred);
    NOISE_ANGLE_COS = cos(FARUtil::kAngleNoise);
    id_tracker_ = 1;
    last_connect_pos_ = Point3D(0, 0, 0);
    /* Initialize Terrian Planner */
    tp_params_.world_frame = FARUtil::worldFrameId;
    tp_params_.voxel_size = FARUtil::kLeafSize;
    tp_params_.radius = FARUtil::kNearDist * 2.0f;
    tp_params_.inflate_size = FARUtil::kObsInflate;
    terrain_planner_.Init(nh, tp_params_);
}
// 更新导航节点的位置
void DynamicGraph::UpdateRobotPosition(const Point3D& robot_pos) {
    robot_pos_ = robot_pos;
    terrain_planner_.SetLocalTerrainObsCloud(FARUtil::local_terrain_obs_);
    if (odom_node_ptr_ == NULL) {
        this->CreateNavNodeFromPoint(robot_pos_, odom_node_ptr_, true);
        this->AddNodeToGraph(odom_node_ptr_);
        if (FARUtil::IsDebug) ROS_INFO("DG: Odom node has been initilaized.");
    } else {
        this->UpdateNodePosition(odom_node_ptr_, robot_pos_);
    }
    odom_node_ptr_->terrain_type = MapHandler::GetTerrainTypeAt(odom_node_ptr_->position);
    FARUtil::odom_pos = odom_node_ptr_->position;
    terrain_planner_.VisualPaths();
}
// 是否需要创建内部导航点
bool DynamicGraph::IsInterNavpointNecessary() {
    if (cur_internav_ptr_ == NULL) {  // create nearest nav point
        last_connect_pos_ = FARUtil::free_odom_p;
        return true;
    }
    const auto it = odom_node_ptr_->edge_votes.find(cur_internav_ptr_->id);
    if (is_bridge_internav_ || it == odom_node_ptr_->edge_votes.end() ||
        !this->IsInternavInRange(cur_internav_ptr_)) {
        float min_dist = FARUtil::kINF;
        for (const auto& internav_ptr : internav_near_nodes_) {
            const float cur_dist = (internav_ptr->position - last_connect_pos_).norm();
            if (cur_dist < min_dist) min_dist = cur_dist;
        }
        if (is_bridge_internav_) {
            ROS_INFO("is_bridge_internav_");
        }
        if (it == odom_node_ptr_->edge_votes.end()) {
            ROS_INFO("it == odom_node_ptr_->edge_votes.end()");
        }
        if (!this->IsInternavInRange(cur_internav_ptr_)) {
            ROS_INFO("!this->IsInternavInRange(cur_internav_ptr_)");
        }
        if (min_dist > FARUtil::kNavClearDist && min_dist < FARUtil::kINF) return true;
    }
    if ((FARUtil::free_odom_p - last_connect_pos_).norm() > FARUtil::kNearDist ||
        (it != odom_node_ptr_->edge_votes.end() && it->second.back() == 1)) {
        last_connect_pos_ = FARUtil::free_odom_p;
    }
    return false;
}
// 从轮廓中提取ct点  创造导航点
bool DynamicGraph::ExtractGraphNodes(const CTNodeStack& new_ctnodes) {
    if (new_ctnodes.empty()) return false;
    NavNodePtr new_node_ptr = NULL;
    new_nodes_.clear();
    // if (this->IsInterNavpointNecessary()) {  // check wheter or not need inter navigation points
    //     if (FARUtil::IsDebug) ROS_INFO("DG: One trajectory node has been created.");
    //     this->CreateNavNodeFromPoint(last_connect_pos_, new_node_ptr, false, true);
    //     new_node_ptr->terrain_type = MapHandler::GetTerrainTypeAt(new_node_ptr->position);
    //     new_nodes_.push_back(new_node_ptr);
    //     last_connect_pos_ = FARUtil::free_odom_p;
    //     if (is_bridge_internav_) is_bridge_internav_ = false;
    // }
    for (const auto& ctnode_ptr : new_ctnodes) {
        bool is_near_new = false;
        if (this->IsAValidNewNode(ctnode_ptr, is_near_new)) {
            this->CreateNewNavNodeFromContour(ctnode_ptr, new_node_ptr);
            if (!is_near_new) {
                new_node_ptr->is_block_frontier = true;
            }
            new_nodes_.push_back(new_node_ptr);
        }
    }
    if (new_nodes_.empty())
        return false;
    else
        return true;
}
// 更新导航图  各种连接
void DynamicGraph::UpdateNavGraph(
    const NodePtrStack& new_nodes, const bool& is_freeze_vgraph, NodePtrStack& clear_node) {
    // clear false positive node detection
    clear_node.clear();
    // 清理无效的节点和验证轨迹连接
    if (!is_freeze_vgraph) {
        // 节点评估和清理
        // 检查节点是否有效
        // 更新节点的位置和表面方向
        // 验证轮廓匹配状态
        for (const auto& node_ptr : extend_match_nodes_) {
            // is_odom以及is_goal的导航点
            if (FARUtil::IsStaticNode(node_ptr) || node_ptr == cur_internav_ptr_)
                continue;                             // 避免误删重要节点
            if (!this->ReEvaluateCorner(node_ptr)) {  // 如果节点无效
                if (this->SetNodeToClear(node_ptr)) {
                    // 根据被投票的数量决定是否设置is_merged，就是当前的点够不够清理阈值
                    clear_node.push_back(node_ptr);
                }
            } else {
                // 节点有效，清理投票-1
                this->ReduceDumperCounter(node_ptr);
            }
        }
        // 轨迹连接的地形验证
        // 检查范围surround_internav_nodes_
        // 条件：动态环境且存在当前中间导航点
        // re-evaluate trajectory edge using terrain planner
        if (!FARUtil::IsStaticEnv && cur_internav_ptr_ != NULL) {
            NodePtrStack internav_check_nodes = surround_internav_nodes_;
            if (!FARUtil::IsTypeInStack(cur_internav_ptr_, internav_check_nodes)) {
                internav_check_nodes.push_back(cur_internav_ptr_);
            }
            for (const auto& sur_internav_ptr : internav_check_nodes) {
                const NodePtrStack copy_traj_connects = sur_internav_ptr->trajectory_connects;
                for (const auto& tnode_ptr : copy_traj_connects) {
                    // 地形验证通过，记录为有效
                    if (this->ReEvaluateConnectUsingTerrian(sur_internav_ptr, tnode_ptr)) {
                        this->RecordValidTrajEdge(sur_internav_ptr, tnode_ptr);
                    } else {
                        this->RemoveInValidTrajEdge(sur_internav_ptr, tnode_ptr);
                    }
                }
            }
        }
    }
    // clear merged nodes in stacks
    // 清理is_merged的点
    this->ClearMergedNodesInGraph();
    // add matched margin nodes into near and wide near nodes
    // 将和ct点匹配到的边缘节点margin_near_nodes_节点加入near_nav_nodes_和wide_near_nodes_中
    this->UpdateNearNodesWithMatchedMarginNodes(
        margin_near_nodes_, near_nav_nodes_, wide_near_nodes_);
    // check-add connections to odom node with wider near nodes
    // 用于检查和里程计节点之间连接有效性的节点列表。
    NodePtrStack codom_check_list = wide_near_nodes_;
    // 把新点加入有效检查列表中
    codom_check_list.insert(
        codom_check_list.end(), new_nodes.begin(), new_nodes.end());  // add new nodes to check list
    // 检查odom点的连接
    for (const auto& conode_ptr : codom_check_list) {
        if (conode_ptr->is_odom) continue;
        // 判断多边形连接，如果能连接，就加入边 poly_connects 以及 connect_nodes
        // odom不进行轮廓连接判断
        if (this->IsValidConnect(odom_node_ptr_, conode_ptr, false) &&
            this->ConnectOdomSteep(odom_node_ptr_, conode_ptr)) {
            this->AddPolyEdge(odom_node_ptr_, conode_ptr),
                this->AddEdge(odom_node_ptr_, conode_ptr);
        } else {
            this->ErasePolyEdge(odom_node_ptr_, conode_ptr),
                this->EraseEdge(conode_ptr, odom_node_ptr_);
        }
    }
    if (!is_freeze_vgraph) {
        // Adding new nodes to near nodes stack
        // 把新点加入导航图，以及near_nav_nodes_
        for (const auto& new_node_ptr : new_nodes) {
            this->AddNodeToGraph(new_node_ptr);
            new_node_ptr->is_near_nodes = true;
            near_nav_nodes_.push_back(new_node_ptr);
            // 更新cur_internav_ptr_，并建立轨迹连接 和last_internav_ptr_
            if (new_node_ptr->is_navpoint) this->UpdateCurInterNavNode(new_node_ptr);
            if (new_node_ptr->ctnode != NULL) {
                // 匹配导航点和其ct点
                ContourGraph::MatchCTNodeWithNavNode(new_node_ptr->ctnode, new_node_ptr);
            }
        }
        // connect outrange contour nodes
        for (const auto& out_node_ptr : out_contour_nodes_) {
            const NavNodePtr matched_node =
                ContourGraph::MatchOutrangeNodeWithCTNode(out_node_ptr, near_nav_nodes_);
            const auto it = out_contour_nodes_map_.find(out_node_ptr);
            if (matched_node != NULL) {
                this->RecordContourVote(out_node_ptr, matched_node);
                it->second.second.insert(matched_node);
            }
            for (const auto& reached_node_ptr : it->second.second) {
                if (reached_node_ptr != matched_node) {
                    this->DeleteContourVote(out_node_ptr, reached_node_ptr);
                }
            }
        }
        // reconnect between near nodes
        // 连接near_nav_nodes_之间的边
        NodePtrStack outside_break_nodes;
        for (std::size_t i = 0; i < near_nav_nodes_.size(); i++) {
            const NavNodePtr nav_ptr1 = near_nav_nodes_[i];
            if (nav_ptr1->is_odom) continue;
            // re-evaluate nodes which are not in near
            const NodePtrStack copy_connect_nodes = nav_ptr1->connect_nodes;
            for (const auto& cnode : copy_connect_nodes) {
                if (cnode->is_odom || cnode->is_near_nodes || FARUtil::IsOutsideGoal(cnode) ||
                    FARUtil::IsTypeInStack(cnode, nav_ptr1->contour_connects))
                    continue;
                if (this->IsValidConnect(nav_ptr1, cnode, false)) {
                    this->AddPolyEdge(nav_ptr1, cnode), this->AddEdge(nav_ptr1, cnode);
                } else {
                    this->ErasePolyEdge(nav_ptr1, cnode), this->EraseEdge(nav_ptr1, cnode);
                    outside_break_nodes.push_back(cnode);
                }
            }
            for (std::size_t j = 0; j < near_nav_nodes_.size(); j++) {
                const NavNodePtr nav_ptr2 = near_nav_nodes_[j];
                if (i == j || j > i || nav_ptr2->is_odom) continue;
                if (this->IsValidConnect(nav_ptr1, nav_ptr2, true)) {
                    this->AddPolyEdge(nav_ptr1, nav_ptr2), this->AddEdge(nav_ptr1, nav_ptr2);
                } else {
                    this->ErasePolyEdge(nav_ptr1, nav_ptr2), this->EraseEdge(nav_ptr1, nav_ptr2);
                }
            }
            for (const auto& oc_node_ptr : out_contour_nodes_) {
                if (!oc_node_ptr->is_contour_match || !nav_ptr1->is_contour_match) continue;
                if (ContourGraph::IsNavNodesConnectFromContour(nav_ptr1, oc_node_ptr)) {
                    this->RecordContourVote(nav_ptr1, oc_node_ptr);
                } else {
                    this->DeleteContourVote(nav_ptr1, oc_node_ptr);
                }
            }
            this->TopTwoContourConnector(nav_ptr1);
        }
        // update out range break nodes connects
        for (const auto& node_ptr : near_nav_nodes_) {
            for (const auto& ob_node_ptr : outside_break_nodes) {
                if (this->IsValidConnect(node_ptr, ob_node_ptr, false)) {
                    this->AddPolyEdge(node_ptr, ob_node_ptr), this->AddEdge(node_ptr, ob_node_ptr);
                } else {
                    this->ErasePolyEdge(node_ptr, ob_node_ptr),
                        this->EraseEdge(node_ptr, ob_node_ptr);
                }
            }
        }
        // Analysisig frontier nodes
        for (const auto& node_ptr : near_nav_nodes_) {
            if (this->IsNodeFullyCovered(node_ptr)) {
                node_ptr->is_covered = true;
            } else {
                node_ptr->is_covered = false;
            }
            if (this->IsFrontierNode(node_ptr)) {
                node_ptr->is_frontier = true;
            } else {
                node_ptr->is_frontier = false;
            }
        }
    }
}
/*
1. 基础几何检查
2. 轮廓连接验证 (如果is_check_contour=true)
3. 普通边连接验证
4. 投票结果评估
5. 轨迹连接检查
6. 紧密区域轮廓连接
 */
bool DynamicGraph::IsValidConnect(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2, const bool& is_check_contour) {
    // 根据terrain_type决定是否有连接的条件。
    if (!this->IsConnectByTerrain(node_ptr1, node_ptr2)) return false;
    // 如果两个点的距离很近，可以连接
    const float dist = (node_ptr1->position - node_ptr2->position).norm();
    if (dist < FARUtil::kEpsilon) return true;

    // 特殊情况：里程计节点与导航点的近距离连接 大约0.5m
    if ((node_ptr1->is_odom || node_ptr2->is_odom) &&
        (node_ptr1->is_navpoint || node_ptr2->is_navpoint)) {
        if (dist < FARUtil::kNavClearDist) return true;
    }

    /* check contour connection from node1 to node2 */
    // 轮廓连接验证
    // 轮廓连接是基于环境的几何结构，想对宽松
    // 普通的连接是根据几何的碰撞检测，方向约束，多边形碰撞等，非常严格
    if (is_check_contour) {
        // 判断两个导航节点是否可以通过它们对应的轮廓节点进行连接。
        // 判断两个导航节点是否通过环境轮廓结构可达，就是沿着轮廓边连接
        // 验证两个导航节点之间的连接是否在地形上可行，主要验证两点之间的高度和坡度行不行
        if (this->IsBoundaryConnect(node_ptr1, node_ptr2) ||
            (ContourGraph::IsNavNodesConnectFromContour(node_ptr1, node_ptr2) &&
                IsOnTerrainConnect(node_ptr1, node_ptr2, true))) {
            this->RecordContourVote(node_ptr1, node_ptr2);
        } else if (node_ptr1->is_contour_match && node_ptr2->is_contour_match) {
            this->DeleteContourVote(node_ptr1, node_ptr2);
        }
    }

    // 多边形连接验证  严格的几何检查 几何碰撞、方向约束、地形匹配
    bool is_connect = false;
    /* check polygon connections */
    // odom的投票连接阈值更小
    const int vote_queue_size = (node_ptr1->is_odom || node_ptr2->is_odom)
                                    ? std::ceil(dg_params_.votes_size / 3.0f)
                                    : dg_params_.votes_size;
    // 凸点检查  连接是否可行，碰到障碍物  检测不与任何障碍物多边形碰撞 并且通过地形验证，高度和坡度
    if (IsConvexConnect(node_ptr1, node_ptr2) && this->IsInDirectConstraint(node_ptr1, node_ptr2) &&
        ContourGraph::IsNavNodesConnectFreePolygon(node_ptr1, node_ptr2) &&
        IsOnTerrainConnect(node_ptr1, node_ptr2, false)) {
        // 检查是否有资格多边形连接
        if (this->IsPolyMatchedForConnect(node_ptr1, node_ptr2)) {
            RecordPolygonVote(node_ptr1, node_ptr2, vote_queue_size);
        }
    } else {
        DeletePolygonVote(node_ptr1, node_ptr2, vote_queue_size);
    }

    // 投票结果，数量够不够
    if (this->IsPolygonEdgeVoteTrue(node_ptr1, node_ptr2)) {
        if (!this->IsSimilarConnectInDiection(node_ptr1, node_ptr2)) is_connect = true;
        // 如果数量不够，对于is_odom立刻清除edge_votes以及potential_edges，快速连接和关闭连接
    } else if (node_ptr1->is_odom || node_ptr2->is_odom) {
        node_ptr1->edge_votes.erase(node_ptr2->id);
        node_ptr2->edge_votes.erase(node_ptr1->id);
        // clear potential connections
        FARUtil::EraseNodeFromStack(node_ptr2, node_ptr1->potential_edges);
        FARUtil::EraseNodeFromStack(node_ptr1, node_ptr2->potential_edges);
    }

    /* check if exsiting trajectory connection exist */
    // 轨迹连接检查 不需要投票  如果多边形连接失败
    if (!is_connect) {
        if (FARUtil::IsTypeInStack(node_ptr1, node_ptr2->trajectory_connects)) is_connect = true;
        if ((node_ptr1->is_odom || node_ptr2->is_odom) && cur_internav_ptr_ != NULL) {
            if (node_ptr1->is_odom &&
                FARUtil::IsTypeInStack(node_ptr2, cur_internav_ptr_->trajectory_connects)) {
                if (FARUtil::IsInCylinder(cur_internav_ptr_->position, node_ptr2->position,
                        node_ptr1->position, FARUtil::kNearDist)) {
                    is_connect = true;
                }
            } else if (node_ptr2->is_odom &&
                       FARUtil::IsTypeInStack(node_ptr1, cur_internav_ptr_->trajectory_connects)) {
                if (FARUtil::IsInCylinder(cur_internav_ptr_->position, node_ptr1->position,
                        node_ptr2->position, FARUtil::kNearDist)) {
                    is_connect = true;
                }
            }
        }
    }

    // 紧密区域轮廓连接
    /* check for additional contour connection through tight area from current robot position */
    if (!is_connect && (node_ptr1->is_odom || node_ptr2->is_odom) &&
        IsConvexConnect(node_ptr1, node_ptr2) && this->IsInDirectConstraint(node_ptr1, node_ptr2)) {
        if (node_ptr1->is_odom && !node_ptr2->contour_connects.empty()) {
            for (const auto& ctnode_ptr : node_ptr2->contour_connects) {
                if (FARUtil::IsInCylinder(ctnode_ptr->position, node_ptr2->position,
                        node_ptr1->position, FARUtil::kNavClearDist)) {
                    is_connect = true;
                }
            }
        } else if (node_ptr2->is_odom && !node_ptr1->contour_connects.empty()) {
            for (const auto& ctnode_ptr : node_ptr1->contour_connects) {
                if (FARUtil::IsInCylinder(ctnode_ptr->position, node_ptr1->position,
                        node_ptr2->position, FARUtil::kNavClearDist)) {
                    is_connect = true;
                }
            }
        }
    }
    return is_connect;
}
// 地形连接验证  验证两个导航节点之间的连接是否在地形上可行，主要验证两点之间的高度和坡度行不行
// 包括坡度检查、高度验证和地形匹配。
// is_contour：用于区分连接类型：  true：轮廓连接验证    false：多边形连接验证，类似几何碰撞
// 轮廓连接允许一定的地形数据不匹配
// 多边形连接必须通过严格的地形验证，使用投票机制防止因噪声导致的连接不稳定
bool DynamicGraph::IsOnTerrainConnect(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2, const bool& is_contour) {
    if (!node_ptr1->is_active || !node_ptr2->is_active) return true;
    // 三维中心点
    Point3D mid_p = (node_ptr1->position + node_ptr2->position) / 2.0f;
    const Point3D diff_p = node_ptr2->position - node_ptr1->position;
    // 如果两点的距离大于1.6米，而且坡度小于45度  坡度检查
    if (diff_p.norm() > FARUtil::kMatchDist &&
        abs(diff_p.z) / std::hypotf(diff_p.x, diff_p.y) > 1) {
        // 如果不进行轮廓检测，terrain_votes投一票
        if (!is_contour) RemoveInvaildTerrainConnect(node_ptr1, node_ptr2);
        return false;  // slope is too steep > 45 degree
    }
    // 如果进行轮廓连接，而且之间有过轮廓连接的投票记录，已经记录过轮廓连接 返回可以连接
    if (is_contour && node_ptr1->contour_votes.find(node_ptr2->id) !=
                          node_ptr1->contour_votes.end()) {  // recorded contour terrain connection
        return true;
    }

    bool is_match;
    float minH, maxH;
    // 检查中心点附近的平均高度
    const float avg_h =
        MapHandler::NearestHeightOfRadius(mid_p, FARUtil::kMatchDist, minH, maxH, is_match);
    if (!is_match && (is_contour || !node_ptr1->is_frontier || !node_ptr2->is_frontier)) {
        if (!is_contour) RemoveInvaildTerrainConnect(node_ptr1, node_ptr2);
        return false;
    }
    // 如果匹配到高度，如果附近的高度差太大，就不行
    if (is_match && (maxH - minH > FARUtil::kMarginHeight ||
                        abs(minH + FARUtil::vehicle_height - mid_p.z) > FARUtil::kTolerZ / 2.0f)) {
        if (!is_contour) RemoveInvaildTerrainConnect(node_ptr1, node_ptr2);
        return false;
    }
    // 如果不是轮廓连接，则如果terrain_votes得票够多，返回false
    if (!is_contour) {
        if (is_match) RecordVaildTerrainConnect(node_ptr1, node_ptr2);
        const auto it = node_ptr1->terrain_votes.find(node_ptr2->id);
        if (it != node_ptr1->terrain_votes.end() && it->second > dg_params_.finalize_thred) {
            return false;
        }
    }
    return true;
}
// 判断节点是否被完全覆盖is_covered
bool DynamicGraph::IsNodeFullyCovered(const NavNodePtr& node_ptr) {
    if (FARUtil::IsFreeNavNode(node_ptr) || node_ptr->is_covered) return true;
    NodePtrStack check_odom_list = internav_near_nodes_;
    check_odom_list.push_back(odom_node_ptr_);
    for (const auto& near_optr : check_odom_list) {
        const float cur_dist = (node_ptr->position - near_optr->position).norm();
        if (cur_dist < FARUtil::kMatchDist) return true;
        if (node_ptr->free_direct != NodeFreeDirect::PILLAR) {
            // TODO: concave nodes will not be marked as covered based on current implementation
            const auto it = near_optr->edge_votes.find(node_ptr->id);
            if (it != near_optr->edge_votes.end() && FARUtil::IsVoteTrue(it->second)) {
                const Point3D diff_p = near_optr->position - node_ptr->position;
                if (FARUtil::IsInCoverageDirPairs(diff_p, node_ptr)) {
                    return true;
                }
            }
        }
    }
    return false;
}
// 判断是不是前沿点
bool DynamicGraph::IsFrontierNode(const NavNodePtr& node_ptr) {
    if (node_ptr->is_contour_match) {
        if (node_ptr->is_block_frontier || node_ptr->is_covered ||
            node_ptr->free_direct != NodeFreeDirect::CONVEX ||
            node_ptr->ctnode->poly_ptr->perimeter < dg_params_.frontier_perimeter_thred) {
            node_ptr->frontier_votes.push_back(0);  // non convex frontier or too small
        } else {
            node_ptr->frontier_votes.push_back(1);  // convex frontier
        }
    } else if (!FARUtil::IsPointInMarginRange(
                   node_ptr->position)) {       // if not in margin range, the node won't be deleted
        node_ptr->frontier_votes.push_back(0);  // non convex frontier
    }
    if (node_ptr->frontier_votes.size() > dg_params_.finalize_thred) {
        node_ptr->frontier_votes.pop_front();
    }
    bool is_frontier = FARUtil::IsVoteTrue(node_ptr->frontier_votes);
    if (!node_ptr->is_frontier && is_frontier &&
        node_ptr->frontier_votes.size() == dg_params_.finalize_thred) {
        if (!FARUtil::IsPointNearNewPoints(node_ptr->position, true)) {
            is_frontier = false;
        }
    }
    return is_frontier;
}
// 判断是否存在一个更优的相似方向连接。  防止在一个方向上建立过多连接  只在某个方向保留最近连接
// true：存在，拒绝当前连接
bool DynamicGraph::IsSimilarConnectInDiection(
    const NavNodePtr& node_ptr_from, const NavNodePtr& node_ptr_to) {
    // TODO: check for connection loss  is_odom允许连接
    if (node_ptr_from->is_odom || node_ptr_to->is_odom) return false;
    if (FARUtil::IsTypeInStack(
            node_ptr_to, node_ptr_from->contour_connects)) {  // release for contour connection
        return false;
    }
    // check from to to node connection
    // 检查从 from 出发是否有更短连
    if (this->IsAShorterConnectInDir(node_ptr_from, node_ptr_to)) {
        return true;
    }
    if (this->IsAShorterConnectInDir(node_ptr_to, node_ptr_from)) {
        return true;
    }
    return false;
}
// 两点连接是否会碰到障碍物，是否可行  可行：true
bool DynamicGraph::IsInDirectConstraint(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    // check for odom -> frontier connections
    if ((node_ptr1->is_odom && node_ptr2->is_frontier) ||
        (node_ptr2->is_odom && node_ptr1->is_frontier))
        return true;
    // check node1 -> node2
    // 如果点1是轮廓点  就是检查连接是否会和障碍物碰撞
    // 检查从 node_ptr1 出发的方向是否可行
    if (node_ptr1->free_direct != NodeFreeDirect::PILLAR) {
        Point3D diff_1to2 = (node_ptr2->position - node_ptr1->position);
        if (!FARUtil::IsOutReducedDirs(diff_1to2, node_ptr1->surf_dirs)) {
            return false;
        }
    }
    // check node1 -> node2
    // 检查从 node_ptr2 出发的反向是否可行
    if (node_ptr2->free_direct != NodeFreeDirect::PILLAR) {
        Point3D diff_2to1 = (node_ptr1->position - node_ptr2->position);
        if (!FARUtil::IsOutReducedDirs(diff_2to1, node_ptr2->surf_dirs)) {
            return false;
        }
    }
    return true;
}

bool DynamicGraph::IsInContourDirConstraint(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    // 排出自由节点
    if (FARUtil::IsFreeNavNode(node_ptr1) || FARUtil::IsFreeNavNode(node_ptr2)) return false;
    // check node1 -> node2
    if (node_ptr1->is_finalized && node_ptr1->free_direct != NodeFreeDirect::PILLAR) {
        const Point3D diff_1to2 = node_ptr2->position - node_ptr1->position;
        if (!FARUtil::IsInContourDirPairs(diff_1to2, node_ptr1->surf_dirs)) {
            if (node_ptr1->contour_connects.size() < 2) {
                this->ResetNodeFilters(node_ptr1);
            }
            return false;
        }
    }
    // check node1 -> node2
    if (node_ptr2->is_finalized && node_ptr2->free_direct != NodeFreeDirect::PILLAR) {
        const Point3D diff_2to1 = node_ptr1->position - node_ptr2->position;
        if (!FARUtil::IsInContourDirPairs(diff_2to1, node_ptr2->surf_dirs)) {
            if (node_ptr2->contour_connects.size() < 2) {
                this->ResetNodeFilters(node_ptr2);
            }
            return false;
        }
    }
    return true;
}
// 检查从 from 出发是否有更短连
bool DynamicGraph::IsAShorterConnectInDir(
    const NavNodePtr& node_ptr_from, const NavNodePtr& node_ptr_to) {
    bool is_nav_connect = false;
    bool is_cover_connect = false;
    if (node_ptr_from->is_navpoint && node_ptr_to->is_navpoint) is_nav_connect = true;
    if (node_ptr_from->is_covered && node_ptr_to->is_covered) is_cover_connect = true;
    if (node_ptr_from->connect_nodes.empty()) return false;
    Point3D ref_dir, ref_diff;
    const Point3D diff_p = node_ptr_to->position - node_ptr_from->position;
    const Point3D connect_dir = diff_p.normalize();
    const float dist = diff_p.norm();
    for (const auto& cnode : node_ptr_from->connect_nodes) {
        if (is_nav_connect && !cnode->is_navpoint) continue;
        if (is_cover_connect && !cnode->is_covered) continue;
        if (FARUtil::IsTypeInStack(cnode, node_ptr_from->contour_connects)) continue;
        ref_diff = cnode->position - node_ptr_from->position;
        if (cnode->is_odom || ref_diff.norm() < FARUtil::kEpsilon) continue;
        ref_dir = ref_diff.normalize();
        if ((connect_dir * ref_dir) > CONNECT_ANGLE_COS && dist > ref_diff.norm()) {
            return true;
        }
    }
    return false;
}
// 更新节点的位置 pos_filter_vec 通过多次观测，使用RANSAC算法逐步收敛节点位置到最准确的估计值
bool DynamicGraph::UpdateNodePosition(const NavNodePtr& node_ptr, const Point3D& new_pos) {
    if (FARUtil::IsFreeNavNode(node_ptr)) {
        this->InitNodePosition(node_ptr, new_pos);
        return true;
    }
    if (node_ptr->is_finalized) return true;  // finalized node
    node_ptr->pos_filter_vec.push_back(new_pos);
    if (node_ptr->pos_filter_vec.size() > dg_params_.pool_size) {
        node_ptr->pos_filter_vec.pop_front();
    }
    // calculate mean nav node position using RANSACS
    std::size_t inlier_size = 0;
    Point3D mean_p = FARUtil::RANSACPoisiton(
        node_ptr->pos_filter_vec, dg_params_.filter_pos_margin, inlier_size);
    if (node_ptr->pos_filter_vec.size() > 1)
        mean_p.z = node_ptr->position.z;  // keep z value with terrain updates
    node_ptr->position = mean_p;
    if (inlier_size > dg_params_.finalize_thred) {
        return true;
    }
    return false;
}

void DynamicGraph::InitNodePosition(const NavNodePtr& node_ptr, const Point3D& new_pos) {
    node_ptr->pos_filter_vec.clear();
    node_ptr->position = new_pos;
    node_ptr->pos_filter_vec.push_back(new_pos);
}
// 更新导航节点的表面方向信息
bool DynamicGraph::UpdateNodeSurfDirs(const NavNodePtr& node_ptr, PointPair cur_dirs) {
    if (FARUtil::IsFreeNavNode(node_ptr)) {
        node_ptr->surf_dirs = {Point3D(0, 0, -1), Point3D(0, 0, -1)};
        node_ptr->free_direct = NodeFreeDirect::PILLAR;
        return true;
    }
    // [新增] 如果节点已经是 PILLAR，跳过表面方向计算
    if (node_ptr->free_direct == NodeFreeDirect::PILLAR) {
        node_ptr->surf_dirs = {Point3D(0, 0, -1), Point3D(0, 0, -1)};
        return true;
    }
    if (node_ptr->is_finalized) return true;  // finalized node
    FARUtil::CorrectDirectOrder(node_ptr->surf_dirs, cur_dirs);
    node_ptr->surf_dirs_vec.push_back(cur_dirs);
    if (node_ptr->surf_dirs_vec.size() > dg_params_.pool_size) {
        node_ptr->surf_dirs_vec.pop_front();
    }
    // calculate mean surface corner direction using RANSACS
    std::size_t inlier_size = 0;
    const PointPair mean_dir = FARUtil::RANSACSurfDirs(
        node_ptr->surf_dirs_vec, dg_params_.filter_dirs_margin, inlier_size);
    if (mean_dir.first == Point3D(0, 0, -1) || mean_dir.second == Point3D(0, 0, -1)) {
        node_ptr->surf_dirs = {Point3D(0, 0, -1), Point3D(0, 0, -1)};
        node_ptr->free_direct = NodeFreeDirect::PILLAR;
    } else {
        node_ptr->surf_dirs = mean_dir;
        this->ReEvaluateConvexity(node_ptr);
    }
    if (inlier_size > dg_params_.finalize_thred) {
        return true;
    }
    return false;
}

void DynamicGraph::ReEvaluateConvexity(const NavNodePtr& node_ptr) {
    if (!node_ptr->is_contour_match || node_ptr->ctnode->poly_ptr->is_pillar) return;
    bool is_wall = false;
    const Point3D topo_dir = FARUtil::SurfTopoDirect(node_ptr->surf_dirs, is_wall);
    if (!is_wall) {
        const Point3D ctnode_p = node_ptr->ctnode->position;
        const Point3D ev_p = ctnode_p + topo_dir * FARUtil::kLeafSize;
        if (FARUtil::IsConvexPoint(node_ptr->ctnode->poly_ptr, ev_p)) {
            node_ptr->free_direct = NodeFreeDirect::CONVEX;
        } else {
            node_ptr->free_direct = NodeFreeDirect::CONCAVE;
        }
    }
}

void DynamicGraph::TopTwoContourConnector(const NavNodePtr& node_ptr) {
    std::vector<int> votesc;
    for (const auto& vote : node_ptr->contour_votes) {
        if (FARUtil::IsVoteTrue(vote.second, false)) {
            votesc.push_back(std::accumulate(vote.second.begin(), vote.second.end(), 0));
        }
    }
    std::sort(votesc.begin(), votesc.end(), std::greater<int>());
    for (const auto& cnode_ptr : node_ptr->potential_contours) {
        const auto it = node_ptr->contour_votes.find(cnode_ptr->id);
        // DEBUG
        if (it == node_ptr->contour_votes.end())
            ROS_ERROR("DG: contour potential node matching error");
        const int itc = std::accumulate(it->second.begin(), it->second.end(), 0);
        if (FARUtil::VoteRankInVotes(itc, votesc) < 2 && FARUtil::IsVoteTrue(it->second, false)) {
            DynamicGraph::AddContourConnect(node_ptr, cnode_ptr);
            this->AddEdge(node_ptr, cnode_ptr);
        } else if (DynamicGraph::DeleteContourConnect(node_ptr, cnode_ptr) &&
                   !FARUtil::IsTypeInStack(cnode_ptr, node_ptr->poly_connects)) {
            this->EraseEdge(node_ptr, cnode_ptr);
        }
    }
}
// 给两个点之间contour_votes投一票  轮廓连接投一票，同时加入potential_contours潜在的连接边中
void DynamicGraph::RecordContourVote(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    if (node_ptr1 == node_ptr2) return;
    const auto it1 = node_ptr1->contour_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->contour_votes.find(node_ptr1->id);
    if (FARUtil::IsDebug) {
        if ((it1 == node_ptr1->contour_votes.end()) != (it2 == node_ptr2->contour_votes.end())) {
            ROS_ERROR_THROTTLE(1.0, "DG: Critical! Contour edge votes queue error.");
        }
    }
    if (it1 == node_ptr1->contour_votes.end() || it2 == node_ptr2->contour_votes.end()) {
        // init contour connection votes
        std::deque<int> vote_queue1, vote_queue2;
        vote_queue1.push_back(1), vote_queue2.push_back(1);
        node_ptr1->contour_votes.insert({node_ptr2->id, vote_queue1});
        node_ptr2->contour_votes.insert({node_ptr1->id, vote_queue2});
        if (!FARUtil::IsTypeInStack(node_ptr1, node_ptr2->potential_contours) &&
            !FARUtil::IsTypeInStack(node_ptr2, node_ptr1->potential_contours)) {
            node_ptr1->potential_contours.push_back(node_ptr2);
            node_ptr2->potential_contours.push_back(node_ptr1);
        }
    } else {
        if (FARUtil::IsDebug) {
            if (it1->second.size() != it2->second.size())
                ROS_ERROR_THROTTLE(1.0, "DG: contour connection votes are not equal.");
        }
        it1->second.push_back(1), it2->second.push_back(1);
        if (it1->second.size() > dg_params_.votes_size) {
            it1->second.pop_front(), it2->second.pop_front();
        }
    }
}
// 多边形连接，edge_votes投1票，potential_edges潜在节点加入，潜在多边形连接
void DynamicGraph::RecordPolygonVote(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2,
    const int& queue_size, const bool& is_reset) {
    if (node_ptr1 == node_ptr2) return;
    const auto it1 = node_ptr1->edge_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->edge_votes.find(node_ptr1->id);
    if (FARUtil::IsDebug) {
        if ((it1 == node_ptr1->edge_votes.end()) != (it2 == node_ptr2->edge_votes.end())) {
            ROS_ERROR_THROTTLE(1.0, "DG: Critical! Polygon edge votes queue error.");
        }
    }
    if (it1 == node_ptr1->edge_votes.end() || it2 == node_ptr2->edge_votes.end()) {
        // init polygon edge votes
        std::deque<int> vote_queue1, vote_queue2;
        vote_queue1.push_back(1), vote_queue2.push_back(1);
        node_ptr1->edge_votes.insert({node_ptr2->id, vote_queue1});
        node_ptr2->edge_votes.insert({node_ptr1->id, vote_queue2});
        if (!FARUtil::IsTypeInStack(node_ptr1, node_ptr2->potential_edges) &&
            !FARUtil::IsTypeInStack(node_ptr2, node_ptr1->potential_edges)) {
            node_ptr1->potential_edges.push_back(node_ptr2);
            node_ptr2->potential_edges.push_back(node_ptr1);
        }
    } else {
        if (FARUtil::IsDebug) {
            if (it1->second.size() != it2->second.size())
                ROS_ERROR_THROTTLE(1.0, "DG: Polygon edge votes are not equal.");
        }
        if (is_reset) it1->second.clear(), it2->second.clear();
        it1->second.push_back(1), it2->second.push_back(1);
        if (it1->second.size() > queue_size) {
            it1->second.pop_front(), it2->second.pop_front();
        }
    }
}

void DynamicGraph::FillPolygonEdgeConnect(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2, const int& queue_size) {
    if (node_ptr1 == node_ptr2) return;
    const auto it1 = node_ptr1->edge_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->edge_votes.find(node_ptr1->id);
    if (it1 == node_ptr1->edge_votes.end() || it2 == node_ptr2->edge_votes.end()) {
        std::deque<int> vote_queue1(queue_size, 1);
        std::deque<int> vote_queue2(queue_size, 1);
        node_ptr1->edge_votes.insert({node_ptr2->id, vote_queue1});
        node_ptr2->edge_votes.insert({node_ptr1->id, vote_queue2});
        if (!FARUtil::IsTypeInStack(node_ptr1, node_ptr2->potential_edges) &&
            !FARUtil::IsTypeInStack(node_ptr2, node_ptr1->potential_edges)) {
            node_ptr1->potential_edges.push_back(node_ptr2);
            node_ptr2->potential_edges.push_back(node_ptr1);
        }
        // Add connections
        if (!FARUtil::IsTypeInStack(node_ptr2, node_ptr1->poly_connects) &&
            !FARUtil::IsTypeInStack(node_ptr1, node_ptr2->poly_connects)) {
            node_ptr1->poly_connects.push_back(node_ptr2);
            node_ptr2->poly_connects.push_back(node_ptr1);
        }
    }
}

void DynamicGraph::FillContourConnect(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2, const int& queue_size) {
    if (node_ptr1 == node_ptr2) return;
    const auto it1 = node_ptr1->contour_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->contour_votes.find(node_ptr1->id);
    std::deque<int> vote_queue1(queue_size, 1);
    std::deque<int> vote_queue2(queue_size, 1);
    if (it1 == node_ptr1->contour_votes.end() || it2 == node_ptr2->contour_votes.end()) {
        // init polygon edge votes
        node_ptr1->contour_votes.insert({node_ptr2->id, vote_queue1});
        node_ptr2->contour_votes.insert({node_ptr1->id, vote_queue2});
        if (!FARUtil::IsTypeInStack(node_ptr1, node_ptr2->potential_contours) &&
            !FARUtil::IsTypeInStack(node_ptr2, node_ptr1->potential_contours)) {
            node_ptr1->potential_contours.push_back(node_ptr2);
            node_ptr2->potential_contours.push_back(node_ptr1);
        }
        // Add contours
        DynamicGraph::AddContourConnect(node_ptr1, node_ptr2);
    }
}

void DynamicGraph::FillTrajConnect(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    if (node_ptr1 == node_ptr2) return;
    const auto it1 = node_ptr1->trajectory_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->trajectory_votes.find(node_ptr1->id);
    if (it1 == node_ptr1->trajectory_votes.end() || it2 == node_ptr2->trajectory_votes.end()) {
        node_ptr1->trajectory_votes.insert({node_ptr2->id, 0});
        node_ptr2->trajectory_votes.insert({node_ptr1->id, 0});
        // Add connection
        if (!FARUtil::IsTypeInStack(node_ptr2, node_ptr1->trajectory_connects) &&
            !FARUtil::IsTypeInStack(node_ptr1, node_ptr2->trajectory_connects)) {
            node_ptr1->trajectory_connects.push_back(node_ptr2);
            node_ptr2->trajectory_connects.push_back(node_ptr1);
        }
    }
}
// edge_votes投0
void DynamicGraph::DeletePolygonVote(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2,
    const int& queue_size, const bool& is_reset) {
    const auto it1 = node_ptr1->edge_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->edge_votes.find(node_ptr1->id);
    if (it1 == node_ptr1->edge_votes.end() || it2 == node_ptr2->edge_votes.end()) return;
    if (is_reset) it1->second.clear(), it2->second.clear();
    it1->second.push_back(0), it2->second.push_back(0);
    if (it1->second.size() > queue_size) {
        it1->second.pop_front(), it2->second.pop_front();
    }
}

/* Delete Contour edge for given two navigation nodes */
// 轮廓连接contour_votes减少一票，投0，并即使更新，移除旧投票
void DynamicGraph::DeleteContourVote(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    const auto it1 = node_ptr1->contour_votes.find(node_ptr2->id);
    const auto it2 = node_ptr2->contour_votes.find(node_ptr1->id);
    if (it1 == node_ptr1->contour_votes.end() || it2 == node_ptr2->contour_votes.end())
        return;  // no connection (not counter init) in the first place
    it1->second.push_back(0), it2->second.push_back(0);
    if (it1->second.size() > dg_params_.votes_size) {
        it1->second.pop_front(), it2->second.pop_front();
    }
}
// 判断一个点是否应该被激活is_active
bool DynamicGraph::IsActivateNavNode(const NavNodePtr& node_ptr) {
    if (node_ptr->is_active) return true;
    if (FARUtil::IsPointNearNewPoints(node_ptr->position, true)) {
        node_ptr->is_active = true;
        return true;
    }
    if (FARUtil::IsFreeNavNode(node_ptr)) {
        const bool is_nearby =
            (node_ptr->position - odom_node_ptr_->position).norm() < FARUtil::kNearDist ? true
                                                                                        : false;
        if (is_nearby) {
            node_ptr->is_active = true;
            return true;
        }
        if (FARUtil::IsTypeInStack(node_ptr, odom_node_ptr_->connect_nodes)) {
            node_ptr->is_active = true;
            return true;
        }
        bool is_connects_activate = true;
        for (const auto& cnode_ptr : node_ptr->connect_nodes) {
            if (!cnode_ptr->is_active) {
                is_connects_activate = false;
                break;
            }
        }
        if ((is_connects_activate && !node_ptr->connect_nodes.empty())) {
            node_ptr->is_active = true;
            return true;
        }
    }
    return false;
}
// 更新局部范围内节点栈: near nodes, wide near nodes etc.,
void DynamicGraph::UpdateGlobalNearNodes() {
    /* update nearby navigation nodes stack --> near_nav_nodes_ */
    near_nav_nodes_.clear(), wide_near_nodes_.clear(), extend_match_nodes_.clear();
    margin_near_nodes_.clear();
    internav_near_nodes_.clear(), surround_internav_nodes_.clear();
    for (const auto& node_ptr : globalGraphNodes_) {
        node_ptr->is_near_nodes = false;
        node_ptr->is_wide_near = false;
        // 点是否在半径30m 传感器范围  且  在neighbor_obs_indices_索引范围内（就是传感器半径多一点）
        if (FARUtil::IsNodeInExtendMatchRange(node_ptr) &&
            (!node_ptr->is_active ||
                MapHandler::IsNavPointOnTerrainNeighbor(node_ptr->position, true))) {
            // 是目标点但不是内部导航点
            if (FARUtil::IsOutsideGoal(node_ptr)) continue;
            // 由于这个判断必定成立，因此extend_match_nodes_的条件就是在半径30m的点
            if (this->IsActivateNavNode(node_ptr) || node_ptr->is_boundary)
                extend_match_nodes_.push_back(node_ptr);
            // 高度范围+传感器范围  &&
            if (FARUtil::IsNodeInLocalRange(node_ptr) && IsPointOnTerrain(node_ptr->position)) {
                wide_near_nodes_.push_back(node_ptr);
                node_ptr->is_wide_near = true;
                if (node_ptr->is_active || node_ptr->is_boundary) {
                    near_nav_nodes_.push_back(node_ptr);
                    node_ptr->is_near_nodes = true;
                    if (node_ptr->is_navpoint) {
                        node_ptr->position.intensity = node_ptr->fgscore;
                        internav_near_nodes_.push_back(node_ptr);
                        // 半径2.5m
                        if ((node_ptr->position - odom_node_ptr_->position).norm() <
                            FARUtil::kLocalPlanRange / 2.0f) {
                            surround_internav_nodes_.push_back(node_ptr);
                        }
                    }
                }
            } else if (node_ptr->is_active || node_ptr->is_boundary) {
                // 由于转道不规则地形了，这里几乎是用不到了
                margin_near_nodes_.push_back(node_ptr);
            }
        }
    }
    // 遍历里程计节点的连接的节点以及连接点的连接点，其必定是wide_near_nodes_
    for (const auto& cnode_ptr :
        odom_node_ptr_->connect_nodes) {  // add additional odom connections to wide near stack
        if (FARUtil::IsOutsideGoal(cnode_ptr)) continue;
        if (!cnode_ptr->is_wide_near) {
            wide_near_nodes_.push_back(cnode_ptr);
            cnode_ptr->is_wide_near = true;
        }
        for (const auto& c2node_ptr : cnode_ptr->connect_nodes) {
            if (!c2node_ptr->is_wide_near && !FARUtil::IsOutsideGoal(c2node_ptr)) {
                wide_near_nodes_.push_back(c2node_ptr);
                c2node_ptr->is_wide_near = true;
            }
        }
    }
    // find the nearest inter_nav node that connect to odom
    // 中间导航节点cur_internav_ptr_的选择和更新的原理
    // 寻找并更新与里程计节点（机器人当前位置）连通的最优中间导航节点
    if (!internav_near_nodes_.empty()) {
        std::sort(internav_near_nodes_.begin(), internav_near_nodes_.end(), nodeptr_icomp());
        for (std::size_t i = 0; i < internav_near_nodes_.size(); i++) {
            const NavNodePtr temp_internav_ptr = internav_near_nodes_[i];
            if (FARUtil::IsTypeInStack(temp_internav_ptr, odom_node_ptr_->potential_edges) &&
                this->IsInternavInRange(temp_internav_ptr)) {
                // 如果导航点和当前导航点足够近，就更新导航点，如果附近没有导航点就桥连接
                if (cur_internav_ptr_ == NULL || temp_internav_ptr == cur_internav_ptr_ ||
                    (temp_internav_ptr->position - cur_internav_ptr_->position).norm() <
                        FARUtil::kNearDist ||
                    FARUtil::IsTypeInStack(temp_internav_ptr, cur_internav_ptr_->connect_nodes)) {
                    this->UpdateCurInterNavNode(temp_internav_ptr);
                } else {
                    is_bridge_internav_ = true;
                }
                break;
            }
        }
    }
}
// 重新评估导航节点的有效性和更新状态，如果无效就清理
// 返回true表示将诶点依然有效，返回false表示无效，多次要被清理
// 主要是清理is_navpoint点
bool DynamicGraph::ReEvaluateCorner(const NavNodePtr node_ptr) {
    if (node_ptr->is_boundary) return true;
    if (node_ptr->is_navpoint) {
        // 静态下如果是surround_internav_nodes_则一定有效
        if (FARUtil::IsTypeInStack(node_ptr, surround_internav_nodes_) &&
            this->IsNodeInTerrainOccupy(node_ptr)) {
            return false;
        }
        return true;
    }
    // 检查是否有新点
    const bool is_near_new = FARUtil::IsPointNearNewPoints(node_ptr->position, false);
    if (is_near_new) {  // if nearby env changes;
        // 重置位置和表面方向的滑动窗口
        this->ResetNodeFilters(node_ptr);
        // 如果不是和ct点匹配的导航点，重置contour_votes和edge_votes
        if (!node_ptr->is_contour_match) this->ResetNodeConnectVotes(node_ptr);
    }
    //对于非和ct点匹配的点，如果在局部范围内或者有新点，则返回当前无效？
    if (!node_ptr->is_contour_match) {
        if (FARUtil::IsPointInMarginRange(node_ptr->position) || is_near_new) return false;
        return true;
    }
    if (node_ptr->is_finalized) return true;

    // 更新位置以及表面方向
    bool is_pos_cov = false;
    bool is_dirs_cov = false;
    if (node_ptr->is_contour_match) {
        is_pos_cov = this->UpdateNodePosition(node_ptr, node_ptr->ctnode->position);
        is_dirs_cov = this->UpdateNodeSurfDirs(node_ptr, node_ptr->ctnode->surf_dirs);
        if (FARUtil::IsDebug)
            ROS_ERROR_COND(
                node_ptr->free_direct == NodeFreeDirect::UNKNOW, "DG: node free space is unknown.");
    }
    if (is_pos_cov && is_dirs_cov) node_ptr->is_finalized = true;

    return true;
}

bool DynamicGraph::ReEvaluateConnectUsingTerrian(
    const NavNodePtr& node_ptr1, const NavNodePtr node_ptr2) {
    PointStack terrain_path;
    if (terrain_planner_.PlanPathFromNodeToNode(node_ptr1, node_ptr2, terrain_path)) {
        return true;
    }
    return false;
}
