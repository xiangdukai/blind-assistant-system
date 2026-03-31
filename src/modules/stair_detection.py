"""
楼梯检测模块
基于霍夫原生ρ-θ极坐标特征优化
通过ρ-θ特征聚类+统计实现台阶数精准计算
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from sklearn.cluster import DBSCAN


class StairDetector:
    """楼梯检测器"""
    def __init__(self, config: dict):
        """
        初始化楼梯检测器
        Args:
            config: 配置字典
                - canny_threshold1: Canny边缘检测低阈值，默认30
                - canny_threshold2: Canny边缘检测高阈值，默认100
                - hough_threshold: 霍夫变换累加阈值，默认45
                - min_line_length: 霍夫变换最小线长，默认12
                - max_line_gap: 霍夫变换最大线间隙，默认2
                - dbscan_eps: DBSCAN聚类邻域半径，默认0.03
                - dbscan_min_samples: DBSCAN最小样本数，默认3
                - min_valid_line_threshold: 最小有效簇线段数阈值，默认3
                - min_valid_step_threshold: 最小有效台阶数（类簇数）阈值，默认3
                - line_count_close_threshold: 候补线段数“相差不大”阈值，默认2
                - theta_band_width_deg: 第一阶段theta分桶宽度（度），默认10.0
                - debug: 是否启用调试模式，默认False
                - parallel_error_threshold: 平行性校验相对误差阈值，默认15%
        """
        # S52 边缘检测参数
        self.canny_threshold1 = config.get('canny_threshold1', 30)
        self.canny_threshold2 = config.get('canny_threshold2', 100)
        # S53 霍夫变换参数
        self.hough_threshold = config.get('hough_threshold', 45)
        self.min_line_length = config.get('min_line_length', 12)
        self.max_line_gap = config.get('max_line_gap', 2)
        # S53 DBSCAN聚类参数 
        self.dbscan_eps = config.get('dbscan_eps', 0.03)
        self.dbscan_min_samples = config.get('dbscan_min_samples', 3)
        # S54 阈值参数：线段数阈值与台阶数阈值分离
        self.min_valid_line_threshold = config.get('min_valid_line_threshold', 1)
        self.min_valid_step_threshold = config.get(
            'min_valid_step_threshold',
            config.get('min_valid_line_threshold', 3)
        )
        self.line_count_close_threshold = config.get('line_count_close_threshold', 3)
        self.parallel_error_threshold = config.get('parallel_error_threshold', 0.15)
        # 平行性绝对角度阈值（度），用于theta分组与平行性判定
        self.parallel_angle_threshold_deg = config.get('parallel_angle_threshold_deg', 10.0)
        # 两阶段聚类第一阶段：按theta分桶
        self.theta_band_width_deg = config.get('theta_band_width_deg', 10.0)
        # 调试开关：打印DBSCAN噪声比例
        self.debug = config.get('debug', False)

    def detect(self, image: np.ndarray, detections: List[Dict], depth_map: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        楼梯检测主流程
        Args:
            image: 输入BGR图像
            detections: YOLO检测结果列表，需包含class_name、bbox、confidence字段
            depth_map: 可选，深度图，用于计算楼梯距离
        Returns:
            检测结果字典，无有效楼梯返回None
        """
        # ========== 前置：YOLO楼梯目标筛选 ==========
        staircase_detections = [d for d in detections if d['class_name'] == 'staircase']
        if not staircase_detections:
            return None
        
        # 选取置信度最高的楼梯目标
        best_detection = max(staircase_detections, key=lambda d: d['confidence'])
        x1, y1, x2, y2 = best_detection["bbox"]
        roi_height = y2 - y1
        roi_width = x2 - x1
        if roi_height <= 0 or roi_width <= 0:
            return None

        # ========== 预处理：裁切+灰度化+高斯滤波 ==========
        roi = image[y1:y2, x1:x2]
        # 灰度化公式：I(x,y)=0.299×R + 0.587×G + 0.114×B
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ========== 边缘检测+闭运算 ==========
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        # 形态学闭运算，滤除杂乱线条干扰
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # ========== S53 霍夫变换直线检测 ==========
        lines = cv2.HoughLinesP(
            closed_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        if lines is None or len(lines) == 0:
            return None
        if self.debug:
            debug_img = self.visualize_hough_lines_on_edges(closed_edges, lines)
            cv2.imwrite("tests/test_images/debug_hough_on_edges.png", debug_img)  

        # ========== 核心：ρ-θ特征聚类+台阶数统计 ==========
        step_count, valid_clusters, valid_center_lines = self._cluster_by_rho_theta(lines, roi_height, roi_width)
        # ========== 楼梯方向判断（上楼/下楼） ==========
        direction = self._determine_direction(valid_center_lines, image.shape[0])

        # ========== 深度信息计算（对齐说明书场景播报要求） ==========
        stair_depth = None
        if depth_map is not None:
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size > 0:
                valid_depth = depth_roi[depth_roi > 0]
                stair_depth = np.median(valid_depth) if valid_depth.size > 0 else None

        # ========== 结果输出 ==========
        return {
            'detected': True,
            'direction': direction,
            'num_steps': step_count,
            'valid_clusters': valid_clusters,
            'valid_center_lines': valid_center_lines,
            'original_bbox': [x1, y1, x2, y2],
            'depth': stair_depth,
            'detection_confidence': best_detection['confidence']
        }

    def _cluster_by_rho_theta(self, lines: np.ndarray, roi_height: int, roi_width: int) -> tuple[int, Dict, np.ndarray]:
        """
        3基于霍夫原生ρ-θ极坐标特征实现聚类与台阶数统计
        Args:
            lines: 霍夫变换输出的所有直线，shape=(N,1,4)
            roi_height: 裁切ROI的高度
            roi_width: 裁切ROI的宽度
        Returns:
            修正后的台阶数、有效簇字典、有效簇中心直线数组
        """
        # ========== 步骤1：ρ-θ特征提取 ==========
        line_features = []
        line_raw_info = []
        # ROI对角线长度，用于ρ归一化
        max_diagonal = np.sqrt(roi_width ** 2 + roi_height ** 2)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            # 跳过无长度的无效直线
            if dx == 0 and dy == 0:
                continue

            # ========== 核心：直线的ρ-θ极坐标参数计算 ==========
            # 1. 计算直线法线方向的θ（弧度）
            # 直线方向向量为(dx, dy)，法线向量为(-dy, dx)，计算法线与x轴夹角
            normal_vec = np.array([-dy, dx])
            # 计算θ，范围[-π, π]
            theta = np.arctan2(normal_vec[1], normal_vec[0])
            # 统一θ到[0, π)范围，避免同一直线重复表示
            if theta < 0:
                theta += np.pi
            # 特殊情况：theta=π时等价于theta=0
            if theta >= np.pi:
                theta -= np.pi

            # 2. 计算ρ：原点到直线的垂直距离
            # 直线方程：A*x + B*y + C = 0 → ρ = |C| / √(A²+B²)
            A = dy
            B = -dx
            C = dx*y1 - dy*x1
            rho = abs(A*0 + B*0 + C) / np.sqrt(A**2 + B**2)

            # 过滤ROI外的无效直线（ρ超过对角线的为异常值）
            if rho > max_diagonal:
                continue

            # ========== 步骤2：ρ-θ特征归一化，严格映射到[0,1] ==========
            theta_norm = theta / np.pi  # θ归一化：[0, π] → [0, 1]
            rho_norm = rho / max_diagonal  # ρ归一化：[0, max_diagonal] → [0, 1]

            # 保存特征与原始信息
            line_features.append([theta_norm, rho_norm])
            line_raw_info.append({
                'raw_line': [x1, y1, x2, y2],
                'theta_rad': theta,
                'theta_deg': np.degrees(theta),
                'rho': rho,
                'theta_norm': theta_norm,
                'rho_norm': rho_norm
            })

        # 无效特征校验（两阶段聚类至少需要2条线）
        if len(line_features) < 2:
            return 0, {}, np.array([])
        feature_array = np.array(line_features)

        # ========== 步骤3：两阶段聚类（先theta分桶，再rho聚类） ==========
        # 阶段1：按theta分桶，避免同方向线段被过度合并
        theta_deg_array = np.array([item['theta_deg'] for item in line_raw_info])
        rho_norm_array = feature_array[:, 1]
        theta_band_width = max(self.theta_band_width_deg, 1e-6)
        theta_band_ids = np.floor(theta_deg_array / theta_band_width).astype(np.int32)

        cluster_labels = np.full(len(line_raw_info), -1, dtype=np.int32)
        next_global_label = 0

        # 阶段2：在每个theta桶内仅按rho_norm做DBSCAN
        rho_eps = max(0.015, min(self.dbscan_eps, 0.06))
        rho_min_samples = max(2, self.dbscan_min_samples - 1)
        for band_id in np.unique(theta_band_ids):
            band_indices = np.where(theta_band_ids == band_id)[0]
            if len(band_indices) < 2:
                continue

            band_rho_data = rho_norm_array[band_indices].reshape(-1, 1)

            band_labels = DBSCAN(eps=rho_eps, min_samples=rho_min_samples).fit(band_rho_data).labels_

            # 桶内全噪声时放宽一次，减少-1
            if np.all(band_labels == -1):
                relaxed_rho_eps = max(rho_eps * 2.0, rho_eps + 0.03)
                relaxed_min_samples = 2
                relaxed_labels = DBSCAN(
                    eps=relaxed_rho_eps,
                    min_samples=relaxed_min_samples
                ).fit(band_rho_data).labels_
                if np.any(relaxed_labels != -1):
                    band_labels = relaxed_labels

            for local_label in np.unique(band_labels):
                if local_label == -1:
                    continue
                member_indices = band_indices[band_labels == local_label]
                cluster_labels[member_indices] = next_global_label
                next_global_label += 1

        # 全局兜底：若两阶段后仍全噪声，按rho一维做一次放宽聚类
        if np.all(cluster_labels == -1):
            global_rho_data = rho_norm_array.reshape(-1, 1)
            fallback_eps = max(rho_eps * 2.5, rho_eps + 0.05)
            fallback_labels = DBSCAN(eps=fallback_eps, min_samples=2).fit(global_rho_data).labels_
            if np.any(fallback_labels != -1):
                cluster_labels = fallback_labels.astype(np.int32)

        # 调试可视化：聚类完成后按标签着色散点图
        if self.debug:
            scatter_img = self.visualize_rho_theta_scatter(feature_array, cluster_labels)
            cv2.imwrite("tests/test_images/debug_rho_theta_scatter.png", scatter_img)

        # ========== 步骤4：聚类结果整理 ==========
        cluster_dict = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # 跳过噪声点
                continue
            if label not in cluster_dict:
                cluster_dict[label] = {
                    'lines': [],
                    'line_count': 0,
                    'avg_theta_deg': 0.0,
                    'avg_rho': 0.0,
                    'center_line': None
                }
            # 保存当前直线信息
            cluster_dict[label]['lines'].append(line_raw_info[idx])
            cluster_dict[label]['line_count'] += 1
            cluster_dict[label]['avg_theta_deg'] += line_raw_info[idx]['theta_deg']
            cluster_dict[label]['avg_rho'] += line_raw_info[idx]['rho']

        # 计算每个簇的平均特征与中心直线
        for label in cluster_dict:
            cluster = cluster_dict[label]
            line_count = cluster['line_count']
            cluster['avg_theta_deg'] /= line_count
            cluster['avg_rho'] /= line_count
            # 计算簇的中心直线（所有直线的坐标均值）
            raw_lines = np.array([item['raw_line'] for item in cluster['lines']])
            cluster['center_line'] = np.mean(raw_lines, axis=0).astype(np.int32)

        # ========== 过滤无效簇（聚类后） ==========
        valid_clusters = {}
        for label, cluster in cluster_dict.items():
            # 1. 过滤线段数小于阈值的簇
            if cluster['line_count'] < self.min_valid_line_threshold:
                continue
            # 2. 平行性校验：簇内角度标准差阈值
            theta_list = [item['theta_deg'] for item in cluster['lines']]
            theta_std = np.std(theta_list)
            if theta_std <= self.parallel_angle_threshold_deg:
                valid_clusters[label] = cluster

        # ========== 步骤1：生成初选集合S_初选 ==========
        # 分组中同时维护：
        # - line_count：用于加权更新组theta
        # - cluster_count：该组归并的簇数量（用于表示台阶数）
        theta_groups = []
        for cluster in valid_clusters.values():
            cluster_theta = cluster['avg_theta_deg']
            cluster_line_count = len(cluster['lines'])
            assigned = False

            for group in theta_groups:
                theta_diff = abs(cluster_theta - group['theta_deg'])
                theta_diff = min(theta_diff, 180 - theta_diff)
                if theta_diff <= self.parallel_angle_threshold_deg:
                    old_line_count = group['line_count']
                    new_line_count = old_line_count + cluster_line_count
                    group['theta_deg'] = (
                        group['theta_deg'] * old_line_count + cluster_theta * cluster_line_count
                    ) / new_line_count
                    group['line_count'] = new_line_count
                    group['cluster_count'] += 1
                    assigned = True
                    break

            if not assigned:
                theta_groups.append({
                    'theta_deg': cluster_theta,
                    'line_count': cluster_line_count,
                    'cluster_count': 1
                })

        # 初选集合保留每个平行线组的统计信息
        S_primary = theta_groups
        if not S_primary:
            return 0, {}, np.array([])

        # ========== 步骤2：生成候选集合S_候选 ==========
        # 候补集合同时保留line_count与cluster_count
        S_candidate = [
            {
                'line_count': group['line_count'],
                'cluster_count': group['cluster_count']
            }
            for group in S_primary
            if group['cluster_count'] >= self.min_valid_step_threshold
        ]
        if not S_candidate:
            return 0, {}, np.array([])

        # ========== 步骤3：台阶数修正 ==========
        # 最终step_count基于cluster_count计算
        candidate_cluster_counts = [item['cluster_count'] for item in S_candidate]
        candidate_line_counts = [item['line_count'] for item in S_candidate]

        # 若|S_候选|≤3:
        # 1) line_count相差大 -> 取line_count最大项对应的cluster_count
        # 2) line_count相差不大 -> 取cluster_count均值
        # 若|S_候选|>3, 按1.5σ区间保留后取中位数
        if len(candidate_cluster_counts) <= 3:
            if len(candidate_cluster_counts) == 1:
                corrected_step_count = int(candidate_cluster_counts[0])
            else:
                line_count_span = max(candidate_line_counts) - min(candidate_line_counts)
                if line_count_span > self.line_count_close_threshold:
                    best_item = max(S_candidate, key=lambda item: (item['line_count'], item['cluster_count']))
                    corrected_step_count = int(best_item['cluster_count'])
                else:
                    corrected_step_count = int(round(np.mean(candidate_cluster_counts)))
        else:
            mu = np.mean(candidate_cluster_counts)
            sigma = np.std(candidate_cluster_counts)
            lower_bound = mu - 1.5 * sigma
            upper_bound = mu + 1.5 * sigma
            S_final = [
                n for n in candidate_cluster_counts
                if lower_bound <= n <= upper_bound
            ]
            S_final = S_final if S_final else candidate_cluster_counts
            corrected_step_count = int(np.median(S_final))

        # 提取有效簇的中心直线
        valid_center_lines = np.array([cluster['center_line'] for cluster in valid_clusters.values()])

        return corrected_step_count, valid_clusters, valid_center_lines

    def _determine_direction(self, valid_center_lines: np.ndarray, image_height: int) -> str:
        """
        判断楼梯方向：上楼/下楼
        Args:
            valid_center_lines: 有效簇中心直线数组
            image_height: 原始图像高度
        Returns:
            'up' 上楼 / 'down' 下楼
        """
        if len(valid_center_lines) == 0:
            return 'unknown'
        # 计算所有有效直线的y坐标中点
        y_mid_list = [(line[1] + line[3]) / 2 for line in valid_center_lines]
        avg_y = np.mean(y_mid_list)
        # 线段集中在图像下半部分→上楼，集中在上半部分→下楼
        return 'up' if avg_y > image_height / 2 else 'down'

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        可视化检测结果
        Args:
            image: 原始输入图像
            result: detect方法返回的检测结果
        Returns:
            可视化后的图像
        """
        vis_img = image.copy()
        if result is None or not result['detected']:
            return vis_img

        # 绘制楼梯检测框
        x1, y1, x2, y2 = result['original_bbox']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制有效台阶中心直线（映射回原图坐标）
        for line in result['valid_center_lines']:
            lx1, ly1, lx2, ly2 = line
            lx1 += x1
            lx2 += x1
            ly1 += y1
            ly2 += y1
            cv2.line(vis_img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

        # 绘制检测信息
        info_text = f"Steps: {result['num_steps']}  Direction: {result['direction']}"
        if result['depth'] is not None:
            info_text += f"  Depth: {result['depth']:.1f}m"
        cv2.putText(vis_img, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return vis_img

    def visualize_hough_lines_on_edges(
        self,
        closed_edges: np.ndarray,
        lines: Optional[np.ndarray],
        line_color: Tuple[int, int, int] = (0, 255, 0),
        line_thickness: int = 1
    ) -> np.ndarray:
        """
        调试可视化：将霍夫变换得到的线段绘制到闭运算后的边缘图上。
        Args:
            closed_edges: 闭运算后的边缘图（单通道或三通道）
            lines: HoughLinesP输出，shape=(N,1,4)
            line_color: 线段颜色（BGR）
            line_thickness: 线段粗细
        Returns:
            叠加线段后的三通道可视化图像
        """
        if closed_edges is None or closed_edges.size == 0:
            return np.array([])

        if len(closed_edges.shape) == 2:
            debug_img = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = closed_edges.copy()

        if lines is None or len(lines) == 0:
            return debug_img

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), line_color, line_thickness)

        cv2.putText(
            debug_img,
            f"Hough lines: {len(lines)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        return debug_img

    def visualize_rho_theta_scatter(
        self,
        feature_array: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        canvas_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        调试可视化：绘制(theta_norm, rho_norm)散点图。
        Args:
            feature_array: shape=(N,2)，列顺序为[theta_norm, rho_norm]
            cluster_labels: 可选，聚类标签数组，长度应为N
            canvas_size: 画布尺寸 (width, height)
        Returns:
            散点图BGR图像
        """
        width, height = canvas_size
        scatter = np.full((height, width, 3), 255, dtype=np.uint8)

        # 坐标轴边距
        margin = 50
        x0, y0 = margin, height - margin
        x1, y1 = width - margin, margin

        # 画坐标轴
        cv2.line(scatter, (x0, y0), (x1, y0), (0, 0, 0), 2)
        cv2.line(scatter, (x0, y0), (x0, y1), (0, 0, 0), 2)

        # 网格与刻度
        for i in range(6):
            t = i / 5.0
            gx = int(x0 + (x1 - x0) * t)
            gy = int(y0 - (y0 - y1) * t)
            cv2.line(scatter, (gx, y0), (gx, y1), (230, 230, 230), 1)
            cv2.line(scatter, (x0, gy), (x1, gy), (230, 230, 230), 1)
            cv2.putText(scatter, f"{t:.1f}", (gx - 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
            cv2.putText(scatter, f"{t:.1f}", (x0 - 35, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        # 绘制散点：x=theta_norm, y=rho_norm
        if feature_array is not None and len(feature_array) > 0:
            theta_vals = np.clip(feature_array[:, 0], 0.0, 1.0)
            rho_vals = np.clip(feature_array[:, 1], 0.0, 1.0)

            # 默认标签（未传入时统一按0处理）
            if cluster_labels is None or len(cluster_labels) != len(feature_array):
                labels = np.zeros(len(feature_array), dtype=np.int32)
            else:
                labels = cluster_labels

            # 每个标签对应一个颜色，噪声(-1)用灰色
            palette = [
                (255, 0, 0), (0, 128, 255), (0, 180, 0), (180, 0, 180),
                (255, 140, 0), (0, 200, 200), (120, 60, 240), (60, 60, 60)
            ]

            for theta_norm, rho_norm, label in zip(theta_vals, rho_vals, labels):
                px = int(x0 + (x1 - x0) * float(theta_norm))
                py = int(y0 - (y0 - y1) * float(rho_norm))
                if int(label) == -1:
                    point_color = (150, 150, 150)
                else:
                    point_color = palette[int(label) % len(palette)]
                cv2.circle(scatter, (px, py), 3, point_color, -1)

        cv2.putText(scatter, "theta_norm", (x1 - 100, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(scatter, "rho_norm", (10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(scatter, f"N={0 if feature_array is None else len(feature_array)}", (x0, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        return scatter


if __name__ == "__main__":
    # 实施例六默认参数初始化
    config = {
        'canny_threshold1': 30,
        'canny_threshold2': 100,
        'hough_threshold': 45,
        'min_line_length': 12,
        'max_line_gap': 10,
        'dbscan_eps': 0.03,
        'dbscan_min_samples': 3,
        'min_valid_line_threshold': 1,
        'min_valid_step_threshold': 3,
        'line_count_close_threshold': 3,
        'theta_band_width_deg': 10.0,
        'parallel_angle_threshold_deg': 10.0,
        'debug': True,
        'parallel_error_threshold': 0.15
    }
    detector = StairDetector(config)
    print("楼梯检测模块初始化完成")
    print(f"实施例默认配置：{config}")

    # 测试：可通过命令行参数手动指定图片路径；未指定时回退到默认目录搜索
    project_root = Path(__file__).resolve().parents[2]
    candidate_dirs = [
        project_root / 'testimage',
        project_root / 'tests' / 'test_images'
    ]
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    image_path = None
    if len(sys.argv) > 1:
        user_path = Path(sys.argv[1]).expanduser()
        if not user_path.is_absolute():
            user_path = (project_root / user_path).resolve()
        if user_path.exists() and user_path.is_file() and user_path.suffix.lower() in image_exts:
            image_path = user_path
        else:
            print(f"指定图片无效或不存在: {user_path}")
            print("请传入有效图片路径，例如: python src/modules/stair_detection.py tests/test_images/xxx.jpg")

    if image_path is None:
        for d in candidate_dirs:
            if not d.exists():
                continue
            images = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]
            if images:
                image_path = images[0]
                break

    if image_path is None:
        print("未找到测试图片，请将图片放入 testimage 或 tests/test_images 目录")
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"图片读取失败: {image_path}")
        else:
            h, w = image.shape[:2]
            detections = [{
                'class_name': 'staircase',
                'bbox': [0, 0, w, h],
                'confidence': 1.0
            }]

            result = detector.detect(image=image, detections=detections, depth_map=None)
            vis_image = detector.visualize(image, result)

            output_path = image_path.with_name(f"vis_{image_path.stem}.png")
            cv2.imwrite(str(output_path), vis_image)

            print(f"测试图片: {image_path}")
            print(f"detect结果: {result['num_steps'] if result else '无'} steps, direction={result['direction'] if result else 'N/A'}")
            print(f"visualize结果已保存: {output_path}")
