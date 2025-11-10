import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------- WGS-84 常量 ----------------------
WGS84_A = 6378137.0
WGS84_F = 1.0/298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
C = 299792458.0  # m/s

# ---------------------- 数据类 ----------------------
@dataclass
class SPPSolution:
    """SPP单历元解"""
    epoch: int
    x: float
    y: float
    z: float
    lat: float
    lon: float
    h: float
    clock_bias: float
    nsat: int
    pdop: float
    hdop: float
    vdop: float
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([
            self.epoch, self.x, self.y, self.z, 
            self.lat, self.lon, self.h, self.clock_bias,
            self.nsat, self.pdop, self.hdop, self.vdop
        ])
    
    @property
    def xyz(self) -> np.ndarray:
        """ECEF坐标"""
        return np.array([self.x, self.y, self.z])
    
    @property
    def llh(self) -> Tuple[float, float, float]:
        """经纬高"""
        return (self.lon, self.lat, self.h)

@dataclass
class ErrorStatistics:
    """误差统计"""
    e_err: np.ndarray
    n_err: np.ndarray
    u_err: np.ndarray
    err_2d: np.ndarray
    err_3d: np.ndarray
    stats: dict

# ---------------------- 坐标转换类 ----------------------
class CoordinateConverter:
    """坐标转换工具类"""
    
    @staticmethod
    def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """ECEF转经纬高 (度, 度, 米)"""
        r = math.hypot(x, y)
        if r < 1e-12 and abs(z) < 1e-12:
            return 0.0, 0.0, -WGS84_A
        
        lon = math.degrees(math.atan2(y, x))
        lat = math.atan2(z, r * (1 - WGS84_E2))
        
        for _ in range(10):
            sinl = math.sin(lat)
            N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sinl * sinl)
            h = r / math.cos(lat) - N
            lat_new = math.atan2(z, r * (1.0 - WGS84_E2 * N / (N + h)))
            if abs(lat_new - lat) < 1e-12:
                lat = lat_new
                break
            lat = lat_new
        
        sinl = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sinl * sinl)
        h = r / math.cos(lat) - N
        
        return math.degrees(lat), lon, h
    
    @staticmethod
    def lla_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
        """经纬高转ECEF"""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sinl, cosl = math.sin(lat), math.cos(lat)
        sinL, cosL = math.sin(lon), math.cos(lon)
        
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sinl * sinl)
        x = (N + h) * cosl * cosL
        y = (N + h) * cosl * sinL
        z = (N * (1 - WGS84_E2) + h) * sinl
        
        return np.array([x, y, z], dtype=float)
    
    @staticmethod
    def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
        """ECEF到ENU的旋转矩阵"""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sl, cl = math.sin(lat), math.cos(lat)
        sL, cL = math.sin(lon), math.cos(lon)
        
        return np.array([
            [-sL,      cL,     0.0],
            [-sl*cL,  -sl*sL,  cl ],
            [ cl*cL,   cl*sL,  sl ]
        ], dtype=float)

# ---------------------- DOP计算类 ----------------------
class DOPCalculator:
    """DOP值计算"""
    
    @staticmethod
    def compute_dops(H: np.ndarray, lat_deg: float, lon_deg: float) -> Tuple[float, float, float]:
        """计算PDOP/HDOP/VDOP"""
        if H.shape[0] < 4:
            return np.nan, np.nan, np.nan
        
        try:
            Q = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        
        Q_xyz = Q[0:3, 0:3]
        R = CoordinateConverter.ecef_to_enu_rotation(lat_deg, lon_deg)
        Q_enu = R @ Q_xyz @ R.T
        
        var_e, var_n, var_u = Q_enu[0, 0], Q_enu[1, 1], Q_enu[2, 2]
        
        if min(var_e, var_n, var_u) < 0:
            return np.nan, np.nan, np.nan
        
        HDOP = math.sqrt(var_e + var_n)
        VDOP = math.sqrt(var_u)
        PDOP = math.sqrt(var_e + var_n + var_u)
        
        return PDOP, HDOP, VDOP

# ---------------------- 数据加载类 ----------------------
class DataLoader:
    """数据加载工具类"""
    
    @staticmethod
    def load_csv(path: str) -> np.ndarray:
        """加载CSV文件"""
        return np.genfromtxt(path, delimiter=',')
    
    @staticmethod
    def split_sat_positions_matrix(sat_pos_mat: np.ndarray) -> List[np.ndarray]:
        """分割卫星位置矩阵"""
        max_sats, cols = sat_pos_mat.shape
        assert cols % 3 == 0, "satellite_positions.csv columns must be multiple of 3"
        num_epochs = cols // 3
        
        return [sat_pos_mat[:, 3*k:3*k+3] for k in range(num_epochs)]
    
    @staticmethod
    def load_truth_from_nav_hpposecef(path: str) -> Optional[np.ndarray]:
        """从NAV-HPPOSECEF加载真值"""
        try:
            # 检测分隔符
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.lstrip().startswith("#"):
                        delim = ',' if ',' in line else None
                        break
            
            # 读取数据
            data = np.genfromtxt(path, delimiter=delim, names=True, dtype=None, encoding=None)
            names = set(data.dtype.names or [])
            
            if 'ecefX' in names and 'ecefY' in names and 'ecefZ' in names:
                # NOTE: NAV-HPPOSECEF 中的坐标单位为 cm，转换为 m
                ecef_x = np.asarray(data['ecefX'], float) * 0.01
                ecef_y = np.asarray(data['ecefY'], float) * 0.01
                ecef_z = np.asarray(data['ecefZ'], float) * 0.01
            else:
                raise ValueError("NAV-HPPOSECEF.csv must contain ecefX, ecefY, ecefZ")
            
            # 批量转换ECEF到LLH
            n = len(ecef_x)
            llh_data = np.zeros((n, 3), dtype=float)
            converter = CoordinateConverter()
            
            for i in range(n):
                if np.isfinite([ecef_x[i], ecef_y[i], ecef_z[i]]).all():
                    lat, lon, h = converter.ecef_to_lla(ecef_x[i], ecef_y[i], ecef_z[i])
                    llh_data[i] = [lon, lat, h]
                else:
                    llh_data[i] = np.nan
            
            return llh_data
            
        except Exception as e:
            logger.error(f"Failed to load truth data: {e}")
            return None

# ---------------------- LS求解器类 ----------------------
class LeastSquaresSolver:
    """最小二乘求解器"""
    
    def __init__(self, max_iter: int = 20, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.converter = CoordinateConverter()
    
    def solve_epoch(self, p_corr: np.ndarray, sat_pos: np.ndarray, 
                    x0: Optional[np.ndarray] = None) -> Tuple:
        """
        求解单历元
        
        Returns:
            (solution, H, nsat, success)
        """
        # 筛选有效卫星
        valid = np.isfinite(p_corr) & np.all(np.isfinite(sat_pos), axis=1)
        idx = np.where(valid)[0]
        
        if idx.size < 4:
            return None, None, 0, False
        
        P = p_corr[idx]
        S = sat_pos[idx]
        
        # 初始状态
        xr = np.zeros(3) if x0 is None else np.array(x0[:3], dtype=float)
        cb = 0.0 if (x0 is None or len(x0) < 4) else float(x0[3])
        
        H_last = None
        success = False
        
        # 迭代求解
        for iteration in range(self.max_iter):
            # 向量化计算几何距离
            diff = S - xr
            rho = np.linalg.norm(diff, axis=1)
            
            # 单位视线向量
            u = (xr - S) / rho[:, np.newaxis]
            
            # 残差
            pred = rho + cb
            v = P - pred
            
            # 构建设计矩阵
            H = np.column_stack([u, np.ones(len(idx))])
            
            # LS更新
            try:
                dx = np.linalg.lstsq(H, v, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            
            xr += dx[:3]
            cb += dx[3]
            H_last = H
            
            if np.linalg.norm(dx) < self.tol:
                success = True
                break
        
        if not success and H_last is not None:
            success = True
        
        return (xr[0], xr[1], xr[2], cb), H_last, idx.size, success

# ---------------------- 误差分析类 ----------------------
class ErrorAnalyzer:
    """误差分析工具"""
    
    def __init__(self):
        self.converter = CoordinateConverter()
    
    def compute_statistics(self, ls_xyz: np.ndarray, truth_xyz: np.ndarray,
                          ref_lat: float, ref_lon: float) -> ErrorStatistics:
        """计算误差统计"""
        R = self.converter.ecef_to_enu_rotation(ref_lat, ref_lon)
        ref_xyz = truth_xyz[0]
        
        # 转换到ENU
        ls_enu = (R @ (ls_xyz - ref_xyz).T).T
        truth_enu = (R @ (truth_xyz - ref_xyz).T).T
        
        # 计算误差
        delta = ls_enu - truth_enu
        e_err, n_err, u_err = delta[:, 0], delta[:, 1], delta[:, 2]
        
        # 2D和3D误差
        err_2d = np.sqrt(e_err**2 + n_err**2)
        err_3d = np.linalg.norm(delta, axis=1)
        
        # 统计量
        valid_mask = np.isfinite(err_3d)
        stats = {}
        
        if valid_mask.sum() > 0:
            for prefix, err in [('e', e_err), ('n', n_err), ('u', u_err),
                               ('2d', err_2d), ('3d', err_3d)]:
                valid_err = err[valid_mask]
                stats.update({
                    f'{prefix}_mean': np.mean(valid_err),
                    f'{prefix}_std': np.std(valid_err),
                    f'{prefix}_rms': np.sqrt(np.mean(valid_err**2))
                })
        
        return ErrorStatistics(e_err, n_err, u_err, err_2d, err_3d, stats)

# ---------------------- 可视化类 ----------------------
class Visualizer:
    """可视化工具"""
    
    def __init__(self, output_prefix: str):
        self.output_prefix = output_prefix
        self.converter = CoordinateConverter()
    
    def _print_error_stats(self, error_data: ErrorStatistics):
        """打印误差统计到日志"""
        if not error_data or not error_data.stats:
            return
        st = error_data.stats
        logger.info("[Error Statistics]")
        logger.info(f"  East  - Mean: {st['e_mean']:7.3f} m, Std: {st['e_std']:6.3f} m, RMS: {st['e_rms']:6.3f} m")
        logger.info(f"  North - Mean: {st['n_mean']:7.3f} m, Std: {st['n_std']:6.3f} m, RMS: {st['n_rms']:6.3f} m")
        logger.info(f"  Up    - Mean: {st['u_mean']:7.3f} m, Std: {st['u_std']:6.3f} m, RMS: {st['u_rms']:6.3f} m")
        logger.info(f"  2D    - Mean: {st['2d_mean']:7.3f} m, Std: {st['2d_std']:6.3f} m, RMS: {st['2d_rms']:6.3f} m")
        logger.info(f"  3D    - Mean: {st['3d_mean']:7.3f} m, Std: {st['3d_std']:6.3f} m, RMS: {st['3d_rms']:6.3f} m")

    def generate_plots(self, solutions: List[SPPSolution], 
                      truth_llh: Optional[np.ndarray] = None):
        """生成所有图表"""
        valid_sols = [s for s in solutions if np.isfinite([s.lat, s.lon, s.h]).all()]
        
        if not valid_sols:
            logger.warning("No valid solutions for plotting")
            return
        
        lats = np.array([s.lat for s in valid_sols])
        lons = np.array([s.lon for s in valid_sols])
        hs = np.array([s.h for s in valid_sols])
        xyz = np.array([s.xyz for s in valid_sols])
        
        # 计算ENU
        ref_lat, ref_lon, ref_h = lats[0], lons[0], hs[0]
        ref_xyz = xyz[0]
        R = self.converter.ecef_to_enu_rotation(ref_lat, ref_lon)
        enu = (R @ (xyz - ref_xyz).T).T
        
        # 处理真值
        t_lon = t_lat = t_h = t_enu = None
        error_data = None
        
        if truth_llh is not None and len(truth_llh) >= 1:
            n = min(len(valid_sols), len(truth_llh))
            t_lon, t_lat, t_h = truth_llh[:n, 0], truth_llh[:n, 1], truth_llh[:n, 2]
            
            t_xyz = np.array([
                self.converter.lla_to_ecef(t_lat[i], t_lon[i], t_h[i])
                for i in range(n)
            ])
            t_enu = (R @ (t_xyz - ref_xyz).T).T
            
            # 计算误差
            analyzer = ErrorAnalyzer()
            error_data = analyzer.compute_statistics(xyz[:n], t_xyz, ref_lat, ref_lon)
            
            self._print_error_stats(error_data)
        
        # 生成各类图表
        self._plot_2d_track(lons, lats, t_lon, t_lat)
        self._plot_3d_enu(enu, t_enu)
        
        if error_data is not None:
            epochs = np.array([s.epoch for s in valid_sols[:len(error_data.e_err)]])
            self._plot_error_timeseries(epochs, error_data)
            self._plot_error_distributions(error_data)
            self._plot_error_summary(error_data)
    
    def _plot_2d_track(self, lons, lats, t_lon=None, t_lat=None):
        """绘制2D轨迹"""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        if len(lons) >= 2:
            ax.plot(lons, lats, 'k.-', linewidth=1, markersize=3, label='LS')
        else:
            ax.scatter(lons, lats, c='k', s=14, label='LS')
        
        if t_lon is not None:
            if len(t_lon) >= 2:
                ax.plot(t_lon, t_lat, 'b.-', linewidth=1, markersize=2, label='Truth')
            else:
                ax.scatter(t_lon, t_lat, c='b', s=16, label='Truth')
        
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title("Geographic Track (Lon-Lat)")
        ax.grid(alpha=0.4)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(f"{self.output_prefix}_2d.png", dpi=150)
        plt.show()
        plt.close(fig)
    
    def _plot_3d_enu(self, enu, t_enu=None):
        """绘制3D ENU轨迹"""
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(enu) >= 2:
            ax.plot(enu[:, 0], enu[:, 1], enu[:, 2], 
                   'r.-', linewidth=1, markersize=3, label='LS')
        else:
            ax.scatter(enu[:, 0], enu[:, 1], enu[:, 2], c='r', s=14, label='LS')
        
        if t_enu is not None:
            if len(t_enu) >= 2:
                ax.plot(t_enu[:, 0], t_enu[:, 1], t_enu[:, 2],
                       'b.-', linewidth=1, markersize=2, label='Truth')
            else:
                ax.scatter(t_enu[:, 0], t_enu[:, 1], t_enu[:, 2], 
                          c='b', s=16, label='Truth')
        
        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.set_zlabel("U (m)")
        ax.set_title("Track in ENU Frame")
        
        # 设置等比例坐标轴
        ranges = [enu[:, i].max() - enu[:, i].min() for i in range(3)]
        max_range = max(ranges) if np.isfinite(ranges).all() else 1.0
        centers = [(enu[:, i].max() + enu[:, i].min()) / 2.0 for i in range(3)]
        
        for i, (c, setter) in enumerate(zip(centers, 
                                            [ax.set_xlim, ax.set_ylim, ax.set_zlim])):
            setter(c - max_range / 2, c + max_range / 2)
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_prefix}_enu_3d.png", dpi=150)
        plt.show()
        plt.close()
    
    def _plot_error_timeseries(self, epochs, error_data: ErrorStatistics):
        """绘制误差时间序列（统一样式）"""
        # 统一配色与标记
        palette = ['#1f77b4', '#2ca02c', '#d62728']  # E, N, U
        markers = ['o', 's', '^']
        labels = ['East Error (m)', 'North Error (m)', 'Up Error (m)']
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        
        for ax, err, color, marker, ylabel in zip(
            axes,
            [error_data.e_err, error_data.n_err, error_data.u_err],
            palette, markers, labels
        ):
            ax.set_facecolor('#f9fbff')
            ax.plot(
                epochs, err, color=color, marker=marker, linestyle='-',
                linewidth=1.4, markersize=3.5, markerfacecolor='white',
                markeredgewidth=0.8, alpha=0.9, label=ylabel
            )
            ax.axhline(0, color='#444', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
            ax.legend(loc='upper right', frameon=True, framealpha=0.85)
        
        axes[0].set_title('Position Errors in ENU Frame')
        axes[2].set_xlabel('Epoch')
        
        fig.tight_layout()
        fig.savefig(f"{self.output_prefix}_error.png", dpi=150)
        plt.show()
        plt.close()
        
        # 2D/3D误差
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].set_facecolor('#f9fbff')
        axes[1].set_facecolor('#f9fbff')
        
        axes[0].plot(
            epochs, error_data.err_2d, color='#ff7f0e', marker='o', linestyle='-',
            linewidth=1.4, markersize=3.5, markerfacecolor='white',
            markeredgewidth=0.8, alpha=0.9, label='2D Error (m)'
        )
        axes[0].grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
        axes[0].set_ylabel('2D Error (m)')
        axes[0].set_title('2D and 3D Position Errors')
        axes[0].legend(loc='upper right', frameon=True, framealpha=0.85)
        
        axes[1].plot(
            epochs, error_data.err_3d, color='#17becf', marker='^', linestyle='-',
            linewidth=1.4, markersize=3.5, markerfacecolor='white',
            markeredgewidth=0.8, alpha=0.9, label='3D Error (m)'
        )
        axes[1].grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
        axes[1].set_ylabel('3D Error (m)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(loc='upper right', frameon=True, framealpha=0.85)
        
        fig.tight_layout()
        fig.savefig(f"{self.output_prefix}_error_2d3d.png", dpi=150)
        plt.show()
        plt.close()
    
    def _plot_error_distributions(self, error_data: ErrorStatistics):
        """绘制误差分布直方图（统一样式+均值与±1σ标注）"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        
        # ENU误差分布
        for idx, (err, name, color) in enumerate([
            (error_data.e_err, 'East', '#1f77b4'),
            (error_data.n_err, 'North', '#2ca02c'),
            (error_data.u_err, 'Up', '#d62728')
        ]):
            valid = err[np.isfinite(err)]
            ax = axes[0, idx]
            ax.set_facecolor('#f9fbff')
            ax.hist(
                valid, bins=40, density=True, color=color, alpha=0.6,
                edgecolor='white', linewidth=0.5
            )
            # 均值与±1σ
            if valid.size > 0:
                mu, sigma = np.mean(valid), np.std(valid)
                ax.axvline(mu, color=color, linestyle='-', linewidth=1.5, alpha=0.9, label='Mean')
                ax.axvspan(mu - sigma, mu + sigma, color=color, alpha=0.15, label='±1σ')
            ax.axvline(0, color='#444', linestyle='--', linewidth=0.9, alpha=0.7)
            ax.set_xlabel(f'{name} Error (m)')
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Error Distribution')
            ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
            ax.legend(loc='upper right', frameon=True, framealpha=0.85)
        
        # 2D/3D误差分布
        for idx, (err, name, color) in enumerate([
            (error_data.err_2d, '2D', '#ff7f0e'),
            (error_data.err_3d, '3D', '#17becf')
        ]):
            valid = err[np.isfinite(err)]
            ax = axes[1, idx]
            ax.set_facecolor('#f9fbff')
            ax.hist(
                valid, bins=40, density=True, color=color, alpha=0.6,
                edgecolor='white', linewidth=0.5
            )
            if valid.size > 0:
                mu, sigma = np.mean(valid), np.std(valid)
                ax.axvline(mu, color=color, linestyle='-', linewidth=1.5, alpha=0.9, label='Mean')
                ax.axvspan(mu - sigma, mu + sigma, color=color, alpha=0.15, label='±1σ')
            ax.set_xlabel(f'{name} Error (m)')
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Error Distribution')
            ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
            ax.legend(loc='upper right', frameon=True, framealpha=0.85)
        
        axes[1, 2].axis('off')
        
        fig.tight_layout()
        fig.savefig(f"{self.output_prefix}_error_hist.png", dpi=150)
        plt.show()
        plt.close()
    
    def _plot_error_summary(self, error_data: ErrorStatistics):
        """绘制误差统计汇总（更新配色/边框/透明度，添加数值标签）"""
        if not error_data.stats:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#f9fbff')
        
        components = ['East', 'North', 'Up', '2D', '3D']
        means = [error_data.stats[f'{k}_mean'] for k in ['e', 'n', 'u', '2d', '3d']]
        stds = [error_data.stats[f'{k}_std'] for k in ['e', 'n', 'u', '2d', '3d']]
        rms = [error_data.stats[f'{k}_rms'] for k in ['e', 'n', 'u', '2d', '3d']]
        
        x = np.arange(len(components))
        width = 0.25
        
        colors = ['#4e79a7', '#f28e2b', '#59a14f']
        edge = '#2f2f2f'
        
        bars_mean = ax.bar(x - width, means, width, label='Mean',
                           color=colors[0], edgecolor=edge, alpha=0.9)
        bars_std  = ax.bar(x, stds, width, label='Std Dev',
                           color=colors[1], edgecolor=edge, alpha=0.9)
        bars_rms  = ax.bar(x + width, rms, width, label='RMS',
                           color=colors[2], edgecolor=edge, alpha=0.9)
        
        # 添加数值标签（若 Matplotlib 版本支持）
        for bars in (bars_mean, bars_std, bars_rms):
            try:
                ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
            except Exception:
                pass
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Error (m)')
        ax.set_title('Error Statistics Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.6)
        
        fig.tight_layout()
        fig.savefig(f"{self.output_prefix}_statistics.png", dpi=150)
        plt.show()
        plt.close()

# ---------------------- 报告生成类 ----------------------
class ReportGenerator:
    """结果报告生成"""
    
    @staticmethod
    def print_summary(solutions: List[SPPSolution], 
                     error_data: Optional[ErrorStatistics]):
        """打印统计摘要"""
        valid_sols = [s for s in solutions if np.isfinite([s.lat, s.lon, s.h]).all()]
        
        # 改造标题为盒式样式
        print("\n" + "╔" + "═"*78 + "╗")
        title = "GNSS SPP POSITIONING RESULTS SUMMARY"
        print("║" + title.center(78) + "║")
        print("╚" + "═"*78 + "╝")
        print(f"\nTotal Epochs Processed: {len(valid_sols)}")
        print("="*80)
        
        # 定位精度统计
        if error_data and error_data.stats:
            ReportGenerator._print_accuracy_stats(error_data)
            ReportGenerator._print_enu_stats(error_data)
        
        # 坐标范围
        if valid_sols:
            ReportGenerator._print_coordinate_ranges(valid_sols)
        
        # DOP统计
        ReportGenerator._print_dop_stats(valid_sols)
        
        # 卫星数统计
        ReportGenerator._print_satellite_stats(valid_sols)
        
        print("="*80 + "\n")
    
    @staticmethod
    def _print_accuracy_stats(error_data: ErrorStatistics):
        """打印精度统计"""
        st = error_data.stats
        print("\nPOSITIONING ACCURACY STATISTICS".center(80))
        print("="*80)
        print(f"{'Metric':<25} {'2D Error (m)':<20} {'3D Error (m)':<20}")
        print("-" * 80)
        print(f"{'Mean':<25} {st['2d_mean']:>18.3f}  {st['3d_mean']:>18.3f}")
        print(f"{'RMS':<25} {st['2d_rms']:>18.3f}  {st['3d_rms']:>18.3f}")
        print(f"{'Standard Deviation':<25} {st['2d_std']:>18.3f}  {st['3d_std']:>18.3f}")
        
        # 计算最大、最小和95百分位
        valid_2d = error_data.err_2d[np.isfinite(error_data.err_2d)]
        valid_3d = error_data.err_3d[np.isfinite(error_data.err_3d)]
        
        if len(valid_2d) > 0 and len(valid_3d) > 0:
            print(f"{'Maximum':<25} {np.max(valid_2d):>18.3f}  {np.max(valid_3d):>18.3f}")
            print(f"{'Minimum':<25} {np.min(valid_2d):>18.3f}  {np.min(valid_3d):>18.3f}")
            print(f"{'95th Percentile':<25} {np.percentile(valid_2d, 95):>18.3f}  "
                  f"{np.percentile(valid_3d, 95):>18.3f}")
    
    @staticmethod
    def _print_enu_stats(error_data: ErrorStatistics):
        """打印ENU分量统计"""
        st = error_data.stats
        print("\n" + "="*80)
        print("ENU COMPONENT ACCURACY".center(80))
        print("="*80)
        print(f"{'Component':<15} {'Mean (m)':<15} {'RMS (m)':<15} {'Std Dev (m)':<15}")
        print("-" * 80)
        
        for comp, prefix in [('East', 'e'), ('North', 'n'), ('Up', 'u')]:
            print(f"{comp:<15} {st[f'{prefix}_mean']:>13.3f}  "
                  f"{st[f'{prefix}_rms']:>13.3f}  {st[f'{prefix}_std']:>13.3f}")
    
    @staticmethod
    def _print_coordinate_ranges(solutions: List[SPPSolution]):
        """打印坐标范围"""
        lats = np.array([s.lat for s in solutions])
        lons = np.array([s.lon for s in solutions])
        hs = np.array([s.h for s in solutions])
        
        print("\n" + "="*80)
        print("COORDINATE RANGES".center(80))
        print("="*80)
        print(f"Latitude Range:  {np.min(lats):.6f}°  to  {np.max(lats):.6f}°")
        print(f"Longitude Range: {np.min(lons):.6f}°  to  {np.max(lons):.6f}°")
        print(f"Altitude Range:  {np.min(hs):.3f} m  to  {np.max(hs):.3f} m")
    
    @staticmethod
    def _print_dop_stats(solutions: List[SPPSolution]):
        """打印DOP统计"""
        pdops = [s.pdop for s in solutions if np.isfinite(s.pdop)]
        hdops = [s.hdop for s in solutions if np.isfinite(s.hdop)]
        vdops = [s.vdop for s in solutions if np.isfinite(s.vdop)]
        
        if pdops and hdops and vdops:
            print("\n" + "="*80)
            print("DOP STATISTICS".center(80))
            print("="*80)
            print(f"{'Metric':<15} {'PDOP':<15} {'HDOP':<15} {'VDOP':<15}")
            print("-" * 80)
            print(f"{'Mean':<15} {np.mean(pdops):>13.3f}  "
                  f"{np.mean(hdops):>13.3f}  {np.mean(vdops):>13.3f}")
            print(f"{'Median':<15} {np.median(pdops):>13.3f}  "
                  f"{np.median(hdops):>13.3f}  {np.median(vdops):>13.3f}")
            print(f"{'Minimum':<15} {np.min(pdops):>13.3f}  "
                  f"{np.min(hdops):>13.3f}  {np.min(vdops):>13.3f}")
            print(f"{'Maximum':<15} {np.max(pdops):>13.3f}  "
                  f"{np.max(hdops):>13.3f}  {np.max(vdops):>13.3f}")
    
    @staticmethod
    def _print_satellite_stats(solutions: List[SPPSolution]):
        """打印卫星数统计"""
        nsats = [s.nsat for s in solutions if np.isfinite(s.nsat)]
        
        if nsats:
            print("\n" + "="*80)
            print("SATELLITE VISIBILITY".center(80))
            print("="*80)
            print(f"Average Satellites:  {np.mean(nsats):.1f}")
            print(f"Minimum Satellites:  {min(nsats)}")
            print(f"Maximum Satellites:  {max(nsats)}")
    
    @staticmethod
    def save_error_and_dop_csv(solutions: List[SPPSolution],
                               error_data: Optional[ErrorStatistics],
                               out_path: str):
        """保存误差和DOP数据到CSV"""
        valid_sols = [s for s in solutions if np.isfinite([s.lat, s.lon, s.h]).all()]
        
        if not valid_sols:
            logger.warning("No valid solutions to export")
            return
        
        # 先收集基础列
        epochs = [s.epoch for s in valid_sols]
        nsats  = [s.nsat for s in valid_sols]
        pdops  = [s.pdop for s in valid_sols]
        hdops  = [s.hdop for s in valid_sols]
        vdops  = [s.vdop for s in valid_sols]
        
        # 统一长度（若有真值误差则按较短者截断所有列）
        if error_data is not None:
            n = min(len(epochs), len(error_data.e_err))
            epochs = epochs[:n]
            nsats  = nsats[:n]
            pdops  = pdops[:n]
            hdops  = hdops[:n]
            vdops  = vdops[:n]
            data_dict = {
                'epoch': epochs,
                'nsat': nsats,
                'PDOP': pdops,
                'HDOP': hdops,
                'VDOP': vdops,
                'east_error_m': error_data.e_err[:n],
                'north_error_m': error_data.n_err[:n],
                'up_error_m': error_data.u_err[:n],
                '2d_error_m': error_data.err_2d[:n],
                '3d_error_m': error_data.err_3d[:n]
            }
        else:
            data_dict = {
                'epoch': epochs,
                'nsat': nsats,
                'PDOP': pdops,
                'HDOP': hdops,
                'VDOP': vdops
            }
        
        df = pd.DataFrame(data_dict)
        df.to_csv(out_path, index=False, float_format='%.6f')
        logger.info(f"Saved error/DOP data to {out_path}")
        
        # 保存统计摘要（不需截断，沿用原逻辑）
        if error_data and error_data.stats:
            ReportGenerator._save_summary_csv(error_data, valid_sols, out_path)
    
    @staticmethod
    def _save_summary_csv(error_data: ErrorStatistics,
                         solutions: List[SPPSolution],
                         base_path: str):
        """保存统计摘要CSV"""
        stats_path = base_path.rsplit('.', 1)[0] + '_summary.csv'
        st = error_data.stats
        
        stats_data = []
        
        # 误差统计
        for comp, prefix in [('East', 'e'), ('North', 'n'), ('Up', 'u'),
                            ('2D', '2d'), ('3D', '3d')]:
            stats_data.append({
                'Component': comp,
                'Mean_m': st[f'{prefix}_mean'],
                'RMS_m': st[f'{prefix}_rms'],
                'Std_m': st[f'{prefix}_std']
            })
        
        # DOP统计
        pdops = [s.pdop for s in solutions if np.isfinite(s.pdop)]
        hdops = [s.hdop for s in solutions if np.isfinite(s.hdop)]
        vdops = [s.vdop for s in solutions if np.isfinite(s.vdop)]
        
        if pdops and hdops and vdops:
            for name, vals in [('PDOP', pdops), ('HDOP', hdops), ('VDOP', vdops)]:
                stats_data.append({
                    'Component': f'{name}_mean',
                    'Mean_m': np.mean(vals),
                    'RMS_m': np.median(vals),
                    'Std_m': np.std(vals)
                })
        
        df = pd.DataFrame(stats_data)
        df.to_csv(stats_path, index=False, float_format='%.6f')
        logger.info(f"Saved summary statistics to {stats_path}")

# ---------------------- 主处理器类 ----------------------
class SPPProcessor:
    """SPP主处理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.loader = DataLoader()
        self.solver = LeastSquaresSolver(
            max_iter=config.get('max_iter', 20),
            tol=config.get('tol', 1e-7)
        )
        self.converter = CoordinateConverter()
        self.dop_calc = DOPCalculator()
        
    def process(self) -> List[SPPSolution]:
        """主处理流程"""
        # 加载数据
        logger.info("Loading data...")
        P = self.loader.load_csv(self.config['pseudoranges'])
        CLK = self.loader.load_csv(self.config['sat_clk'])
        ION = self.loader.load_csv(self.config['ion'])
        TRP = self.loader.load_csv(self.config['trop'])
        SATM = self.loader.load_csv(self.config['sat_pos'])
        
        # 验证数据
        self._validate_data(P, CLK, ION, TRP, SATM)
        
        # 分割卫星位置
        sat_per_epoch = self.loader.split_sat_positions_matrix(SATM)
        num_epochs = len(sat_per_epoch)
        
        logger.info(f"Processing {num_epochs} epochs...")
        
        # 初始值
        x0 = self._get_initial_state()
        
        # 逐历元求解
        solutions = []
        for k in range(num_epochs):
            sol = self._process_epoch(k, P[:, k], CLK[:, k], ION[:, k], 
                                     TRP[:, k], sat_per_epoch[k], x0)
            solutions.append(sol)
            
            # 更新初值
            if sol and np.isfinite([sol.x, sol.y, sol.z, sol.clock_bias]).all():
                x0 = np.array([sol.x, sol.y, sol.z, sol.clock_bias])
        
        logger.info(f"Completed processing {len(solutions)} epochs")
        return solutions
    
    def _validate_data(self, P, CLK, ION, TRP, SATM):
        """验证输入数据"""
        Smax, E = P.shape
        
        if any(arr.shape != (Smax, E) for arr in [CLK, ION, TRP]):
            raise ValueError("Data shape mismatch")
        
        if SATM.shape != (Smax, 3*E):
            raise ValueError("Satellite position shape mismatch")
    
    def _get_initial_state(self) -> Optional[np.ndarray]:
        """获取初始状态"""
        llh0 = self.config.get('llh0')
        if llh0:
            lat, lon, h = [float(v) for v in llh0.split()]
            xyz = self.converter.lla_to_ecef(lat, lon, h)
            return np.array([xyz[0], xyz[1], xyz[2], 0.0])
        return None
    
    def _process_epoch(self, k: int, P_k, CLK_k, ION_k, TRP_k, 
                      sat_pos_k, x0) -> SPPSolution:
        """处理单历元"""
        # 校正伪距
        P_corr = P_k + CLK_k - ION_k - TRP_k
        
        # LS求解
        sol, H, ns, ok = self.solver.solve_epoch(P_corr, sat_pos_k, x0)
        
        if not ok or sol is None:
            return SPPSolution(
                epoch=k+1, x=np.nan, y=np.nan, z=np.nan,
                lat=np.nan, lon=np.nan, h=np.nan, clock_bias=np.nan,
                nsat=0, pdop=np.nan, hdop=np.nan, vdop=np.nan
            )
        
        xs, ys, zs, cb = sol
        lat, lon, h = self.converter.ecef_to_lla(xs, ys, zs)
        
        # 计算DOP
        pdop, hdop, vdop = (np.nan, np.nan, np.nan)
        if H is not None and ns >= 4 and np.isfinite([lat, lon]).all():
            pdop, hdop, vdop = self.dop_calc.compute_dops(H, lat, lon)
        
        return SPPSolution(
            epoch=k+1, x=xs, y=ys, z=zs,
            lat=lat, lon=lon, h=h, clock_bias=cb,
            nsat=ns, pdop=pdop, hdop=hdop, vdop=vdop
        )

# ---------------------- 主函数 ----------------------
def main():
    """主函数"""
    # 配置
    config = {
        'pseudoranges': "/Users/yupudem2/Downloads/4203/pseudoranges_meas.csv",
        'sat_clk': "/Users/yupudem2/Downloads/4203/satellite_clock_bias.csv",
        'ion': "/Users/yupudem2/Downloads/4203/ionospheric_delay.csv",
        'trop': "/Users/yupudem2/Downloads/4203/tropospheric_delay.csv",
        'sat_pos': "/Users/yupudem2/Downloads/4203/satellite_positions.csv",
        'out_path': "/Users/yupudem2/Downloads/4203/solution.csv",
        'truth_path': "/Users/yupudem2/Downloads/4203/NAV-HPPOSECEF.csv",
        'llh0': None,
        'max_iter': 10,
        'tol': 1e-5
    }
    
    # 处理SPP
    processor = SPPProcessor(config)
    solutions = processor.process()
    
    # 保存结果
    logger.info(f"Saving results to {config['out_path']}...")
    out_array = np.array([s.to_array() for s in solutions])
    header = "epoch,x,y,z,lat_deg,lon_deg,h_m,clock_bias_m,nsat,PDOP,HDOP,VDOP"
    np.savetxt(config['out_path'], out_array, delimiter=",", 
               header=header, comments="", fmt="%.10f")
    
    # 加载真值
    loader = DataLoader()
    truth_llh = loader.load_truth_from_nav_hpposecef(config['truth_path'])
    
    # 误差分析
    error_data = None
    if truth_llh is not None:
        valid_sols = [s for s in solutions if np.isfinite(s.llh).all()]
        if valid_sols:
            xyz = np.array([s.xyz for s in valid_sols])
            converter = CoordinateConverter()
            t_xyz = np.array([
                converter.lla_to_ecef(truth_llh[i, 1], truth_llh[i, 0], truth_llh[i, 2])
                for i in range(min(len(valid_sols), len(truth_llh)))
            ])
            
            analyzer = ErrorAnalyzer()
            error_data = analyzer.compute_statistics(
                xyz[:len(t_xyz)], t_xyz,
                valid_sols[0].lat, valid_sols[0].lon
            )
    
    # 生成图表
    output_prefix = config['out_path'].rsplit('.', 1)[0]
    visualizer = Visualizer(output_prefix)
    visualizer.generate_plots(solutions, truth_llh)
    
    # 打印报告
    ReportGenerator.print_summary(solutions, error_data)
    
    # 保存误差/DOP CSV
    error_dop_csv = output_prefix + '_error_dop.csv'
    ReportGenerator.save_error_and_dop_csv(solutions, error_data, error_dop_csv)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()