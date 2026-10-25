"""
Plate Saturation & Equilibration Latency Calculator
Calculates dead volume latency, 99% full saturation time, and flow velocity
based on plate geometry (pure rectangular chambers, no mesh) and multi-channel gas flow.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class SaturationMetrics:
    """Standardized metrics object for UI data binding."""
    dead_volume_latency_s: float
    full_saturation_s: float
    linear_velocity_cm_s: float
    row_progress_s: List[float] = field(default_factory=list)  # Legacy: column times, bottom -> top
    d_eff_cm2_s: float = 0.0
    micro_radial_diffusion_s: float = 0.0
    geometry_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dead_volume_latency_s": self.dead_volume_latency_s,
            "full_saturation_s": self.full_saturation_s,
            "linear_velocity_cm_s": self.linear_velocity_cm_s,
            "row_progress_s": self.row_progress_s,
            "d_eff_cm2_s": self.d_eff_cm2_s,
            "micro_radial_diffusion_s": self.micro_radial_diffusion_s,
            "geometry_summary": self.geometry_summary,
        }


@dataclass
class PlateDesignConfig:
    name: str = "Standard Rectangular Plate"
    rows: int = 4                      # Parallel flow channels (image rows)
    cols: int = 3                      # Serial chambers per channel (left bottom to right top)
    plate_total_len_cm: float = 13.6   # Total flow channel active length (cm)
    gap_len_cm: float = 0.2            # Inter-chamber gap along each flow channel (cm)
    chamber_r_cm: float = 0.35         # Semi-cylindrical cross-section radius (cm)
    tube_id_cm: float = 0.6            # Gas inlet tube inner diameter (cm)
    tube_len_cm: float = 55.0          # Gas inlet tube length (cm)
    mix_chamber_w_cm: float = 0.7      # Top buffer/mixing chamber width (cm)
    mix_chamber_h_cm: float = 0.45     # Top buffer/mixing chamber height (cm)
    has_mesh: bool = False             # Mesh separator disabled by design


class SaturationCalculator:
    """
    Core calculator for plate gas flushing and saturation kinetics.
    Supports direct (rows, cols, total_flow_sccm) initialization or PlateDesignConfig.
    """

    # Binary diffusion coefficients at 298.15 K, 1 atm (cm^2/s)
    D_CO2_N2 = 0.165
    D_CO2_O2 = 0.160
    D_O2_N2 = 0.205

    def __init__(
        self,
        config: Optional[PlateDesignConfig] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        total_flow_sccm: float = 200.0,
        **kwargs
    ):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = PlateDesignConfig()
            if rows is not None:
                self.cfg.rows = max(1, int(rows))
            if cols is not None:
                self.cfg.cols = max(1, int(cols))
        
        self.total_flow_sccm = max(10.0, float(total_flow_sccm))

    def maximum_flow(self):
        """Return the configured 18 cm/s flow ceiling rounded down to 10 sccm."""
        area = self.get_geometry_details()["single_cross_area_cm2"]
        return math.floor(18.0 * area * self.cfg.rows * 60.0 / 10.0) * 10.0

    def carrier_flows(self, anaesthetic=0.0):
        total = self.maximum_flow()
        anaesthetic = float(anaesthetic)
        if not math.isfinite(anaesthetic) or not 0 <= anaesthetic <= total:
            raise ValueError("Anaesthetic flow exceeds the calculated total flow.")
        oxygen = round((total - anaesthetic) * 0.2, 1)
        nitrogen = round(total - anaesthetic - oxygen, 1)
        return {"o2": oxygen, "anes": anaesthetic, "n2": nitrogen}

    def get_geometry_details(self) -> Dict[str, float]:
        """Calculates volume and section metrics based on current plate geometry."""
        n_rows = self.cfg.rows
        n_cols = self.cfg.cols

        total_gaps = (n_cols - 1) * self.cfg.gap_len_cm
        h_cell = (self.cfg.plate_total_len_cm - total_gaps) / max(1, n_cols)
        single_cross_area = 0.5 * math.pi * (self.cfg.chamber_r_cm ** 2)

        v_single_cell = single_cross_area * h_cell
        v_gap_single = single_cross_area * self.cfg.gap_len_cm
        v_channel = n_cols * v_single_cell + (n_cols - 1) * v_gap_single
        v_exp_total = n_rows * v_channel

        mix_len = (4.5 / 4.0) * n_rows
        v_mix = self.cfg.mix_chamber_w_cm * self.cfg.mix_chamber_h_cm * mix_len
        v_tube = math.pi * ((self.cfg.tube_id_cm / 2.0) ** 2) * self.cfg.tube_len_cm

        return {
            "h_cell_cm": h_cell,
            "single_cross_area_cm2": single_cross_area,
            "v_single_cell_ml": v_single_cell,
            "v_exp_total_ml": v_exp_total,
            "v_mix_ml": v_mix,
            "v_tube_ml": v_tube,
            "mix_len_cm": mix_len
        }

    def compute_effective_diffusivity(self, q_anes: float, q_o2: float, q_n2: float) -> float:
        """Effective diffusivity using Blanc's law."""
        q_tot = q_anes + q_o2 + q_n2
        if q_tot <= 0:
            return self.D_CO2_N2
        x_anes = q_anes / q_tot
        x_o2 = q_o2 / q_tot
        x_n2 = q_n2 / q_tot

        if x_anes < 1e-4:
            return self.D_O2_N2
        denom = (x_n2 / self.D_CO2_N2) + (x_o2 / self.D_CO2_O2)
        return (1.0 - x_anes) / denom if denom > 0 else self.D_CO2_N2

    def compute_metrics(
        self,
        q_anes_sccm: float = 0.0,
        q_o2_sccm: Optional[float] = None,
        q_n2_sccm: float = 0.0
    ) -> SaturationMetrics:
        """Computes dead volume latency, 99% saturation time, and flow speed."""
        if q_o2_sccm is None:
            q_o2_sccm = max(0.0, self.total_flow_sccm - q_anes_sccm - q_n2_sccm)
        q_total_sccm = q_anes_sccm + q_o2_sccm + q_n2_sccm
        if q_total_sccm <= 0:
            q_total_sccm = self.total_flow_sccm

        q_total_cm3_s = q_total_sccm / 60.0
        q_channel_cm3_s = q_total_cm3_s / max(1, self.cfg.rows)
        geo = self.get_geometry_details()
        d_eff = self.compute_effective_diffusivity(q_anes_sccm, q_o2_sccm, q_n2_sccm)

        # 1. Dead volume latency (tube + mixing buffer)
        t_dead_delay = (geo["v_tube_ml"] + geo["v_mix_ml"]) / q_total_cm3_s

        # 2. Turnover multiplier (without mesh separator: 2.50)
        turnover_factor = 2.15 if self.cfg.has_mesh else 2.50

        # 3. Time to reach 99% full saturation
        t_flush_total = (turnover_factor * geo["v_exp_total_ml"]) / q_total_cm3_s
        t_saturated_total = t_dead_delay + t_flush_total

        # Column-indexed times: C1 is left/bottom, C(cols) right/top inlet.
        # Each image row is a parallel path; gas crosses columns in reverse order.
        # Include only the gaps upstream of the current chamber.
        row_saturation_times = []  # Legacy field name, indexed by image column.
        for col in range(self.cfg.cols):
            chambers_from_inlet = self.cfg.cols - col
            distance_cm = (chambers_from_inlet * geo["h_cell_cm"]
                           + (chambers_from_inlet - 1) * self.cfg.gap_len_cm)
            volume_ml = self.cfg.rows * geo["single_cross_area_cm2"] * distance_cm
            t_col_sat = t_dead_delay + turnover_factor * volume_ml / q_total_cm3_s
            row_saturation_times.append(round(t_col_sat, 2))

        # Positive speed magnitude; flow is right/top -> left/bottom.
        u_down_cm_s = q_channel_cm3_s / max(1e-4, geo["single_cross_area_cm2"])

        return SaturationMetrics(
            dead_volume_latency_s=round(t_dead_delay, 2),
            full_saturation_s=round(t_saturated_total, 2),
            linear_velocity_cm_s=round(u_down_cm_s, 2),
            row_progress_s=row_saturation_times,
            d_eff_cm2_s=round(d_eff, 4),
            micro_radial_diffusion_s=round(self.cfg.chamber_r_cm ** 2 / (2.0 * max(1e-4, d_eff)), 3),
            geometry_summary=geo
        )

    # Legacy method compatibility
    def calculate_saturation(
        self,
        q_anes_sccm: float,
        q_o2_sccm: float,
        q_n2_sccm: float
    ) -> Dict[str, Any]:
        m = self.compute_metrics(q_anes_sccm, q_o2_sccm, q_n2_sccm)
        return {
            "q_total_sccm": q_anes_sccm + q_o2_sccm + q_n2_sccm,
            "tube_delay_sec": m.dead_volume_latency_s,
            "row_saturation_times_sec": m.row_progress_s,
            "total_saturated_time_sec": m.full_saturation_s,
            "total_saturated_time_min": round(m.full_saturation_s / 60.0, 3),
            "flow_speed_in_chamber_cm_s": m.linear_velocity_cm_s,
            "geometry_summary": m.geometry_summary
        }


# Aliases for backwards compatibility
GasSaturationCalculator = SaturationCalculator
