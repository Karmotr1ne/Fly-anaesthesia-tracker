export interface RawTrackingPoint {
  frame: number;
  chamber_id: number;
  fly_id: number;
  timestamp_s?: number;
  x_px: number | null;
  y_px: number | null;
  norm_x: number | null;
  norm_y: number | null;
  area?: number | null;
}

export interface CleanedKinematicPoint {
  frame: number;
  chamber_id: number;
  timestamp_s: number;
  x_raw: number | null;
  y_raw: number | null;
  x_clean: number | null;
  y_clean: number | null;
  norm_pos: number;
  speed: number;
  activity_bin: number;
  is_occluded?: boolean;
  is_jump?: boolean;
}

export interface AnesthesiaResult {
  chamber_id: number;
  induction_time_sec: number | null;
  is_sedated: boolean;
  baseline_speed: number;
  pre_sedation_activity: number;
  binned_activity: { time_sec: number; speed: number; is_still: boolean }[];
}

export interface ChamberBox {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PipelineParameters {
  fps: number;
  bodyLengthPx: number;
  bodyLengthThresh: number;
  maxSpeedPxPerFrame: number;
  occlusionDispThresh: number;
  occlusionVarThresh: number;
  savgolWindow: number;
  savgolPoly: number;
  
  // Grid parameters
  gridRows: number;
  gridCols: number;
  gridOrder: 'column_first' | 'row_first';

  // Anesthesia Kinetics Parameters
  anesthesiaBinSec: number;
  anesthesiaWindowBins: number;
  anesthesiaThreshold: number;
}
