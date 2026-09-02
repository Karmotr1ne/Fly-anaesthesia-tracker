import {
  RawTrackingPoint,
  CleanedKinematicPoint,
  AnesthesiaResult,
  PipelineParameters
} from '../types';

export const DEFAULT_PIPELINE_PARAMS: PipelineParameters = {
  fps: 30,
  bodyLengthPx: 12,
  bodyLengthThresh: 0.5,
  maxSpeedPxPerFrame: 45,
  occlusionDispThresh: 80,
  occlusionVarThresh: 8,
  savgolWindow: 7,
  savgolPoly: 2,
  
  // Grid parameters
  gridRows: 4,
  gridCols: 2,
  gridOrder: 'column_first',

  // Anesthesia Kinetics Parameters
  anesthesiaBinSec: 5,
  anesthesiaWindowBins: 24, // 24 * 5s = 120s
  anesthesiaThreshold: 0.01
};

/**
 * 1D Savitzky-Golay smoothing filter implementation in TypeScript
 */
function savitzkyGolay(data: number[], windowSize: number = 7): number[] {
  const n = data.length;
  if (n < windowSize) return [...data];

  // Convolution coefficients for quadratic polynomial (window=7, order=2)
  // [-2, 3, 6, 7, 6, 3, -2] / 21
  const coeffs7 = [-2, 3, 6, 7, 6, 3, -2];
  const denom = 21;
  const half = Math.floor(windowSize / 2);
  const out = new Array(n);

  for (let i = 0; i < n; i++) {
    if (i < half || i >= n - half) {
      out[i] = data[i]; // Boundary preservation
    } else {
      let sum = 0;
      for (let j = -half; j <= half; j++) {
        sum += data[i + j] * coeffs7[j + half];
      }
      out[i] = sum / denom;
    }
  }
  return out;
}

/**
 * Module 2: Kinematic Cleaning & Artifact Clamping
 * Robust pipeline:
 * 1. Physical 1% to 99% Percentile Bounds Filtering
 * 2. Occlusion Trap Clamping (plug/mesh shadow relocation)
 * 3. Velocity Spike Jump Rejection
 * 4. Savitzky-Golay Trajectory Smoothing on valid data (no artificial secondary interpolation)
 * 5. Normalized Vertical Position & Micro-movement Inactivity Speed Filter
 */
export function cleanRawTrajectory(
  rawPoints: RawTrackingPoint[],
  params: PipelineParameters = DEFAULT_PIPELINE_PARAMS
): CleanedKinematicPoint[] {
  const byChamber = new Map<number, RawTrackingPoint[]>();
  rawPoints.forEach((pt) => {
    if (!byChamber.has(pt.chamber_id)) byChamber.set(pt.chamber_id, []);
    byChamber.get(pt.chamber_id)!.push(pt);
  });

  const cleanedAll: CleanedKinematicPoint[] = [];
  const distThresh = params.bodyLengthThresh * params.bodyLengthPx;

  byChamber.forEach((points, chId) => {
    points.sort((a, b) => a.frame - b.frame);
    const n = points.length;

    // 1. Physical 1% to 99% Percentile Bounds
    const validXs = points.filter((p) => p.x_px !== null).map((p) => p.x_px!) as number[];
    const validYs = points.filter((p) => p.y_px !== null).map((p) => p.y_px!) as number[];

    if (validXs.length < 15) {
      points.forEach((p) => {
        cleanedAll.push({
          frame: p.frame,
          chamber_id: chId,
          timestamp_s: p.timestamp_s ?? p.frame / params.fps,
          x_raw: p.x_px,
          y_raw: p.y_px,
          x_clean: p.x_px,
          y_clean: p.y_px,
          norm_pos: 0.5,
          speed: 0,
          activity_bin: 0
        });
      });
      return;
    }

    validXs.sort((a, b) => a - b);
    validYs.sort((a, b) => a - b);
    const minX = validXs[Math.floor(validXs.length * 0.01)];
    const maxX = validXs[Math.floor(validXs.length * 0.99)];
    const minY = validYs[Math.floor(validYs.length * 0.01)];
    const maxY = validYs[Math.floor(validYs.length * 0.99)];
    const centerX = (minX + maxX) / 2.0;
    const spanX = maxX - minX > 0 ? maxX - minX : 1.0;

    const xClean = new Array<number | null>(n);
    const yClean = new Array<number | null>(n);
    const isOccluded = new Array<boolean>(n).fill(false);
    const isJump = new Array<boolean>(n).fill(false);

    // Initial Out-of-bounds Filter
    for (let i = 0; i < n; i++) {
      const px = points[i].x_px;
      const py = points[i].y_px;
      if (px === null || py === null || px < minX - 15 || px > maxX + 15 || py < minY - 15 || py > maxY + 15) {
        xClean[i] = null;
        yClean[i] = null;
      } else {
        xClean[i] = px;
        yClean[i] = py;
      }
    }

    // 2. Occlusion Trap Clamping (detection of jump + low local variance)
    const win = 5;
    for (let i = 1; i < n; i++) {
      if (xClean[i] !== null && xClean[i - 1] !== null) {
        const dx = Math.abs(xClean[i]! - xClean[i - 1]!);
        if (dx > params.occlusionDispThresh) {
          // Check local variance
          const slice: number[] = [];
          for (let k = Math.max(0, i - Math.floor(win / 2)); k <= Math.min(n - 1, i + Math.floor(win / 2)); k++) {
            if (xClean[k] !== null) slice.push(xClean[k]!);
          }
          const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
          const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / slice.length;
          const std = Math.sqrt(variance);

          if (std < params.occlusionVarThresh) {
            isOccluded[i] = true;
            // Clamp to nearest boundary edge
            xClean[i] = xClean[i]! < centerX ? minX : maxX;
          }
        }
      }
    }

    // 3. Velocity Spike & Jump Outliers
    for (let i = 1; i < n; i++) {
      if (xClean[i] !== null && xClean[i - 1] !== null && yClean[i] !== null && yClean[i - 1] !== null) {
        const stepDist = Math.hypot(xClean[i]! - xClean[i - 1]!, yClean[i]! - yClean[i - 1]!);
        if (stepDist > params.maxSpeedPxPerFrame) {
          isJump[i] = true;
          xClean[i] = null;
          yClean[i] = null;
        }
      }
    }

    // 4. Savitzky-Golay Trajectory Smoothing Filter (Directly applied to valid coordinates without redundant interpolation)
    const validIdxs: number[] = [];
    const validDataX: number[] = [];
    const validDataY: number[] = [];
    for (let k = 0; k < n; k++) {
      if (xClean[k] !== null && yClean[k] !== null) {
        validIdxs.push(k);
        validDataX.push(xClean[k]!);
        validDataY.push(yClean[k]!);
      }
    }

    if (validDataX.length > params.savgolWindow) {
      const smoothedX = savitzkyGolay(validDataX, params.savgolWindow);
      const smoothedY = savitzkyGolay(validDataY, params.savgolWindow);
      validIdxs.forEach((origIdx, pos) => {
        xClean[origIdx] = smoothedX[pos];
        yClean[origIdx] = smoothedY[pos];
      });
    }

    // 5. Build Cleaned Kinematic Points
    for (let k = 0; k < n; k++) {
      const curX = xClean[k];
      const curY = yClean[k];
      let speed = 0;
      let normPos = 0.5;

      if (curX !== null) {
        normPos = Math.min(1.0, Math.max(0.0, (curX - minX) / spanX));
      }

      if (k > 0 && curX !== null && curY !== null && xClean[k - 1] !== null && yClean[k - 1] !== null) {
        const dx = curX - xClean[k - 1]!;
        const dy = curY - yClean[k - 1]!;
        let stepDist = Math.hypot(dx, dy);
        // Eliminate micro-jitter below 0.5 body lengths
        if (stepDist < distThresh) stepDist = 0;
        speed = stepDist * params.fps;
      }

      const tSec = points[k].timestamp_s ?? points[k].frame / params.fps;

      cleanedAll.push({
        frame: points[k].frame,
        chamber_id: chId,
        timestamp_s: Math.round(tSec * 1000) / 1000,
        x_raw: points[k].x_px,
        y_raw: points[k].y_px,
        x_clean: curX !== null ? Math.round(curX * 100) / 100 : null,
        y_clean: curY !== null ? Math.round(curY * 100) / 100 : null,
        norm_pos: Math.round(normPos * 1000) / 1000,
        speed: Math.round(speed * 100) / 100,
        activity_bin: speed > 0.1 ? 1 : 0,
        is_occluded: isOccluded[k],
        is_jump: isJump[k]
      });
    }
  });

  cleanedAll.sort((a, b) => a.frame - b.frame || a.chamber_id - b.chamber_id);
  return cleanedAll;
}

/**
 * Module 3: Stationary Detection Engine (Vectorized sliding window maximum filter)
 */
export function slidingWindowMaxFilter(
  activitySeries: number[],
  windowSize: number,
  threshold: number
): boolean[] {
  const n = activitySeries.length;
  const isStationary = new Array<boolean>(n).fill(false);
  if (n === 0 || windowSize <= 0) return isStationary;

  for (let t = 0; t <= n - windowSize; t++) {
    let maxVal = -Infinity;
    for (let w = 0; w < windowSize; w++) {
      const val = activitySeries[t + w];
      if (val > maxVal) maxVal = val;
    }
    if (maxVal < threshold) {
      isStationary[t] = true;
    }
  }
  return isStationary;
}

/**
 * Module 4: Anesthesia & Sedation Kinetics Evaluation
 */
export function evaluateAnesthesiaKinetics(
  cleanedPoints: CleanedKinematicPoint[],
  params: PipelineParameters = DEFAULT_PIPELINE_PARAMS
): AnesthesiaResult[] {
  const byChamber = new Map<number, CleanedKinematicPoint[]>();
  cleanedPoints.forEach((pt) => {
    if (!byChamber.has(pt.chamber_id)) byChamber.set(pt.chamber_id, []);
    byChamber.get(pt.chamber_id)!.push(pt);
  });

  const results: AnesthesiaResult[] = [];
  const framesPerBin = Math.max(1, Math.round(params.anesthesiaBinSec * params.fps));

  byChamber.forEach((points, chId) => {
    points.sort((a, b) => a.frame - b.frame);
    const totalFrames = points.length;
    const numBins = Math.ceil(totalFrames / framesPerBin);
    const binnedSpeeds: number[] = new Array(numBins);
    const timeSecAxis: number[] = new Array(numBins);

    for (let b = 0; b < numBins; b++) {
      const startF = b * framesPerBin;
      const endF = Math.min(totalFrames, (b + 1) * framesPerBin);
      let sumSpeed = 0;
      let count = 0;
      for (let k = startF; k < endF; k++) {
        sumSpeed += points[k].speed;
        count++;
      }
      binnedSpeeds[b] = count > 0 ? sumSpeed / count : 0;
      timeSecAxis[b] = b * params.anesthesiaBinSec;
    }

    // Baseline speed (first 10% of recording or initial 60 seconds)
    const initBins = Math.max(1, Math.min(12, Math.floor(numBins * 0.1)));
    const baselineSpeed = binnedSpeeds.slice(0, initBins).reduce((a, b) => a + b, 0) / initBins;

    // Apply Module 3 Sliding Window Max Filter (W = 120s, 24 bins)
    const stationaryMask = slidingWindowMaxFilter(
      binnedSpeeds,
      params.anesthesiaWindowBins,
      params.anesthesiaThreshold
    );

    let inductionTimeSec: number | null = null;
    let isSedated = false;
    let preSedationActivity = 0;

    const firstOnsetIdx = stationaryMask.findIndex((val) => val === true);
    if (firstOnsetIdx !== -1) {
      inductionTimeSec = timeSecAxis[firstOnsetIdx];
      isSedated = true;
      const preBins = binnedSpeeds.slice(0, Math.max(1, firstOnsetIdx));
      preSedationActivity = preBins.reduce((a, b) => a + b, 0) / preBins.length;
    } else {
      preSedationActivity = binnedSpeeds.reduce((a, b) => a + b, 0) / binnedSpeeds.length;
    }

    const binnedActivityTimeline = binnedSpeeds.map((spd, idx) => ({
      time_sec: timeSecAxis[idx],
      speed: Math.round(spd * 100) / 100,
      is_still: stationaryMask[idx]
    }));

    results.push({
      chamber_id: chId,
      induction_time_sec: inductionTimeSec !== null ? Math.round(inductionTimeSec * 10) / 10 : null,
      is_sedated: isSedated,
      baseline_speed: Math.round(baselineSpeed * 100) / 100,
      pre_sedation_activity: Math.round(preSedationActivity * 100) / 100,
      binned_activity: binnedActivityTimeline
    });
  });

  results.sort((a, b) => a.chamber_id - b.chamber_id);
  return results;
}

