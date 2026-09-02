import { RawTrackingPoint, PipelineParameters, ChamberBox } from '../types';

export interface SymmetricGridConfig {
  chamber_id: number;
  roi: [number, number, number, number];
  row: number;
  col: number;
}

export function generateSymmetricChambers(
  frameWidth: number = 880,
  frameHeight: number = 500,
  firstRoi: [number, number, number, number] = [44, 40, 420, 115],
  rows: number = 4,
  cols: number = 2,
  minGap: number = 5,
  order: 'column_first' | 'row_first' = 'column_first'
): ChamberBox[] {
  const [fx1, fy1, fx2, fy2] = firstRoi;
  const tubeW = Math.max(10, fx2 - fx1);
  const tubeH = Math.max(10, fy2 - fy1);

  const availableW = frameWidth - 2 * fx1;
  const colPitch = cols > 1 ? Math.floor((availableW - tubeW) / (cols - 1)) : tubeW + minGap;

  const availableH = frameHeight - 2 * fy1;
  const rowPitch = rows > 1 ? Math.floor((availableH - tubeH) / (rows - 1)) : tubeH + minGap;

  const chambers: ChamberBox[] = [];
  let currentId = 1;

  if (order === 'column_first') {
    for (let c = 0; c < cols; c++) {
      const curX1 = fx1 + c * colPitch;
      const curX2 = Math.min(frameWidth, curX1 + tubeW);
      for (let r = 0; r < rows; r++) {
        const curY1 = fy1 + r * rowPitch;
        const curY2 = Math.min(frameHeight, curY1 + tubeH);
        chambers.push({
          id: currentId,
          x1: Math.round(curX1),
          y1: Math.round(curY1),
          x2: Math.round(curX2),
          y2: Math.round(curY2)
        });
        currentId++;
      }
    }
  } else {
    for (let r = 0; r < rows; r++) {
      const curY1 = fy1 + r * rowPitch;
      const curY2 = Math.min(frameHeight, curY1 + tubeH);
      for (let c = 0; c < cols; c++) {
        const curX1 = fx1 + c * colPitch;
        const curX2 = Math.min(frameWidth, curX1 + tubeW);
        chambers.push({
          id: currentId,
          x1: Math.round(curX1),
          y1: Math.round(curY1),
          x2: Math.round(curX2),
          y2: Math.round(curY2)
        });
        currentId++;
      }
    }
  }

  return chambers;
}

export function generateSimulationDataset(
  preset: 'anesthesia' | 'recovery_assay' | 'noisy_climbing',
  numChambers: number = 8,
  totalSeconds: number = 180,
  fps: number = 30
): { points: RawTrackingPoint[]; metadata: { title: string; desc: string; fps: number; totalFrames: number } } {
  const totalFrames = Math.floor(totalSeconds * fps);
  const points: RawTrackingPoint[] = [];

  // Multi-Chamber physical coordinate boxes
  const rows = Math.min(8, Math.ceil(numChambers / 2));
  const cols = Math.ceil(numChambers / rows);
  const chamberBounds = generateSymmetricChambers(880, 500, [45, 40, 420, 115], rows, cols).slice(0, numChambers);

  for (let chIdx = 0; chIdx < numChambers; chIdx++) {
    const box = chamberBounds[chIdx];
    const chamberId = box.id;
    const spanX = Math.max(10, box.x2 - box.x1);
    let currentX = box.x1 + spanX * (0.2 + 0.6 * Math.random());
    let currentY = box.y1 + (box.y2 - box.y1) * 0.5;
    let isKnockedDown = false;
    let knockdownFrame = Infinity;
    let recoveryFrame = Infinity;

    if (preset === 'anesthesia') {
      // Knockdown occurs between 40s and 120s
      knockdownFrame = Math.floor((35 + chIdx * 12 + Math.random() * 10) * fps);
    } else if (preset === 'recovery_assay') {
      // Sedated at start, gradually wakes up upon wash-out
      knockdownFrame = 0;
      recoveryFrame = Math.floor((60 + chIdx * 14 + Math.random() * 10) * fps);
      isKnockedDown = true;
    }

    for (let f = 0; f < totalFrames; f++) {
      let speed = 0;
      let xPx: number | null = currentX;
      let yPx: number | null = currentY;
      const tSec = f / fps;

      if (preset === 'anesthesia' && f >= knockdownFrame) {
        isKnockedDown = true;
      } else if (preset === 'recovery_assay' && f >= recoveryFrame) {
        isKnockedDown = false;
      }

      if (isKnockedDown) {
        // Animal is sedated with resting micro-drift < 0.03px
        speed = 0.02 * (Math.random() - 0.5);
        xPx = Math.min(box.x2 - 5, Math.max(box.x1 + 5, currentX + speed));
        yPx = currentY + 0.02 * (Math.random() - 0.5);
      } else {
        // Active climbing / negative geotaxis
        const walkDirection = Math.random() < 0.6 ? 1 : -1;
        const stepSize = Math.random() * 3.5 + 0.5;
        currentX += walkDirection * stepSize;

        if (currentX > box.x2 - 10) currentX = box.x2 - 10;
        if (currentX < box.x1 + 10) currentX = box.x1 + 10;

        currentY = box.y1 + (box.y2 - box.y1) * 0.5 + (Math.random() - 0.5) * 6;
        xPx = currentX;
        yPx = currentY;
      }

      // Inject realistic plug occlusion trap artifacts for benchmark testing
      if (preset === 'noisy_climbing' || f % 180 > 140 && f % 180 < 165 && Math.random() < 0.7 && !isKnockedDown) {
        xPx = box.x1 + 2;
        yPx = box.y1 + 10;
      }

      // Inject speed spike
      if (Math.random() < 0.003 && !isKnockedDown) {
        xPx = box.x1 + Math.random() * spanX;
      }

      // Inject missing frame
      if (Math.random() < 0.008) {
        xPx = null;
        yPx = null;
      }

      points.push({
        frame: f,
        chamber_id: chamberId,
        fly_id: 1,
        timestamp_s: Math.round(tSec * 1000) / 1000,
        x_px: xPx !== null ? Math.round(xPx * 100) / 100 : null,
        y_px: yPx !== null ? Math.round(yPx * 100) / 100 : null,
        norm_x: xPx !== null ? Math.round(((xPx - box.x1) / spanX) * 10000) / 10000 : null,
        norm_y: yPx !== null ? Math.round(((yPx - box.y1) / (box.y2 - box.y1)) * 10000) / 10000 : null,
        area: 55 + Math.random() * 15
      });
    }
  }

  const titles = {
    anesthesia: `Volatile Anesthetic Induction Kinetics (${numChambers} Chambers)`,
    recovery_assay: `Post-Anesthetic Washout Recovery (${numChambers} Chambers)`,
    noisy_climbing: `Locomotor Assay with Boundary Occlusion Traps (${numChambers} Chambers)`
  };

  const descs = {
    anesthesia: `High-throughput ${numChambers}-Chamber volatile anesthetic (Isoflurane / Halothane / CO2) knockdown assay quantifying distinct induction times and 120s sliding window stillness.`,
    recovery_assay: `Washout recovery kinetics tracking latency to first spontaneous movement post-anesthesia.`,
    noisy_climbing: `Raw multi-chamber tracking benchmark with plug shadows, speed spikes, and dropped frames to evaluate the Savitzky-Golay and jump filter pipeline.`
  };

  return {
    points,
    metadata: {
      title: titles[preset],
      desc: descs[preset],
      fps,
      totalFrames
    }
  };
}
