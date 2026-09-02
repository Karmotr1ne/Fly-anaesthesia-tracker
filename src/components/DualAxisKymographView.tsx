import React, { useState, useMemo } from 'react';
import { CleanedKinematicPoint, PipelineParameters } from '../types';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';
import { BarChart2, Layers, Flame, Sliders, Image, Download } from 'lucide-react';

interface DualAxisKymographViewProps {
  cleanedPoints: CleanedKinematicPoint[];
  params: PipelineParameters;
}

export const DualAxisKymographView: React.FC<DualAxisKymographViewProps> = ({
  cleanedPoints,
  params
}) => {
  const [selectedChamber, setSelectedChamber] = useState<number>(0);
  const [gridSizeTime, setGridSizeTime] = useState<number>(40);
  const [gridSizePos, setGridSizePos] = useState<number>(20);

  // Group by Chamber for Dual Y-Axis Chart
  const chamberPoints = useMemo(() => {
    return cleanedPoints.filter((p) => p.chamber_id === selectedChamber);
  }, [cleanedPoints, selectedChamber]);

  // Build smoothed activity and normalized position dataset
  const dualAxisData = useMemo(() => {
    // Downsample for fast responsive chart rendering (sample every 2 frames)
    const filtered = chamberPoints.filter((_, idx) => idx % 2 === 0);
    return filtered.map((p) => ({
      time_sec: Math.round((p.frame / params.fps) * 10) / 10,
      activity: p.speed,
      norm_pos: p.norm_pos !== undefined ? Math.round(p.norm_pos * 100) / 100 : 0.5
    }));
  }, [chamberPoints, params.fps]);

  // Compute Space-Time Density Heatmap Matrix for Normalized Kymograph (across all 8 chambers)
  const kymographMatrix = useMemo(() => {
    if (cleanedPoints.length === 0) return [];
    const maxFrame = Math.max(...cleanedPoints.map((p) => p.frame));
    const totalSec = Math.max(1, maxFrame / params.fps);

    const timeBins = gridSizeTime;
    const posBins = gridSizePos;
    const matrix: number[][] = Array.from({ length: posBins }, () => new Array(timeBins).fill(0));

    cleanedPoints.forEach((p) => {
      const tSec = p.frame / params.fps;
      const tIdx = Math.min(timeBins - 1, Math.floor((tSec / totalSec) * timeBins));
      const pIdx = Math.min(posBins - 1, Math.floor(p.norm_pos * posBins));
      if (tIdx >= 0 && tIdx < timeBins && pIdx >= 0 && pIdx < posBins) {
        matrix[pIdx][tIdx]++;
      }
    });

    // Find maximum count for log normalization
    let maxCount = 1;
    for (let r = 0; r < posBins; r++) {
      for (let c = 0; c < timeBins; c++) {
        if (matrix[r][c] > maxCount) maxCount = matrix[r][c];
      }
    }

    return { matrix, maxCount, timeBins, posBins, totalSec };
  }, [cleanedPoints, gridSizeTime, gridSizePos, params.fps]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-lg font-bold text-slate-800">Module 6: Scientific Visualization &amp; Kymograph</h2>
            <span className="bg-purple-100 text-purple-800 text-xs px-2.5 py-0.5 rounded-full font-semibold">
              Publication Ready Visuals
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Renders synchronized dual Y-axis dynamic plots (Activity vs. Vertical Height 0.0 Ground to 1.0 Top) and population
            normalized Space-Time Kymographs for export in scientific reports.
          </p>
        </div>

        {/* Chamber selector buttons */}
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-600">Active Tube:</span>
          <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200">
            {Array.from({ length: 8 }).map((_, i) => (
              <button
                key={i}
                onClick={() => setSelectedChamber(i)}
                className={`px-2.5 py-1 text-xs font-bold rounded-md transition ${
                  selectedChamber === i
                    ? 'bg-purple-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                CH{i}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 1. Dual Y-Axis Dynamic Plot (Activity Area vs. Position Line) */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-purple-600" />
              <span>Dual Y-Axis Behavioral Dynamic Plot: Activity &amp; Vertical Position (CH {selectedChamber})</span>
            </h3>
            <p className="text-xs text-slate-500">
              Left Axis (Cyan Area): Locomotor Activity (px/s) | Right Axis (Crimson Line): Vertical Position (0.0 Ground to 1.0 Top)
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center space-x-1.5 text-sky-700 font-medium">
              <span className="w-2.5 h-2.5 bg-sky-400 rounded-sm inline-block"></span>
              <span>Activity (Left)</span>
            </span>
            <span className="flex items-center space-x-1.5 text-red-700 font-medium">
              <span className="w-2.5 h-2.5 bg-red-500 rounded-sm inline-block"></span>
              <span>Position (Right)</span>
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={dualAxisData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="time_sec"
                label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -4, fontSize: 11 }}
                tick={{ fontSize: 10 }}
              />

              {/* Left Y Axis: Activity */}
              <YAxis
                yAxisId="left"
                label={{ value: 'Activity (px/s)', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#0284c7' }}
                tick={{ fontSize: 10, fill: '#0284c7' }}
              />

              {/* Right Y Axis: Vertical Position (0.0 to 1.0) */}
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0, 1]}
                ticks={[0, 0.5, 1]}
                tickFormatter={(v) => (v === 0 ? '0.0 (Btm)' : v === 1 ? '1.0 (Top)' : '0.5')}
                label={{ value: 'Position (au)', angle: 90, position: 'insideRight', fontSize: 11, fill: '#dc2626' }}
                tick={{ fontSize: 10, fill: '#dc2626' }}
              />

              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                labelFormatter={(v) => `Time: ${v}s`}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />

              <Area
                yAxisId="left"
                type="monotone"
                dataKey="activity"
                name="Locomotor Activity (px/s)"
                fill="#00A2E8"
                fillOpacity={0.25}
                stroke="#00A2E8"
                strokeWidth={1.5}
                isAnimationActive={false}
              />

              <Line
                yAxisId="right"
                type="monotone"
                dataKey="norm_pos"
                name="Vertical Height (au)"
                stroke="#E53935"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. Population Normalized Space-Time Kymograph (Hexbin Heatmap) */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-2">
              <Flame className="w-4 h-4 text-amber-500" />
              <span>Population Space-Time Distribution Kymograph (Hexbin Density Map)</span>
            </h3>
            <p className="text-xs text-slate-500">
              Aggregates all 8 chambers into a normalized height map (0.0 Ground to 1.0 Top) over time, showing collective geotaxis or sedation clustering.
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs text-slate-500">
            <span>Grid Resolution: {gridSizeTime} x {gridSizePos} bins</span>
          </div>
        </div>

        {/* Heatmap Visual Canvas */}
        {kymographMatrix && kymographMatrix.matrix ? (
          <div className="space-y-2">
            <div className="flex items-stretch space-x-2">
              {/* Vertical Position Y-Axis Label */}
              <div className="flex flex-col justify-between text-[11px] font-mono text-slate-500 py-1 text-right w-20 select-none">
                <span>1.0 (Top)</span>
                <span>0.75</span>
                <span>0.50 (Mid)</span>
                <span>0.25</span>
                <span>0.0 (Btm)</span>
              </div>

              {/* Heatmap Grid */}
              <div className="flex-1 bg-slate-950 p-2 rounded-lg border border-slate-800 shadow-inner flex flex-col gap-0.5">
                {/* Render from top row (pos=1.0) down to bottom row (pos=0.0) */}
                {Array.from({ length: kymographMatrix.posBins }).map((_, rIdx) => {
                  const row = kymographMatrix.posBins - 1 - rIdx;
                  return (
                    <div key={row} className="flex flex-1 gap-0.5 h-3">
                      {Array.from({ length: kymographMatrix.timeBins }).map((_, col) => {
                        const count = kymographMatrix.matrix[row][col] || 0;
                        const intensity = count > 0 ? Math.min(1.0, Math.log10(count + 1) / Math.log10(kymographMatrix.maxCount + 1)) : 0;
                        
                        // Infernomap color interpolation: from dark purple/black to orange to yellow
                        let bg = '#090d16';
                        if (intensity > 0.8) bg = '#fde047'; // bright yellow
                        else if (intensity > 0.6) bg = '#fb923c'; // orange
                        else if (intensity > 0.35) bg = '#e11d48'; // red
                        else if (intensity > 0.15) bg = '#7e22ce'; // purple
                        else if (intensity > 0) bg = '#312e81'; // dark blue

                        return (
                          <div
                            key={col}
                            className="flex-1 rounded-[1px] transition-colors"
                            style={{ backgroundColor: bg }}
                            title={`Time bin: ${Math.round((col / kymographMatrix.timeBins) * kymographMatrix.totalSec)}s | Pos: ${(row / kymographMatrix.posBins).toFixed(2)} | Count: ${count}`}
                          />
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Time Axis Label & Legend */}
            <div className="flex justify-between items-center text-xs text-slate-500 pl-24 pr-4">
              <span>0s</span>
              <span className="font-semibold text-slate-700">Time (seconds)</span>
              <span>{Math.round(kymographMatrix.totalSec)}s</span>
            </div>

            {/* Color scale bar */}
            <div className="flex items-center justify-end space-x-2 text-[11px] text-slate-500 pt-1 pr-4">
              <span>Low Density</span>
              <div className="w-32 h-2.5 rounded bg-gradient-to-r from-indigo-950 via-purple-700 via-rose-600 to-amber-300 border border-slate-300"></div>
              <span>High Density (Fly Occurrence)</span>
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-400">Loading Kymograph...</p>
        )}
      </div>
    </div>
  );
};
