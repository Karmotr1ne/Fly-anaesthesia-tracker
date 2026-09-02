import React, { useState, useMemo } from 'react';
import { RawTrackingPoint, CleanedKinematicPoint, PipelineParameters } from '../types';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceArea
} from 'recharts';
import { Sparkles, ShieldAlert, CheckCircle2, Sliders, Info, Zap } from 'lucide-react';

interface KinematicCleaningViewProps {
  rawPoints: RawTrackingPoint[];
  cleanedPoints: CleanedKinematicPoint[];
  params: PipelineParameters;
  setParams: React.Dispatch<React.SetStateAction<PipelineParameters>>;
}

export const KinematicCleaningView: React.FC<KinematicCleaningViewProps> = ({
  rawPoints,
  cleanedPoints,
  params,
  setParams
}) => {
  const [selectedChamber, setSelectedChamber] = useState<number>(0);
  const [showRawOverlay, setShowRawOverlay] = useState<boolean>(true);
  const [timeWindowSec, setTimeWindowSec] = useState<[number, number]>([0, 60]);

  // Filter trajectory for selected chamber
  const chamberCleaned = useMemo(() => {
    return cleanedPoints.filter((p) => p.chamber_id === selectedChamber);
  }, [cleanedPoints, selectedChamber]);

  // Prepare chart dataset
  const chartData = useMemo(() => {
    const minFrame = timeWindowSec[0] * params.fps;
    const maxFrame = timeWindowSec[1] * params.fps;

    return chamberCleaned
      .filter((p) => p.frame >= minFrame && p.frame <= maxFrame)
      .map((p) => ({
        time_sec: Math.round((p.frame / params.fps) * 10) / 10,
        raw_x: p.x_raw,
        clean_x: p.x_clean,
        norm_pos: p.norm_pos !== undefined ? Math.round(p.norm_pos * 100) : null,
        speed: p.speed,
        is_occluded: p.is_occluded,
        is_jump: p.is_jump
      }));
  }, [chamberCleaned, timeWindowSec, params.fps]);

  // Statistics for the selected chamber
  const stats = useMemo(() => {
    const occludedCount = chamberCleaned.filter((p) => p.is_occluded).length;
    const jumpCount = chamberCleaned.filter((p) => p.is_jump).length;
    const validClean = chamberCleaned.filter((p) => p.x_clean !== null).length;
    const validRaw = chamberCleaned.filter((p) => p.x_raw !== null).length;
    const detectionRate = ((validRaw / Math.max(1, chamberCleaned.length)) * 100).toFixed(1);

    return { occludedCount, jumpCount, validClean, detectionRate };
  }, [chamberCleaned]);

  return (
    <div className="space-y-6">
      {/* Overview Header */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-lg font-bold text-slate-800">Module 2: Kinematic Cleaning &amp; Artifact Clamping</h2>
            <span className="bg-emerald-100 text-emerald-800 text-xs px-2.5 py-0.5 rounded-full font-semibold">
              Savitzky-Golay (w={params.savgolWindow}, p={params.savgolPoly})
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Removes plug shadow occlusion traps, suppresses speed spike jumps, and performs polynomial trajectory smoothing with 0.5 body-length thresholding.
          </p>
        </div>

        {/* Chamber switcher */}
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-600">Chamber:</span>
          <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200">
            {Array.from({ length: 8 }).map((_, i) => (
              <button
                key={i}
                onClick={() => setSelectedChamber(i)}
                className={`px-2.5 py-1 text-xs font-bold rounded-md transition ${
                  selectedChamber === i
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                CH{i}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Metric Cards Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-500 text-xs font-medium">
            <ShieldAlert className="w-4 h-4 text-amber-500" />
            <span>Occlusion Traps Clamped</span>
          </div>
          <p className="text-2xl font-bold text-slate-800 mt-1">{stats.occludedCount} <span className="text-xs font-normal text-slate-400">frames</span></p>
          <p className="text-[11px] text-amber-700 mt-0.5">Plug edge relocation</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-500 text-xs font-medium">
            <Zap className="w-4 h-4 text-red-500" />
            <span>Speed Spikes Rejected</span>
          </div>
          <p className="text-2xl font-bold text-slate-800 mt-1">{stats.jumpCount} <span className="text-xs font-normal text-slate-400">frames</span></p>
          <p className="text-[11px] text-red-700 mt-0.5">&gt; {params.maxSpeedPxPerFrame} px/frame cutoff</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-500 text-xs font-medium">
            <Sparkles className="w-4 h-4 text-emerald-500" />
            <span>Smoothed Kinematic Frames</span>
          </div>
          <p className="text-2xl font-bold text-slate-800 mt-1">{stats.validClean} <span className="text-xs font-normal text-slate-400">frames</span></p>
          <p className="text-[11px] text-emerald-700 mt-0.5">Savitzky-Golay quadratic</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-500 text-xs font-medium">
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            <span>Raw Detection Rate</span>
          </div>
          <p className="text-2xl font-bold text-emerald-600 mt-1">{stats.detectionRate}%</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Valid video centroid frames</p>
        </div>
      </div>

      {/* Main Trajectory Comparison Chart */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-800">
              Trajectory Reconstruction: Raw Coordinates vs. Cleaned &amp; Smoothed (CH {selectedChamber})
            </h3>
            <p className="text-xs text-slate-500">
              Gray dashed curve shows noisy raw detection; Vibrant emerald curve shows filtered &amp; Savitzky-Golay smoothed trajectory.
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <label className="flex items-center space-x-1.5 cursor-pointer text-slate-700 font-medium">
              <input
                type="checkbox"
                checked={showRawOverlay}
                onChange={(e) => setShowRawOverlay(e.target.checked)}
                className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
              />
              <span>Overlay Raw Trajectory</span>
            </label>

            <div className="flex items-center space-x-1 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200">
              <span className="text-slate-500">Window:</span>
              <select
                value={`${timeWindowSec[0]}-${timeWindowSec[1]}`}
                onChange={(e) => {
                  const [s, end] = e.target.value.split('-').map(Number);
                  setTimeWindowSec([s, end]);
                }}
                className="bg-transparent font-semibold text-slate-800 focus:outline-none cursor-pointer"
              >
                <option value="0-60">0s - 60s</option>
                <option value="60-120">60s - 120s</option>
                <option value="120-180">120s - 180s</option>
                <option value="0-180">Full Recording (180s)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Recharts Trajectory Plot */}
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="time_sec"
                label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -4, fontSize: 11 }}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                label={{ value: 'X Coordinate (px)', angle: -90, position: 'insideLeft', fontSize: 11 }}
                tick={{ fontSize: 10 }}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                labelFormatter={(v) => `Time: ${v}s`}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />

              {showRawOverlay && (
                <Line
                  type="monotone"
                  dataKey="raw_x"
                  name="Raw X Coordinate (px)"
                  stroke="#94a3b8"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  dot={false}
                  isAnimationActive={false}
                />
              )}

              <Line
                type="monotone"
                dataKey="clean_x"
                name="Cleaned & Smoothed X (px)"
                stroke="#059669"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Parameters & Algorithm Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">1. Boundary &amp; Occlusion</h4>
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs text-slate-700">
                <span>Occlusion Displacement:</span>
                <strong className="font-mono">{params.occlusionDispThresh} px</strong>
              </div>
              <input
                type="range"
                min="40"
                max="140"
                value={params.occlusionDispThresh}
                onChange={(e) => setParams((p) => ({ ...p, occlusionDispThresh: parseInt(e.target.value) }))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
            <div>
              <div className="flex justify-between text-xs text-slate-700">
                <span>Occlusion Std Variance:</span>
                <strong className="font-mono">{params.occlusionVarThresh} px</strong>
              </div>
              <input
                type="range"
                min="2"
                max="20"
                value={params.occlusionVarThresh}
                onChange={(e) => setParams((p) => ({ ...p, occlusionVarThresh: parseInt(e.target.value) }))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">2. Outlier Jump Filter</h4>
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs text-slate-700">
                <span>Max Step Velocity:</span>
                <strong className="font-mono">{params.maxSpeedPxPerFrame} px/frame</strong>
              </div>
              <input
                type="range"
                min="20"
                max="80"
                value={params.maxSpeedPxPerFrame}
                onChange={(e) => setParams((p) => ({ ...p, maxSpeedPxPerFrame: parseInt(e.target.value) }))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
            <div>
              <div className="flex justify-between text-xs text-slate-700">
                <span>Micro-Movement Cutoff:</span>
                <strong className="font-mono">{params.bodyLengthThresh} body len ({(params.bodyLengthThresh * params.bodyLengthPx).toFixed(1)}px)</strong>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={params.bodyLengthThresh}
                onChange={(e) => setParams((p) => ({ ...p, bodyLengthThresh: parseFloat(e.target.value) }))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">3. Savitzky-Golay Smoothing</h4>
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs text-slate-700">
                <span>Window Size:</span>
                <strong className="font-mono">{params.savgolWindow} frames</strong>
              </div>
              <input
                type="range"
                min="3"
                max="15"
                step="2"
                value={params.savgolWindow}
                onChange={(e) => setParams((p) => ({ ...p, savgolWindow: parseInt(e.target.value) }))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
            <div>
              <div className="flex justify-between text-xs text-slate-700">
                <span>Polynomial Order:</span>
                <strong className="font-mono">Order {params.savgolPoly} (Quadratic)</strong>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Preserves sharp turning peaks while eliminating camera pixel jitter.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
