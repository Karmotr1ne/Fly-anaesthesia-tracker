import React, { useState } from 'react';
import { AnesthesiaResult, PipelineParameters } from '../types';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  ReferenceLine
} from 'recharts';
import { Clock, Sliders, CheckCircle2, AlertTriangle, PlayCircle, BarChart3 } from 'lucide-react';

interface AnesthesiaKineticsViewProps {
  anesthesiaResults: AnesthesiaResult[];
  params: PipelineParameters;
  setParams: React.Dispatch<React.SetStateAction<PipelineParameters>>;
}

export const AnesthesiaKineticsView: React.FC<AnesthesiaKineticsViewProps> = ({
  anesthesiaResults,
  params,
  setParams
}) => {
  const [selectedChamber, setSelectedChamber] = useState<number>(0);

  const activeResult = anesthesiaResults.find((r) => r.chamber_id === selectedChamber) || anesthesiaResults[0];

  // Bar chart dataset across all 8 chambers
  const latencyData = anesthesiaResults.map((r) => ({
    chamber: `CH ${r.chamber_id}`,
    id: r.chamber_id,
    induction_sec: r.induction_time_sec ?? 180,
    is_sedated: r.is_sedated,
    baseline_speed: r.baseline_speed,
    pre_speed: r.pre_sedation_activity
  }));

  const sedatedCount = anesthesiaResults.filter((r) => r.is_sedated).length;
  const validTimes = anesthesiaResults.filter((r) => r.induction_time_sec !== null).map((r) => r.induction_time_sec!);
  const meanInductionTime = validTimes.length > 0 ? (validTimes.reduce((a, b) => a + b, 0) / validTimes.length).toFixed(1) : 'N/A';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-lg font-bold text-slate-800">Module 4: Anesthesia &amp; Sedation Kinetics</h2>
            <span className="bg-amber-100 text-amber-800 text-xs px-2.5 py-0.5 rounded-full font-semibold">
              Sliding Window Max Filter (W={params.anesthesiaWindowBins * params.anesthesiaBinSec}s)
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Integrates the Core Stationary Detection Engine to determine the exact induction latency when locomotion
            sustainedly drops below threshold ({params.anesthesiaThreshold} px/s) for {params.anesthesiaWindowBins * params.anesthesiaBinSec} seconds (24 x 5s bins).
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-right">
            <span className="text-[11px] text-slate-400 block uppercase font-bold">Sedation Rate</span>
            <span className="text-lg font-extrabold text-slate-800">
              {sedatedCount} / {anesthesiaResults.length} <span className="text-xs font-normal text-slate-500">({((sedatedCount / Math.max(1, anesthesiaResults.length)) * 100).toFixed(0)}%)</span>
            </span>
          </div>
          <div className="h-8 w-px bg-slate-200"></div>
          <div className="text-right">
            <span className="text-[11px] text-slate-400 block uppercase font-bold">Mean Induction Time</span>
            <span className="text-lg font-extrabold text-amber-600">{meanInductionTime} s</span>
          </div>
        </div>
      </div>

      {/* Latency Comparison Across All Chambers (Bar Chart) */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-1.5">
              <BarChart3 className="w-4 h-4 text-amber-600" />
              <span>Chamber-by-Chamber Anesthesia Induction Latency (Knockdown Time)</span>
            </h3>
            <p className="text-xs text-slate-500">Click any bar to inspect its single-animal 5-second binned velocity timeline.</p>
          </div>
          <span className="text-xs text-slate-400 font-mono">Cutoff Window: W = 120s</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={latencyData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="chamber" tick={{ fontSize: 11 }} />
              <YAxis
                label={{ value: 'Induction Time (seconds)', angle: -90, position: 'insideLeft', fontSize: 11 }}
                tick={{ fontSize: 10 }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                formatter={(val: any, name: any, item: any) => [
                  item.payload.is_sedated ? `${val}s (Sedated)` : 'Not sedated',
                  'Induction Latency'
                ]}
              />
              <Bar
                dataKey="induction_sec"
                radius={[4, 4, 0, 0]}
                onClick={(entry) => setSelectedChamber(entry.id)}
                className="cursor-pointer"
              >
                {latencyData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.id === selectedChamber ? '#f59e0b' : entry.is_sedated ? '#3b82f6' : '#cbd5e1'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Selected Chamber Detail: 5-Second Binned Timeline & Stationary Mask */}
      {activeResult && (
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div className="flex items-center space-x-3">
              <span className="w-8 h-8 rounded-lg bg-amber-500 text-white font-bold flex items-center justify-center text-sm shadow-sm">
                CH{activeResult.chamber_id}
              </span>
              <div>
                <h4 className="text-sm font-bold text-slate-800">
                  Individual Knockdown Timeline: 5s Binned Activity &amp; Inactivity Window
                </h4>
                <p className="text-xs text-slate-500">
                  Blue line = 5s binned average speed (px/s); Red dashed line = Validated Sedation Onset Time Point.
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3 text-xs">
              <div className="bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
                <span className="text-slate-500">Induction Onset: </span>
                <strong className="text-amber-600 font-mono">
                  {activeResult.induction_time_sec !== null ? `${activeResult.induction_time_sec}s` : 'Did not reach complete shutdown'}
                </strong>
              </div>
              <div className="bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
                <span className="text-slate-500">Baseline Speed: </span>
                <strong className="text-slate-800 font-mono">{activeResult.baseline_speed} px/s</strong>
              </div>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={activeResult.binned_activity} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="time_sec"
                  label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -4, fontSize: 11 }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  label={{ value: 'Mean Speed (px/s)', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  labelFormatter={(v) => `Time: ${v}s`}
                />

                {activeResult.induction_time_sec !== null && (
                  <ReferenceLine
                    x={activeResult.induction_time_sec}
                    stroke="#ef4444"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    label={{ value: `Onset ${activeResult.induction_time_sec}s`, fill: '#ef4444', fontSize: 11, position: 'top' }}
                  />
                )}

                <Line
                  type="monotone"
                  dataKey="speed"
                  name="5s Mean Speed"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Kinetics Control Parameters */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex justify-between text-xs text-slate-700">
            <span>Temporal Bin Size:</span>
            <strong className="font-mono">{params.anesthesiaBinSec} seconds</strong>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            value={params.anesthesiaBinSec}
            onChange={(e) => setParams((p) => ({ ...p, anesthesiaBinSec: parseInt(e.target.value) }))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-[11px] text-slate-400">Standard 5s bins smooth short locomotor pauses.</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex justify-between text-xs text-slate-700">
            <span>Inactivity Window (W):</span>
            <strong className="font-mono">{params.anesthesiaWindowBins * params.anesthesiaBinSec}s ({params.anesthesiaWindowBins} bins)</strong>
          </div>
          <input
            type="range"
            min="6"
            max="48"
            value={params.anesthesiaWindowBins}
            onChange={(e) => setParams((p) => ({ ...p, anesthesiaWindowBins: parseInt(e.target.value) }))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-[11px] text-slate-400">Literature standard 120-second continuous stillness window.</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex justify-between text-xs text-slate-700">
            <span>Stillness Speed Cutoff:</span>
            <strong className="font-mono">{params.anesthesiaThreshold} px/s</strong>
          </div>
          <input
            type="range"
            min="0.005"
            max="0.1"
            step="0.005"
            value={params.anesthesiaThreshold}
            onChange={(e) => setParams((p) => ({ ...p, anesthesiaThreshold: parseFloat(e.target.value) }))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-[11px] text-slate-400">Threshold for total physiological behavioral cessation.</p>
        </div>
      </div>
    </div>
  );
};
