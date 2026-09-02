import React from 'react';
import { AnesthesiaResult, CleanedKinematicPoint } from '../types';
import {
  Activity,
  Sparkles,
  Clock,
  BarChart2,
  CheckCircle2,
  Layers,
  ArrowRight,
  TrendingUp,
  ShieldCheck
} from 'lucide-react';
import { ActiveTab } from './Navbar';

interface SummaryMetricsCardsProps {
  cleanedPoints: CleanedKinematicPoint[];
  anesthesiaResults: AnesthesiaResult[];
  setActiveTab: (tab: ActiveTab) => void;
}

export const SummaryMetricsCards: React.FC<SummaryMetricsCardsProps> = ({
  cleanedPoints,
  anesthesiaResults,
  setActiveTab
}) => {
  // Summary calculations
  const totalFrames = Math.max(...cleanedPoints.map((p) => p.frame), 0);
  const totalChambers = anesthesiaResults.length || 8;
  const sedatedCount = anesthesiaResults.filter((r) => r.is_sedated).length;
  const meanSpeed = (
    cleanedPoints.reduce((a, b) => a + b.speed, 0) / Math.max(1, cleanedPoints.length)
  ).toFixed(2);

  const validTimes = anesthesiaResults.filter((r) => r.induction_time_sec !== null).map((r) => r.induction_time_sec!);
  const meanInductionTime = validTimes.length > 0 ? (validTimes.reduce((a, b) => a + b, 0) / validTimes.length).toFixed(1) : 'N/A';

  const modules = [
    {
      id: 'calibrator',
      num: 'Module 1',
      title: 'Vision Tracking & Multi-Chamber Auto-Snap',
      desc: 'Adaptive tube alignment, temporal median background subtraction, and Darkness Mass Score centroid extraction.',
      status: 'Calibrated',
      icon: Activity,
      color: 'blue'
    },
    {
      id: 'cleaning',
      num: 'Module 2',
      title: 'Kinematic Cleaning & Artifact Clamping',
      desc: '1%-99% quantile physical bounds, plug shadow occlusion trap relocation, jump rejection, and Savitzky-Golay trajectory smoothing.',
      status: 'Cleaned',
      icon: Sparkles,
      color: 'emerald'
    },
    {
      id: 'anesthesia',
      num: 'Module 3',
      title: 'Stationary Engine & Anesthesia Kinetics',
      desc: 'Sliding window max filter operator (W=120s, 24x 5s bins) to pinpoint sedation onset and knockdown dynamics.',
      status: `${sedatedCount}/${totalChambers} Sedated`,
      icon: Clock,
      color: 'amber'
    },
    {
      id: 'kymograph',
      num: 'Module 4',
      title: 'Dual Y-Axis & Space-Time Kymograph',
      desc: 'Synchronized dual Y-axis dynamic plots (Activity vs Height) and population normalized hexbin density maps.',
      status: 'Rendered',
      icon: BarChart2,
      color: 'indigo'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner KPI Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Total Chambers</span>
            <p className="text-xl font-bold text-slate-800">{totalChambers} Tubes <span className="text-xs font-normal text-slate-400">({totalFrames} frames)</span></p>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Mean Locomotion</span>
            <p className="text-xl font-bold text-emerald-600">{meanSpeed} <span className="text-xs font-normal text-slate-400">px/s</span></p>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Sedation Knockdown</span>
            <p className="text-xl font-bold text-slate-800">
              {sedatedCount} / {totalChambers} <span className="text-xs font-normal text-amber-600 font-semibold">({((sedatedCount / totalChambers) * 100).toFixed(0)}%)</span>
            </p>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Mean Induction Time</span>
            <p className="text-xl font-bold text-indigo-600">{meanInductionTime} {meanInductionTime !== 'N/A' && <span className="text-xs font-normal text-slate-400">sec</span>}</p>
          </div>
        </div>
      </div>

      {/* 4 Modular Step Cards */}
      <div className="space-y-3">
        <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          <span>Integrated 4-Module Functional Flow (v0.1)</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {modules.map((mod) => {
            const Icon = mod.icon;
            return (
              <div
                key={mod.id}
                onClick={() => setActiveTab(mod.id as ActiveTab)}
                className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300 transition cursor-pointer flex flex-col justify-between group"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                      {mod.num}
                    </span>
                    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                      {mod.status}
                    </span>
                  </div>

                  <h4 className="text-sm font-bold text-slate-800 mt-2 flex items-center space-x-1.5 group-hover:text-emerald-600 transition-colors">
                    <Icon className="w-4 h-4 text-slate-600 group-hover:text-emerald-600" />
                    <span>{mod.title}</span>
                  </h4>
                  <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">{mod.desc}</p>
                </div>

                <div className="flex items-center justify-between text-xs font-semibold text-slate-600 pt-4 mt-3 border-t border-slate-100 group-hover:text-emerald-600">
                  <span>Inspect module</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Comprehensive Results Table Across All Chambers */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden space-y-2">
        <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h4 className="text-sm font-bold text-slate-800">
              Consolidated Statistical Summary (*_results_summary.csv)
            </h4>
            <p className="text-xs text-slate-500">
              Integrated quantitative readouts across kinematic cleaning and anesthesia induction kinetics.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-100 text-slate-600 font-bold uppercase text-[10px] tracking-wider border-b border-slate-200">
              <tr>
                <th className="px-4 py-2.5">Chamber</th>
                <th className="px-4 py-2.5">Induction Knockdown (s)</th>
                <th className="px-4 py-2.5">Sedated State</th>
                <th className="px-4 py-2.5">Baseline Locomotion (px/s)</th>
                <th className="px-4 py-2.5">Pre-Sedation Speed (px/s)</th>
                <th className="px-4 py-2.5">Total Stillness Bins</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {anesthesiaResults.map((an) => {
                const stillBins = an.binned_activity.filter((b) => b.is_still).length;

                return (
                  <tr key={an.chamber_id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-2.5 font-bold text-slate-900 font-sans">
                      CH {an.chamber_id}
                    </td>
                    <td className="px-4 py-2.5 text-amber-600 font-bold">
                      {an.induction_time_sec !== null ? `${an.induction_time_sec}s` : 'N/A'}
                    </td>
                    <td className="px-4 py-2.5">
                      {an.is_sedated ? (
                        <span className="inline-block px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 text-[10px] font-sans font-semibold">
                          Sedated
                        </span>
                      ) : (
                        <span className="inline-block px-2 py-0.5 rounded bg-slate-100 text-slate-600 text-[10px] font-sans">
                          Active
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2.5">{an.baseline_speed}</td>
                    <td className="px-4 py-2.5">{an.pre_sedation_activity}</td>
                    <td className="px-4 py-2.5 text-slate-600">{stillBins} bins ({stillBins * 5}s)</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
