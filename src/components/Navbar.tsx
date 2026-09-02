import React from 'react';
import {
  Activity,
  Layers,
  Sparkles,
  Clock,
  Zap,
  BarChart2,
  FileCode,
  Download,
  RotateCcw
} from 'lucide-react';

export type ActiveTab =
  | 'workbench'
  | 'calibrator'
  | 'cleaning'
  | 'anesthesia'
  | 'kymograph'
  | 'python_hub';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  preset: 'anesthesia' | 'recovery_assay' | 'noisy_climbing';
  setPreset: (preset: 'anesthesia' | 'recovery_assay' | 'noisy_climbing') => void;
  onReset: () => void;
  onExportCSV: () => void;
  onOpenImportModal?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  preset,
  setPreset,
  onReset,
  onExportCSV,
  onOpenImportModal
}) => {
  const tabs = [
    { id: 'workbench', label: 'Executive Dashboard', icon: Layers, badge: 'Overview' },
    { id: 'calibrator', label: '1. Multi-Chamber Calibrator', icon: Activity, badge: 'Auto-Snap' },
    { id: 'cleaning', label: '2. Kinematic Cleaning', icon: Sparkles, badge: 'Filter' },
    { id: 'anesthesia', label: '3. Anesthesia Kinetics', icon: Clock, badge: 'W=120s' },
    { id: 'kymograph', label: '4. Dual-Axis & Kymograph', icon: BarChart2, badge: 'Graphics' },
    { id: 'python_hub', label: 'Python Suite & Release', icon: FileCode, badge: 'v0.1' }
  ];

  return (
    <header className="bg-slate-900 text-slate-100 border-b border-slate-800 sticky top-0 z-40 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and System Title */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-600 flex items-center justify-center text-white shadow-md">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-white">Drosophila Anesthesia Tracker</span>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  v0.1 Release
                </span>
              </div>
              <p className="text-xs text-slate-400">High-Throughput Anesthesia Kinetics &amp; Locomotor Analysis</p>
            </div>
          </div>

          {/* Dataset preset selector & Global Actions */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-slate-800 px-3 py-1.5 rounded-md border border-slate-700 text-xs">
              <span className="text-slate-400 font-medium">Assay:</span>
              <select
                id="dataset-preset-select"
                value={preset}
                onChange={(e) => setPreset(e.target.value as any)}
                className="bg-transparent text-slate-100 font-semibold focus:outline-none cursor-pointer"
              >
                <option value="anesthesia" className="bg-slate-800 text-white">
                  Volatile Anesthesia Induction (Isoflurane / CO2)
                </option>
                <option value="recovery_assay" className="bg-slate-800 text-white">
                  Post-Anesthetic Washout Recovery
                </option>
                <option value="noisy_climbing" className="bg-slate-800 text-white">
                  Locomotion Benchmark with Occlusion Traps
                </option>
              </select>
            </div>

            {onOpenImportModal && (
              <button
                id="btn-nav-import-video"
                onClick={onOpenImportModal}
                className="flex items-center space-x-1 px-3 py-1.5 rounded-md bg-blue-600/80 hover:bg-blue-500 text-white text-xs font-medium shadow-sm transition cursor-pointer"
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Import Video</span>
              </button>
            )}

            <button
              id="btn-regenerate-simulation"
              onClick={onReset}
              title="Regenerate random simulation data"
              className="flex items-center space-x-1 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Simulate</span>
            </button>

            <button
              id="btn-export-all-csv"
              onClick={onExportCSV}
              className="flex items-center space-x-1 px-3.5 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 overflow-x-auto pb-2 scrollbar-none">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id as ActiveTab)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-md text-xs font-medium transition-all whitespace-nowrap cursor-pointer ${
                  isActive
                    ? 'bg-slate-800 text-white border-b-2 border-emerald-400 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
                {tab.badge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                      isActive ? 'bg-emerald-950 text-emerald-300' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
