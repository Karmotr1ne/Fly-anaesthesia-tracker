import React, { useState, useMemo, useEffect } from 'react';
import { Navbar, ActiveTab } from './components/Navbar';
import { ChamberCalibratorView } from './components/ChamberCalibratorView';
import { KinematicCleaningView } from './components/KinematicCleaningView';
import { AnesthesiaKineticsView } from './components/AnesthesiaKineticsView';
import { DualAxisKymographView } from './components/DualAxisKymographView';
import { PythonPackageHub } from './components/PythonPackageHub';
import { SummaryMetricsCards } from './components/SummaryMetricsCards';
import { VideoImportModal } from './components/VideoImportModal';
import { generateSimulationDataset } from './utils/simulation';
import {
  cleanRawTrajectory,
  evaluateAnesthesiaKinetics,
  DEFAULT_PIPELINE_PARAMS
} from './utils/kinematics';
import { PipelineParameters, ChamberBox } from './types';
import confetti from 'canvas-confetti';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workbench');
  const [preset, setPreset] = useState<'anesthesia' | 'recovery_assay' | 'noisy_climbing'>('anesthesia');
  const [params, setParams] = useState<PipelineParameters>(DEFAULT_PIPELINE_PARAMS);
  const [isImportModalOpen, setIsImportModalOpen] = useState<boolean>(false);

  const [activeChambers, setActiveChambers] = useState<ChamberBox[]>([
    { id: 1, x1: 50, y1: 45, x2: 410, y2: 120 },
    { id: 2, x1: 52, y1: 155, x2: 412, y2: 230 },
    { id: 3, x1: 48, y1: 265, x2: 408, y2: 340 },
    { id: 4, x1: 50, y1: 375, x2: 410, y2: 450 },
    { id: 5, x1: 460, y1: 42, x2: 820, y2: 118 },
    { id: 6, x1: 458, y1: 152, x2: 818, y2: 228 },
    { id: 7, x1: 462, y1: 262, x2: 822, y2: 338 },
    { id: 8, x1: 460, y1: 372, x2: 820, y2: 448 }
  ]);

  // Simulation Raw Dataset State
  const [simulationData, setSimulationData] = useState(() => {
    return generateSimulationDataset('anesthesia', 8, 180, 30);
  });

  // Re-generate dataset whenever preset changes
  useEffect(() => {
    const data = generateSimulationDataset(preset, 8, 180, params.fps);
    setSimulationData(data);
  }, [preset, params.fps]);

  const handleReset = () => {
    const data = generateSimulationDataset(preset, 8, 180, params.fps);
    setSimulationData(data);
  };

  const handleSessionImport = (sessionData: {
    videoName: string;
    fps: number;
    totalFrames: number;
    chambers: ChamberBox[];
    diffThresh: number;
  }) => {
    setActiveChambers(sessionData.chambers);
    setParams((p) => ({ ...p, fps: sessionData.fps }));
    const data = generateSimulationDataset(preset, 8, Math.floor(sessionData.totalFrames / sessionData.fps), sessionData.fps);
    setSimulationData(data);
  };

  // Pipeline Execution: Module 2 (Cleaning) -> Module 3 (Stationary) -> Module 4 (Anesthesia Kinetics)
  const cleanedPoints = useMemo(() => {
    return cleanRawTrajectory(simulationData.points, params);
  }, [simulationData.points, params]);

  const anesthesiaResults = useMemo(() => {
    return evaluateAnesthesiaKinetics(cleanedPoints, params);
  }, [cleanedPoints, params]);

  // CSV Exporter
  const handleExportCSV = () => {
    try {
      // Build summary CSV
      const headers = [
        'chamber_id',
        'induction_time_sec',
        'is_sedated',
        'baseline_speed',
        'pre_sedation_activity',
        'stillness_bins_count'
      ];

      const rows = anesthesiaResults.map((an) => {
        const stillBins = an.binned_activity.filter((b) => b.is_still).length;
        return [
          an.chamber_id,
          an.induction_time_sec ?? '',
          an.is_sedated ? 1 : 0,
          an.baseline_speed,
          an.pre_sedation_activity,
          stillBins
        ].join(',');
      });

      const csvContent = [headers.join(','), ...rows].join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `Drosophila_${preset}_results_summary.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      confetti({ particleCount: 60, spread: 60, origin: { y: 0.8 } });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex flex-col font-sans antialiased">
      {/* Top Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        preset={preset}
        setPreset={setPreset}
        onReset={handleReset}
        onExportCSV={handleExportCSV}
        onOpenImportModal={() => setIsImportModalOpen(true)}
      />

      <VideoImportModal
        isOpen={isImportModalOpen}
        onClose={() => setIsImportModalOpen(false)}
        onConfirm={handleSessionImport}
        initialChambers={activeChambers}
      />

      {/* Main View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === 'workbench' && (
          <SummaryMetricsCards
            cleanedPoints={cleanedPoints}
            anesthesiaResults={anesthesiaResults}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'calibrator' && (
          <ChamberCalibratorView
            params={params}
            setParams={setParams}
            onImportSession={handleSessionImport}
          />
        )}

        {activeTab === 'cleaning' && (
          <KinematicCleaningView
            rawPoints={simulationData.points}
            cleanedPoints={cleanedPoints}
            params={params}
            setParams={setParams}
          />
        )}

        {activeTab === 'anesthesia' && (
          <AnesthesiaKineticsView
            anesthesiaResults={anesthesiaResults}
            params={params}
            setParams={setParams}
          />
        )}

        {activeTab === 'kymograph' && (
          <DualAxisKymographView cleanedPoints={cleanedPoints} params={params} />
        )}

        {activeTab === 'python_hub' && <PythonPackageHub />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 border-t border-slate-800 py-4 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            <strong>Drosophila Anesthesia Tracker</strong> &bull; v0.1 Release
          </span>
          <span className="text-slate-500 font-mono">
            PyQt6 Async ThreadPool / OpenCV / Scipy / Matplotlib / PyInstaller Standalone Release
          </span>
        </div>
      </footer>
    </div>
  );
}
