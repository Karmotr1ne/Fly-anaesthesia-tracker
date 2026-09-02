import React, { useState, useRef } from 'react';
import {
  Upload,
  Sparkles,
  Sliders,
  CheckCircle2,
  X,
  Play,
  Pause,
  RefreshCw,
  Video,
  FileSpreadsheet,
  Layers,
  Crosshair,
  Eye,
  Info,
  Grid
} from 'lucide-react';
import { ChamberBox, PipelineParameters } from '../types';
import { generateSymmetricChambers } from '../utils/simulation';

interface VideoImportModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (sessionData: {
    videoName: string;
    fps: number;
    totalFrames: number;
    chambers: ChamberBox[];
    diffThresh: number;
    rows: number;
    cols: number;
    order: 'column_first' | 'row_first';
  }) => void;
  initialChambers: ChamberBox[];
}

export const VideoImportModal: React.FC<VideoImportModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  initialChambers
}) => {
  const [selectedFile, setSelectedFile] = useState<string>('sample_isoflurane_trial_8ch.mp4');
  const [fps, setFps] = useState<number>(30);
  const [totalFrames, setTotalFrames] = useState<number>(5400); // 3 minutes at 30 fps
  const [currentFrame, setCurrentFrame] = useState<number>(45);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [diffThresh, setDiffThresh] = useState<number>(14);
  const [showMask, setShowMask] = useState<boolean>(false);
  const [selectedChamber, setSelectedChamber] = useState<number>(1);
  const [linkMode, setLinkMode] = useState<'single' | 'col' | 'row' | 'all'>('all');
  const [rows, setRows] = useState<number>(4);
  const [cols, setCols] = useState<number>(2);
  const [order, setOrder] = useState<'column_first' | 'row_first'>('column_first');
  const [chambers, setChambers] = useState<ChamberBox[]>(
    initialChambers.length > 0 ? initialChambers : generateSymmetricChambers(880, 500, [45, 40, 420, 115], 4, 2, 5, 'column_first')
  );
  const [autoSnapped, setAutoSnapped] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file.name);
      setFps(30);
      setTotalFrames(Math.max(1800, Math.min(10800, Math.floor(file.size / 20000))));
      setCurrentFrame(0);
      handleRegenerateGrid();
    }
  };

  const handleRegenerateGrid = () => {
    const newChambers = generateSymmetricChambers(
      880,
      500,
      [45, 40, 45 + Math.floor(750 / cols), 40 + Math.floor(400 / rows)],
      rows,
      cols,
      5,
      order
    );
    setChambers(newChambers);
    setSelectedChamber(1);
    setAutoSnapped(false);
  };

  const handleAutoSnap = () => {
    setChambers((prev) =>
      prev.map((c, i) => {
        const offset = Math.sin(i * 1.2) * 3;
        return {
          ...c,
          y1: Math.round(c.y1 + offset),
          y2: Math.round(c.y2 + offset)
        };
      })
    );
    setAutoSnapped(true);
  };

  const handleAdjust = (dx: number, dy: number, dw: number, dh: number) => {
    const selIdx = chambers.findIndex((c) => c.id === selectedChamber);
    const selRow = selIdx >= 0 ? (order === 'column_first' ? selIdx % rows : Math.floor(selIdx / cols)) : 0;
    const selCol = selIdx >= 0 ? (order === 'column_first' ? Math.floor(selIdx / rows) : selIdx % cols) : 0;

    setChambers((prev) =>
      prev.map((c, idx) => {
        const curRow = order === 'column_first' ? idx % rows : Math.floor(idx / cols);
        const curCol = order === 'column_first' ? Math.floor(idx / rows) : idx % cols;

        let match = false;
        if (linkMode === 'all') match = true;
        else if (linkMode === 'col' && curCol === selCol) match = true;
        else if (linkMode === 'row' && curRow === selRow) match = true;
        else if (linkMode === 'single' && c.id === selectedChamber) match = true;

        if (match) {
          return {
            ...c,
            x1: Math.max(10, c.x1 + dx),
            y1: Math.max(10, c.y1 + dy),
            x2: Math.min(860, c.x2 + dx + dw),
            y2: Math.min(480, c.y2 + dy + dh)
          };
        }
        return c;
      })
    );
  };

  const handleSaveAndStartTracking = () => {
    onConfirm({
      videoName: selectedFile,
      fps,
      totalFrames,
      chambers,
      diffThresh,
      rows,
      cols,
      order
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-5xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/60">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600/20 text-blue-400 border border-blue-500/30 flex items-center justify-center">
              <Video className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center space-x-2">
                <span>Multi-Chamber Video Import & Grid Calibration</span>
                <span className="bg-blue-500/20 text-blue-300 text-[11px] font-mono px-2 py-0.5 rounded-full border border-blue-500/30">
                  {rows * cols} Chambers ({rows}x{cols})
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Configure arbitrary symmetric grid dimensions (1..N) and calibrate glass tube bounding boxes
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Video View & Interactive Chamber Overlay */}
          <div className="lg:col-span-2 space-y-4">
            {/* Top Toolbar / Video Source Selection */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950 p-3 rounded-xl border border-slate-800">
              <div className="flex items-center space-x-2">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept=".mp4,.avi,.mov,.mkv,.csv"
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center space-x-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold transition"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload Video</span>
                </button>

                <select
                  value={selectedFile}
                  onChange={(e) => setSelectedFile(e.target.value)}
                  className="bg-slate-900 text-slate-200 border border-slate-700 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-blue-500"
                >
                  <option value="sample_isoflurane_trial_8ch.mp4">Sample: Isoflurane Knockdown (8-Chamber MP4)</option>
                  <option value="sample_sleep_circadian_24h.mp4">Sample: Circadian Sleep 24h Assay</option>
                  <option value="sample_negative_geotaxis_climbing.mp4">Sample: Negative Geotaxis Climbing</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={handleAutoSnap}
                  className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-bold shadow transition"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Auto-Snap</span>
                </button>
                <button
                  onClick={() => setShowMask(!showMask)}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                    showMask
                      ? 'bg-purple-900/40 border-purple-500 text-purple-300'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>{showMask ? 'Mask View' : 'Raw Stream'}</span>
                </button>
              </div>
            </div>

            {/* Chamber Video Canvas */}
            <div className="relative bg-slate-950 p-2 rounded-xl border border-slate-800 shadow-inner flex flex-col items-center">
              <svg
                viewBox="0 0 880 500"
                className="w-full max-h-[380px] bg-slate-900 rounded-lg border border-slate-800 select-none shadow-md"
              >
                <defs>
                  <pattern id="modalGrid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255, 255, 255, 0.04)" strokeWidth="1" />
                  </pattern>
                  <linearGradient id="modalTubeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#1e293b" stopOpacity="0.85" />
                    <stop offset="50%" stopColor="#334155" stopOpacity="0.45" />
                    <stop offset="100%" stopColor="#1e293b" stopOpacity="0.85" />
                  </linearGradient>
                </defs>

                <rect width="880" height="500" fill="url(#modalGrid)" />

                {/* Center Divider Groove */}
                {cols > 1 && (
                  <>
                    <line x1="435" y1="20" x2="435" y2="480" stroke="#475569" strokeWidth="2" strokeDasharray="4 4" />
                    <text x="435" y="16" fill="#64748b" fontSize="10" textAnchor="middle" fontFamily="monospace">
                      CENTER GROOVE
                    </text>
                  </>
                )}

                {/* N Chamber Tubes */}
                {chambers.map((c) => {
                  const isSelected = c.id === selectedChamber;
                  const width = c.x2 - c.x1;
                  const height = c.y2 - c.y1;

                  // Simulated fly coordinate
                  const flyProgress = 0.2 + ((c.id * 17 + currentFrame * 0.4) % 100) / 130;
                  const flyX = c.x1 + width * flyProgress;
                  const flyY = c.y1 + height * 0.5;

                  return (
                    <g
                      key={c.id}
                      onClick={() => setSelectedChamber(c.id)}
                      className="cursor-pointer transition-all"
                    >
                      {/* Tube outline */}
                      <rect
                        x={c.x1}
                        y={c.y1}
                        width={width}
                        height={height}
                        rx="5"
                        fill={showMask ? '#090d16' : 'url(#modalTubeGrad)'}
                        stroke={isSelected ? '#f59e0b' : '#10b981'}
                        strokeWidth={isSelected ? '2.5' : '1.5'}
                      />

                      {/* Plug / Mesh end caps */}
                      <line
                        x1={c.x1 + width * 0.08}
                        y1={c.y1}
                        x2={c.x1 + width * 0.08}
                        y2={c.y2}
                        stroke="#f97316"
                        strokeWidth="1"
                        strokeDasharray="2 2"
                      />
                      <line
                        x1={c.x2 - width * 0.08}
                        y1={c.y1}
                        x2={c.x2 - width * 0.08}
                        y2={c.y2}
                        stroke="#38bdf8"
                        strokeWidth="1"
                        strokeDasharray="2 2"
                      />

                      {/* 1-based Label */}
                      <text
                        x={c.x1 + 6}
                        y={c.y1 + 14}
                        fill={isSelected ? '#fbbf24' : '#94a3b8'}
                        fontSize="10"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        CH {c.id} {isSelected ? '★' : ''}
                      </text>

                      {/* Detected Centroid */}
                      {showMask ? (
                        <ellipse
                          cx={flyX}
                          cy={flyY}
                          rx="6"
                          ry="3.5"
                          fill="#ec4899"
                          opacity="0.9"
                          filter="blur(1px)"
                        />
                      ) : (
                        <>
                          <circle cx={flyX} cy={flyY} r="3" fill="#ef4444" />
                          <circle cx={flyX} cy={flyY} r="7" fill="none" stroke="#10b981" strokeWidth="1" />
                          <line x1={flyX - 5} y1={flyY} x2={flyX + 5} y2={flyY} stroke="#ef4444" strokeWidth="0.8" />
                          <line x1={flyX} y1={flyY - 5} x2={flyX} y2={flyY + 5} stroke="#ef4444" strokeWidth="0.8" />
                        </>
                      )}
                    </g>
                  );
                })}
              </svg>

              {/* Video Playback & Frame Scrubber */}
              <div className="w-full mt-3 px-2 flex items-center space-x-3 text-xs text-slate-300">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded-md border border-slate-700 text-white"
                >
                  {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                </button>

                <span className="font-mono text-[11px] text-slate-400">
                  Frame: {currentFrame} / {totalFrames}
                </span>

                <input
                  type="range"
                  min="0"
                  max={totalFrames}
                  value={currentFrame}
                  onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                  className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />

                <span className="font-mono text-[11px] text-emerald-400">
                  {(currentFrame / fps).toFixed(1)}s
                </span>
              </div>
            </div>
          </div>

          {/* Right 1 Col: Chamber Adjustment & Vision Parameter Panel */}
          <div className="space-y-4">
            {/* Grid Dimension Configuration */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <h3 className="text-xs font-bold text-white flex items-center space-x-1.5">
                <Grid className="w-4 h-4 text-blue-400" />
                <span>Grid Dimensions (Symmetric Align)</span>
              </h3>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] text-slate-400">Rows (1-32):</label>
                  <input
                    type="number"
                    min="1"
                    max="32"
                    value={rows}
                    onChange={(e) => setRows(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded px-2 py-1 mt-0.5"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-400">Cols (1-16):</label>
                  <input
                    type="number"
                    min="1"
                    max="16"
                    value={cols}
                    onChange={(e) => setCols(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded px-2 py-1 mt-0.5"
                  />
                </div>
              </div>

              <div>
                <label className="text-[10px] text-slate-400">Numbering Order:</label>
                <select
                  value={order}
                  onChange={(e) => setOrder(e.target.value as any)}
                  className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded px-2 py-1 mt-0.5"
                >
                  <option value="column_first">Column-first (1..N down columns)</option>
                  <option value="row_first">Row-first (1..N across rows)</option>
                </select>
              </div>

              <button
                onClick={handleRegenerateGrid}
                className="w-full py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition flex items-center justify-center space-x-1"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Regenerate Symmetric Grid</span>
              </button>
            </div>

            {/* Box Calibration Controls */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-white flex items-center space-x-1.5">
                  <Crosshair className="w-4 h-4 text-blue-400" />
                  <span>ROI Nudge & Selection</span>
                </h3>
                {autoSnapped && (
                  <span className="text-[10px] text-emerald-400 font-semibold flex items-center space-x-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Snapped</span>
                  </span>
                )}
              </div>

              {/* Link Mode */}
              <div>
                <label className="text-[11px] font-semibold text-slate-400">Link Mode:</label>
                <div className="grid grid-cols-2 gap-1 mt-1">
                  {[
                    { id: 'single', label: 'Single Tube' },
                    { id: 'col', label: 'Active Column' },
                    { id: 'row', label: 'Active Row' },
                    { id: 'all', label: `All ${chambers.length} Tubes` }
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setLinkMode(m.id as any)}
                      className={`px-2 py-1 rounded text-[11px] font-medium border ${
                        linkMode === m.id
                          ? 'bg-blue-600/30 border-blue-500 text-blue-300 font-bold'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Active Chamber */}
              <div>
                <label className="text-[11px] font-semibold text-slate-400">Active Chamber (1..N):</label>
                <div className="flex flex-wrap gap-1 mt-1 max-h-24 overflow-y-auto p-1 bg-slate-900 rounded border border-slate-800">
                  {chambers.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setSelectedChamber(c.id)}
                      className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${
                        selectedChamber === c.id
                          ? 'bg-amber-500 text-slate-950 border-amber-400'
                          : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                      }`}
                    >
                      {c.id}
                    </button>
                  ))}
                </div>
              </div>

              {/* D-Pad Micro Adjust */}
              <div>
                <label className="text-[11px] font-semibold text-slate-400">Nudge Coordinates (2px):</label>
                <div className="grid grid-cols-3 gap-1 mt-1 max-w-[150px] mx-auto text-center">
                  <div></div>
                  <button
                    onClick={() => handleAdjust(0, -2, 0, 0)}
                    className="px-2 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-200 font-bold"
                  >
                    ▲
                  </button>
                  <div></div>

                  <button
                    onClick={() => handleAdjust(-2, 0, 0, 0)}
                    className="px-2 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-200 font-bold"
                  >
                    ◀
                  </button>
                  <div className="flex items-center justify-center text-[10px] text-slate-500 font-mono">2px</div>
                  <button
                    onClick={() => handleAdjust(2, 0, 0, 0)}
                    className="px-2 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-200 font-bold"
                  >
                    ▶
                  </button>

                  <div></div>
                  <button
                    onClick={() => handleAdjust(0, 2, 0, 0)}
                    className="px-2 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-200 font-bold"
                  >
                    ▼
                  </button>
                  <div></div>
                </div>
              </div>

              {/* Expansion & Shrink */}
              <div className="flex items-center justify-between pt-2 border-t border-slate-800">
                <button
                  onClick={() => handleAdjust(-2, -1, 4, 2)}
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 rounded"
                >
                  Expand (+)
                </button>
                <button
                  onClick={() => handleAdjust(2, 1, -4, -2)}
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 rounded"
                >
                  Shrink (-)
                </button>
              </div>
            </div>

            {/* Vision Thresholds */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <h3 className="text-xs font-bold text-white flex items-center space-x-1.5">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <span>Detection Parameters</span>
              </h3>

              <div>
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Darkness Diff Threshold:</span>
                  <strong className="font-mono text-emerald-400">{diffThresh} px</strong>
                </div>
                <input
                  type="range"
                  min="5"
                  max="35"
                  value={diffThresh}
                  onChange={(e) => setDiffThresh(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer mt-1 accent-emerald-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Video Framerate (FPS):</span>
                  <strong className="font-mono text-blue-400">{fps} Hz</strong>
                </div>
                <input
                  type="number"
                  min="1"
                  max="120"
                  value={fps}
                  onChange={(e) => setFps(Math.max(1, parseInt(e.target.value) || 30))}
                  className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded px-2.5 py-1 mt-1 font-mono focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Info className="w-4 h-4 text-blue-400" />
            <span>Calibrated {chambers.length} ROIs are automatically synced across downstream analytics modules.</span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition"
            >
              Cancel
            </button>
            <button
              onClick={handleSaveAndStartTracking}
              className="flex items-center space-x-1.5 px-5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-md transition"
            >
              <CheckCircle2 className="w-4 h-4" />
              <span>Confirm & Initialize Pipeline</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
