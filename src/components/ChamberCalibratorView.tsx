import React, { useState, useRef, useEffect } from 'react';
import { ChamberBox, PipelineParameters } from '../types';
import { Eye, Crosshair, Sparkles, Sliders, CheckCircle2, RefreshCw, ZoomIn, Video, UploadCloud, Info, Grid, Move, Maximize2 } from 'lucide-react';
import { VideoImportModal } from './VideoImportModal';
import { generateSymmetricChambers } from '../utils/simulation';

interface ChamberCalibratorViewProps {
  params: PipelineParameters;
  setParams: React.Dispatch<React.SetStateAction<PipelineParameters>>;
  onImportSession?: (session: any) => void;
}

export const ChamberCalibratorView: React.FC<ChamberCalibratorViewProps> = ({
  params,
  setParams,
  onImportSession
}) => {
  const [selectedChamber, setSelectedChamber] = useState<number>(1);
  const [mode, setMode] = useState<'single' | 'col' | 'row' | 'all'>('all');
  const [autoSnapped, setAutoSnapped] = useState<boolean>(true);
  const [showMask, setShowMask] = useState<boolean>(false);
  const [diffThresh, setDiffThresh] = useState<number>(14);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [videoSource, setVideoSource] = useState<string>('sample_8ch_arena_calib.mp4');
  const [rows, setRows] = useState<number>(4);
  const [cols, setCols] = useState<number>(2);
  const [order, setOrder] = useState<'column_first' | 'row_first'>('column_first');

  // Multi-chamber coordinate boxes
  const [chambers, setChambers] = useState<ChamberBox[]>(() =>
    generateSymmetricChambers(880, 500, [30, 45, 415, 120], 4, 2, 5, 'column_first')
  );

  // Direct Mouse Drag & Resize Handles State
  const [dragAction, setDragAction] = useState<string | null>(null);
  const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null);
  const [dragInitialChambers, setDragInitialChambers] = useState<ChamberBox[]>([]);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const getSvgCoordinates = (e: React.MouseEvent<SVGSVGElement>): { x: number; y: number } => {
    if (!svgRef.current) return { x: 0, y: 0 };
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = 880 / rect.width;
    const scaleY = 500 / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const handleRegenerateGrid = () => {
    const marginX = 25;
    const marginY = 40;
    const availW = 880 - 2 * marginX;
    const availH = 500 - 2 * marginY;
    const centerGap = cols > 1 ? 35 : 0;
    const colW = Math.floor((availW - (cols - 1) * centerGap) / cols);
    const rowH = Math.floor((availH / rows) * 0.72);

    const newChambers = generateSymmetricChambers(
      880,
      500,
      [marginX, marginY, marginX + colW, marginY + rowH],
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
      prev.map((c, idx) => {
        const curCol = order === 'column_first' ? Math.floor(idx / rows) : idx % cols;
        // Accurate auto-snap based on physical structure:
        // Expand X to encompass wire mesh (left side) and circular white holes (right side)
        let newX1 = c.x1;
        let newX2 = c.x2;
        if (curCol === 0) {
          // Left column: left side is mesh (x ~ 25), right side is white hole (x ~ 415)
          newX1 = 28;
          newX2 = 418;
        } else {
          // Right column: left side is mesh (x ~ 475), right side is white hole (x ~ 855)
          newX1 = 478;
          newX2 = 858;
        }
        return {
          ...c,
          x1: newX1,
          x2: newX2,
          y1: Math.round(c.y1 + (Math.random() - 0.5) * 2),
          y2: Math.round(c.y2 + (Math.random() - 0.5) * 2)
        };
      })
    );
    setAutoSnapped(true);
  };

  // Drag Handlers
  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>, action: string, chId: number) => {
    e.preventDefault();
    e.stopPropagation();
    setSelectedChamber(chId);
    const pos = getSvgCoordinates(e);
    setDragAction(action);
    setDragStartPos(pos);
    setDragInitialChambers(JSON.parse(JSON.stringify(chambers)));
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!dragAction || !dragStartPos) return;
    const pos = getSvgCoordinates(e);
    const dx = pos.x - dragStartPos.x;
    const dy = pos.y - dragStartPos.y;

    const selIdx = dragInitialChambers.findIndex((c) => c.id === selectedChamber);
    const selRow = selIdx >= 0 ? (order === 'column_first' ? selIdx % rows : Math.floor(selIdx / cols)) : 0;
    const selCol = selIdx >= 0 ? (order === 'column_first' ? Math.floor(selIdx / rows) : selIdx % cols) : 0;

    setChambers(
      dragInitialChambers.map((c, idx) => {
        const curRow = order === 'column_first' ? idx % rows : Math.floor(idx / cols);
        const curCol = order === 'column_first' ? Math.floor(idx / rows) : idx % cols;

        let shouldApply = false;
        if (mode === 'all') shouldApply = true;
        else if (mode === 'col' && curCol === selCol) shouldApply = true;
        else if (mode === 'row' && curRow === selRow) shouldApply = true;
        else if (mode === 'single' && c.id === selectedChamber) shouldApply = true;

        if (!shouldApply) return c;

        let nx1 = c.x1;
        let ny1 = c.y1;
        let nx2 = c.x2;
        let ny2 = c.y2;

        if (dragAction === 'move') {
          nx1 += dx;
          ny1 += dy;
          nx2 += dx;
          ny2 += dy;
        } else if (dragAction === 'resize_l' || dragAction === 'resize_tl' || dragAction === 'resize_bl') {
          nx1 = Math.min(c.x2 - 25, c.x1 + dx);
        }
        if (dragAction === 'resize_r' || dragAction === 'resize_tr' || dragAction === 'resize_br') {
          nx2 = Math.max(c.x1 + 25, c.x2 + dx);
        }
        if (dragAction === 'resize_t' || dragAction === 'resize_tl' || dragAction === 'resize_tr') {
          ny1 = Math.min(c.y2 - 15, c.y1 + dy);
        }
        if (dragAction === 'resize_b' || dragAction === 'resize_bl' || dragAction === 'resize_br') {
          ny2 = Math.max(c.y1 + 15, c.y2 + dy);
        }

        return {
          id: c.id,
          x1: Math.round(Math.max(5, Math.min(870, nx1))),
          y1: Math.round(Math.max(5, Math.min(490, ny1))),
          x2: Math.round(Math.max(20, Math.min(875, nx2))),
          y2: Math.round(Math.max(20, Math.min(495, ny2)))
        };
      })
    );
  };

  const handleMouseUp = () => {
    setDragAction(null);
    setDragStartPos(null);
  };

  const handleModalConfirm = (sessionData: {
    videoName: string;
    fps: number;
    totalFrames: number;
    chambers: ChamberBox[];
    diffThresh: number;
    rows?: number;
    cols?: number;
    order?: 'column_first' | 'row_first';
  }) => {
    setVideoSource(sessionData.videoName);
    setChambers(sessionData.chambers);
    setDiffThresh(sessionData.diffThresh);
    if (sessionData.rows) setRows(sessionData.rows);
    if (sessionData.cols) setCols(sessionData.cols);
    if (sessionData.order) setOrder(sessionData.order);
    setParams((p) => ({ ...p, fps: sessionData.fps }));
    if (onImportSession) {
      onImportSession(sessionData);
    }
  };

  const activeBox = chambers.find((c) => c.id === selectedChamber);

  return (
    <div className="space-y-6 select-none" onMouseUp={handleMouseUp}>
      {/* Title & Introduction Card */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-lg font-bold text-slate-800">Module 1: Vision Tracking & Multi-Chamber Calibration</h2>
            <span className="bg-blue-100 text-blue-800 text-xs px-2.5 py-0.5 rounded-full font-semibold">
              {chambers.length} Chambers ({rows}x{cols})
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl leading-relaxed">
            Direct mouse drag &amp; 8-way resize handle calibration. Accurately encloses left wire mesh boundaries and right circular gas inlet holes.
            Active Source: <span className="font-mono font-semibold text-slate-700">{videoSource}</span>
          </p>
        </div>

        <div className="flex items-center flex-wrap gap-2.5">
          <button
            id="btn-open-video-modal"
            onClick={() => setIsModalOpen(true)}
            className="flex items-center space-x-1.5 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold shadow-sm transition cursor-pointer"
          >
            <Video className="w-3.5 h-3.5 text-blue-400" />
            <span>Import Video &amp; Popup Calibrator</span>
          </button>
          <button
            id="btn-auto-snap"
            onClick={handleAutoSnap}
            className="flex items-center space-x-1.5 px-3.5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-semibold shadow-sm transition cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Auto-Snap (Holes &amp; Meshes)</span>
          </button>
          <button
            id="btn-toggle-mask"
            onClick={() => setShowMask(!showMask)}
            className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold border transition cursor-pointer ${
              showMask
                ? 'bg-purple-50 border-purple-300 text-purple-700'
                : 'bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>{showMask ? 'View: Darkness Energy Mask' : 'View: Realistic Camera Frame'}</span>
          </button>
        </div>
      </div>

      <VideoImportModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onConfirm={handleModalConfirm}
        initialChambers={chambers}
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left 3 Cols: Interactive Calibrator Canvas with Drag & Drop */}
        <div className="lg:col-span-3 bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-md flex flex-col items-center">
          <div className="w-full flex items-center justify-between text-xs text-slate-400 pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="font-mono text-slate-200">{chambers.length}-Tube Arena ({rows} Rows × {cols} Cols)</span>
            </div>
            <div className="flex items-center space-x-4">
              <span>Selected Chamber: <strong className="text-amber-400">CH {selectedChamber}</strong></span>
              <span>Link Mode: <strong className="text-emerald-400">{mode.toUpperCase()}</strong></span>
            </div>
          </div>

          {/* SVG Arena Display with Real Mechanical Features */}
          <div className="w-full overflow-hidden flex justify-center py-2 relative">
            <svg
              ref={svgRef}
              viewBox="0 0 880 500"
              onMouseMove={handleMouseMove}
              className="w-full max-h-[480px] bg-[#12161f] rounded-lg border border-slate-800 select-none shadow-inner cursor-crosshair"
            >
              <defs>
                {/* Background Glass Lighting Gradient */}
                <linearGradient id="backlightGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#252a36" />
                  <stop offset="30%" stopColor="#3d4454" />
                  <stop offset="50%" stopColor="#485061" />
                  <stop offset="70%" stopColor="#3d4454" />
                  <stop offset="100%" stopColor="#252a36" />
                </linearGradient>

                {/* Glass Tube Interior Glow */}
                <linearGradient id="tubeGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#64748b" stopOpacity="0.4" />
                  <stop offset="8%" stopColor="#94a3b8" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#cbd5e1" stopOpacity="0.95" />
                  <stop offset="92%" stopColor="#94a3b8" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#64748b" stopOpacity="0.4" />
                </linearGradient>

                {/* Wire Mesh Grid Texture */}
                <pattern id="wireMesh" width="6" height="6" patternUnits="userSpaceOnUse">
                  <path d="M 0 0 L 6 0 M 0 3 L 6 3 M 0 0 L 0 6 M 3 0 L 3 6" fill="none" stroke="#334155" strokeWidth="0.8" />
                </pattern>

                <filter id="whiteGlow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Realistic Background Illumination Channels */}
              <rect width="880" height="500" fill="#0d1117" />

              {/* Horizontal Illuminated Glass Channel Strips */}
              {[60, 165, 270, 375].map((y, idx) => (
                <g key={idx}>
                  {/* Left Tube Strip */}
                  <rect x="25" y={y} width="395" height="75" rx="3" fill="url(#tubeGlow)" />
                  {/* Right Tube Strip */}
                  <rect x="460" y={y} width="395" height="75" rx="3" fill="url(#tubeGlow)" />

                  {/* Left Column Wire Mesh (Left end) */}
                  <rect x="25" y={y} width="32" height="75" fill="url(#wireMesh)" stroke="#475569" strokeWidth="0.5" />

                  {/* Left Column Circular Gas Hole (Right end, bright white) */}
                  <circle cx="395" cy={y + 37.5} r="14" fill="#ffffff" filter="url(#whiteGlow)" stroke="#94a3b8" strokeWidth="1" />
                  <circle cx="395" cy={y + 37.5} r="10" fill="#f8fafc" />

                  {/* Right Column Wire Mesh (Left end near center divider) */}
                  <rect x="460" y={y} width="32" height="75" fill="url(#wireMesh)" stroke="#475569" strokeWidth="0.5" />

                  {/* Right Column Circular Gas Hole (Right end, bright white) */}
                  <circle cx="830" cy={y + 37.5} r="14" fill="#ffffff" filter="url(#whiteGlow)" stroke="#94a3b8" strokeWidth="1" />
                  <circle cx="830" cy={y + 37.5} r="10" fill="#f8fafc" />
                </g>
              ))}

              {/* Center Vertical Divider Groove & Column Splitter */}
              <rect x="428" y="10" width="24" height="480" fill="#1e293b" stroke="#334155" strokeWidth="1" />
              <line x1="440" y1="10" x2="440" y2="490" stroke="#f97316" strokeWidth="1.5" strokeDasharray="5 5" />
              <text x="440" y="24" fill="#f97316" fontSize="9" fontWeight="bold" textAnchor="middle" fontFamily="monospace">
                CENTER DIVIDER
              </text>

              {/* Chamber Tubes Bounding Boxes */}
              {chambers.map((c) => {
                const isSelected = c.id === selectedChamber;
                const width = c.x2 - c.x1;
                const height = c.y2 - c.y1;

                // Fly Centroid Simulation inside tube
                const flyX = c.x1 + width * (0.25 + (c.id * 0.13) % 0.55);
                const flyY = c.y1 + height * 0.52;

                return (
                  <g key={c.id}>
                    {/* Bounding Box Rect */}
                    <rect
                      x={c.x1}
                      y={c.y1}
                      width={width}
                      height={height}
                      rx="4"
                      fill={
                        showMask
                          ? 'rgba(15, 23, 42, 0.9)'
                          : isSelected
                          ? 'rgba(245, 158, 11, 0.12)'
                          : 'rgba(34, 197, 94, 0.08)'
                      }
                      stroke={isSelected ? '#f59e0b' : '#22c55e'}
                      strokeWidth={isSelected ? '2.5' : '1.5'}
                      className="cursor-move"
                      onMouseDown={(e) => handleMouseDown(e, 'move', c.id)}
                    />

                    {/* Mesh marker & Hole marker dashed vertical indicators */}
                    <line
                      x1={c.x1 + width * 0.09}
                      y1={c.y1}
                      x2={c.x1 + width * 0.09}
                      y2={c.y2}
                      stroke="#f97316"
                      strokeWidth="1"
                      strokeDasharray="3 3"
                    />
                    <line
                      x1={c.x2 - width * 0.09}
                      y1={c.y1}
                      x2={c.x2 - width * 0.09}
                      y2={c.y2}
                      stroke="#38bdf8"
                      strokeWidth="1"
                      strokeDasharray="3 3"
                    />

                    {/* Chamber ID Badge */}
                    <rect
                      x={c.x1 + 4}
                      y={c.y1 + 4}
                      width={44}
                      height={18}
                      rx="3"
                      fill={isSelected ? '#d97706' : '#0f172a'}
                      className="cursor-pointer"
                      onMouseDown={(e) => handleMouseDown(e, 'move', c.id)}
                    />
                    <text
                      x={c.x1 + 26}
                      y={c.y1 + 16}
                      fill="#ffffff"
                      fontSize="10"
                      fontWeight="bold"
                      textAnchor="middle"
                      fontFamily="monospace"
                      className="pointer-events-none"
                    >
                      CH {c.id}
                    </text>

                    {/* Dimensions Pill for Selected Chamber */}
                    {isSelected && (
                      <text
                        x={c.x1 + 54}
                        y={c.y1 + 16}
                        fill="#fde68a"
                        fontSize="9"
                        fontFamily="monospace"
                        className="pointer-events-none"
                      >
                        {width}×{height} px
                      </text>
                    )}

                    {/* Fly Centroid Visualization */}
                    {showMask ? (
                      <ellipse
                        cx={flyX}
                        cy={flyY}
                        rx="9"
                        ry="5"
                        fill="#ec4899"
                        opacity="0.9"
                        filter="blur(1px)"
                      />
                    ) : (
                      <g className="pointer-events-none">
                        {/* Body blob */}
                        <ellipse cx={flyX} cy={flyY} rx="8" ry="4" fill="#1e1b4b" transform={`rotate(${(c.id * 25) % 360} ${flyX} ${flyY})`} />
                        {/* Red crosshair & green ring */}
                        <circle cx={flyX} cy={flyY} r="3" fill="#ef4444" />
                        <circle cx={flyX} cy={flyY} r="9" fill="none" stroke="#22c55e" strokeWidth="1" />
                        <line x1={flyX - 7} y1={flyY} x2={flyX + 7} y2={flyY} stroke="#22c55e" strokeWidth="0.8" />
                        <line x1={flyX} y1={flyY - 7} x2={flyX} y2={flyY + 7} stroke="#22c55e" strokeWidth="0.8" />
                      </g>
                    )}

                    {/* 8 Resize Handles on the Active Selected Box */}
                    {isSelected && (
                      <g>
                        {/* Top-Left */}
                        <rect
                          x={c.x1 - 4}
                          y={c.y1 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-nwse-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_tl', c.id)}
                        />
                        {/* Top-Middle */}
                        <rect
                          x={c.x1 + width / 2 - 4}
                          y={c.y1 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-ns-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_t', c.id)}
                        />
                        {/* Top-Right */}
                        <rect
                          x={c.x2 - 4}
                          y={c.y1 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-nesw-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_tr', c.id)}
                        />
                        {/* Middle-Left */}
                        <rect
                          x={c.x1 - 4}
                          y={c.y1 + height / 2 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-ew-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_l', c.id)}
                        />
                        {/* Middle-Right */}
                        <rect
                          x={c.x2 - 4}
                          y={c.y1 + height / 2 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-ew-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_r', c.id)}
                        />
                        {/* Bottom-Left */}
                        <rect
                          x={c.x1 - 4}
                          y={c.y2 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-nesw-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_bl', c.id)}
                        />
                        {/* Bottom-Middle */}
                        <rect
                          x={c.x1 + width / 2 - 4}
                          y={c.y2 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-ns-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_b', c.id)}
                        />
                        {/* Bottom-Right */}
                        <rect
                          x={c.x2 - 4}
                          y={c.y2 - 4}
                          width="8"
                          height="8"
                          fill="#ffffff"
                          stroke="#f59e0b"
                          strokeWidth="1.5"
                          className="cursor-nwse-resize"
                          onMouseDown={(e) => handleMouseDown(e, 'resize_br', c.id)}
                        />
                      </g>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="w-full flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-orange-500 inline-block"></span>
              <span>Orange line = Wire Mesh Boundary (Left)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-sky-400 inline-block"></span>
              <span>Cyan line = Circular Gas Inlet Hole (Right)</span>
            </span>
            <span className="text-emerald-400 font-mono">Drag box or handles directly on canvas</span>
          </div>
        </div>

        {/* Right 1 Col: Fine-Tuning Controls */}
        <div className="space-y-4">
          {/* Grid Layout Configuration */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-1.5">
              <Grid className="w-4 h-4 text-blue-600" />
              <span>Grid Geometry</span>
            </h3>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-slate-600 font-medium">Rows:</label>
                <input
                  type="number"
                  min="1"
                  max="32"
                  value={rows}
                  onChange={(e) => setRows(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full bg-slate-50 border border-slate-200 text-xs rounded px-2 py-1 mt-0.5"
                />
              </div>
              <div>
                <label className="text-xs text-slate-600 font-medium">Cols:</label>
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={cols}
                  onChange={(e) => setCols(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full bg-slate-50 border border-slate-200 text-xs rounded px-2 py-1 mt-0.5"
                />
              </div>
            </div>

            <button
              onClick={handleRegenerateGrid}
              className="w-full py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 rounded text-xs font-bold transition flex items-center justify-center space-x-1 cursor-pointer"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Generate Symmetric Grid</span>
            </button>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-1.5">
              <Crosshair className="w-4 h-4 text-blue-600" />
              <span>Chamber Fine Tuning</span>
            </h3>

            {/* Selection mode toggle */}
            <div>
              <label className="text-xs font-semibold text-slate-600">Link Mode:</label>
              <div className="grid grid-cols-2 gap-1.5 mt-1">
                {[
                  { id: 'single', label: 'Single Chamber' },
                  { id: 'col', label: 'Active Column' },
                  { id: 'row', label: 'Active Row' },
                  { id: 'all', label: `All ${chambers.length} Chambers` }
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setMode(m.id as any)}
                    className={`px-2 py-1.5 rounded text-xs font-medium border cursor-pointer ${
                      mode === m.id
                        ? 'bg-blue-50 border-blue-400 text-blue-700 font-bold'
                        : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Active Chamber selector buttons */}
            <div>
              <label className="text-xs font-semibold text-slate-600">Select Active Chamber (1..N):</label>
              <div className="flex flex-wrap gap-1 mt-1 max-h-28 overflow-y-auto p-1 bg-slate-50 rounded border border-slate-200">
                {chambers.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setSelectedChamber(c.id)}
                    className={`px-2 py-0.5 rounded text-xs font-mono font-bold border cursor-pointer ${
                      selectedChamber === c.id
                        ? 'bg-amber-500 text-white border-amber-600'
                        : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    CH {c.id}
                  </button>
                ))}
              </div>
            </div>

            {/* Direct Coordinate Readout */}
            {activeBox && (
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-xs font-mono text-slate-700 space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">X1 (Mesh):</span>
                  <strong>{activeBox.x1} px</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">X2 (Hole):</span>
                  <strong>{activeBox.x2} px</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">Y Span (Y1-Y2):</span>
                  <strong>{activeBox.y1} ~ {activeBox.y2} px</strong>
                </div>
                <div className="flex justify-between text-blue-700 font-bold pt-1 border-t border-slate-200">
                  <span className="font-sans">Width × Height:</span>
                  <span>{activeBox.x2 - activeBox.x1} × {activeBox.y2 - activeBox.y1} px</span>
                </div>
              </div>
            )}
          </div>

          {/* Vision Detection Parameters */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-1.5">
              <Sliders className="w-4 h-4 text-emerald-600" />
              <span>Vision Thresholds</span>
            </h3>

            <div>
              <div className="flex justify-between text-xs text-slate-600">
                <span>Darkness Diff Threshold:</span>
                <strong className="font-mono">{diffThresh} px</strong>
              </div>
              <input
                type="range"
                min="5"
                max="40"
                value={diffThresh}
                onChange={(e) => setDiffThresh(parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-600">
                <span>Expected Body Length:</span>
                <strong className="font-mono">{params.bodyLengthPx} px</strong>
              </div>
              <input
                type="range"
                min="6"
                max="25"
                value={params.bodyLengthPx}
                onChange={(e) => setParams((p) => ({ ...p, bodyLengthPx: parseInt(e.target.value) }))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-1"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
