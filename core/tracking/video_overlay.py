"""Render trajectory annotations without importing scientific plotting backends."""
from core import schema
from collections import defaultdict, deque
from pathlib import Path
import csv
import cv2
import numpy as np


def _overlay_rows(data):
    """Yield frame-ordered rows, streaming CSV sources without a DataFrame."""
    if isinstance(data, (str, Path)):
        previous = -1
        with Path(data).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = {schema.field(name): name for name in reader.fieldnames or ()}
            clean = {'x_clean', 'y_clean'}.issubset(columns)
            x_name, y_name = ('x_clean', 'y_clean') if clean else ('x', 'y')
            for row in reader:
                frame = int(float(row[columns['frame']]))
                if frame < previous:
                    raise ValueError('Overlay CSV must be ordered by frame.')
                previous = frame
                def value(name, default=''):
                    return row.get(columns.get(name, ''), default)
                yield (frame, int(float(value('chamber_id'))),
                       float(value(x_name) or 'nan'), float(value(y_name) or 'nan'),
                       float(value('is_interpolated') or 0), value('state'), value('segment_id'))
    else:
        ordered = schema.normalise(data).sort_values('frame', kind='stable')
        clean = {'x_clean', 'y_clean'}.issubset(ordered.columns)
        x_name, y_name = ('x_clean', 'y_clean') if clean else ('x', 'y')
        count = len(ordered)
        yield from zip(ordered.frame.to_numpy(dtype=np.int64),
            ordered.chamber_id.to_numpy(dtype=np.int64), ordered[x_name].to_numpy(), ordered[y_name].to_numpy(),
            ordered.is_interpolated.to_numpy() if 'is_interpolated' in ordered else np.zeros(count),
            ordered.state.to_numpy() if 'state' in ordered else ['']*count,
            ordered.segment_id.to_numpy() if 'segment_id' in ordered else ['']*count)


def render_trajectory_overlay(data, source_path, output_path, fps_fallback=30.0,
                              scale=1.0, target_width=1080,
                              progress_callback=None, cancel_callback=None):
    source,target=Path(source_path),Path(output_path)
    if source.resolve()==target.resolve():
        raise ValueError("Overlay output must not overwrite the source video")
    capture=cv2.VideoCapture(str(source))
    writer=None
    temporary=target.with_name(target.stem+'.partial'+target.suffix)
    committed=False
    rows = _overlay_rows(data)
    try:
        if not capture.isOpened():
            raise IOError(f"Cannot open video: {source}")
        width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:
            fps=fps_fallback
        factor=min(1.0,target_width/width) if scale is None else float(scale)
        if not np.isfinite(factor) or factor<=0:
            raise ValueError("Overlay scale must be finite and positive")
        out_width,out_height=int(width*factor),int(height*factor)
        if out_width<2 or out_height<2:
            raise ValueError("Invalid overlay video dimensions")
        if factor==1.0 and (out_width%2 or out_height%2):
            raise ValueError("The MP4 encoder requires even source dimensions to preserve native resolution")
        if factor!=1.0:
            out_width-=out_width%2
            out_height-=out_height%2
        sx,sy=out_width/width,out_height/height
        target.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,(out_width,out_height))
        if not writer.isOpened():
            raise IOError(f"Cannot initialize overlay video writer: {target}")
        pending = next(rows, None)
        trails=defaultdict(lambda:deque(maxlen=30))
        colors=((50,205,50),(255,180,0),(0,180,255),(200,80,220),(255,120,120),(80,220,220))
        frame_index=0
        while True:
            if cancel_callback and cancel_callback():
                raise InterruptedError("Overlay rendering cancelled")
            ok,frame=capture.read()
            if not ok:
                break
            if (out_width,out_height)!=(width,height):
                frame=cv2.resize(frame,(out_width,out_height),interpolation=cv2.INTER_AREA)
            while pending is not None and pending[0] < frame_index:
                pending = next(rows, None)
            while pending is not None and pending[0] == frame_index:
                _, chamber, x, y, flag, state, segment = pending
                pending = next(rows, None)
                chamber = int(chamber)
                color=colors[(chamber-1)%len(colors)]
                if not np.isfinite(x) or not np.isfinite(y) or not (0<=x<=width and 0<=y<=height):
                    trails[chamber].clear()
                    continue
                point=(int(np.clip(x*sx,0,out_width-1)),int(np.clip(y*sy,0,out_height-1)))
                trail=trails[chamber]
                if trail and (trail[-1][0]!=frame_index-1 or trail[-1][2]!=segment):
                    trail.clear()
                trail.append((frame_index,point,segment))
                if len(trail)>1:
                    points=np.asarray([item[1] for item in trail],dtype=np.int32)
                    cv2.polylines(frame,[points],False,color,1,cv2.LINE_AA)
                cv2.circle(frame,point,4,color,1 if bool(flag) else -1,cv2.LINE_AA)
                label=f"CH{chamber}"+(f" {state}" if isinstance(state,str) and state else '')
                cv2.putText(frame,label,(min(out_width-1,point[0]+6),max(12,point[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)
            writer.write(frame)
            frame_index+=1
            if progress_callback and (frame_index%100==0 or frame_index==total):
                progress_callback(frame_index,total)
        if frame_index==0:
            raise ValueError("No source video frames were decoded for the overlay")
        if cancel_callback and cancel_callback():
            raise InterruptedError("Overlay rendering cancelled")
        writer.release()
        writer=None
        temporary.replace(target)
        committed=True
        return str(target)
    finally:
        rows.close()
        capture.release()
        if writer is not None:
            writer.release()
        if not committed and temporary.is_file():
            temporary.unlink()
