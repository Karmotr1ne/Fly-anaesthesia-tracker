"""Signed Cartesian velocity and wrapped trajectory turning rate."""
import numpy as np


def add_directional_velocity(data):
    data = data.copy()
    count = len(data)
    vx, vy, omega = (np.full(count, np.nan) for _ in range(3))
    for _, indices in data.groupby('chamber_id', sort=False).indices.items():
        g = data.iloc[indices]
        if not ({'x_clean', 'y_clean'}.issubset(g.columns) or {'x', 'y'}.issubset(g.columns)):
            continue
        x = g['x_clean' if 'x_clean' in g else 'x'].to_numpy(dtype=float)
        y = g['y_clean' if 'y_clean' in g else 'y'].to_numpy(dtype=float)
        t = g.time.to_numpy(dtype=float)
        dt = np.diff(t)
        valid = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & np.isfinite(y[1:]) & np.isfinite(y[:-1]) & np.isfinite(dt) & (dt > 0)
        if 'frame' in g:
            valid &= np.diff(g.frame.to_numpy()) == 1
        if 'segment_id' in g:
            segment = g.segment_id.to_numpy()
            valid &= segment[1:] == segment[:-1]
        for flag in ('is_rejected',):
            if flag in g:
                rejected = g[flag].fillna(1).to_numpy() != 0
                valid &= ~rejected[1:] & ~rejected[:-1]
        a, b = np.full(len(g), np.nan), np.full(len(g), np.nan)
        np.divide(np.diff(x), dt, out=a[1:], where=valid)
        # Image y grows downwards; Cartesian y grows upwards.
        np.divide(-np.diff(y), dt, out=b[1:], where=valid)
        heading = np.arctan2(b, a)
        heading[np.hypot(a, b) <= 1e-6] = np.nan
        delta = np.diff(heading)
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        rate = np.full(len(g), np.nan)
        # Headings describe intervals, so compare their midpoint timestamps.
        heading_dt = dt.copy()
        if len(dt) > 1:
            heading_dt[1:] = (dt[1:] + dt[:-1]) / 2
        np.divide(np.degrees(delta), heading_dt, out=rate[1:], where=valid & np.isfinite(delta))
        vx[indices], vy[indices], omega[indices] = a, b, rate
    data['horizontal_velocity(px/s)'] = vx
    data['vertical_velocity(px/s)'] = vy
    data['turning_angular_velocity(deg/s)'] = omega
    data['vertical_direction'] = np.where(np.isnan(vy), 'Unknown', np.where(vy > 0, 'Up', np.where(vy < 0, 'Down', 'Stationary')))
    return data


def plot_velocity(data, path, angular=False):
    import matplotlib.pyplot as plt
    if angular is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        try:
            fields = ('horizontal_velocity(px/s)', 'vertical_velocity(px/s)', 'turning_angular_velocity(deg/s)')
            labels = ('Horizontal (px/s; right +)', 'Vertical (px/s; up +)', 'Turning (deg/s; CCW +)')
            for ax, field, label in zip(axes, fields, labels):
                for chamber, group in data.groupby('chamber_id'):
                    ax.plot(group.time, group[field], linewidth=.7, label=f'CH{chamber}')
                ax.axhline(0, color='grey', linewidth=.5)
                ax.set_ylabel(label)
            axes[0].legend(ncol=4, fontsize=8)
            axes[-1].set_xlabel('Time (s)')
            fig.tight_layout()
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        if angular:
            for chamber, group in data.groupby('chamber_id'):
                ax.plot(group.time, group['turning_angular_velocity(deg/s)'], linewidth=.7, label=f'CH{chamber}')
            ax.legend()
            ax.set_ylabel('Turning angular velocity (deg/s; CCW positive)')
        else:
            for chamber, group in data.groupby('chamber_id'):
                ax.plot(group.time, group['horizontal_velocity(px/s)'], label=f'CH{chamber} horizontal', linewidth=.7)
                ax.plot(group.time, group['vertical_velocity(px/s)'], label=f'CH{chamber} vertical (up positive)', linewidth=.7)
            ax.set_ylabel('Velocity (px/s)')
            ax.legend()
        ax.axhline(0, color='grey', linewidth=.5)
        ax.set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def plot_boxplot_summary(data, path, fields):
    """Each box contains one median per chamber/time bin, not pooled frames."""
    import matplotlib.pyplot as plt
    fields = [(name, label) for name, label in fields if name in data]
    if not fields:
        raise ValueError('No selected numeric fields for boxplot summary.')
    valid = data.loc[np.isfinite(data.time)].copy()
    if valid.empty:
        raise ValueError('No finite timestamps for boxplot summary.')
    start, end = float(valid.time.min()), float(valid.time.max())
    width = max(1., (end - start) / 30.)
    valid['_time_bin'] = np.floor((valid.time - start) / width).astype(int).clip(upper=29)
    fig, axes = plt.subplots(len(fields), 1, figsize=(13, max(4, 3*len(fields))), squeeze=False, sharex=True)
    try:
        for ax, (name, label) in zip(axes[:, 0], fields):
            values = valid.groupby(['_time_bin', 'chamber_id'])[name].median()
            samples, positions = [], []
            for bin_id, group in values.groupby(level=0):
                finite = group.to_numpy(dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    samples.append(finite)
                    positions.append(start + (bin_id + .5)*width)
            if samples:
                boxes = ax.boxplot(samples, positions=positions, widths=width*.65,
                    whis=1.5, showfliers=True, patch_artist=True, manage_ticks=False)
                for box in boxes['boxes']:
                    box.set_facecolor('#DBEAFE')
            ax.set_ylabel(label)
            ax.axhline(0, color='grey', linewidth=.5)
        axes[-1, 0].set_xlabel(f'Time (s); bin width {width:.1f} s; one value per chamber')
        fig.suptitle('Chamber distribution: median, IQR and 1.5 IQR whiskers')
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
