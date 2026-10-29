# Time-binned chamber distribution plots.
import numpy as np


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
