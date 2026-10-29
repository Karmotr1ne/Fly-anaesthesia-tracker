"""One evaluation/export path shared by the desktop worker and CLI."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import pandas as pd
from core import schema
from core.session.layout import OutputTransaction, SessionPaths, session_root, experiment_name
from .anaesthesia_analyser import AnaesthesiaAnalyser
from .spatial_preference import protocol_windows, spatial_statistics


def _evaluate_with_overlay(source, params, *, plots=False, heatmaps=False,
                     cancelled=None, output_dir=None, overlay_video=None):
    source = Path(source).resolve()
    def check():
        if cancelled and cancelled():
            raise InterruptedError("Evaluation cancelled.")

    check()
    data = pd.read_csv(source)
    fields = {schema.field(name) for name in data.columns}
    if source.stem.lower().endswith("_raw") and not schema.has_kinematics(fields):
        raise ValueError("Raw contains uncleaned detections. Run Tracking and select its Position CSV.")
    analyser = AnaesthesiaAnalyser(replace(params))
    evaluated = analyser.process_dataframe(data)
    if evaluated.empty:
        raise ValueError("No valid trajectory; analysis results were not exported")
    check()
    summary = analyser.extract_summary(evaluated, _prepared=True)
    paths = SessionPaths(session_root(source))
    if output_dir is None:
        paths.ensure()
        output_dir, commit_root = paths.analysis, paths.root
        metadata = paths.logs / (experiment_name(source) + "_evaluation_config.json")
    else:
        output_dir = Path(output_dir).resolve()
        commit_root = output_dir
        metadata = output_dir / (experiment_name(source) + "_evaluation_config.json")
    prefix = Path(output_dir) / experiment_name(source)
    output = prefix.with_name(prefix.name + "_evaluated.csv")
    warnings = []
    windows, protocol_source = None, None
    if heatmaps:
        windows, protocol_source, notes = protocol_windows(source)
        warnings.extend(notes)
    with OutputTransaction(commit_root) as transaction:
        schema.write_csv(evaluated, transaction.stage(output))
        schema.write_csv(summary, transaction.stage(str(prefix) + "_summary.csv"))
        check()
        if plots or heatmaps or overlay_video:
            from .visualiser import ScientificVisualiser
            visualiser = ScientificVisualiser(analyser.fps)
            figures = []
            if plots:
                figures.extend((
                    (visualiser.plot_activity_position_overview, evaluated, "_activity_position.png", {}),
                    (visualiser.plot_kymograph_hexbin, evaluated, "_kymograph.png", {}),
                    (visualiser.plot_survival_kinetics, summary, "_survival.png", {})))
            if heatmaps:
                figures.append((visualiser.plot_spatial_preferences, evaluated, "_spatial_preferences.png",
                    {"floor_height": params.floor_height, "ceiling_height": params.ceiling_height,
                     "windows": windows}))
            for method, frame, suffix, options in figures:
                check()
                staged = transaction.stage(str(prefix) + suffix)
                try:
                    method(frame, str(staged), **options)
                except Exception as exc:
                    if staged.is_file():
                        staged.unlink()
                    warnings.append(f"{suffix}: {exc}")
            if overlay_video:
                check()
                visualiser.render_overlay_video(evaluated, str(overlay_video),
                    str(transaction.stage(str(prefix) + "_overlay.mp4")), cancel_callback=cancelled)
        effective = asdict(analyser.params)
        effective["fps"] = analyser.fps
        transaction.stage(metadata).write_text(json.dumps({
            "input": str(source), "parameters": effective,
            "plots": plots, "heatmaps": heatmaps, "warnings": warnings,
            "spatial_protocol_source": protocol_source, "spatial_windows": windows,
            "output": str(output)}, indent=2), encoding="utf-8")
        check()
        transaction.commit(cancelled)
    return evaluated, output, warnings


def evaluate_dataset(source, params, *, plots=False, heatmaps=False,
                     cancelled=None, output_dir=None, overlay_video=None,
                     summary_only=False, progress=None, angular_plot=False, component_plot=False, median_plot=False):
    """Bound analysis memory to one chamber; commit each video's CSVs separately.

    Return the small summary table, output path and optional plot warnings.
    """
    if overlay_video:
        if angular_plot or component_plot or median_plot:
            raise ValueError('Export the analysis plots separately from the CLI video overlay.')
        if summary_only:
            raise ValueError('Summary-only mode cannot export an overlay.')
        # Explicit CLI overlay export retains the existing full-video renderer.
        evaluated, output, warnings = _evaluate_with_overlay(
            source, params, plots=plots, heatmaps=heatmaps, cancelled=cancelled,
            output_dir=output_dir, overlay_video=overlay_video)
        analyser = AnaesthesiaAnalyser(replace(params))
        return analyser.extract_summary(evaluated, _prepared=True), output, warnings
    import numpy as np
    params.validate()
    source = Path(source).resolve()
    paths = SessionPaths(session_root(source))
    directory = Path(output_dir).resolve() if output_dir is not None else paths.analysis
    root = directory if output_dir is not None else paths.root
    from core.statistics.collector import SummaryCollector
    analysis_id = experiment_name(source)
    prefix = directory / analysis_id
    output = prefix.with_name(prefix.name + ('_summary.csv' if summary_only else '_evaluated.csv'))
    metadata = (directory if output_dir is not None else paths.logs) / (analysis_id + '_evaluation_config.json')
    warnings, summaries, chambers = [], [], {}
    rates = set()
    plot_parts = []
    preference_statistics = []
    windows, protocol_source = None, None
    if heatmaps and not summary_only:
        windows, protocol_source, notes = protocol_windows(source)
        warnings.extend(notes)

    def notify(message, fraction):
        if cancelled and cancelled():
            raise InterruptedError('Evaluation cancelled.')
        if progress:
            progress(message, fraction)

    with OutputTransaction(root) as transaction:
        notify('Reading Position CSV; splitting chambers', 0.)
        with pd.read_csv(source, chunksize=20000) as reader:
            for chunk in reader:
                notify('Reading Position CSV; splitting chambers', .05)
                chunk = schema.normalise(chunk)
                if source.stem.lower().endswith('_raw') and not schema.has_kinematics(chunk.columns):
                    raise ValueError('Raw contains uncleaned detections. Select the Position CSV.')
                if 'chamber_id' not in chunk:
                    raise ValueError('Missing chamber id.')
                ids = pd.to_numeric(chunk.chamber_id, errors='raise')
                if not np.isfinite(ids).all() or not (ids == np.floor(ids)).all():
                    raise ValueError('Chamber IDs must be finite integers.')
                chunk['chamber_id'] = ids.astype('int64')
                if 'fps' in chunk:
                    rates.update(pd.to_numeric(chunk.fps, errors='coerce').dropna().unique())
                    if any(not np.isfinite(rate) or rate <= 0 for rate in rates):
                        raise ValueError('Frame rate must be finite and positive.')
                    if len(rates) > 1:
                        raise ValueError('A dataset must have one consistent frame rate.')
                for chamber, group in chunk.groupby('chamber_id', sort=False):
                    spool = transaction.directory / f'chamber_{chamber}.csv'
                    group.to_csv(spool, mode='a', header=not spool.exists(), index=False, encoding='utf-8')
                    chambers[int(chamber)] = spool
        if not chambers:
            raise ValueError('No valid trajectory; analysis results were not exported')
        analyser = AnaesthesiaAnalyser(replace(params))
        written = False
        evaluated_stage = transaction.stage(output) if not summary_only else None
        for index, (chamber, spool) in enumerate(sorted(chambers.items()), 1):
            notify(f'Analysing CH{chamber} ({index}/{len(chambers)})', .1 + .8*(index-1)/len(chambers))
            data = pd.read_csv(spool)
            evaluated = analyser.process_dataframe(data)
            del data
            spool.unlink()
            if evaluated.empty:
                continue
            summary = analyser.extract_summary(evaluated, _prepared=True)
            summaries.append(summary)
            if not summary_only:
                schema.write_csv(evaluated, evaluated_stage, mode='a', header=not written)
                written = True
            if not summary_only and (plots or heatmaps or angular_plot or component_plot):
                if heatmaps:
                    # Aggregate before plot decimation: short steps retain all frames.
                    preference_statistics.extend(spatial_statistics(
                        evaluated, windows, analyser.fps, params.floor_height, params.ceiling_height))
                # Plot bounded samples, preserving a panel for every chamber.
                stride = max(1, (len(evaluated) + 9999) // 10000)
                plot_parts.append(evaluated.iloc[::stride].copy())
            del evaluated
        if not summaries:
            raise ValueError('No valid trajectory; analysis results were not exported')
        summary = pd.concat(summaries, ignore_index=True)
        summary['analysis_id'] = analysis_id
        summary_path = Path(str(prefix) + '_summary.csv')
        schema.write_csv(summary, transaction.stage(summary_path))
        if plot_parts:
            from .visualiser import ScientificVisualiser
            from .directional_velocity import plot_velocity
            visualiser = ScientificVisualiser(analyser.fps)
            plot_data = pd.concat(plot_parts, ignore_index=True)
            plot_parts.clear()
            figures = []
            if plots:
                figures.extend(((visualiser.plot_activity_position_overview, '_activity_position.png', {}),
                                (visualiser.plot_kymograph_hexbin, '_kymograph.png', {})))
            if heatmaps:
                figures.append((visualiser.plot_spatial_preferences, '_spatial_preferences.png',
                    dict(floor_height=params.floor_height, ceiling_height=params.ceiling_height,
                         windows=windows, statistics=preference_statistics)))
            if angular_plot and component_plot:
                figures.append((plot_velocity, '_directional_motion.png', dict(angular=None)))
            elif angular_plot:
                figures.append((plot_velocity, '_angular_velocity.png', dict(angular=True)))
            elif component_plot:
                figures.append((plot_velocity, '_component_velocity.png', dict(angular=False)))
            datasets = [('', plot_data)]
            for tag, frame in datasets:
                for method, suffix, options in figures:
                    notify(f'Plotting experiment{tag}: {suffix}', .9)
                    staged = transaction.stage(str(prefix) + tag + suffix)
                    try:
                        method(frame, str(staged), **options)
                    except Exception as exc:
                        staged.unlink(missing_ok=True)
                        warnings.append(f'{tag}{suffix}: {exc}')
            if median_plot:
                from .directional_velocity import plot_boxplot_summary
                fields = []
                if plots:
                    fields.extend((('speed', 'Speed (px/s)'), ('position', 'Normalised height')))
                if heatmaps and not plots:
                    fields.append(('position', 'Normalised height'))
                if component_plot:
                    fields.extend((('horizontal_velocity(px/s)', 'Horizontal (px/s)'),
                                   ('vertical_velocity(px/s)', 'Vertical (px/s; up +)')))
                if angular_plot:
                    fields.append(('turning_angular_velocity(deg/s)', 'Turning (deg/s)'))
                staged = transaction.stage(str(prefix) + '_boxplot_summary.png')
                notify('Plotting box-and-whisker summary', .9)
                try:
                    plot_boxplot_summary(plot_data, staged, fields)
                except Exception as exc:
                    staged.unlink(missing_ok=True)
                    warnings.append(f'Boxplot summary: {exc}')
            del datasets, plot_data
        if plots and not summary_only:
            from .visualiser import ScientificVisualiser
            staged = transaction.stage(str(prefix) + '_survival.png')
            try:
                ScientificVisualiser(analyser.fps).plot_survival_kinetics(summary, str(staged))
            except Exception as exc:
                staged.unlink(missing_ok=True)
                warnings.append(f'_survival.png: {exc}')
        effective = asdict(analyser.params)
        effective['fps'] = analyser.fps
        transaction.stage(metadata).write_text(json.dumps(dict(
            input=str(source), analysis_id=analysis_id, boxplot_summary=median_plot, parameters=effective, plots=plots and not summary_only,
            heatmaps=heatmaps and not summary_only, summary_only=summary_only,
            storage='per_chamber_analysis; incremental_csv', warnings=warnings,
            angular_plot=angular_plot, component_plot=component_plot,
            spatial_protocol_source=protocol_source, spatial_windows=windows,
            output=str(output), summary=str(summary_path)), indent=2), encoding='utf-8')
        notify('Saving complete video CSVs to Session', .95)
        transaction.commit(cancelled)
    notify(f'Saved: {output}', 1.)
    return summary, output, warnings
