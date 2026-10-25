import os
import re
import shutil
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd


class SessionDataExtractorApp:
    # Half-open intervals include every minute of the ending hour:
    # day:   08:00:00 <= time < 11:00:00
    # dawn:  11:00:00 <= time < 14:00:00
    # night: 14:00:00 <= time < 17:00:00
    SUMMARY_TIME_WINDOWS = (
        ("day", 8, 11),
        ("dawn", 11, 14),
        ("night", 14, 17),
    )

    TIMESTAMP_PATTERN = re.compile(
        r"(?<!\d)(\d{8}_\d{6})(?!\d)"
    )

    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Session CSV Extractor & Aggregator")
        self.root.geometry("900x700")
        self.root.minsize(820, 620)

        # Map output metrics to candidate input column names.
        self.target_metrics = {
            "sedation_latency": [
                "sedation_latency_sec",
                "sedation_latency",
            ],
            "anesthesia_latency": [
                "anesthesia_latency_sec",
                "anesthesia_latency",
            ],
            "speed": [
                "baseline_speed",
                "speed",
                "velocity",
            ],
        }

        self.loaded_subfolders = []
        self._build_ui()

    def _build_ui(self):
        pad_opts = {"padx": 10, "pady": 5}

        # Workspace and folder selection.
        dir_frame = ttk.LabelFrame(
            self.root,
            text=" 1. Workspace & Folder Selection ",
        )
        dir_frame.pack(fill="x", **pad_opts)

        top_row = ttk.Frame(dir_frame)
        top_row.pack(fill="x", padx=5, pady=4)

        ttk.Label(
            top_row,
            text="Parent Workspace:",
        ).pack(side="left", padx=2)

        self.dir_path_var = tk.StringVar()
        self.dir_entry = ttk.Entry(
            top_row,
            textvariable=self.dir_path_var,
            width=50,
        )
        self.dir_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5,
        )

        ttk.Button(
            top_row,
            text="Browse Workspace...",
            command=self._select_workspace,
        ).pack(side="left", padx=2)

        list_label_frame = ttk.Frame(dir_frame)
        list_label_frame.pack(
            fill="x",
            padx=5,
            pady=(4, 0),
        )

        ttk.Label(
            list_label_frame,
            text=(
                "Detected Folders "
                "(Hold Shift or Ctrl to select multiple folders):"
            ),
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        list_container = ttk.Frame(dir_frame)
        list_container.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=4,
        )

        self.folder_listbox = tk.Listbox(
            list_container,
            selectmode=tk.EXTENDED,
            height=6,
            exportselection=False,
        )

        list_scroll_y = ttk.Scrollbar(
            list_container,
            orient="vertical",
            command=self.folder_listbox.yview,
        )
        list_scroll_x = ttk.Scrollbar(
            list_container,
            orient="horizontal",
            command=self.folder_listbox.xview,
        )

        self.folder_listbox.configure(
            yscrollcommand=list_scroll_y.set,
            xscrollcommand=list_scroll_x.set,
        )

        self.folder_listbox.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        list_scroll_y.grid(
            row=0,
            column=1,
            sticky="ns",
        )
        list_scroll_x.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        btn_select_row = ttk.Frame(dir_frame)
        btn_select_row.pack(fill="x", padx=5, pady=4)

        ttk.Button(
            btn_select_row,
            text="Select All",
            command=self._select_all_folders,
        ).pack(side="left", padx=3)

        ttk.Button(
            btn_select_row,
            text="Clear Selection",
            command=self._clear_selected_folders,
        ).pack(side="left", padx=3)

        self.folder_count_lbl = ttk.Label(
            btn_select_row,
            text="0 folder(s) loaded",
            foreground="gray",
        )
        self.folder_count_lbl.pack(side="right", padx=5)

        out_row = ttk.Frame(dir_frame)
        out_row.pack(fill="x", padx=5, pady=4)

        ttk.Label(
            out_row,
            text="Output Directory:",
        ).pack(side="left", padx=2)

        self.out_path_var = tk.StringVar()
        ttk.Entry(
            out_row,
            textvariable=self.out_path_var,
            width=50,
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=5,
        )

        ttk.Button(
            out_row,
            text="Browse Output...",
            command=self._select_out_dir,
        ).pack(side="left", padx=2)

        # File scanning options.
        config_frame = ttk.LabelFrame(
            self.root,
            text=" 2. File Scan Mode Configuration ",
        )
        config_frame.pack(fill="x", **pad_opts)

        file_frame = ttk.Frame(config_frame)
        file_frame.pack(fill="x", padx=5, pady=4)

        ttk.Label(
            file_frame,
            text="File Mode:",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=2)

        self.scan_summary = tk.BooleanVar(value=True)
        self.scan_cleaned = tk.BooleanVar(value=False)

        self.chk_summary = ttk.Checkbutton(
            file_frame,
            text="Summary CSVs (Extract and group metrics)",
            variable=self.scan_summary,
        )
        self.chk_summary.pack(side="left", padx=10)

        self.chk_cleaned = ttk.Checkbutton(
            file_frame,
            text="Cleaned CSVs (Copy directly)",
            variable=self.scan_cleaned,
        )
        self.chk_cleaned.pack(side="left", padx=10)

        # Execution controls and log.
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", **pad_opts)

        self.run_btn = ttk.Button(
            action_frame,
            text="Execute & Process",
            command=self._start_processing_thread,
        )
        self.run_btn.pack(side="right", padx=5)

        log_frame = ttk.LabelFrame(
            self.root,
            text=" Execution Output Log ",
        )
        log_frame.pack(
            fill="both",
            expand=True,
            **pad_opts,
        )

        self.log_text = ScrolledText(
            log_frame,
            height=13,
            state="normal",
            font=("Consolas", 9),
        )
        self.log_text.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

    def _select_workspace(self):
        path = filedialog.askdirectory(
            title="Select Parent Workspace Directory"
        )
        if not path:
            return

        self.dir_path_var.set(path)

        if not self.out_path_var.get():
            self.out_path_var.set(path)

        self._load_subdirectories(path)

    def _load_subdirectories(self, base_path):
        self.folder_listbox.delete(0, tk.END)
        self.loaded_subfolders = []

        try:
            for item in sorted(os.listdir(base_path)):
                full_path = os.path.join(base_path, item)

                if os.path.isdir(full_path):
                    self.loaded_subfolders.append(full_path)
                    self.folder_listbox.insert(tk.END, item)

            count = len(self.loaded_subfolders)
            self.folder_count_lbl.config(
                text=f"{count} folder(s) loaded"
            )

            if count:
                self._select_all_folders()

        except Exception as exc:
            self._log(
                f"[Error] Failed to read subdirectories: {exc}"
            )

    def _select_all_folders(self):
        self.folder_listbox.select_set(0, tk.END)

    def _clear_selected_folders(self):
        self.folder_listbox.selection_clear(0, tk.END)

    def _select_out_dir(self):
        path = filedialog.askdirectory(
            title="Select Destination Output Directory"
        )
        if path:
            self.out_path_var.set(path)

    def _append_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _log(self, message):
        # Schedule all Tkinter widget updates on the main thread.
        if threading.current_thread() is threading.main_thread():
            self._append_log(message)
        else:
            self.root.after(0, self._append_log, message)

    def _get_target_directories(self):
        selected_indices = self.folder_listbox.curselection()

        if not selected_indices:
            root_dir = self.dir_path_var.get().strip()
            return [root_dir] if root_dir else []

        return [
            self.loaded_subfolders[index]
            for index in selected_indices
        ]

    def _extract_column_header_implicit(self, file_path):
        """
        Read the nearest ancestor folder containing YYYYMMDD_HHMMSS.
        Return only the date and time, without the Session prefix.
        """
        folder_path = os.path.dirname(os.path.abspath(file_path))
        fallback_name = os.path.basename(folder_path) or "Unknown"

        while folder_path:
            folder_name = os.path.basename(folder_path)
            match = self.TIMESTAMP_PATTERN.search(folder_name)

            if match:
                return match.group(1)

            parent_path = os.path.dirname(folder_path)
            if parent_path == folder_path:
                break

            folder_path = parent_path

        # Preserve a traceable fallback when the timestamp is missing.
        fallback_name = re.sub(
            r"session",
            "",
            fallback_name,
            flags=re.IGNORECASE,
        )
        return fallback_name.strip(" _-") or "Unknown"

    def _get_summary_time_info(self, identifier):
        """
        Return (group name, parsed datetime).
        Missing or invalid timestamps remain unclassified.
        """
        match = self.TIMESTAMP_PATTERN.search(identifier)

        if not match:
            return "unclassified", None

        try:
            experiment_time = datetime.strptime(
                match.group(1),
                "%Y%m%d_%H%M%S",
            )
        except ValueError:
            return "unclassified", None

        for group, start_hour, end_hour in self.SUMMARY_TIME_WINDOWS:
            if start_hour <= experiment_time.hour < end_hour:
                return group, experiment_time

        return "unclassified", experiment_time

    def _summary_column_sort_key(self, column_name):
        """Sort by group, full experiment timestamp, and column name."""
        group, experiment_time = self._get_summary_time_info(
            column_name
        )

        group_order = {
            name: index
            for index, (name, _, _) in enumerate(
                self.SUMMARY_TIME_WINDOWS
            )
        }

        return (
            group_order.get(group, len(group_order)),
            experiment_time is None,
            experiment_time if experiment_time is not None else datetime.max,
            column_name,
        )

    @staticmethod
    def _available_output_path(out_dir, filename):
        """Append a numeric suffix instead of overwriting an existing file."""
        destination = os.path.join(out_dir, filename)
        name_root, extension = os.path.splitext(destination)
        counter = 1

        while os.path.exists(destination):
            destination = f"{name_root}_{counter}{extension}"
            counter += 1

        return destination

    def _copy_cleaned_file_safely(self, file_path, out_dir):
        """Copy a cleaned CSV using its date and time as the prefix."""
        identifier = self._extract_column_header_implicit(file_path)
        filename = f"{identifier}_{os.path.basename(file_path)}"

        destination = self._available_output_path(
            out_dir,
            filename,
        )
        shutil.copy2(file_path, destination)
        return destination

    def _start_processing_thread(self):
        target_dirs = self._get_target_directories()
        out_dir = self.out_path_var.get().strip()

        if not target_dirs or any(
            not os.path.isdir(path) for path in target_dirs
        ):
            messagebox.showerror(
                "Error",
                "Please select valid input folders.",
            )
            return

        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror(
                "Error",
                "Please select a valid output directory.",
            )
            return

        # Read Tkinter variables before starting the worker thread.
        scan_summary = self.scan_summary.get()
        scan_cleaned = self.scan_cleaned.get()

        if not scan_summary and not scan_cleaned:
            messagebox.showerror(
                "Error",
                "Please check at least one file scanning option.",
            )
            return

        self.run_btn.config(state="disabled")
        self.log_text.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self._process_folders,
            args=(
                target_dirs,
                out_dir,
                scan_summary,
                scan_cleaned,
            ),
            daemon=True,
        )
        thread.start()

    def _process_folders(
        self,
        target_dirs,
        out_dir,
        scan_summary,
        scan_cleaned,
    ):
        try:
            self._log(
                f"[*] Scanning {len(target_dirs)} selected folder(s)..."
            )

            summary_files = []
            cleaned_files = []
            seen_files = set()

            for folder_dir in target_dirs:
                for current_root, directories, files in os.walk(folder_dir):
                    directories.sort()

                    for filename in sorted(files):
                        lower_name = filename.lower()
                        if not lower_name.endswith(".csv"):
                            continue

                        file_path = os.path.join(
                            current_root,
                            filename,
                        )
                        file_key = os.path.normcase(
                            os.path.realpath(file_path)
                        )

                        # Avoid repeated processing of overlapping selections.
                        if file_key in seen_files:
                            continue
                        seen_files.add(file_key)

                        # Exclude metric tables exported by this application.
                        is_metric_export = any(
                            re.fullmatch(
                                rf"{re.escape(metric)}_summary"
                                rf"(?:_\d+)?\.csv",
                                lower_name,
                            )
                            for metric in self.target_metrics
                        )

                        if (
                            scan_summary
                            and "summary" in lower_name
                            and not is_metric_export
                        ):
                            summary_files.append(file_path)

                        if scan_cleaned and "cleaned" in lower_name:
                            cleaned_files.append(file_path)

            # Process only the file list captured before any output is written.
            if scan_cleaned:
                self._log(
                    f"\n[*] Found {len(cleaned_files)} cleaned file(s)."
                )

                copied_count = 0
                for file_path in cleaned_files:
                    destination = self._copy_cleaned_file_safely(
                        file_path,
                        out_dir,
                    )
                    copied_count += 1
                    self._log(
                        f"   [+] Copied: {os.path.basename(destination)}"
                    )

                self._log(
                    f"[+] Copied {copied_count} cleaned file(s)."
                )

            if scan_summary:
                self._process_summary_files(
                    summary_files,
                    out_dir,
                )

            self._log("\n[+] Processing finished.")
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Completed",
                    "Processing finished. Please review the log "
                    "for skipped files and warnings.\n"
                    f"Destination: {out_dir}",
                ),
            )

        except Exception as exc:
            error_message = str(exc)
            self._log(f"\n[Fatal Error] {error_message}")
            self.root.after(
                0,
                lambda text=error_message: messagebox.showerror(
                    "Execution Error",
                    f"An unexpected error occurred:\n{text}",
                ),
            )

        finally:
            self.root.after(
                0,
                lambda: self.run_btn.config(state="normal"),
            )

    def _process_summary_files(self, summary_files, out_dir):
        self._log(
            f"\n[*] Found {len(summary_files)} summary file(s)."
        )

        if not summary_files:
            return

        metric_series = {
            metric: {}
            for metric in self.target_metrics
        }
        group_counts = {
            group: 0
            for group, _, _ in self.SUMMARY_TIME_WINDOWS
        }
        group_counts["unclassified"] = 0

        # Reserve one identifier per source file across all output metrics.
        used_headers = set()
        processed_count = 0

        for file_path in sorted(summary_files):
            identifier = self._extract_column_header_implicit(
                file_path
            )
            group, experiment_time = self._get_summary_time_info(
                identifier
            )

            if experiment_time is None:
                self._log(
                    "[Check] Missing or invalid folder timestamp; "
                    f"assigned to unclassified: {file_path}"
                )
            elif group == "unclassified":
                self._log(
                    f"[Check] Timestamp {experiment_time:%Y-%m-%d %H:%M:%S} "
                    f"is outside all windows: {file_path}"
                )
            else:
                self._log(
                    f"   [Group] {identifier} -> {group}"
                )

            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                self._log(
                    f"   [Read Error] {file_path}: {exc}"
                )
                continue

            chamber_col = next(
                (
                    column
                    for column in ("chamber_id", "chamber", "id")
                    if column in df.columns
                ),
                None,
            )

            if chamber_col is None:
                self._log(
                    f"   [Skip] Chamber column missing: {file_path}"
                )
                continue

            missing_chambers = df[chamber_col].isna()
            missing_count = int(missing_chambers.sum())

            if missing_count:
                self._log(
                    f"   [Warning] Removed {missing_count} row(s) "
                    f"with missing chamber IDs: {file_path}"
                )
                df = df.loc[~missing_chambers].copy()

            duplicates = df.duplicated(
                subset=[chamber_col],
                keep="first",
            )
            duplicate_count = int(duplicates.sum())

            if duplicate_count:
                self._log(
                    f"   [Warning] Found {duplicate_count} duplicate "
                    f"chamber row(s); keeping the first: {file_path}"
                )
                df = df.loc[~duplicates].copy()

            if df.empty:
                self._log(
                    f"   [Skip] No valid chamber rows: {file_path}"
                )
                continue

            df = df.set_index(chamber_col)

            matched_metrics = {}
            for metric, candidates in self.target_metrics.items():
                matched_column = next(
                    (
                        column
                        for column in candidates
                        if column in df.columns
                    ),
                    None,
                )

                if matched_column is None:
                    self._log(
                        f"   [Warning] Metric '{metric}' missing: "
                        f"{file_path}"
                    )
                else:
                    matched_metrics[metric] = matched_column

            if not matched_metrics:
                self._log(
                    f"   [Skip] No target metrics found: {file_path}"
                )
                continue

            # Valid timestamps produce headers such as day_20261017_100000.
            base_header = f"{group}_{identifier}"
            unique_header = base_header
            counter = 1

            while unique_header in used_headers:
                unique_header = f"{base_header}_{counter}"
                counter += 1

            if unique_header != base_header:
                self._log(
                    f"   [Warning] Repeated experiment identifier; "
                    f"using '{unique_header}': {file_path}"
                )

            used_headers.add(unique_header)

            for metric, matched_column in matched_metrics.items():
                metric_series[metric][unique_header] = (
                    df[matched_column].copy()
                )

            group_counts[group] += 1
            processed_count += 1

        self._log(
            f"\n[*] Extracted metrics from "
            f"{processed_count} summary file(s)."
        )

        for group, count in group_counts.items():
            self._log(f"   {group}: {count}")

        self._log("\n[*] Writing metric tables...")

        for metric, columns in metric_series.items():
            if not columns:
                continue

            merged_df = pd.DataFrame(columns)

            try:
                merged_df = merged_df.sort_index()
            except TypeError:
                # Keep mixed numeric/text chamber IDs sortable.
                merged_df = merged_df.sort_index(
                    key=lambda index: index.map(str)
                )

            sorted_columns = sorted(
                merged_df.columns,
                key=self._summary_column_sort_key,
            )
            merged_df = merged_df.reindex(
                columns=sorted_columns
            )
            merged_df.index.name = "chamber_id"

            output_path = self._available_output_path(
                out_dir,
                f"{metric}_summary.csv",
            )
            merged_df.to_csv(output_path)

            self._log(
                f"   [+] Saved: {os.path.basename(output_path)}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = SessionDataExtractorApp(root)
    root.mainloop()