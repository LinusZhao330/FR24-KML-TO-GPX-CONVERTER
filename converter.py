import os
import re
import sys
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import click
import xml.etree.ElementTree as ET


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


@dataclass
class TrackPoint:
    time: datetime
    lat: float
    lon: float
    ele: float


@dataclass
class ConversionResult:
    success: bool
    message: str
    point_count: int = 0
    output_path: Optional[str] = None
    warnings: Tuple[str, ...] = tuple()


def parse_iso8601(text: str) -> datetime:
    cleaned = text.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


def to_iso8601(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def try_parse_alt_from_description(html: str) -> Optional[float]:
    if not html:
        return None
    match = re.search(r"Altitude:\s*</?[^>]*>\s*([\d.]+)\s*ft", html, re.IGNORECASE)
    if match:
        feet = float(match.group(1))
        return feet * 0.3048
    return None


def extract_points(kml_file: str) -> Tuple[List[TrackPoint], Optional[str]]:
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
    except Exception as exc:  # noqa: BLE001
        return [], f"Failed to parse KML: {exc}"

    route_folder = None
    for folder in root.findall(".//kml:Document/kml:Folder", KML_NS):
        name_el = folder.find("kml:name", KML_NS)
        if name_el is not None and (name_el.text or "").strip() == "Route":
            route_folder = folder
            break

    if route_folder is None:
        return [], "No Folder named 'Route' found"

    points: List[TrackPoint] = []
    for placemark in route_folder.findall("kml:Placemark", KML_NS):
        ts_el = placemark.find(".//kml:TimeStamp/kml:when", KML_NS)
        coord_el = placemark.find(".//kml:Point/kml:coordinates", KML_NS)
        desc_el = placemark.find(".//kml:description", KML_NS)
        if ts_el is None or coord_el is None or not coord_el.text:
            continue

        try:
            when = parse_iso8601(ts_el.text)
        except Exception:  # noqa: BLE001
            continue

        try:
            lon_str, lat_str, alt_str = re.split(r"\s*,\s*", coord_el.text.strip())
            lat = float(lat_str)
            lon = float(lon_str)
            alt = float(alt_str)
        except Exception:  # noqa: BLE001
            continue

        altitude_override = try_parse_alt_from_description(desc_el.text if desc_el is not None else "")
        if altitude_override is not None:
            alt = altitude_override

        points.append(TrackPoint(time=when, lat=lat, lon=lon, ele=alt))

    points.sort(key=lambda p: p.time)
    if not points:
        return [], "No valid points under 'Route'"

    return points, None


def apply_target_date(points: List[TrackPoint], target_date: str) -> Tuple[List[TrackPoint], Optional[str]]:
    if not target_date:
        return points, None

    try:
        year, month, day = map(int, target_date.strip().split("-"))
        tz8 = timezone(timedelta(hours=8))
    except Exception as exc:  # noqa: BLE001
        return points, f"Invalid target date, using original timestamps ({exc})"

    adjusted: List[TrackPoint] = []
    for point in points:
        local_time = point.time.astimezone(tz8)
        try:
            new_local = local_time.replace(year=year, month=month, day=day)
        except ValueError as exc:  # Invalid calendar date
            return points, f"Target date is out of range ({exc})"
        adjusted.append(
            TrackPoint(
                time=new_local.astimezone(timezone.utc),
                lat=point.lat,
                lon=point.lon,
                ele=point.ele,
            )
        )

    return adjusted, None


def interpolate_segment(start: TrackPoint, end: TrackPoint, interval: int) -> List[TrackPoint]:
    interpolated: List[TrackPoint] = []
    if interval <= 0:
        return interpolated

    total_seconds = (end.time - start.time).total_seconds()
    if total_seconds <= interval:
        return interpolated

    steps = int(total_seconds // interval)
    for step in range(1, steps):
        current_time = start.time + timedelta(seconds=interval * step)
        if current_time >= end.time:
            break
        fraction = (current_time - start.time).total_seconds() / total_seconds
        interpolated.append(
            TrackPoint(
                time=current_time,
                lat=start.lat + (end.lat - start.lat) * fraction,
                lon=start.lon + (end.lon - start.lon) * fraction,
                ele=start.ele + (end.ele - start.ele) * fraction,
            )
        )

    return interpolated


def apply_interpolation(points: List[TrackPoint], interval: int) -> List[TrackPoint]:
    if interval <= 0:
        return points

    expanded: List[TrackPoint] = []
    for idx in range(len(points) - 1):
        current_point = points[idx]
        expanded.append(current_point)
        expanded.extend(interpolate_segment(current_point, points[idx + 1], interval))

    expanded.append(points[-1])
    return expanded


def write_gpx(points: List[TrackPoint], gpx_file: str) -> Optional[str]:
    gpx = ET.Element("gpx", attrib={"version": "1.1", "xmlns": "http://www.topografix.com/GPX/1/1"})
    track = ET.SubElement(gpx, "trk")
    segment = ET.SubElement(track, "trkseg")

    for point in points:
        trkpt = ET.SubElement(
            segment,
            "trkpt",
            attrib={"lat": f"{point.lat:.6f}", "lon": f"{point.lon:.6f}"},
        )
        ET.SubElement(trkpt, "ele").text = f"{point.ele:.2f}"
        ET.SubElement(trkpt, "time").text = to_iso8601(point.time)

    try:
        xml_bytes = ET.tostring(gpx, encoding="utf-8")
        with open(gpx_file, "wb") as handle:
            handle.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            handle.write(xml_bytes)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to write GPX: {exc}"

    return None


def convert(kml_file: str, gpx_file: str, interval: int = 60, target_date: str = "") -> ConversionResult:
    points, error = extract_points(kml_file)
    if error:
        return ConversionResult(success=False, message=error)

    warnings: List[str] = []

    adjusted_points, warning = apply_target_date(points, target_date)
    if warning:
        warnings.append(warning)
    else:
        points = adjusted_points

    points = apply_interpolation(points, interval)

    write_error = write_gpx(points, gpx_file)
    if write_error:
        return ConversionResult(success=False, message=write_error)

    message = f"Wrote GPX: {gpx_file} (points: {len(points)})"
    return ConversionResult(
        success=True,
        message=message,
        point_count=len(points),
        output_path=gpx_file,
        warnings=tuple(warnings),
    )


def should_pause_after_run() -> bool:
    if getattr(sys, "frozen", False) and not sys.argv[1:]:
        stdin = getattr(sys, "stdin", None)
        return bool(stdin and stdin.isatty())
    return False


def get_default_output_path(kml_path: str) -> str:
    return os.path.splitext(kml_path)[0] + ".gpx"


def should_launch_gui() -> bool:
    args = sys.argv[1:]
    if "--cli" in args:
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg != "--cli"]
        return False
    if "--gui" in args:
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg != "--gui"]
        return True
    return getattr(sys, "frozen", False) and not args


def reveal_in_explorer(path: str) -> None:
    try_path = path if os.path.isdir(path) else os.path.dirname(path)
    if not try_path:
        try_path = os.getcwd()

    try:
        if sys.platform.startswith("win"):
            if os.path.isfile(path):
                subprocess.run(["explorer", "/select,", os.path.normpath(path)], check=False)
            else:
                os.startfile(os.path.normpath(try_path))  # type: ignore[arg-type]
        elif sys.platform == "darwin":
            subprocess.run(["open", path if os.path.isfile(path) else try_path], check=False)
        else:
            subprocess.run(["xdg-open", try_path], check=False)
    except Exception:  # noqa: BLE001
        pass


def format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def run_gui() -> bool:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except Exception:  # noqa: BLE001
        print("Tkinter is not available; GUI mode cannot be used.")
        return False

    class ConverterGUI:
        def __init__(self) -> None:
            self.tk = tk
            self.ttk = ttk
            self.filedialog = filedialog
            self.messagebox = messagebox
            self.scrolledtext = scrolledtext

            self.root = tk.Tk()
            self.root.title("FR24 KML to GPX Converter")
            self.root.configure(bg="#e5e9f2")

            self.kml_var = tk.StringVar()
            self.gpx_var = tk.StringVar()
            self.interval_var = tk.StringVar(value="60")
            self.date_var = tk.StringVar()
            self.open_after_var = tk.BooleanVar(value=True)
            self.status_var = tk.StringVar(value="Ready - default interpolation 60 s")

            self._last_default_output: str = ""
            self._last_preview_points: List[TrackPoint] = []
            self._busy = False
            self._setup_style()
            self._build_ui()

        def _setup_style(self) -> None:
            style = self.ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")

            style.configure("App.TFrame", background="#f5f7fb")
            style.configure("Header.TFrame", background="#1e3a8a")
            style.configure("HeaderTitle.TLabel", background="#1e3a8a", foreground="#ffffff", font=("Segoe UI", 17, "bold"))
            style.configure("HeaderSubtitle.TLabel", background="#1e3a8a", foreground="#bfdbfe", font=("Segoe UI", 10))
            style.configure("Card.TLabelframe", background="#f5f7fb", borderwidth=0)
            style.configure("Card.TLabelframe.Label", background="#f5f7fb", foreground="#1f2937", font=("Segoe UI", 10, "bold"))
            style.configure("Subtitle.TLabel", background="#f5f7fb", foreground="#4b5563", font=("Segoe UI", 10))
            style.configure("Status.TLabel", background="#f5f7fb", foreground="#465266", font=("Segoe UI", 9))
            style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#2563eb", foreground="#ffffff")
            style.map(
                "Accent.TButton",
                background=[("pressed", "#1e40af"), ("active", "#1d4ed8")],
                foreground=[("disabled", "#cbd5f5"), ("active", "#ffffff")],
            )
            style.configure("Horizontal.TProgressbar", background="#2563eb", troughcolor="#dbeafe", bordercolor="#dbeafe", lightcolor="#2563eb", darkcolor="#1d4ed8")

        def _build_ui(self) -> None:
            main = self.ttk.Frame(self.root, padding=(24, 22, 24, 22), style="App.TFrame")
            main.pack(fill="both", expand=True)
            main.columnconfigure(0, weight=1)
            main.rowconfigure(7, weight=1)

            header = self.ttk.Frame(main, style="Header.TFrame", padding=(24, 18, 24, 18))
            header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
            header.columnconfigure(0, weight=1)
            self.ttk.Label(header, text="FR24 KML to GPX", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
            self.ttk.Label(
                header,
                text="Convert FlightRadar24 exports into GPX tracks with interpolation and timezone controls.",
                style="HeaderSubtitle.TLabel",
            ).grid(row=1, column=0, sticky="w", pady=(6, 0))

            files_group = self.ttk.Labelframe(main, text="Files", padding=(18, 14), style="Card.TLabelframe")
            files_group.grid(row=1, column=0, sticky="ew", pady=(0, 12))
            files_group.columnconfigure(1, weight=1)

            self.ttk.Label(files_group, text="KML file:").grid(row=0, column=0, sticky="w")
            self.kml_entry = self.ttk.Entry(files_group, textvariable=self.kml_var, width=52)
            self.kml_entry.grid(row=0, column=1, sticky="ew", padx=(10, 8))
            self.ttk.Button(files_group, text="Browse...", command=self.browse_kml).grid(row=0, column=2, sticky="ew")

            self.ttk.Label(files_group, text="Output GPX:").grid(row=1, column=0, sticky="w", pady=(10, 0))
            self.gpx_entry = self.ttk.Entry(files_group, textvariable=self.gpx_var, width=52)
            self.gpx_entry.grid(row=1, column=1, sticky="ew", padx=(10, 8), pady=(10, 0))
            self.ttk.Button(files_group, text="Save As...", command=self.browse_gpx).grid(row=1, column=2, sticky="ew", pady=(10, 0))

            settings_group = self.ttk.Labelframe(main, text="Options", padding=(18, 14), style="Card.TLabelframe")
            settings_group.grid(row=2, column=0, sticky="ew", pady=(0, 12))
            settings_group.columnconfigure(1, weight=1)

            self.ttk.Label(settings_group, text="Interpolation (seconds):").grid(row=0, column=0, sticky="w")
            spinbox_cls = self.ttk.Spinbox if hasattr(self.ttk, "Spinbox") else self.tk.Spinbox
            self.interval_input = spinbox_cls(
                settings_group,
                from_=0,
                to=7200,
                increment=15,
                textvariable=self.interval_var,
                width=10,
            )
            self.interval_input.grid(row=0, column=1, sticky="w", padx=(10, 0))

            self.ttk.Label(settings_group, text="Target date (UTC+8, YYYY-MM-DD):").grid(row=1, column=0, sticky="w", pady=(10, 0))
            self.ttk.Entry(settings_group, textvariable=self.date_var, width=18).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
            self.ttk.Button(settings_group, text="Use preview date", command=self.use_preview_date).grid(row=1, column=2, sticky="e", pady=(10, 0))

            self.ttk.Checkbutton(
                settings_group,
                text="Open folder after conversion",
                variable=self.open_after_var,
            ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))

            helper = self.ttk.Label(
                main,
                text="Tip: Preview loads the flight summary and auto-fills the target date.",
                style="Subtitle.TLabel",
            )
            helper.grid(row=3, column=0, sticky="w", pady=(0, 8))

            actions_frame = self.ttk.Frame(main, style="App.TFrame")
            actions_frame.grid(row=4, column=0, sticky="ew")
            actions_frame.columnconfigure((0, 1, 2), weight=1)

            self.preview_btn = self.ttk.Button(actions_frame, text="Preview", command=self.on_preview)
            self.preview_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

            self.convert_btn = self.ttk.Button(actions_frame, text="Convert", style="Accent.TButton", command=self.on_convert)
            self.convert_btn.grid(row=0, column=1, sticky="ew", padx=8)

            self.reset_btn = self.ttk.Button(actions_frame, text="Reset", command=self.on_reset)
            self.reset_btn.grid(row=0, column=2, sticky="ew", padx=(8, 0))

            progress_frame = self.ttk.Frame(main, style="App.TFrame")
            progress_frame.grid(row=5, column=0, sticky="ew", pady=(16, 6))
            progress_frame.columnconfigure(0, weight=1)

            self.progress = self.ttk.Progressbar(progress_frame, mode="indeterminate")
            self.progress.grid(row=0, column=0, sticky="ew")
            self.progress.grid_remove()

            self.status_label = self.ttk.Label(main, textvariable=self.status_var, style="Status.TLabel")
            self.status_label.grid(row=6, column=0, sticky="w")

            log_group = self.ttk.Labelframe(main, text="Activity log", padding=(18, 14), style="Card.TLabelframe")
            log_group.grid(row=7, column=0, sticky="nsew", pady=(10, 0))
            log_group.columnconfigure(0, weight=1)
            log_group.rowconfigure(0, weight=1)

            self.log_text = self.scrolledtext.ScrolledText(
                log_group,
                height=10,
                width=60,
                wrap="word",
                font=("Consolas", 9),
                background="#111827",
                foreground="#f8fafc",
                relief="flat",
                borderwidth=0,
                state="disabled",
                insertbackground="#f8fafc",
                padx=10,
                pady=10,
            )
            self.log_text.grid(row=0, column=0, sticky="nsew")

        def set_busy(self, busy: bool) -> None:
            self._busy = busy
            state = "disabled" if busy else "normal"
            for widget in (self.preview_btn, self.convert_btn, self.reset_btn, self.kml_entry, self.gpx_entry, self.interval_input):
                try:
                    widget.configure(state=state)
                except Exception:  # noqa: BLE001
                    pass
            if busy:
                self.progress.grid()
                self.progress.start(8)
            else:
                self.progress.stop()
                self.progress.grid_remove()

        def append_log(self, message: str, level: str = "info") -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {
                "info": "INFO",
                "success": "OK",
                "warning": "WARN",
                "error": "ERR",
            }.get(level, "INFO")
            colored = {
                "info": "#f8fafc",
                "success": "#34d399",
                "warning": "#facc15",
                "error": "#f87171",
            }.get(level, "#f8fafc")

            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"[{timestamp}] {prefix}: {message}\n", level)
            self.log_text.tag_config(level, foreground=colored)
            self.log_text.configure(state="disabled")
            self.log_text.see("end")

        def clear_log(self) -> None:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")

        def _apply_target_date_from_points(self, points: List[TrackPoint], *, announce: bool, context: str) -> None:
            tz8 = timezone(timedelta(hours=8))
            suggested = points[0].time.astimezone(tz8).strftime("%Y-%m-%d")
            self.date_var.set(suggested)
            if announce:
                self.set_status(f"{context}: {suggested}", "success")
            self.append_log(f"{context}: {suggested}")

        def _auto_populate_date_from_path(self, path: str, announce: bool = False) -> bool:
            points, error = extract_points(path)
            if error:
                if announce:
                    self.set_status(error, "warning")
                self.append_log(error, "warning")
                return False
            self._last_preview_points = points
            self._apply_target_date_from_points(points, announce=announce, context="Target date auto-filled")
            return True

        def set_status(self, message: str, status: str = "info") -> None:
            colors = {
                "info": "#475569",
                "success": "#047857",
                "warning": "#b45309",
                "error": "#b91c1c",
            }
            self.status_var.set(message)
            self.status_label.configure(foreground=colors.get(status, "#475569"))

        def browse_kml(self) -> None:
            path = self.filedialog.askopenfilename(
                title="Select KML file",
                filetypes=[("KML files", "*.kml"), ("All files", "*.*")],
            )
            if not path:
                return
            self.kml_var.set(path)
            default_output = get_default_output_path(path)
            current_output = self.gpx_var.get().strip()
            if not current_output or current_output == self._last_default_output:
                self.gpx_var.set(default_output)
            self._last_default_output = default_output
            self.append_log(f"Selected KML: {path}")
            if self._auto_populate_date_from_path(path, announce=True):
                self.set_status("KML ready. Target date auto-filled.", "success")

        def browse_gpx(self) -> None:
            initial = self.gpx_var.get().strip() or "output.gpx"
            path = self.filedialog.asksaveasfilename(
                title="Save GPX as",
                defaultextension=".gpx",
                initialfile=os.path.basename(initial),
                filetypes=[("GPX files", "*.gpx"), ("All files", "*.*")],
            )
            if path:
                self.gpx_var.set(path)
                self.set_status("Ready", "info")
                self.append_log(f"Selected output: {path}")

        def parse_interval(self) -> Optional[int]:
            value = self.interval_var.get().strip()
            if not value:
                return 60
            try:
                number = int(value)
            except ValueError:
                return None
            if number < 0:
                return None
            return number

        def use_preview_date(self) -> None:
            if self._last_preview_points:
                self._apply_target_date_from_points(self._last_preview_points, announce=True, context="Target date from preview")
                return

            path = self.kml_var.get().strip()
            if not path:
                self.messagebox.showinfo("Preview required", "Load a KML file first so the date can be suggested.")
                return
            if not self._auto_populate_date_from_path(path, announce=True):
                self.messagebox.showerror("Cannot fetch date", "Unable to determine date from KML. Please run Preview first.")
                return

        def on_preview(self) -> None:
            if self._busy:
                return
            kml_path = self.kml_var.get().strip()
            if not kml_path:
                self.messagebox.showwarning("Missing KML", "Please select a KML file to preview.")
                return
            if not os.path.isfile(kml_path):
                self.messagebox.showerror("Invalid KML", "The selected KML file does not exist.")
                return

            self.set_status("Analyzing KML...", "info")
            points, error = extract_points(kml_path)
            if error:
                self.set_status(error, "error")
                self.append_log(error, "error")
                self.messagebox.showerror("Preview failed", error)
                return

            self._last_preview_points = points
            self._apply_target_date_from_points(points, announce=False, context="Preview target date")
            first_point = points[0]
            last_point = points[-1]
            duration = last_point.time - first_point.time
            altitude_min = min(p.ele for p in points)
            altitude_max = max(p.ele for p in points)
            preview_lines = [
                f"Preview: {len(points)} points detected",
                f"  First point (UTC): {first_point.time.isoformat()}",
                f"  Last point (UTC):  {last_point.time.isoformat()}",
                f"  Duration: {format_timedelta(duration)}",
                f"  Altitude range: {altitude_min:.0f} m → {altitude_max:.0f} m",
            ]
            self.append_log("\n".join(preview_lines), "success")
            self.set_status(f"Preview ready ({len(points)} points)", "success")

        def on_convert(self) -> None:
            if self._busy:
                return

            kml_path = self.kml_var.get().strip()
            if not kml_path:
                self.messagebox.showwarning("Missing KML", "Please select a KML file to convert.")
                return
            if not os.path.isfile(kml_path):
                self.messagebox.showerror("Invalid KML", "The selected KML file does not exist.")
                return

            gpx_path = self.gpx_var.get().strip() or get_default_output_path(kml_path)
            if not gpx_path.lower().endswith(".gpx"):
                gpx_path += ".gpx"

            interval_value = self.parse_interval()
            if interval_value is None:
                self.messagebox.showwarning("Invalid interval", "Interpolation interval must be a non-negative integer.")
                self.interval_var.set("60")
                return

            target_date = self.date_var.get().strip()

            self.append_log(f"Starting conversion: interval={interval_value}s, target_date='{target_date or 'n/a'}'")
            self.set_status("Conversion running...", "info")
            self.set_busy(True)

            thread = threading.Thread(
                target=self._convert_async,
                args=(kml_path, gpx_path, interval_value, target_date, self.open_after_var.get()),
                daemon=True,
            )
            thread.start()

        def _convert_async(self, kml_path: str, gpx_path: str, interval: int, target_date: str, open_after: bool) -> None:
            result = convert(kml_path, gpx_path, interval, target_date)
            self.root.after(0, lambda: self._on_conversion_finished(result, open_after))

        def _on_conversion_finished(self, result: ConversionResult, open_after: bool) -> None:
            self.set_busy(False)

            if result.success:
                self.gpx_var.set(result.output_path or self.gpx_var.get())
                self.set_status(f"Conversion complete ({result.point_count} points)", "success")
                self.append_log(result.message, "success")
                for warning in result.warnings:
                    self.append_log(warning, "warning")
                self.messagebox.showinfo("Conversion complete", result.message)
                if open_after and result.output_path:
                    reveal_in_explorer(result.output_path)
            else:
                self.set_status(result.message, "error")
                self.append_log(result.message, "error")
                self.messagebox.showerror("Conversion failed", result.message)

        def on_reset(self) -> None:
            if self._busy:
                return
            self.kml_var.set("")
            self.gpx_var.set("")
            self.interval_var.set("60")
            self.date_var.set("")
            self.open_after_var.set(True)
            self._last_default_output = ""
            self._last_preview_points = []
            self.clear_log()
            self.set_status("Reset complete. Select a KML file to begin.", "info")

        def run(self) -> None:
            self.root.minsize(640, 520)
            self.root.mainloop()

    ConverterGUI().run()
    return True


@click.command()
@click.option("--kml", "kml_path", prompt="Enter KML file path", type=click.Path(exists=True, dir_okay=False))
@click.option("--gpx", "gpx_path", prompt="Enter output GPX path", default=lambda: "output.gpx", show_default=True)
@click.option("--interval", prompt="Interpolation interval (seconds), Enter to skip", default=60, type=int, show_default=True)
@click.option("--date", "target_date", prompt="Target date (UTC+8, YYYY-MM-DD), Enter to skip", default="", show_default=False)
def cli(kml_path: str, gpx_path: str, interval: int, target_date: str) -> None:
    if gpx_path.strip() == "output.gpx":
        gpx_path = get_default_output_path(kml_path)

    result = convert(kml_path, gpx_path, interval, target_date)
    if result.success:
        click.echo(result.message)
    else:
        click.secho(result.message, fg="red", err=True)
    for warning in result.warnings:
        click.secho(f"Warning: {warning}", fg="yellow")

    exit_code = 0 if result.success else 1
    if should_pause_after_run():
        prompt = "Conversion finished, press Enter to exit..." if result.success else "Process finished, press Enter to exit..."
        try:
            input(prompt)
        except EOFError:
            pass
    if exit_code:
        sys.exit(exit_code)


def run_cli() -> None:
    try:
        cli.main(standalone_mode=False)
    except KeyboardInterrupt:
        print("Cancelled")
        sys.exit(1)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except SystemExit as exc:
        sys.exit(exc.code)


if __name__ == "__main__":
    if should_launch_gui():
        if run_gui():
            sys.exit(0)
    run_cli()
