import bisect
import os
import re
import sys
import math
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import click
import requests
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
    points: Optional[List[TrackPoint]] = None


@dataclass(frozen=True)
class LocationCandidate:
    query: str
    display_name: str
    lat: float
    lon: float


@dataclass
class NavigationPlan:
    success: bool
    message: str
    points: List[TrackPoint]
    start_label: str = ""
    end_label: str = ""
    distance_km: float = 0.0
    duration_s: float = 0.0
    warnings: Tuple[str, ...] = tuple()
    start_candidates: Tuple[LocationCandidate, ...] = tuple()
    dest_candidates: Tuple[LocationCandidate, ...] = tuple()


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
        points=points,
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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def compute_total_distance(points: List[TrackPoint]) -> float:
    total = 0.0
    if len(points) < 2:
        return total
    for start, end in zip(points, points[1:]):
        total += haversine_distance(start.lat, start.lon, end.lat, end.lon)
    return total


def geocode_location(query: str, limit: int = 5) -> Tuple[List[LocationCandidate], Optional[str]]:
    cleaned = (query or "").strip()
    if not cleaned:
        return [], "Empty search query."

    headers = {"User-Agent": "FR24-to-GPX/1.0 (+https://github.com/zzk90/FR24_to_GPX)"}
    params = {"q": cleaned, "format": "jsonv2", "limit": str(limit), "addressdetails": "0"}
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=12)
    except requests.RequestException as exc:  # noqa: BLE001
        return [], f"Geocoding request failed: {exc}"

    if response.status_code != 200:
        return [], f"Geocoding service error (status {response.status_code})."

    try:
        data = response.json()
    except ValueError as exc:  # noqa: BLE001
        return [], f"Unable to parse geocoding response: {exc}"

    candidates: List[LocationCandidate] = []
    for item in data:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        display = item.get("display_name") or cleaned
        candidates.append(LocationCandidate(query=cleaned, display_name=display, lat=lat, lon=lon))

    if not candidates:
        return [], f"No results found for '{cleaned}'."

    warning = None
    if len(candidates) > 1:
        warning = f"Multiple matches found for '{cleaned}'. Using the best match."

    return candidates, warning


def fetch_route_osrm(start: LocationCandidate, end: LocationCandidate) -> Tuple[List[Tuple[float, float]], float, float, Optional[str]]:
    url = f"https://router.project-osrm.org/route/v1/driving/{start.lon},{start.lat};{end.lon},{end.lat}"
    params = {"overview": "full", "geometries": "geojson", "annotations": "false"}
    headers = {"User-Agent": "FR24-to-GPX/1.0 (+https://github.com/zzk90/FR24_to_GPX)"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
    except requests.RequestException as exc:  # noqa: BLE001
        return [], 0.0, 0.0, f"Routing request failed: {exc}"

    if response.status_code != 200:
        return [], 0.0, 0.0, f"Routing service error (status {response.status_code})."

    try:
        payload = response.json()
    except ValueError as exc:  # noqa: BLE001
        return [], 0.0, 0.0, f"Unable to parse routing response: {exc}"

    if payload.get("code") != "Ok":
        message = payload.get("message") or "Routing service returned an error."
        return [], 0.0, 0.0, message

    routes = payload.get("routes") or []
    if not routes:
        return [], 0.0, 0.0, "Routing service returned no routes."

    route = routes[0]
    geometry = route.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    if not coordinates:
        return [], 0.0, 0.0, "Routing data contained no coordinates."

    coords: List[Tuple[float, float]] = []
    for lon, lat in coordinates:
        try:
            coords.append((float(lat), float(lon)))
        except (TypeError, ValueError):
            continue

    if not coords:
        return [], 0.0, 0.0, "Routing coordinates were invalid."

    distance_km = float(route.get("distance", 0.0)) / 1000.0
    duration_s = float(route.get("duration", 0.0))
    return coords, distance_km, duration_s, None


def build_navigation_track(coords: List[Tuple[float, float]], duration: float, interval: int) -> List[TrackPoint]:
    if not coords:
        return []

    if len(coords) == 1:
        coords = [coords[0], coords[0]]

    cumulative: List[float] = [0.0]
    total_distance_m = 0.0
    for idx in range(1, len(coords)):
        lat0, lon0 = coords[idx - 1]
        lat1, lon1 = coords[idx]
        segment = haversine_distance(lat0, lon0, lat1, lon1) * 1000.0
        total_distance_m += max(segment, 0.0)
        cumulative.append(total_distance_m)

    if total_distance_m <= 0.0:
        step = 1.0 if len(coords) <= 1 else 1.0 / max(len(coords) - 1, 1)
        cumulative = [float(i) * step for i in range(len(coords))]
        total_distance_m = cumulative[-1] if cumulative else 0.0
        if total_distance_m <= 0.0:
            total_distance_m = 1.0

    duration = max(float(duration or 0.0), 0.0)
    if duration <= 0.0:
        assumed_speed_mps = 15.0  # ~54 km/h fallback
        duration = max(total_distance_m / assumed_speed_mps, 60.0 if total_distance_m > 0 else 1.0)

    interval_seconds = max(int(interval), 0)
    if interval_seconds == 0 and len(coords) > 1:
        step = duration / max(len(coords) - 1, 1)
        offsets = [min(duration, step * idx) for idx in range(len(coords))]
    elif interval_seconds == 0:
        offsets = [0.0]
    else:
        count = int(duration // interval_seconds)
        offsets = [float(i * interval_seconds) for i in range(count + 1)]
        if offsets[-1] < duration:
            offsets.append(duration)

    start_time = datetime.now(timezone.utc)
    result: List[TrackPoint] = []
    for offset in offsets:
        target_distance = (offset / duration) * total_distance_m if duration > 0 else 0.0
        if target_distance >= total_distance_m:
            lat, lon = coords[-1]
        else:
            index = bisect.bisect_right(cumulative, target_distance)
            if index >= len(coords):
                lat, lon = coords[-1]
            else:
                prev_idx = max(index - 1, 0)
                next_idx = min(index, len(coords) - 1)
                segment_start = cumulative[prev_idx]
                segment_end = cumulative[next_idx]
                if segment_end <= segment_start:
                    ratio = 0.0
                else:
                    ratio = (target_distance - segment_start) / (segment_end - segment_start)
                lat0, lon0 = coords[prev_idx]
                lat1, lon1 = coords[next_idx]
                lat = lat0 + (lat1 - lat0) * ratio
                lon = lon0 + (lon1 - lon0) * ratio
        point_time = start_time + timedelta(seconds=float(offset))
        result.append(TrackPoint(time=point_time, lat=lat, lon=lon, ele=0.0))

    if result:
        last = result[-1]
        end_lat, end_lon = coords[-1]
        if abs(last.lat - end_lat) > 1e-6 or abs(last.lon - end_lon) > 1e-6:
            final_time = start_time + timedelta(seconds=duration)
            result.append(TrackPoint(time=final_time, lat=end_lat, lon=end_lon, ele=0.0))

    return result


def plan_navigation_route(start_query: str, dest_query: str, interval: int) -> NavigationPlan:
    warnings: List[str] = []

    start_candidates, start_warning = geocode_location(start_query)
    if start_warning:
        warnings.append(start_warning)
    if not start_candidates:
        message = start_warning or f"Unable to find location for '{start_query}'."
        return NavigationPlan(success=False, message=message, points=[])

    dest_candidates, dest_warning = geocode_location(dest_query)
    if dest_warning:
        warnings.append(dest_warning)
    if not dest_candidates:
        message = dest_warning or f"Unable to find location for '{dest_query}'."
        return NavigationPlan(success=False, message=message, points=[], start_candidates=tuple(start_candidates))

    start = start_candidates[0]
    dest = dest_candidates[0]

    coords, distance_km, duration_s, route_error = fetch_route_osrm(start, dest)
    if route_error:
        warnings.append(route_error)
        return NavigationPlan(
            success=False,
            message=route_error,
            points=[],
            start_label=start.display_name,
            end_label=dest.display_name,
            warnings=tuple(warnings),
            start_candidates=tuple(start_candidates),
            dest_candidates=tuple(dest_candidates),
        )

    points = build_navigation_track(coords, duration_s, interval)
    if not points:
        message = "Route was found but no points could be generated."
        return NavigationPlan(
            success=False,
            message=message,
            points=[],
            start_label=start.display_name,
            end_label=dest.display_name,
            warnings=tuple(warnings),
            start_candidates=tuple(start_candidates),
            dest_candidates=tuple(dest_candidates),
        )

    message = (
        f"Generated route {start.display_name} → {dest.display_name}. "
        f"Distance: {distance_km:.1f} km, points: {len(points)}."
    )

    return NavigationPlan(
        success=True,
        message=message,
        points=points,
        start_label=start.display_name,
        end_label=dest.display_name,
        distance_km=distance_km,
        duration_s=duration_s,
        warnings=tuple(warnings),
        start_candidates=tuple(start_candidates),
        dest_candidates=tuple(dest_candidates),
    )


def run_gui() -> bool:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except Exception:  # noqa: BLE001
        print("Tkinter is not available; GUI mode cannot be used.")
        return False

    try:
        from tkintermapview import TkinterMapView  # type: ignore import
    except Exception:  # noqa: BLE001
        TkinterMapView = None

    try:
        import folium  # type: ignore import
    except Exception:  # noqa: BLE001
        folium = None

    import tempfile
    import webbrowser

    class ConverterGUI:
        def __init__(self) -> None:
            self.tk = tk
            self.ttk = ttk
            self.filedialog = filedialog
            self.messagebox = messagebox
            self.scrolledtext = scrolledtext
            self.MapViewClass = TkinterMapView
            self.folium = folium
            self.tempfile = tempfile
            self.webbrowser = webbrowser

            self.root = tk.Tk()
            self.root.title("FR24 KML to GPX Converter")
            self.root.configure(bg="#e5e9f2")

            self.kml_var = tk.StringVar()
            self.gpx_var = tk.StringVar()
            self.interval_var = tk.StringVar(value="60")
            self.date_var = tk.StringVar()
            self.open_after_var = tk.BooleanVar(value=True)
            self.nav_start_var = tk.StringVar()
            self.nav_dest_var = tk.StringVar()
            self.nav_interval_var = tk.StringVar(value="60")
            self.nav_summary_var = tk.StringVar(value="No route planned yet.")
            self.nav_detail_var = tk.StringVar(value="Enter start and destination to plan a route.")
            self.status_var = tk.StringVar(value="Ready - default interpolation 60 s")
            self.summary_points_var = tk.StringVar(value="Points: --")
            self.summary_duration_var = tk.StringVar(value="Duration: --")
            self.summary_altitude_var = tk.StringVar(value="Altitude: --")
            self.summary_localtime_var = tk.StringVar(value="Local time (UTC+8): --")
            self.summary_distance_var = tk.StringVar(value="Distance: --")
            self.summary_source_var = tk.StringVar(value="No route loaded")

            self._last_default_output: str = ""
            self._last_preview_points: List[TrackPoint] = []
            self._last_rendered_points: List[TrackPoint] = []
            self._busy = False
            self._last_output_path: str | None = None
            self._nav_points: List[TrackPoint] = []
            self._nav_plan: NavigationPlan | None = None
            if self.MapViewClass is not None:
                self.map_backend = "tkinter"
            elif self.folium is not None:
                self.map_backend = "folium"
            else:
                self.map_backend = None
            self.map_widget = None
            self.map_path = None
            self.map_markers: List[object] = []
            self.map_tab_note = None
            self.map_html_path: str | None = None
            self.map_button = None
            self.fit_map_btn = None
            self.nav_plan_btn = None
            self.nav_export_btn = None
            self.nav_start_entry = None
            self.nav_dest_entry = None
            self.nav_interval_entry = None
            self.nav_swap_btn = None

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
            style.configure("TNotebook", background="#f5f7fb", borderwidth=0)
            style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 6, 14, 4))
            style.map("TNotebook.Tab", background=[("selected", "#e0e7ff")])
            style.configure("Info.TFrame", background="#eef2ff", relief="flat")
            style.configure("InfoTitle.TLabel", background="#eef2ff", foreground="#1f2937", font=("Segoe UI", 11, "bold"))
            style.configure("InfoValue.TLabel", background="#eef2ff", foreground="#1e3a8a", font=("Segoe UI", 10, "bold"))
            style.configure("InfoLabel.TLabel", background="#eef2ff", foreground="#475569", font=("Segoe UI", 9))

        def _build_ui(self) -> None:
            self.root.rowconfigure(0, weight=1)
            self.root.columnconfigure(0, weight=1)

            container = self.ttk.Frame(self.root, style="App.TFrame")
            container.grid(row=0, column=0, sticky="nsew")
            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=1)

            canvas = self.tk.Canvas(container, highlightthickness=0, bg="#f5f7fb")
            vscroll = self.ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vscroll.set)

            canvas.grid(row=0, column=0, sticky="nsew")
            vscroll.grid(row=0, column=1, sticky="ns")

            main = self.ttk.Frame(canvas, padding=(24, 22, 24, 22), style="App.TFrame")
            canvas_window = canvas.create_window((0, 0), window=main, anchor="nw")

            self._scroll_container = container
            self._scroll_canvas = canvas
            self._scrollbar = vscroll
            self._scroll_window = canvas_window

            def _update_scroll_region(_: object) -> None:
                canvas.configure(scrollregion=canvas.bbox("all"))

            def _sync_frame_width(event: object) -> None:
                canvas.itemconfigure(canvas_window, width=canvas.winfo_width())

            def _is_descendant(widget: object, ancestor: object) -> bool:
                if not widget or not ancestor:
                    return False
                try:
                    current = widget
                    while current is not None:
                        if current is ancestor:
                            return True
                        parent_name = current.winfo_parent()
                        if not parent_name:
                            break
                        current = self.root.nametowidget(parent_name)
                except Exception:  # noqa: BLE001
                    return False
                return False

            def _should_ignore_scroll(widget: object) -> bool:
                if widget is None:
                    return False
                try:
                    widget_class = widget.winfo_class()
                except Exception:  # noqa: BLE001
                    widget_class = ""
                if widget_class in {"Text"}:
                    return True
                if self.map_backend == "tkinter" and self.map_widget is not None:
                    if _is_descendant(widget, self.map_widget):
                        return True
                return False

            def _on_mousewheel(event: object) -> str | None:
                widget = getattr(event, "widget", None)
                if _should_ignore_scroll(widget):
                    return None
                delta = getattr(event, "delta", 0)
                if delta:
                    if sys.platform == "darwin":
                        steps = -int(delta)
                    else:
                        steps = -int(delta / 120)
                        if steps == 0:
                            steps = -1 if delta > 0 else 1
                    if steps:
                        canvas.yview_scroll(steps, "units")
                return "break"

            def _on_mousewheel_linux(event: object) -> str | None:
                widget = getattr(event, "widget", None)
                if _should_ignore_scroll(widget):
                    return None
                direction = -1 if getattr(event, "num", 0) == 4 else 1
                canvas.yview_scroll(direction, "units")
                return "break"

            main.bind("<Configure>", _update_scroll_region, add="+")
            canvas.bind("<Configure>", _sync_frame_width, add="+")
            canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")
            canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel, add="+")
            canvas.bind_all("<Button-4>", _on_mousewheel_linux, add="+")
            canvas.bind_all("<Button-5>", _on_mousewheel_linux, add="+")
            self.root.after_idle(lambda: canvas.configure(scrollregion=canvas.bbox("all")))
            main.columnconfigure(0, weight=1)
            main.rowconfigure(9, weight=1)

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

            summary_frame = self.ttk.Frame(main, style="Info.TFrame", padding=(18, 12))
            summary_frame.grid(row=4, column=0, sticky="ew", pady=(0, 12))
            summary_frame.columnconfigure((0, 1, 2), weight=1)

            self.ttk.Label(summary_frame, text="Route overview", style="InfoTitle.TLabel").grid(row=0, column=0, sticky="w")
            self.ttk.Label(summary_frame, textvariable=self.summary_source_var, style="InfoLabel.TLabel").grid(row=0, column=2, sticky="e")

            self.ttk.Label(summary_frame, textvariable=self.summary_points_var, style="InfoValue.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
            self.ttk.Label(summary_frame, textvariable=self.summary_duration_var, style="InfoValue.TLabel").grid(row=1, column=1, sticky="w", pady=(6, 0))
            self.ttk.Label(summary_frame, textvariable=self.summary_distance_var, style="InfoValue.TLabel").grid(row=1, column=2, sticky="w", pady=(6, 0))

            self.ttk.Label(summary_frame, textvariable=self.summary_altitude_var, style="InfoLabel.TLabel").grid(row=2, column=0, sticky="w", pady=(2, 0))
            self.ttk.Label(summary_frame, textvariable=self.summary_localtime_var, style="InfoLabel.TLabel").grid(row=2, column=1, sticky="w", pady=(2, 0))

            self.open_output_btn = self.ttk.Button(summary_frame, text="Open output folder", command=self.open_output_location, state="disabled")
            self.open_output_btn.grid(row=2, column=2, sticky="e", pady=(2, 0))

            nav_group = self.ttk.Labelframe(main, text="Navigation planner", padding=(18, 14), style="Card.TLabelframe")
            nav_group.grid(row=5, column=0, sticky="ew", pady=(0, 12))
            nav_group.columnconfigure(1, weight=1)
            nav_group.columnconfigure(2, weight=0)
            nav_group.columnconfigure(3, weight=0)

            self.ttk.Label(nav_group, text="Start location:").grid(row=0, column=0, sticky="w")
            self.nav_start_entry = self.ttk.Entry(nav_group, textvariable=self.nav_start_var, width=48)
            self.nav_start_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 8))
            self.nav_swap_btn = self.ttk.Button(nav_group, text="Swap", command=self.swap_navigation_endpoints)
            self.nav_swap_btn.grid(row=0, column=3, sticky="e")

            self.ttk.Label(nav_group, text="Destination:").grid(row=1, column=0, sticky="w", pady=(10, 0))
            self.nav_dest_entry = self.ttk.Entry(nav_group, textvariable=self.nav_dest_var, width=48)
            self.nav_dest_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=(10, 0))

            self.ttk.Label(nav_group, text="GPS point interval (s):").grid(row=2, column=0, sticky="w", pady=(10, 0))
            self.nav_interval_entry = self.ttk.Entry(nav_group, textvariable=self.nav_interval_var, width=10)
            self.nav_interval_entry.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
            self.nav_plan_btn = self.ttk.Button(nav_group, text="Plan route", command=self.on_plan_navigation)
            self.nav_plan_btn.grid(row=2, column=2, sticky="ew", padx=(10, 4), pady=(10, 0))
            self.nav_export_btn = self.ttk.Button(nav_group, text="Export GPX", command=self.on_export_navigation, state="disabled")
            self.nav_export_btn.grid(row=2, column=3, sticky="ew", pady=(10, 0))

            self.ttk.Label(nav_group, textvariable=self.nav_summary_var, style="Subtitle.TLabel").grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))
            self.ttk.Label(
                nav_group,
                textvariable=self.nav_detail_var,
                style="InfoLabel.TLabel",
                wraplength=520,
                justify="left",
            ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

            actions_frame = self.ttk.Frame(main, style="App.TFrame")
            actions_frame.grid(row=6, column=0, sticky="ew")
            actions_frame.columnconfigure((0, 1, 2), weight=1)

            self.preview_btn = self.ttk.Button(actions_frame, text="Preview", command=self.on_preview)
            self.preview_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

            self.convert_btn = self.ttk.Button(actions_frame, text="Convert", style="Accent.TButton", command=self.on_convert)
            self.convert_btn.grid(row=0, column=1, sticky="ew", padx=8)

            self.reset_btn = self.ttk.Button(actions_frame, text="Reset", command=self.on_reset)
            self.reset_btn.grid(row=0, column=2, sticky="ew", padx=(8, 0))

            progress_frame = self.ttk.Frame(main, style="App.TFrame")
            progress_frame.grid(row=7, column=0, sticky="ew", pady=(16, 6))
            progress_frame.columnconfigure(0, weight=1)

            self.progress = self.ttk.Progressbar(progress_frame, mode="indeterminate")
            self.progress.grid(row=0, column=0, sticky="ew")
            self.progress.grid_remove()

            self.status_label = self.ttk.Label(main, textvariable=self.status_var, style="Status.TLabel")
            self.status_label.grid(row=8, column=0, sticky="w")

            notebook = self.ttk.Notebook(main)
            notebook.grid(row=9, column=0, sticky="nsew", pady=(10, 0))

            log_tab = self.ttk.Frame(notebook, style="App.TFrame")
            log_tab.columnconfigure(0, weight=1)
            log_tab.rowconfigure(0, weight=1)

            self.log_text = self.scrolledtext.ScrolledText(
                log_tab,
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
            notebook.add(log_tab, text="Activity Log")

            if self.map_backend == "tkinter":
                map_tab = self.ttk.Frame(notebook, style="App.TFrame")
                map_tab.columnconfigure(0, weight=1)
                map_tab.rowconfigure(1, weight=1)

                controls = self.ttk.Frame(map_tab, style="App.TFrame")
                controls.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
                self.fit_map_btn = self.ttk.Button(
                    controls,
                    text="Fit route to view",
                    command=self.fit_map_to_route,
                    state="disabled",
                )
                self.fit_map_btn.pack(side="left")

                self.map_widget = self.MapViewClass(map_tab, corner_radius=0)
                self.map_widget.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
                try:
                    self.map_widget.set_zoom(2)
                    self.map_widget.set_position(20.0, 0.0)
                except Exception:  # noqa: BLE001
                    pass
                notebook.add(map_tab, text="Map Preview")
                self.map_tab_note = None
            elif self.map_backend == "folium":
                info_tab = self.ttk.Frame(notebook, style="App.TFrame")
                info_tab.columnconfigure(0, weight=1)
                info_tab.rowconfigure(1, weight=1)
                self.map_tab_note = self.ttk.Label(
                    info_tab,
                    text="Map preview will open in your browser using folium.",
                    style="Subtitle.TLabel",
                    wraplength=520,
                    anchor="center",
                    justify="center",
                )
                self.map_tab_note.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 12))
                self.map_button = self.ttk.Button(
                    info_tab,
                    text="Open Map Preview (browser)",
                    command=self.open_map_preview,
                    state="disabled",
                )
                self.map_button.grid(row=1, column=0, sticky="n", pady=(0, 24))
                notebook.add(info_tab, text="Map Preview")
            else:
                info_tab = self.ttk.Frame(notebook, style="App.TFrame")
                info_tab.columnconfigure(0, weight=1)
                info_tab.rowconfigure(0, weight=1)
                self.map_tab_note = self.ttk.Label(
                    info_tab,
                    text="Install 'tkintermapview' for an embedded map, or 'folium' for a browser preview.",
                    style="Subtitle.TLabel",
                    wraplength=520,
                    anchor="center",
                    justify="center",
                )
                self.map_tab_note.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)
                notebook.add(info_tab, text="Map Preview")

        def set_busy(self, busy: bool) -> None:
            self._busy = busy
            state = "disabled" if busy else "normal"
            for widget in (
                self.preview_btn,
                self.convert_btn,
                self.reset_btn,
                self.kml_entry,
                self.gpx_entry,
                self.interval_input,
                self.nav_plan_btn,
                self.nav_export_btn,
                self.nav_start_entry,
                self.nav_dest_entry,
                self.nav_interval_entry,
                self.nav_swap_btn,
            ):
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

        def open_map_preview(self) -> None:
            if self.map_backend != "folium":
                return
            if not self.map_html_path:
                self.messagebox.showinfo("Map preview", "Generate a preview first by loading a KML.")
                return
            try:
                self.webbrowser.open_new_tab(os.path.abspath(self.map_html_path))
            except Exception as exc:  # noqa: BLE001
                self.messagebox.showerror("Map preview", f"Unable to open map: {exc}")

        def _clear_map(self) -> None:
            if self.map_backend == "tkinter":
                if not self.map_widget:
                    return
                try:
                    self.map_widget.delete_all_marker()
                    self.map_widget.delete_all_path()
                except Exception:  # noqa: BLE001
                    pass
                self.map_path = None
                self.map_markers = []
                if self.fit_map_btn is not None:
                    self.fit_map_btn.configure(state="disabled")
            elif self.map_backend == "folium":
                previous = self.map_html_path
                self.map_html_path = None
                if previous and os.path.exists(previous):
                    try:
                        os.remove(previous)
                    except Exception:  # noqa: BLE001
                        pass
                if self.map_button is not None:
                    self.map_button.configure(state="disabled")
                if self.map_tab_note is not None:
                    self.map_tab_note.configure(
                        text="Map preview will open in your browser using folium."
                    )

        def _suggest_zoom(self, span: float) -> int:
            thresholds = [
                (0.05, 13),
                (0.2, 11),
                (0.5, 10),
                (1.0, 9),
                (5.0, 7),
                (10.0, 6),
                (20.0, 5),
                (40.0, 4),
                (80.0, 3),
                (160.0, 2),
            ]
            for limit, zoom in thresholds:
                if span <= limit:
                    return zoom
            return 1

        def _update_map(self, points: List[TrackPoint]) -> None:
            if not points or self.map_backend is None:
                return
            coords = [(p.lat, p.lon) for p in points]
            if self.map_backend == "tkinter":
                if not self.map_widget:
                    return
                self._clear_map()
                try:
                    if len(coords) >= 2:
                        self.map_path = self.map_widget.set_path(coords, color="#2563eb", width=4)
                    else:
                        self.map_path = None
                except Exception:  # noqa: BLE001
                    self.map_path = None

                self.map_markers = []
                try:
                    start_marker = self.map_widget.set_marker(coords[0][0], coords[0][1], text="Start")
                    self.map_markers.append(start_marker)
                    if len(coords) > 1:
                        end_marker = self.map_widget.set_marker(coords[-1][0], coords[-1][1], text="End")
                        self.map_markers.append(end_marker)
                except Exception:  # noqa: BLE001
                    pass

                latitudes = [c[0] for c in coords]
                longitudes = [c[1] for c in coords]
                top, bottom = max(latitudes), min(latitudes)
                left, right = min(longitudes), max(longitudes)
                center_lat = (top + bottom) / 2
                center_lon = (left + right) / 2
                lat_span = top - bottom
                lon_span = (right - left) * max(math.cos(math.radians(center_lat)), 0.1)
                span = max(lat_span, lon_span)
                try:
                    if len(coords) >= 2 and hasattr(self.map_widget, "fit_bounding_box"):
                        self.map_widget.fit_bounding_box((top, left), (bottom, right))
                    else:
                        self.map_widget.set_position(center_lat, center_lon)
                        self.map_widget.set_zoom(self._suggest_zoom(span))
                except Exception:  # noqa: BLE001
                    pass
                if self.fit_map_btn is not None:
                    self.fit_map_btn.configure(state="normal")
            elif self.map_backend == "folium":
                self._clear_map()
                try:
                    latitudes = [c[0] for c in coords]
                    longitudes = [c[1] for c in coords]
                    bounds = [
                        [min(latitudes), min(longitudes)],
                        [max(latitudes), max(longitudes)],
                    ]
                    center = [sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)]
                    fmap = self.folium.Map(location=center, zoom_start=6)
                    self.folium.PolyLine(coords, weight=4, color="#2563eb").add_to(fmap)
                    self.folium.Marker(coords[0], tooltip="Start").add_to(fmap)
                    if len(coords) > 1:
                        self.folium.Marker(coords[-1], tooltip="End").add_to(fmap)
                    fmap.fit_bounds(bounds)
                    tmp = self.tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                    fmap.save(tmp.name)
                    self.map_html_path = tmp.name
                    tmp.close()
                    if self.map_button is not None:
                        self.map_button.configure(state="normal")
                    if self.map_tab_note is not None:
                        self.map_tab_note.configure(
                            text=f"Map preview ready ({len(points)} points)."
                        )
                except Exception as exc:  # noqa: BLE001
                    self.map_html_path = None
                    if self.map_button is not None:
                        self.map_button.configure(state="disabled")
                    if self.map_tab_note is not None:
                        self.map_tab_note.configure(text=f"Unable to render map: {exc}")

        def fit_map_to_route(self) -> None:
            if not self._last_rendered_points:
                return
            coords = [(p.lat, p.lon) for p in self._last_rendered_points]
            if self.map_backend == "tkinter" and self.map_widget:
                latitudes = [c[0] for c in coords]
                longitudes = [c[1] for c in coords]
                top, bottom = max(latitudes), min(latitudes)
                left, right = min(longitudes), max(longitudes)
                try:
                    if hasattr(self.map_widget, "fit_bounding_box"):
                        self.map_widget.fit_bounding_box((top, left), (bottom, right))
                    else:
                        span = max(top - bottom, right - left)
                        center_lat = (top + bottom) / 2
                        center_lon = (left + right) / 2
                        self.map_widget.set_position(center_lat, center_lon)
                        self.map_widget.set_zoom(self._suggest_zoom(span))
                except Exception:  # noqa: BLE001
                    pass
            elif self.map_backend == "folium":
                self.open_map_preview()

        def _update_summary(self, points: List[TrackPoint], source: str) -> None:
            if not points:
                return
            self._last_rendered_points = points
            self.summary_points_var.set(f"Points: {len(points):,}")
            duration = points[-1].time - points[0].time
            self.summary_duration_var.set(f"Duration: {format_timedelta(duration)}")
            alt_min = min(p.ele for p in points)
            alt_max = max(p.ele for p in points)
            self.summary_altitude_var.set(f"Altitude: {alt_min:.0f} m -> {alt_max:.0f} m")
            tz8 = timezone(timedelta(hours=8))
            start_local = points[0].time.astimezone(tz8)
            end_local = points[-1].time.astimezone(tz8)
            self.summary_localtime_var.set(
                f"Local time (UTC+8): {start_local.strftime('%Y-%m-%d %H:%M')} — {end_local.strftime('%H:%M')}"
            )
            distance = compute_total_distance(points)
            self.summary_distance_var.set(f"Distance: {distance:,.1f} km")
            self.summary_source_var.set(source)

        def open_output_location(self) -> None:
            if not self._last_output_path:
                self.messagebox.showinfo("Output", "No GPX file has been generated yet.")
                return
            reveal_in_explorer(self._last_output_path)

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
                if announce:
                    self._clear_map()
                return False
            self._last_preview_points = points
            self._apply_target_date_from_points(points, announce=announce, context="Target date auto-filled")
            self._update_summary(points, "Loaded from file")
            self._update_map(points)
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

        def swap_navigation_endpoints(self) -> None:
            start = self.nav_start_var.get()
            dest = self.nav_dest_var.get()
            self.nav_start_var.set(dest)
            self.nav_dest_var.set(start)
            self.append_log("Swapped navigation start and destination.")

        def parse_nav_interval(self) -> Optional[int]:
            value = self.nav_interval_var.get().strip()
            if not value:
                return 60
            try:
                number = int(value)
            except ValueError:
                return None
            if number < 0:
                return None
            return number

        def on_plan_navigation(self) -> None:
            if self._busy:
                return

            start = self.nav_start_var.get().strip()
            dest = self.nav_dest_var.get().strip()
            if not start or not dest:
                self.messagebox.showwarning("Missing information", "Please provide both start and destination.")
                return

            interval_value = self.parse_nav_interval()
            if interval_value is None:
                self.messagebox.showwarning("Invalid interval", "GPS point interval must be a non-negative integer.")
                self.nav_interval_var.set("60")
                return

            self.append_log(f"Planning navigation from '{start}' to '{dest}' (interval {interval_value}s).")
            self.nav_summary_var.set("Planning route...")
            self.nav_detail_var.set("Contacting geocoding and routing services...")
            if self.nav_export_btn is not None:
                self.nav_export_btn.configure(state="disabled")
            self._nav_points = []
            self._nav_plan = None
            self.set_status("Planning navigation route...", "info")
            self.set_busy(True)

            thread = threading.Thread(
                target=self._plan_navigation_async,
                args=(start, dest, interval_value),
                daemon=True,
            )
            thread.start()

        def _plan_navigation_async(self, start: str, dest: str, interval: int) -> None:
            plan = plan_navigation_route(start, dest, interval)
            self.root.after(0, lambda result=plan: self._on_navigation_finished(result))

        def _on_navigation_finished(self, plan: NavigationPlan) -> None:
            self.set_busy(False)

            if plan.success:
                self._nav_points = plan.points
                self._nav_plan = plan
                summary = f"{plan.start_label} -> {plan.end_label}"
                self.nav_summary_var.set(summary)
                duration_text = format_timedelta(timedelta(seconds=plan.duration_s)) if plan.duration_s else "N/A"
                detail = f"Distance: {plan.distance_km:.1f} km | Duration: {duration_text} | Points: {len(plan.points)}"
                self.nav_detail_var.set(detail)
                self._update_summary(plan.points, "Navigation plan")
                self._update_map(plan.points)
                if self.nav_export_btn is not None:
                    self.nav_export_btn.configure(state="normal")
                self.set_status(plan.message, "success")
                self.append_log(plan.message, "success")
            else:
                self._nav_points = []
                self._nav_plan = None
                if plan.start_label or plan.end_label:
                    summary = f"{plan.start_label or 'Start'} -> {plan.end_label or 'Destination'}"
                else:
                    summary = "Navigation planning failed."
                self.nav_summary_var.set(summary)
                self.nav_detail_var.set(plan.message)
                if self.nav_export_btn is not None:
                    self.nav_export_btn.configure(state="disabled")
                self.set_status(plan.message, "error")
                self.append_log(plan.message, "error")
                self.messagebox.showerror("Navigation planner", plan.message)

            for warning in plan.warnings:
                if warning and warning != plan.message:
                    self.append_log(warning, "warning")

        def on_export_navigation(self) -> None:
            if self._busy:
                return
            if not self._nav_points:
                self.messagebox.showinfo("Navigation planner", "Plan a route first before exporting.")
                return

            default_name = "navigation.gpx"
            if self._nav_plan and self._nav_plan.start_label and self._nav_plan.end_label:
                start_slug = re.sub(r"[^A-Za-z0-9]+", "_", self._nav_plan.start_label).strip("_")
                end_slug = re.sub(r"[^A-Za-z0-9]+", "_", self._nav_plan.end_label).strip("_")
                if start_slug and end_slug:
                    default_name = f"{start_slug}_to_{end_slug}.gpx"

            path = self.filedialog.asksaveasfilename(
                title="Save navigation GPX",
                defaultextension=".gpx",
                filetypes=[("GPX files", "*.gpx"), ("All files", "*.*")],
                initialfile=default_name,
            )
            if not path:
                return

            error = write_gpx(self._nav_points, path)
            if error:
                self.append_log(error, "error")
                self.set_status(error, "error")
                self.messagebox.showerror("Navigation planner", error)
                return

            message = f"Wrote navigation GPX: {path} (points: {len(self._nav_points)})"
            self.append_log(message, "success")
            self.set_status(message, "success")
            self.messagebox.showinfo("Navigation planner", message)
            self._last_output_path = path
            if self.open_output_btn is not None:
                self.open_output_btn.configure(state="normal")

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
            self._update_summary(points, "Preview data")
            self._update_map(points)
            first_point = points[0]
            last_point = points[-1]
            duration = last_point.time - first_point.time
            altitude_min = min(p.ele for p in points)
            altitude_max = max(p.ele for p in points)
            distance = compute_total_distance(points)
            preview_lines = [
                f"Preview: {len(points)} points detected",
                f"  First point (UTC): {first_point.time.isoformat()}",
                f"  Last point (UTC):  {last_point.time.isoformat()}",
                f"  Duration: {format_timedelta(duration)}",
                f"  Altitude range: {altitude_min:.0f} m -> {altitude_max:.0f} m",
                f"  Distance: {distance:,.1f} km",
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
            self._last_output_path = None
            if self.open_output_btn is not None:
                self.open_output_btn.configure(state="disabled")

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
                points_for_summary = result.points or self._last_preview_points
                if points_for_summary:
                    self._update_summary(points_for_summary, "Converted output")
                    self._update_map(points_for_summary)
                if result.output_path:
                    self._last_output_path = result.output_path
                    if self.open_output_btn is not None:
                        self.open_output_btn.configure(state="normal")
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
            self.nav_start_var.set("")
            self.nav_dest_var.set("")
            self.nav_interval_var.set("60")
            self.nav_summary_var.set("No route planned yet.")
            self.nav_detail_var.set("Enter start and destination to plan a route.")
            self._last_default_output = ""
            self._last_preview_points = []
            self._last_rendered_points = []
            self._last_output_path = None
            self._nav_points = []
            self._nav_plan = None
            self.summary_points_var.set("Points: --")
            self.summary_duration_var.set("Duration: --")
            self.summary_distance_var.set("Distance: --")
            self.summary_altitude_var.set("Altitude: --")
            self.summary_localtime_var.set("Local time (UTC+8): --")
            self.summary_source_var.set("No route loaded")
            if self.open_output_btn is not None:
                self.open_output_btn.configure(state="disabled")
            if self.nav_export_btn is not None:
                self.nav_export_btn.configure(state="disabled")
            self.clear_log()
            self._clear_map()
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
