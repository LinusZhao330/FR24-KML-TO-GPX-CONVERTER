# FR24 KML to GPX Converter

## Overview
**FR24_to_GPX** is a tool designed to convert flight tracks exported from **FlightRadar24 (FR24)** in KML format into standard **GPX** tracks. It supports track preview, interpolation, timestamp correction, and navigation planning.  
The project includes both a **command-line interface (CLI)** workflow and a **Tkinter-based graphical user interface (GUI)**, suitable for batch processing, interactive previewing, or packaging into an executable program.

## Key Features
- **KML → GPX Conversion**: Parses the Route layer exported by FR24, retaining timestamp, latitude/longitude, and altitude information.  
- **Time Interpolation and Smoothing**: Performs linear interpolation of the track based on a custom time interval to smooth out missing points.  
- **Target Date Adjustment**: Resets all timestamps using UTC+8 as the base time zone for easy reuse of flight records.  
- **Graphical Preview**: Includes logging, summary statistics, and map preview (via TkinterMapView or browser-based folium). After export, you can open the output folder directly.  
- **Navigation Planning (Experimental)**: Uses OpenStreetMap Nominatim and OSRM APIs to generate ground routes based on start, end, and waypoint inputs, then exports them as GPX.  
- **Interactive CLI**: Built with `click`; even without arguments, it interactively prompts for inputs, making it convenient for scripting or batch processing.

## Requirements
- **Python** 3.10 or above (the code uses union type syntax introduced in 3.10).  
- **Core dependencies:** `click`, `requests`.  
- **Optional dependencies:**  
  - `tkinter` (included in most Python distributions; some Linux distros require separate installation).  
  - `tkintermapview` (for embedded map display).  
  - `folium` (used for browser-based map preview if TkinterMapView is unavailable).  
- **Internet access:** Required for route planning, geocoding, and online maps.

## Installation
1. Clone the project:
   ```powershell
   git clone https://github.com/zzk90/FR24_to_GPX.git
   cd FR24_to_GPX
   ```
2. Create a virtual environment (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -U click requests
   # Optional: install map-related dependencies
   pip install -U tkintermapview folium
   ```

## Quick Start

### Command Line
```powershell
python converter.py --kml path\to\flight.kml --gpx flight.gpx --interval 60 --date 2024-08-15
```

**Common options:**

| Option | Description |
| --- | --- |
| `--kml PATH` | Input KML file path (required; if missing, the program will prompt for input). |
| `--gpx PATH` | Output GPX file path. Defaults to the same name as the source file. |
| `--interval SEC` | Interpolation interval (in seconds). `0` = no interpolation. Default is `60`. |
| `--date YYYY-MM-DD` | Rewrites all timestamps based on UTC+8. If omitted, keeps original timestamps. |
| `--gui` / `--cli` | Forces GUI or CLI mode. The packaged executable defaults to GUI. |

After conversion, the command line outputs the generated file path and displays any warnings (e.g., failed altitude or date parsing).

### Graphical Interface
```powershell
python converter.py --gui
```
or simply double-click the packaged executable.

**Main GUI functions:**
- Use **“Select KML”** to choose a file and click **“Preview”** to analyze the track, set target date, and update the summary.  
- **“Interpolation interval”** adjusts interpolation frequency, while **“Target date”** can be edited manually.  
- **“Convert to GPX”** executes the conversion; after completion, **“Open output location”** opens the destination folder.  
- The **“Map preview”** tab displays the track. If `tkintermapview` is unavailable, it falls back to a folium-based browser map.  
- The bottom log window records status updates and warnings for troubleshooting.

## Navigation Planner (Experimental)
The “Navigation planner” section in the GUI allows you to:
1. Enter the start point, end point, and optional waypoints (one per line).  
2. Set export sampling interval.  
3. Click **“Plan route”** to call Nominatim and OSRM for road routing.  
4. Preview the route on the map, then **“Export GPX”** to save it.

> Note: This feature requires a stable internet connection. Frequent requests may trigger rate limits.

## Troubleshooting
- **Cannot open GUI / missing Tkinter**: Install `python3-tk` or the corresponding Tk package for your OS.  
- **Map not displayed in GUI**: Install `tkintermapview`; if still unavailable, install `folium` for browser preview.  
- **Navigation planning failed**: Check your network connection, retry later, or reduce request frequency. You can also convert KML directly in CLI mode.  
- **Missing altitude data**: If the FR24 KML lacks altitude info, the program tries to extract it from the description field. If it still fails, you can manually edit the altitude values in the generated GPX.

## Development & Build
- The main entry point is `converter.py`, which contains all CLI and GUI logic.  
- The repository includes `converter.spec` and `FR24Converter.spec` for **PyInstaller** builds:
  ```powershell
  pyinstaller converter.spec
  ```
- `update_dataclasses.py` helps batch update type definitions when adjusting data structures.  
- Contributions (e.g., improving interpolation, adding new map backends, or enhancing documentation) are welcome.

---

If you encounter issues or have new ideas, feel free to log them and share feedback — together we can make this tool even better.
