"""Rain radar overlay: RainViewer client, tile maths, map layers and the close-up toggle. No network."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import FakeResponse, FakeSession, MADRID
from weather_insights import charts, pipeline, transform
from weather_insights.config import RADAR_MAX_ZOOM
from weather_insights.extract import NetworkError, RainViewerClient, ServerError
from weather_insights.models import RadarFrame

T0 = 1_794_824_400  # a real RainViewer frame time (seconds since 1970, UTC)


def maps_json(host="https://tilecache.rainviewer.com", past=None):
    """The shape of https://api.rainviewer.com/public/weather-maps.json (newest frame last)."""
    if past is None:
        past = [{"time": T0 + 600, "path": "/v2/radar/bbb"}, {"time": T0, "path": "/v2/radar/aaa"}]
    return {"version": "2.0", "generated": T0 + 700, "host": host,
            "radar": {"past": past, "nowcast": []}, "satellite": {"infrared": []}}


def radar_client(*responses):
    return RainViewerClient(session=FakeSession(*responses))


# ---- RainViewerClient ----
def test_frames_are_parsed_and_sorted_oldest_first():
    c = radar_client(FakeResponse(200, maps_json()))
    frames = c.radar_frames()
    assert [f.time for f in frames] == [datetime.fromtimestamp(T0, timezone.utc),
                                        datetime.fromtimestamp(T0 + 600, timezone.utc)]
    assert frames[0].tile_url(7, 62, 48) == "https://tilecache.rainviewer.com/v2/radar/aaa/512/7/62/48/2/1_1.png"
    url, params = c._session.calls[0]
    assert url == "https://api.rainviewer.com/public/weather-maps.json"
    assert params == {}  # no key, nothing secret in the request


def test_no_frames_gives_an_empty_list():
    assert radar_client(FakeResponse(200, maps_json(past=[]))).radar_frames() == []


@pytest.mark.parametrize("payload", [
    {"radar": {"past": []}},  # no host
    maps_json(host="http://tilecache.rainviewer.com"),  # not https: the browser would load it insecurely
    maps_json(past=[{"time": "soon", "path": "/v2/radar/aaa"}]),  # time is not a number
    maps_json(past=[{"time": T0}]),  # no path
    ["not", "an", "object"],
    maps_json(host="https://evil.example"),  # not RainViewer
    maps_json(host="https://rainviewer.com.evil.example"),  # only looks like RainViewer
    maps_json(past=[{"time": T0, "path": "@evil.example/x"}]),  # would turn the host into a login part
    maps_json(past=[{"time": T0, "path": "/v2/radar/{z}"}]),  # braces would break the tile address
])
def test_unexpected_answers_raise_server_error(payload):
    with pytest.raises(ServerError, match="radar service"):
        radar_client(FakeResponse(200, payload)).radar_frames()


def test_http_failures_mention_the_radar_service():
    with pytest.raises(ServerError, match="radar service"):
        radar_client(FakeResponse(503)).radar_frames()


def test_pipeline_wraps_the_client():
    c = radar_client(FakeResponse(200, maps_json()))
    assert len(pipeline.fetch_radar_frames(c)) == 2


def test_minutes_ago():
    frame = RadarFrame(datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc), "x/{z}/{x}/{y}")
    assert frame.minutes_ago(datetime(2026, 11, 16, 12, 30, tzinfo=timezone.utc)) == 30
    assert frame.minutes_ago(datetime(2026, 11, 16, 11, 59, tzinfo=timezone.utc)) == 0  # never negative


# ---- tile maths ----
def test_zoom_zero_tile_is_the_whole_web_map():
    west, south, east, north = transform.tile_bounds(0, 0, 0)
    assert (west, east) == (-180.0, 180.0)
    assert north == pytest.approx(transform.MERCATOR_MAX_LAT) and south == pytest.approx(-transform.MERCATOR_MAX_LAT)


def test_madrid_tile_at_zoom_7_contains_madrid():
    # Worked by hand: x = (180 - 3.70) / 360 * 128 = 62.7 -> 62; y = (1 - asinh(tan 40.42°) / pi) / 2 * 128 = 48.3 -> 48
    assert transform.lonlat_to_tile(MADRID.lat, MADRID.lon, 7) == (62, 48)
    west, south, east, north = transform.tile_bounds(62, 48, 7)
    assert west <= MADRID.lon <= east and south <= MADRID.lat <= north


def test_points_beyond_the_map_edge_are_clamped_onto_it():
    assert transform.lonlat_to_tile(89.9, 179.99, 3) == (7, 0)
    assert transform.lonlat_to_tile(-89.9, -200, 3) == (0, 7)


@pytest.mark.parametrize("map_zoom, expected", [(10.4, RADAR_MAX_ZOOM), (7.0, 7), (3.9, 3), (-1, 0)])
def test_radar_zoom_never_goes_deeper_than_the_service(map_zoom, expected):
    assert transform.radar_zoom(map_zoom) == expected


def test_radar_tiles_cover_the_whole_area_around_the_city():
    tiles = transform.radar_tiles(MADRID.lat, MADRID.lon, 200, map_zoom=10)
    assert {z for z, _, _ in tiles} == {7}
    assert (7, 62, 48) in tiles
    bounds = [transform.tile_bounds(x, y, z) for z, x, y in tiles]
    west, south = min(b[0] for b in bounds), min(b[1] for b in bounds)
    east, north = max(b[2] for b in bounds), max(b[3] for b in bounds)
    dlat = 200 / 111.32
    assert south <= MADRID.lat - dlat and north >= MADRID.lat + dlat
    assert west <= MADRID.lon - 2.3 and east >= MADRID.lon + 2.3  # 200 km is ~2.4° of longitude at Madrid
    assert len(tiles) <= 9  # a handful of pictures, not hundreds


def test_tiles_continue_past_the_180_degree_line():
    # Suva (Fiji) sits at 178.4°E: the area east of it is across the 180° meridian, in column 0 again.
    tiles = transform.radar_tiles(-18.14, 178.44, 200, map_zoom=10)
    columns = {x for _, x, _ in tiles}
    assert 2 ** 7 in columns  # the column just past 180°, kept on the east side of the city
    layer = charts.radar_layers(FRAME, [t for t in tiles if t[1] == 2 ** 7][:1])[0]
    assert "/7/0/" in layer["source"]  # but the picture is fetched as column 0
    assert layer["coordinates"][0][0] == pytest.approx(180.0)  # and drawn east of the meridian


def test_far_north_uses_a_coarser_zoom_instead_of_many_pictures():
    from weather_insights.config import RADAR_MAX_TILES

    tiles = transform.radar_tiles(78.2, 15.6, 300, map_zoom=10)  # Svalbard: longitude degrees are short here
    assert 0 < len(tiles) <= RADAR_MAX_TILES
    assert tiles[0][0] < RADAR_MAX_ZOOM


# ---- map layers ----
FRAME = RadarFrame(datetime.fromtimestamp(T0, timezone.utc), "https://tiles.example/{z}/{x}/{y}.png")


def test_radar_layers_put_each_tile_on_its_own_edges():
    layers = charts.radar_layers(FRAME, [(7, 62, 48), (7, 63, 48)])
    assert [l["source"] for l in layers] == ["https://tiles.example/7/62/48.png", "https://tiles.example/7/63/48.png"]
    west, south, east, north = transform.tile_bounds(62, 48, 7)
    assert layers[0]["coordinates"] == [[west, north], [east, north], [east, south], [west, south]]
    assert all(l["sourcetype"] == "image" and l["below"] == "traces" for l in layers)


def test_no_tiles_no_layers():
    assert charts.radar_layers(FRAME, []) == []


def grid_df():
    rows = [{"lat": MADRID.lat + d, "lon": MADRID.lon + d, "temp": 20.0, "feels_like": 19.0, "humidity": 50.0,
             "wind_speed": 2.0, "clouds": 10.0, "precip_mm": 0.0, "description": "clear sky"} for d in (-0.05, 0.05)]
    return pd.DataFrame(rows)


SMOOTH = {"values": np.array([[20.0, 21.0], [22.0, 23.0]]), "north": 40.5, "south": 40.3, "west": -3.8, "east": -3.6}


def closeup(smooth, radar=None):
    return charts.closeup_map(grid_df(), pd.DataFrame(), (MADRID.lat, MADRID.lon), "temp", 15, 2, smooth,
                              radar=radar)


def test_radar_replaces_the_smooth_weather_surface():
    radar = charts.radar_layers(FRAME, [(7, 62, 48)])
    without, with_radar = closeup(SMOOTH), closeup(SMOOTH, radar)
    assert without.layout.map.layers[0].source.startswith("data:image/png")  # the weather surface
    assert [l.source for l in with_radar.layout.map.layers] == ["https://tiles.example/7/62/48.png"]
    readings = [t for t in with_radar.data if t.marker.colorbar.title.text]
    assert len(readings) == 1 and len(readings[0].lat) == len(grid_df())  # the hoverable reading dots stay


def test_empty_radar_list_keeps_the_weather_surface():
    assert closeup(SMOOTH, radar=[]).layout.map.layers[0].source.startswith("data:image/png")


def test_radar_makes_weather_cells_faint():
    radar = charts.radar_layers(FRAME, [(7, 62, 48)])
    without, with_radar = closeup(None), closeup(None, radar)
    assert without.data[0].marker.opacity == 0.6
    assert with_radar.data[0].marker.opacity == charts.CELLS_UNDER_RADAR_OPACITY
    assert [l.source for l in with_radar.layout.map.layers][-1] == "https://tiles.example/7/62/48.png"


# ---- the toggle in the close-up tab (Streamlit AppTest, no browser) ----
def radar_app():
    import streamlit as st
    from conftest import MADRID
    from weather_insights import ui

    st.session_state["layers"] = ui._radar_controls(MADRID, 3600, 15.0)


@pytest.fixture
def frames(monkeypatch):
    """Two fake frames, served by a fake pipeline; the Streamlit cache is cleared around each test."""
    import streamlit as st

    now = datetime.now(timezone.utc)
    fake = [RadarFrame(now - timedelta(minutes=20), "https://t/old/{z}/{x}/{y}.png"),
            RadarFrame(now - timedelta(minutes=10), "https://t/new/{z}/{x}/{y}.png")]
    monkeypatch.setattr(pipeline, "fetch_radar_frames", lambda client: fake)
    st.cache_data.clear()
    yield fake
    st.cache_data.clear()


def test_radar_is_off_by_default(frames):
    at = AppTest.from_function(radar_app).run()
    assert at.session_state["layers"] is None
    assert not at.select_slider


def test_turning_radar_on_shows_the_latest_frame(frames):
    at = AppTest.from_function(radar_app).run()
    at.toggle(key="closeup_radar").set_value(True).run()
    layers = at.session_state["layers"]
    assert layers and all("/new/7/" in l["source"] for l in layers)
    local = frames[-1].time + timedelta(hours=1)  # the app passes a UTC+1 city
    assert at.select_slider[0].format_func(at.select_slider[0].value) == f"{local:%H:%M}"
    assert any("10 min ago" in c.value and "RainViewer" in c.value for c in at.caption)


def test_slider_picks_an_older_frame_and_keeps_it_on_the_next_rerun(frames):
    at = AppTest.from_function(radar_app).run()
    at.toggle(key="closeup_radar").set_value(True).run()
    at.select_slider[0].set_value(0).run()
    assert all("/old/7/" in l["source"] for l in at.session_state["layers"])
    at.run()  # any other click reruns the page; the chosen picture must stay
    assert all("/old/7/" in l["source"] for l in at.session_state["layers"])
    assert any("20 min ago" in c.value for c in at.caption)


def test_radar_failure_is_a_warning_not_a_crash(monkeypatch):
    import streamlit as st

    def fail(client):
        raise NetworkError("Could not reach the radar service. Check your internet connection.")

    monkeypatch.setattr(pipeline, "fetch_radar_frames", fail)
    st.cache_data.clear()
    at = AppTest.from_function(radar_app).run()
    at.toggle(key="closeup_radar").set_value(True).run()
    assert at.session_state["layers"] is None
    assert "Rain radar unavailable" in at.warning[0].value
    st.cache_data.clear()
