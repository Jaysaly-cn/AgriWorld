import os
import ssl
import json
from agriworld.data_quality import forcing_quality_issue
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
import ee
from google.oauth2 import service_account
from shapely.geometry import Polygon
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from agriworld.nass import (
    CROP_YIELD_SPECS,
    fetch_nass_yield,
    make_yield_key,
    parse_crop_codes,
)
from agriworld.paths import CACHE_ROOT, DATA_ROOT, GEE_CREDENTIALS_PATH, PROXY_URL

os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MAX_WORKERS = int(os.getenv("AGRI_GEE_WORKERS", "4"))
MAX_WORKERS_SMAP = int(os.getenv("AGRI_SMAP_WORKERS", "4"))
MAX_RETRIES = 4
MIN_WEATHER_COVERAGE = 0.95
NASS_CROP_CODES = parse_crop_codes(os.getenv("AGRI_NASS_CROPS", "1,5"))


class StageProgress:
    def __init__(self, name, total, report_every=10):
        self.name = name
        self.total = max(int(total), 1)
        self.report_every = max(int(report_every), 1)
        self.done = 0
        self.success = 0
        self.cached = 0
        self.started = time.time()

    def update(self, ok=True, cached=False):
        self.done += 1
        self.success += int(ok)
        self.cached += int(cached)
        if self.done % self.report_every != 0 and self.done != self.total:
            return
        elapsed = max(time.time() - self.started, 1e-6)
        rate = self.done / elapsed
        eta = (self.total - self.done) / max(rate, 1e-6)
        print(
            f"  {self.name}: {self.done}/{self.total} "
            f"ok={self.success} fail={self.done-self.success} "
            f"cached={self.cached} "
            f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}m "
            f"ETA={eta/60:.1f}m",
            flush=True,
        )


class AgriWorldDataPipeline:
    FORCING_COLS = ['Precip', 'ETo', 'SRAD', 'PAR', 'Tmax', 'Tmin', 'Tmean', 'VPD', 'GDD']
    STATIC_COLS = ['Elevation', 'Slope', 'Aspect', 'Bulk_Density', 'SOC',
                   'Clay_Fraction', 'Sand_Fraction', 'Total_Nitrogen', 'pH',
                   'N_Rate', 'Crop_Type']

    def __init__(self, start_date, end_date, output_dir=DATA_ROOT,
                 cache_root=CACHE_ROOT,
                 credentials_path=GEE_CREDENTIALS_PATH,
                 skip_smap=False):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.year = self.start_date.year
        self.output_dir = output_dir
        # Every remote product is year-dependent, including CDL crop masks.
        # Year-scoped caches prevent one year's weather/LAI from contaminating
        # another when multi-year collection is resumed.
        self.cache_root = cache_root
        self.cache_dir = os.path.join(self.cache_root, str(self.year))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_root, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.expected_days = len(self.date_range)
        self.skip_smap = bool(skip_smap)
        print(
            f"[CONFIG] year={self.year} workers={MAX_WORKERS} "
            f"smap_workers={MAX_WORKERS_SMAP} skip_smap={self.skip_smap}",
            flush=True,
        )

        self.arms_n_rates = {
            'IA': 149.0, 'IL': 168.0, 'NE': 144.0, 'MN': 133.0, 'IN': 160.0
        }
        self.state_fips_map = {
            '19': 'IA', '17': 'IL', '31': 'NE', '27': 'MN', '18': 'IN'
        }
        self.planting_doy = {
            'IA': 121, 'IL': 110, 'NE': 125, 'MN': 130, 'IN': 115
        }

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials missing: {credentials_path}")
        with open(credentials_path, 'r') as f:
            key_data = json.load(f)
        project_id = key_data.get("project_id")
        scopes = [
            'https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform'
        ]
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scopes)
        ee.Initialize(credentials=credentials, project=project_id)

    def fetch_usda_nass_yield(self, year=None, crop_code=1):
        year = year or self.year
        return fetch_nass_yield(year, self.arms_n_rates.keys(), crop_code)

    def fetch_usda_nass_yields(self, year=None, crop_codes=None):
        year = year or self.year
        crop_codes = NASS_CROP_CODES if crop_codes is None else parse_crop_codes(crop_codes)
        return {
            int(crop): self.fetch_usda_nass_yield(year=year, crop_code=int(crop))
            for crop in crop_codes
            if int(crop) in CROP_YIELD_SPECS
        }

    def generate_grids(self, yield_map):
        print("[GRID] Generating county centroid grids...")
        fips_list = list(self.state_fips_map.keys())
        counties_fc = ee.FeatureCollection("TIGER/2018/Counties") \
            .filter(ee.Filter.inList('STATEFP', fips_list))
        features = counties_fc.getInfo()['features']
        records = []
        grid_size = 0.01
        for f in features:
            props = f['properties']
            state_fips = props['STATEFP']
            state_alpha = self.state_fips_map.get(state_fips)
            if not state_alpha:
                continue
            county_name = props['NAME'].upper()
            yield_key = make_yield_key(state_alpha, county_name)
            if yield_key not in yield_map:
                continue
            lon = float(props['INTPTLON'])
            lat = float(props['INTPTLAT'])
            mtrs_id = f"{state_alpha}-{county_name}-CEN"
            poly = Polygon([
                (lon - grid_size / 2, lat - grid_size / 2),
                (lon + grid_size / 2, lat - grid_size / 2),
                (lon + grid_size / 2, lat + grid_size / 2),
                (lon - grid_size / 2, lat + grid_size / 2)
            ])
            records.append({
                'MTRS': mtrs_id, 'State': state_alpha, 'County': county_name,
                'geometry': poly, 'Yield': yield_map[yield_key],
                'Lat': lat, 'Lon': lon
            })
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        print(f"  Matched {len(gdf)} grids with real yields")
        return gdf

    def fetch_weather(self, grid_gdf):
        start_str = self.start_date.strftime('%Y-%m-%d')
        # Earth Engine filterDate uses an exclusive end date.
        end_str = (self.end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        daymet_col = ee.ImageCollection('NASA/ORNL/DAYMET_V4') \
            .filterDate(start_str, end_str) \
            .select(['prcp', 'srad', 'tmax', 'tmin', 'vp'])
        all_dfs = []
        unique_mtrs = grid_gdf['MTRS'].unique()
        print(f"[WEATHER] Extracting Daymet 5-band ({len(unique_mtrs)} grids)...")

        def valid_cached_weather(df, mtrs_id):
            if df is None or df.empty or "date" not in df or "Tmax" not in df:
                return False
            dates = pd.to_datetime(df["date"])
            if not dates.dt.year.eq(self.year).all():
                return False
            min_days = int(0.90 * self.expected_days)
            return (
                len(df) >= min_days and
                df["Tmax"].notna().sum() >= min_days and
                df["Tmin"].notna().sum() >= min_days and
                df["Tmax"].std() > 2.0
            )

        def process_grid(mtrs_id):
            cache_file = os.path.join(self.cache_dir, f"wx3_{mtrs_id}.pkl")
            if os.path.exists(cache_file):
                try:
                    cached = pd.read_pickle(cache_file)
                    if valid_cached_weather(cached, mtrs_id):
                        return cached, True
                    print(f"  Invalid weather cache, refetch: {mtrs_id}")
                except Exception:
                    print(f"  Unreadable weather cache, refetch: {mtrs_id}")
            row = grid_gdf[grid_gdf['MTRS'] == mtrs_id].iloc[0]
            ee_geom = ee.Geometry.Polygon([list(row['geometry'].exterior.coords)])
            last_error = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    def ext(img):
                        date_str = img.date().format('YYYY-MM-dd')
                        reduced = img.reduceRegion(
                            ee.Reducer.mean(), ee_geom, 1000, bestEffort=True
                        )
                        return ee.Feature(None, reduced).set('date_str', date_str)

                    features = (
                        daymet_col.filterBounds(ee_geom).map(ext)
                        .getInfo().get("features", [])
                    )
                    data = []
                    for feat in features:
                        p = feat['properties']
                        if p.get('prcp') is None or p.get('date_str') is None:
                            continue
                        data.append({
                            'MTRS': mtrs_id,
                            'date': pd.to_datetime(p['date_str']),
                            'Precip': p.get('prcp', np.nan),
                            'SRAD_raw': p.get('srad', np.nan),
                            'Tmax': p.get('tmax', np.nan),
                            'Tmin': p.get('tmin', np.nan),
                            'VP': p.get('vp', np.nan),
                        })
                    df = pd.DataFrame(data) if data else pd.DataFrame()
                    if valid_cached_weather(df, mtrs_id):
                        df.to_pickle(cache_file)
                        return df, False
                    last_error = RuntimeError(
                        f"incomplete Daymet response ({len(df)}/{self.expected_days} days)"
                    )
                except Exception as exc:
                    last_error = exc

                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)

            print(f"  WX fail {mtrs_id} after {MAX_RETRIES} attempts: {last_error}")
            return pd.DataFrame(), False

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_grid, m): m for m in unique_mtrs}
            progress = StageProgress("Weather", len(futures))
            for future in as_completed(futures):
                df, cache_hit = future.result()
                if not df.empty:
                    all_dfs.append(df)
                progress.update(not df.empty, cache_hit)
        if not all_dfs:
            raise RuntimeError(f"No valid Daymet weather returned for {self.year}")
        coverage = len(all_dfs) / max(len(unique_mtrs), 1)
        print(
            f"  Weather coverage: {len(all_dfs)}/{len(unique_mtrs)} "
            f"({coverage:.1%})"
        )
        if coverage < MIN_WEATHER_COVERAGE:
            raise RuntimeError(
                f"Daymet coverage {coverage:.1%} is below required "
                f"{MIN_WEATHER_COVERAGE:.0%}; refusing to save partial year {self.year}"
            )
        wx = pd.concat(all_dfs, ignore_index=True)
        wx = self._compute_derived_weather(wx, grid_gdf)
        return wx

    def _compute_derived_weather(self, wx, grid_gdf):
        lat_lookup = dict(zip(grid_gdf['MTRS'], grid_gdf['Lat']))
        wx['_DOY'] = wx['date'].dt.dayofyear
        wx['_lat'] = wx['MTRS'].map(lat_lookup)
        wx['SRAD'] = wx['SRAD_raw'].astype(np.float64) * 0.0864
        wx['PAR'] = wx['SRAD'] * 0.48
        wx['Tmean'] = (wx['Tmax'] + wx['Tmin']) / 2.0
        svp = 0.6108 * np.exp(17.27 * wx['Tmean'] / (wx['Tmean'] + 237.3))
        ea = wx['VP'].astype(np.float64) / 1000.0
        wx['VPD'] = np.maximum(svp - ea, 0)
        wx['GDD'] = np.maximum(wx['Tmean'] - 10.0, 0)
        lat_rad = np.radians(wx['_lat'].values)
        doy = wx['_DOY'].values.astype(np.float64)
        delta = 0.409 * np.sin(2 * np.pi / 365.0 * doy - 1.39)
        dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * doy)
        cos_arg = -np.tan(lat_rad) * np.tan(delta)
        ws = np.arccos(np.clip(cos_arg, -1, 1))
        Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )
        t_range = np.maximum(wx['Tmax'] - wx['Tmin'], 0)
        wx['ETo'] = np.maximum(0.0023 * (wx['Tmean'] + 17.8) * np.sqrt(t_range) * Ra, 0)
        wx.drop(columns=['_DOY', '_lat', 'SRAD_raw', 'VP'], inplace=True)
        wx = wx.drop_duplicates(subset=['MTRS', 'date'], keep='first')
        wx.sort_values(['MTRS', 'date'], inplace=True)
        return wx

    def fetch_static(self, grid_gdf):
        topo = ee.ImageCollection("USGS/3DEP/10m_collection").mosaic()
        elevation_raw = topo.select('elevation')
        elevation = elevation_raw.rename('Elevation')
        slope = ee.Terrain.slope(elevation_raw).rename('Slope')
        aspect = ee.Terrain.aspect(elevation_raw).rename('Aspect')

        soil_img = ee.Image([
            ee.Image("projects/soilgrids-isric/bdod_mean") \
                .select(['bdod_0-5cm_mean']).multiply(0.01).rename('Bulk_Density'),
            ee.Image("projects/soilgrids-isric/soc_mean") \
                .select(['soc_0-5cm_mean']).multiply(0.1).rename('SOC'),
            ee.Image("projects/soilgrids-isric/clay_mean") \
                .select(['clay_0-5cm_mean']).multiply(0.1).rename('Clay_Fraction'),
            ee.Image("projects/soilgrids-isric/sand_mean") \
                .select(['sand_0-5cm_mean']).multiply(0.1).rename('Sand_Fraction'),
            ee.Image("projects/soilgrids-isric/nitrogen_mean") \
                .select(['nitrogen_0-5cm_mean']).multiply(0.01).rename('Total_Nitrogen'),
            ee.Image("projects/soilgrids-isric/phh2o_mean") \
                .select(['phh2o_0-5cm_mean']).multiply(0.1).rename('pH'),
        ])

        cdl_raw = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(self.year, self.year, 'year')) \
            .first().select('cropland')
        crop_mask = cdl_raw.remap(
            ee.List([1, 5, 24, 26, 23, 36, 37]),
            ee.List([1, 1, 1, 1, 1, 1, 1]), 0
        ).eq(1)
        cdl_masked = cdl_raw.updateMask(crop_mask)

        static_img = ee.Image([elevation, slope, aspect]).addBands(soil_img)

        all_dfs = []
        unique_mtrs = grid_gdf['MTRS'].unique()
        print(f"[STATIC] Extracting topo + soil + CDL ({len(unique_mtrs)} grids)...")

        def process_grid(mtrs_id):
            cache_file = os.path.join(self.cache_dir, f"static4_{mtrs_id}.pkl")
            if os.path.exists(cache_file):
                return pd.read_pickle(cache_file)
            row = grid_gdf[grid_gdf['MTRS'] == mtrs_id].iloc[0]
            ee_geom = ee.Geometry.Polygon([list(row['geometry'].exterior.coords)])
            try:
                topo_soil = static_img.reduceRegion(
                    ee.Reducer.mean(), ee_geom, 30, bestEffort=True
                ).getInfo()
                crop_result = cdl_masked.reduceRegion(
                    ee.Reducer.mode(), ee_geom, 30, bestEffort=True
                ).getInfo()
                crop_code = crop_result.get('cropland', 0) if crop_result else 0

                def safe_val(d, key):
                    v = d.get(key)
                    if v is None:
                        return np.nan
                    return float(v)

                df = pd.DataFrame([{
                    'MTRS': mtrs_id,
                    'Elevation': safe_val(topo_soil, 'Elevation'),
                    'Slope': safe_val(topo_soil, 'Slope'),
                    'Aspect': safe_val(topo_soil, 'Aspect'),
                    'Bulk_Density': safe_val(topo_soil, 'Bulk_Density'),
                    'SOC': safe_val(topo_soil, 'SOC'),
                    'Clay_Fraction': safe_val(topo_soil, 'Clay_Fraction'),
                    'Sand_Fraction': safe_val(topo_soil, 'Sand_Fraction'),
                    'Total_Nitrogen': safe_val(topo_soil, 'Total_Nitrogen'),
                    'pH': safe_val(topo_soil, 'pH'),
                    'Crop_Type': int(crop_code) if crop_code else np.nan,
                }])
                df.to_pickle(cache_file)
                return df
            except Exception as e:
                print(f"  Static fail {mtrs_id}: {e}")
                return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_grid, m): m for m in unique_mtrs}
            progress = StageProgress("Static", len(futures))
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)
                progress.update(not df.empty)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def fetch_s2_lai(self, grid_gdf):
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = (self.end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(self.year, self.year, 'year')) \
            .first()
        crop_mask = cdl.select('cropland').remap(
            ee.List([1, 5, 24, 26, 23, 36, 37]),
            ee.List([1, 1, 1, 1, 1, 1, 1]), 0
        ).eq(1)

        def calc_lai(img):
            img_m = img.updateMask(crop_mask)
            evi = img_m.expression(
                '2.5 * ((N-R)/(N+6*R-7.5*B+1))',
                {'N': img_m.select('B8'), 'R': img_m.select('B4'), 'B': img_m.select('B2')}
            )
            lai = evi.multiply(3.618).subtract(0.118).rename('LAI')
            return img_m.addBands(lai).select('LAI')

        s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_str, end_str) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .map(calc_lai)
        all_dfs = []
        unique_mtrs = grid_gdf['MTRS'].unique()
        print(f"[S2-LAI] Extracting Sentinel-2 LAI ({len(unique_mtrs)} grids)...")

        def process_grid(mtrs_id):
            cache_file = os.path.join(self.cache_dir, f"lai2_{mtrs_id}.pkl")
            if os.path.exists(cache_file):
                return pd.read_pickle(cache_file)
            row = grid_gdf[grid_gdf['MTRS'] == mtrs_id].iloc[0]
            ee_geom = ee.Geometry.Polygon([list(row['geometry'].exterior.coords)])
            try:
                def ext(img):
                    date_str = img.date().format('YYYY-MM-dd')
                    reduced = img.reduceRegion(ee.Reducer.mean(), ee_geom, 30, bestEffort=True)
                    return ee.Feature(None, reduced).set('date_str', date_str)
                res = s2_col.filterBounds(ee_geom).map(ext).getInfo()['features']
                data = []
                for feat in res:
                    p = feat['properties']
                    lai_val = p.get('LAI')
                    ds = p.get('date_str')
                    if lai_val is not None and ds is not None:
                        data.append({'MTRS': mtrs_id, 'date': pd.to_datetime(ds), 'LAI': lai_val})
                df = pd.DataFrame(data) if data else pd.DataFrame()
                if not df.empty:
                    df.to_pickle(cache_file)
                return df
            except Exception as e:
                print(f"  S2 fail {mtrs_id}: {e}")
                return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_grid, m): m for m in unique_mtrs}
            progress = StageProgress("S2-LAI", len(futures))
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)
                progress.update(not df.empty)
        if not all_dfs:
            return pd.DataFrame(columns=['MTRS', 'date', 'LAI'])
        lai_df = pd.concat(all_dfs, ignore_index=True)
        lai_df = lai_df.drop_duplicates(subset=['MTRS', 'date'], keep='first')
        lai_df['LAI'] = lai_df['LAI'].clip(lower=0.0)
        return lai_df

    def _build_monthly_windows(self):
        windows = []
        for month in range(1, 13):
            m_start = pd.Timestamp(self.year, month, 1)
            if month == 12:
                m_end = pd.Timestamp(self.year, 12, 31)
            else:
                m_end = pd.Timestamp(self.year, month + 1, 1) - pd.Timedelta(days=1)
            m_start = max(m_start, self.start_date)
            m_end = min(m_end, self.end_date)
            if m_start <= m_end:
                windows.append((m_start.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d')))
        return windows

    def fetch_smap(self, grid_gdf):
        windows = self._build_monthly_windows()
        all_dfs = []
        unique_mtrs = grid_gdf['MTRS'].unique()
        total = len(unique_mtrs)
        print(f"[SMAP] Extracting soil moisture ({total} grids, {len(windows)} monthly windows)...")

        grid_dict = {}
        for _, row in grid_gdf.iterrows():
            grid_dict[row['MTRS']] = ee.Geometry.Polygon([list(row['geometry'].exterior.coords)])

        for w_idx, (w_start, w_end) in enumerate(windows):
            tag = w_start[:7]
            smap_col = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008") \
                .filterDate(w_start, w_end) \
                .select(['sm_surface', 'sm_rootzone'])

            try:
                col_size = smap_col.size().getInfo()
            except Exception:
                col_size = 0
            if col_size == 0:
                continue

            print(f"  Window {w_start} to {w_end} ({col_size} images)")

            def process_grid(mtrs_id, _tag=tag, _w_start=w_start, _w_end=w_end):
                cache_file = os.path.join(self.cache_dir, f"smap2_{mtrs_id}_{_tag}.pkl")
                if os.path.exists(cache_file):
                    return pd.read_pickle(cache_file)
                try:
                    ee_geom = grid_dict[mtrs_id]
                    def ext(img):
                        date_str = img.date().format('YYYY-MM-dd')
                        reduced = img.reduceRegion(
                            ee.Reducer.mean(), ee_geom, 11000, bestEffort=True
                        )
                        return ee.Feature(None, reduced).set('date_str', date_str)
                    res = smap_col.filterBounds(ee_geom).map(ext).getInfo()['features']
                    data = []
                    for feat in res:
                        p = feat['properties']
                        sm_val = p.get('sm_surface')
                        ds = p.get('date_str')
                        if sm_val is not None and ds is not None:
                            data.append({
                                'MTRS': mtrs_id, 'date': pd.to_datetime(ds),
                                'sm_surface': sm_val,
                                'sm_rootzone': p.get('sm_rootzone', np.nan)
                            })
                    df = pd.DataFrame(data) if data else pd.DataFrame()
                    if not df.empty:
                        df.to_pickle(cache_file)
                    return df
                except Exception as e:
                    print(f"    SMAP fail {mtrs_id} ({_tag}): {e}")
                    return pd.DataFrame()

            window_dfs = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_SMAP) as executor:
                futures = {executor.submit(process_grid, m): m for m in unique_mtrs}
                progress = StageProgress(f"SMAP {tag}", len(futures), report_every=25)
                for future in as_completed(futures):
                    df = future.result()
                    if not df.empty:
                        window_dfs.append(df)
                    progress.update(not df.empty)

            if window_dfs:
                all_dfs.extend(window_dfs)

            time.sleep(2)

        if not all_dfs:
            return pd.DataFrame()
        smap_df = pd.concat(all_dfs, ignore_index=True)
        smap_df = smap_df.drop_duplicates(subset=['MTRS', 'date'], keep='first')
        smap_df = smap_df.groupby(['MTRS', 'date']).mean().reset_index()
        return smap_df

    def _enforce_length(self, group):
        if len(group) == self.expected_days:
            return group
        group = group.drop_duplicates(subset=['date'], keep='first')
        if len(group) >= self.expected_days:
            return group.head(self.expected_days)
        all_dates = pd.DataFrame({'date': self.date_range})
        group = pd.merge(all_dates, group, on='date', how='left')
        if 'MTRS' not in group.columns or group['MTRS'].isna().all():
            return group
        group['MTRS'] = group['MTRS'].ffill().bfill()
        return group

    def _fill_static_nans(self, static_df):
        numeric_cols = ['Elevation', 'Slope', 'Aspect', 'Bulk_Density', 'SOC',
                        'Clay_Fraction', 'Sand_Fraction', 'Total_Nitrogen', 'pH']
        for col in numeric_cols:
            if col in static_df.columns:
                median_val = static_df[col].median()
                n_nan = static_df[col].isna().sum()
                if n_nan > 0:
                    print(f"  Filling {n_nan} NaN values in {col} with median={median_val:.3f}")
                    static_df[col] = static_df[col].fillna(median_val)
        if 'Crop_Type' in static_df.columns:
            mode_val = static_df['Crop_Type'].mode()
            if len(mode_val) > 0:
                mode_val = mode_val.iloc[0]
            else:
                mode_val = 1
            n_nan = static_df['Crop_Type'].isna().sum()
            if n_nan > 0:
                print(f"  Filling {n_nan} NaN values in Crop_Type with mode={int(mode_val)}")
                static_df['Crop_Type'] = static_df['Crop_Type'].fillna(mode_val)
        return static_df

    def compile(self):
        yield_maps = self.fetch_usda_nass_yields()
        if not yield_maps:
            raise ValueError("USDA NASS returned no yield data")
        grid_yield_map = {}
        for crop_map in yield_maps.values():
            grid_yield_map.update(crop_map)
        grid_gdf = self.generate_grids(grid_yield_map)
        unique_mtrs = grid_gdf['MTRS'].unique()

        base_records = []
        for mtrs in unique_mtrs:
            df = pd.DataFrame({
                'date': self.date_range, 'MTRS': mtrs,
                'DOY': self.date_range.dayofyear
            })
            base_records.append(df)
        base_matrix = pd.concat(base_records, ignore_index=True)

        wx_df = self.fetch_weather(grid_gdf)
        static_df = self.fetch_static(grid_gdf)
        lai_df = self.fetch_s2_lai(grid_gdf)
        if self.skip_smap:
            print("[SMAP] Skipped by --skip-smap; validation fields will be NaN.")
            smap_df = pd.DataFrame()
        else:
            smap_df = self.fetch_smap(grid_gdf)

        static_df = self._fill_static_nans(static_df)

        merged = pd.merge(base_matrix, wx_df, on=['MTRS', 'date'], how='left')
        merged = pd.merge(merged, lai_df, on=['MTRS', 'date'], how='left')
        if not smap_df.empty:
            merged = pd.merge(merged, smap_df, on=['MTRS', 'date'], how='left')
        merged['LAI_mask'] = merged['LAI'].notna().astype(int)

        interp_cols = list(self.FORCING_COLS)
        missing_forcing_cols = [col for col in interp_cols if col not in merged.columns]
        if missing_forcing_cols:
            raise RuntimeError(
                f"Weather table is missing forcing columns: {missing_forcing_cols}"
            )
        smap_cols = ['sm_surface', 'sm_rootzone']
        for col in smap_cols:
            if col not in merged.columns:
                merged[col] = np.nan

        tensor_dict = {}
        skipped = 0

        for mtrs, group in merged.groupby('MTRS'):
            group = group.sort_values('date').copy()
            group = self._enforce_length(group)
            if len(group) != self.expected_days:
                skipped += 1
                continue

            state_alpha = mtrs.split('-')[0]
            county_name = mtrs.split('-')[1]
            yield_key = make_yield_key(state_alpha, county_name)

            observed_weather_days = int(group["Tmean"].notna().sum())
            if observed_weather_days < int(0.90 * self.expected_days):
                print(
                    f"  Skip {mtrs}: only {observed_weather_days}/"
                    f"{self.expected_days} observed weather days"
                )
                skipped += 1
                continue
            group[interp_cols] = group[interp_cols].interpolate(
                method='linear', limit_direction='both'
            )
            if group[interp_cols].isna().any().any():
                print(f"  Skip {mtrs}: forcing still contains NaN after interpolation")
                skipped += 1
                continue

            planting = self.planting_doy.get(state_alpha, 120)
            weather_issue = forcing_quality_issue(
                group[interp_cols].values,
                doy=group["DOY"].values,
                planting_doy=planting,
            )
            if weather_issue is not None:
                print(f"  Skip {mtrs}: invalid weather ({weather_issue})")
                skipped += 1
                continue

            static_row = static_df[static_df['MTRS'] == mtrs]
            if static_row.empty:
                skipped += 1
                continue

            n_rate = self.arms_n_rates.get(state_alpha, 0.0)
            static_base = static_row[['Elevation', 'Slope', 'Aspect', 'Bulk_Density',
                                       'SOC', 'Clay_Fraction', 'Sand_Fraction',
                                       'Total_Nitrogen', 'pH']].iloc[0].values.astype(np.float32)
            crop_type = static_row['Crop_Type'].iloc[0] if 'Crop_Type' in static_row.columns else 1
            crop_code = int(round(float(crop_type)))
            crop_yield_map = yield_maps.get(crop_code, {})
            if yield_key not in crop_yield_map:
                skipped += 1
                continue
            full_static = np.append(static_base, [n_rate, float(crop_type)]).astype(np.float32)

            gdd_vals = group['GDD'].values
            doy_vals = group['DOY'].values
            gdd_post = np.where(doy_vals >= planting, gdd_vals, 0.0)
            group['GDD_cumsum'] = np.cumsum(gdd_post)

            grid_info = grid_gdf[grid_gdf['MTRS'] == mtrs].iloc[0]

            tensor_dict[mtrs] = {
                "DOY": group['DOY'].values.astype(np.int32),
                "stress_forcing": group[interp_cols].values.astype(np.float32),
                "static_features": full_static,
                "obs_LAI": group['LAI'].fillna(0.0).values.astype(np.float32),
                "mask_LAI": group['LAI_mask'].values.astype(np.float32),
                "target_yield": float(crop_yield_map[yield_key]),
                "val_smap_surface": group['sm_surface'].values.astype(np.float32),
                "val_smap_rootzone": group['sm_rootzone'].values.astype(np.float32),
                "GDD_cumsum": group['GDD_cumsum'].values.astype(np.float32),
                "meta": {
                    "state": state_alpha, "county": county_name,
                    "lat": float(grid_info['Lat']), "lon": float(grid_info['Lon']),
                    "crop_type": crop_code,
                    "yield_crop_type": crop_code,
                    "yield_crop_name": CROP_YIELD_SPECS[crop_code]["name"],
                    "yield_source": "USDA NASS QuickStats county yield",
                    "yield_unit": "bu/acre",
                    "bushel_lb": CROP_YIELD_SPECS[crop_code]["bushel_lb"],
                    "planting_doy": planting,
                }
            }

        if len(tensor_dict) < int(MIN_WEATHER_COVERAGE * len(unique_mtrs)):
            raise RuntimeError(
                f"Only {len(tensor_dict)}/{len(unique_mtrs)} grids passed final QC "
                f"for {self.year}; refusing to save incomplete annual file."
            )

        save_path = os.path.join(
            self.output_dir, f"national_ode_tensors_v2_{self.year}.pkl"
        )
        tmp_path = save_path + ".tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(tensor_dict, f)
        os.replace(tmp_path, save_path)
        print(f"\nCompiled {len(tensor_dict)} grids (skipped {skipped})")
        print(f"Saved to {save_path}")
        self._print_quality_report(tensor_dict)
        return tensor_dict

    def _print_quality_report(self, tensor_dict):
        print(f"\n{'='*70}")
        print("DATA QUALITY REPORT")
        print(f"{'='*70}")
        n = len(tensor_dict)
        yields = [v['target_yield'] for v in tensor_dict.values()]
        print(f"  Grids: {n}")
        print(f"  Yield: mean={np.mean(yields):.1f} std={np.std(yields):.1f} "
              f"range=[{np.min(yields):.1f}, {np.max(yields):.1f}] bu/acre")

        lengths = [v['stress_forcing'].shape[0] for v in tensor_dict.values()]
        print(f"  Temporal length: min={min(lengths)} max={max(lengths)} expected={self.expected_days}")

        for i, col in enumerate(self.FORCING_COLS):
            vals = []
            for v in tensor_dict.values():
                sf = v['stress_forcing']
                if sf.shape[1] > i:
                    vals.append(sf[:, i])
            if vals:
                all_vals = np.concatenate(vals)
                print(f"  {col:>10s}: mean={np.nanmean(all_vals):8.3f} std={np.nanstd(all_vals):8.3f} "
                      f"nan={int(np.isnan(all_vals).sum())}")

        all_static = np.stack([v['static_features'] for v in tensor_dict.values()])
        for i, col in enumerate(self.STATIC_COLS):
            vals = all_static[:, i]
            print(f"  {col:>15s}: mean={np.nanmean(vals):8.3f} "
                  f"range=[{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}] "
                  f"nan={int(np.isnan(vals).sum())}")

        n_valid = sum(np.sum(v['mask_LAI']) for v in tensor_dict.values())
        print(f"  Valid LAI obs: {int(n_valid)} total ({n_valid / n:.1f} per grid)")

        smap_cov = sum(np.nansum(v['val_smap_surface']) > 0 for v in tensor_dict.values())
        print(f"  SMAP coverage: {smap_cov}/{n} grids have SMAP data")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AgriWorld 鏁版嵁绠＄嚎 (鍗曞勾)')
    parser.add_argument('--start', type=str, default='2022-01-01')
    parser.add_argument('--end', type=str, default='2022-12-31')
    parser.add_argument('--output', type=str, default=DATA_ROOT)
    parser.add_argument('--cache', type=str, default=CACHE_ROOT)
    parser.add_argument('--credentials', type=str, default=GEE_CREDENTIALS_PATH)
    parser.add_argument('--skip-smap', action='store_true',
                        help='璺宠繃鏈€鑰楁椂鐨?SMAP 涓嬭浇锛屽厛鐢熸垚璁粌鏁版嵁')
    args = parser.parse_args()

    try:
        pipeline = AgriWorldDataPipeline(
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output,
            cache_root=args.cache,
            credentials_path=args.credentials,
            skip_smap=args.skip_smap,
        )
        pipeline.compile()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

