from django.shortcuts import render
import ee, os, json, traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from joblib import load
import numpy as np



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models")

lr_model = load(os.path.join(model_path, "linear_model.pkl"))



# Read env vars
service_account = os.getenv("EE_SERVICE_ACCOUNT")
key_path = os.getenv("EE_KEY_PATH")

print("SERVICE_ACCOUNT:", service_account)
print("KEY_PATH:", key_path)
print("FILE EXISTS:", os.path.exists(key_path))

# Debug: check key correctness
print("DEBUG: reading key file...")
try:
    with open(key_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("DEBUG: key file successfully read.")
    print("DEBUG: key type:", data.get("type"))
    print("DEBUG: client_email:", data.get("client_email"))
except Exception as e:
    print("DEBUG: error reading key:", e)

# Initialize GEE
try:
    credentials = ee.ServiceAccountCredentials(service_account, key_path)
    ee.Initialize(credentials)
    print("✅ GEE connected successfully!")
except Exception as e:
    print("❌ Error initializing GEE:", e)


@csrf_exempt
def home(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            geometry = data.get('geometry')
            start = data.get("start_date")
            end = data.get("end_date")
            model_choice = data.get('model')
            polygon = ee.Geometry.Polygon(geometry["coordinates"][0])

            # === NDVI ===
            def add_ndvi(img):
                return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))

            ndvi = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterDate(start, end) \
                .filterBounds(polygon) \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
                .map(add_ndvi)

            ndvi_mean = ndvi.select("NDVI").mean()

            # === Wind ===
            def add_wind(img):
                u = img.select("u_component_of_wind_10m")
                v = img.select("v_component_of_wind_10m")
                return img.addBands(u.hypot(v).rename("wind"))

            wind = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterDate(start, end) \
                .map(add_wind).select("wind").mean()

            # === Soil moisture ===
            sm = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterDate(start, end) \
                .select("volumetric_soil_water_layer_1")

            sm_mean = sm.mean()

            # === Slope ===
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem).rename("slope")

            # === Soil texture ===
            soil_texture = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
            sand = soil_texture.select("b0").rename("soil_type")

            # === Precipitation ===
            precip_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                .select("total_precipitation_sum") \
                .filterDate(start, end) \
                .filterBounds(polygon)

            precip_img = ee.Image(
                ee.Algorithms.If(
                    precip_col.size().gt(0),
                    precip_col.sum().multiply(1000).rename("precipitation_mm"),
                    ee.Image.constant(0).rename("precipitation_mm")
                )
            )

            # === Temperature ===
            temp_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                .select("temperature_2m") \
                .filterDate(start, end) \
                .filterBounds(polygon)

            temp_img = ee.Image(
                ee.Algorithms.If(
                    temp_col.size().gt(0),
                    temp_col.mean().subtract(273.15).rename("temperature_C"),
                    ee.Image.constant(0).rename("temperature_C")
                )
            )

            # === Автоматически извлекаем коэффициенты ===
            coef_dict = {name: coef for name, coef in zip(lr_model.feature_names_in_, lr_model.coef_)}
            intercept = lr_model.intercept_

            print("📊 Coefficients loaded automatically:")
            for k, v in coef_dict.items():
                print(f"{k}: {v}")

            # === Привязка GEE изображений к именам признаков ===
            feature_imgs = {
                "NDVI": ndvi_mean,
                "sm_surface": sm_mean,
                "temperature_2m": temp_img,
                "total_precipitation_sum": precip_img,
                "soil_type": sand,
                "wind_speed": wind
            }

            # === Генерация выражения автоматически ===
            # создаём выражение для каждого признака
            risk_img = ee.Image.constant(intercept)
            for name, coef in coef_dict.items():
                img = feature_imgs.get(name)
                if img:
                    risk_img = risk_img.add(img.multiply(coef))
                else:
                    print(f"⚠️ Warning: feature {name} not found in GEE images!")

            # === Нормализация для визуализации ===
            risk_scaled = risk_img.unitScale(
                ee.Number(risk_img.reduceRegion(reducer=ee.Reducer.min(), geometry=polygon, scale=1000).values().reduce(ee.Reducer.min())),
                ee.Number(risk_img.reduceRegion(reducer=ee.Reducer.max(), geometry=polygon, scale=1000).values().reduce(ee.Reducer.max()))
            ).rename("ML_Risk")

            vis = {"min": 0, "max": 1, "palette": ["green", "yellow", "red"]}
            map_id = ee.Image(risk_scaled).getMapId(vis)

            pred = risk_scaled.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=1000
            ).get("ML_Risk").getInfo()


            return JsonResponse({"tile_url": map_id["tile_fetcher"].url_format, "prediction": pred})

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)
    return render(request, "analysis/home.html")



@csrf_exempt
def get_data(request):
    try:
        start = request.GET.get("start")
        end = request.GET.get("end")
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        point = ee.Geometry.Point([lon, lat])

        # NDVI
        sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start, end) \
            .filterBounds(point) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        ndvi = sentinel.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")) \
                       .select("NDVI").mean().reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=point,
                            scale=30,
                            bestEffort=True
                        ).get("NDVI").getInfo()

        # Wind
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start, end)
        wind = era5.map(lambda img: img.expression(
            "sqrt(u*u + v*v)", {
                "u": img.select("u_component_of_wind_10m"),
                "v": img.select("v_component_of_wind_10m")
            }
        ).rename("wind")).select("wind").mean() \
        .reduceRegion(ee.Reducer.mean(), point, 1000).get("wind").getInfo()

        # Soil moisture
        soil = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
            .filterDate(start, end) \
            .select("volumetric_soil_water_layer_1") \
            .mean().reduceRegion(ee.Reducer.mean(), point, 1000) \
            .get("volumetric_soil_water_layer_1").getInfo()

        # Risk
        risk_value = (1 - (ndvi or 0)) * 0.3 + (wind or 0) / 10 * 0.4 + (1 - (soil or 0)) * 0.3
        
        if risk_value > 0.8: risk_level = "High"
        elif risk_value > 0.5: risk_level = "Medium"
        else: risk_level = "Low"

        return JsonResponse({
            "lat": lat, "lon": lon,
            "ndvi": round(ndvi or 0, 3),
            "wind": round(wind or 0, 2),
            "soil_moisture": round(soil or 0, 3),
            "risk": risk_level
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

from django.shortcuts import render
import ee, os, json, traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from joblib import load
import numpy as np
from tempfile import NamedTemporaryFile


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models")

lr_model = load(os.path.join(model_path, "linear_model.pkl"))


# Читаем переменные окружения
service_account = os.environ.get("EE_SERVICE_ACCOUNT")
key_json_str = os.environ.get("EE_KEY_JSON")

# Создаем временный файл с ключом
with NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
    f.write(key_json_str)
    key_path = f.name

# Инициализация GEE
credentials = ee.ServiceAccountCredentials(service_account, key_path)
ee.Initialize(credentials)
print("✅ GEE connected successfully!")


@csrf_exempt
def home(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            geometry = data.get('geometry')
            start = data.get("start_date")
            end = data.get("end_date")
            model_choice = data.get('model')
            polygon = ee.Geometry.Polygon(geometry["coordinates"][0])

            # === NDVI ===
            def add_ndvi(img):
                return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))

            ndvi = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterDate(start, end) \
                .filterBounds(polygon) \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
                .map(add_ndvi)

            ndvi_mean = ndvi.select("NDVI").mean()

            # === Wind ===
            def add_wind(img):
                u = img.select("u_component_of_wind_10m")
                v = img.select("v_component_of_wind_10m")
                return img.addBands(u.hypot(v).rename("wind"))

            wind = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterDate(start, end) \
                .map(add_wind).select("wind").mean()

            # === Soil moisture ===
            sm = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterDate(start, end) \
                .select("volumetric_soil_water_layer_1")

            sm_mean = sm.mean()

            # === Slope ===
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem).rename("slope")

            # === Soil texture ===
            soil_texture = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
            sand = soil_texture.select("b0").rename("soil_type")

            # === Precipitation ===
            precip_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                .select("total_precipitation_sum") \
                .filterDate(start, end) \
                .filterBounds(polygon)

            precip_img = ee.Image(
                ee.Algorithms.If(
                    precip_col.size().gt(0),
                    precip_col.sum().multiply(1000).rename("precipitation_mm"),
                    ee.Image.constant(0).rename("precipitation_mm")
                )
            )

            # === Temperature ===
            temp_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                .select("temperature_2m") \
                .filterDate(start, end) \
                .filterBounds(polygon)

            temp_img = ee.Image(
                ee.Algorithms.If(
                    temp_col.size().gt(0),
                    temp_col.mean().subtract(273.15).rename("temperature_C"),
                    ee.Image.constant(0).rename("temperature_C")
                )
            )

            # === Автоматически извлекаем коэффициенты ===
            coef_dict = {name: coef for name, coef in zip(lr_model.feature_names_in_, lr_model.coef_)}
            intercept = lr_model.intercept_

            print("📊 Coefficients loaded automatically:")
            for k, v in coef_dict.items():
                print(f"{k}: {v}")

            # === Привязка GEE изображений к именам признаков ===
            feature_imgs = {
                "NDVI": ndvi_mean,
                "sm_surface": sm_mean,
                "temperature_2m": temp_img,
                "total_precipitation_sum": precip_img,
                "soil_type": sand,
                "wind_speed": wind
            }

            # === Генерация выражения автоматически ===
            # создаём выражение для каждого признака
            risk_img = ee.Image.constant(intercept)
            for name, coef in coef_dict.items():
                img = feature_imgs.get(name)
                if img:
                    risk_img = risk_img.add(img.multiply(coef))
                else:
                    print(f"⚠️ Warning: feature {name} not found in GEE images!")

            # === Нормализация для визуализации ===
            risk_scaled = risk_img.unitScale(
                ee.Number(risk_img.reduceRegion(reducer=ee.Reducer.min(), geometry=polygon, scale=1000).values().reduce(ee.Reducer.min())),
                ee.Number(risk_img.reduceRegion(reducer=ee.Reducer.max(), geometry=polygon, scale=1000).values().reduce(ee.Reducer.max()))
            ).rename("ML_Risk")

            vis = {"min": 0, "max": 1, "palette": ["green", "yellow", "red"]}
            map_id = ee.Image(risk_scaled).getMapId(vis)

            pred = risk_scaled.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=1000
            ).get("ML_Risk").getInfo()


            return JsonResponse({"tile_url": map_id["tile_fetcher"].url_format, "prediction": pred})

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)
    return render(request, "analysis/home.html")



@csrf_exempt
def get_data(request):
    try:
        start = request.GET.get("start")
        end = request.GET.get("end")
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
        point = ee.Geometry.Point([lon, lat])

        # NDVI
        sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start, end) \
            .filterBounds(point) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        ndvi = sentinel.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")) \
                       .select("NDVI").mean().reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=point,
                            scale=30,
                            bestEffort=True
                        ).get("NDVI").getInfo()

        # Wind
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start, end)
        wind = era5.map(lambda img: img.expression(
            "sqrt(u*u + v*v)", {
                "u": img.select("u_component_of_wind_10m"),
                "v": img.select("v_component_of_wind_10m")
            }
        ).rename("wind")).select("wind").mean() \
        .reduceRegion(ee.Reducer.mean(), point, 1000).get("wind").getInfo()

        # Soil moisture
        soil = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
            .filterDate(start, end) \
            .select("volumetric_soil_water_layer_1") \
            .mean().reduceRegion(ee.Reducer.mean(), point, 1000) \
            .get("volumetric_soil_water_layer_1").getInfo()

        # Risk
        risk_value = (1 - (ndvi or 0)) * 0.3 + (wind or 0) / 10 * 0.4 + (1 - (soil or 0)) * 0.3
        
        if risk_value > 0.8: risk_level = "High"
        elif risk_value > 0.5: risk_level = "Medium"
        else: risk_level = "Low"

        return JsonResponse({
            "lat": lat, "lon": lon,
            "ndvi": round(ndvi or 0, 3),
            "wind": round(wind or 0, 2),
            "soil_moisture": round(soil or 0, 3),
            "risk": risk_level
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


