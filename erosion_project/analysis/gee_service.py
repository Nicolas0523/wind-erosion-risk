# analysis/gee_service.py
import ee

def init_gee():
    service_account = "my-bot-service@cogent-sunspot-472315-q5.iam.gserviceaccount.com"
    credentials = ee.ServiceAccountCredentials(service_account, "key.json")
    ee.Initialize(credentials)

def get_avg_wind(polygon):
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate("2024-01-01", "2024-12-31")

    def add_wind(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        wind = u.pow(2).add(v.pow(2)).sqrt().rename("wind")
        return img.addBands(wind)

    era5_wind = era5.map(add_wind)
    mean_wind = era5_wind.select("wind").mean()
    stats = mean_wind.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=10000
    )
    return stats.getInfo()
