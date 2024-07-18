import juliacall
import juliapkg
import json

juliapkg_path = juliapkg.project()

print(juliapkg_path)

with open(juliapkg_path + "/pyjuliapkg/juliapkg.json", "r") as f:
    juliapkg_data = json.load(f)

print(juliapkg_data)

pkg_list = juliapkg_data.get("packages", [])

if "SnowFlurry" not in pkg_list:
    juliapkg.add("SnowFlurry", "7bd9edc1-4fdc-40a1-a0f6-da58fb4f45ec", version="0.3")
