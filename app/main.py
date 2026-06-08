# app/main.py
import os
import cv2
from core.pipeline import PressurePipeline
from regressor.mobilenet_v3_predictor import MobileNetV3Predictor
from visualization.comparison_visualizer import ComparisonVisualizer


IMG_DIR = "app/data_model/real/test/images"
MODEL_DIR = "app/models"


# --------------------------------------------------
# MODELS
# --------------------------------------------------

mobilenet = MobileNetV3Predictor(
    os.path.join(
        MODEL_DIR,
        "mobilenet_v3_epoch_59.pth"
    )
)

print("MobileNet loaded successfully")


pipeline = PressurePipeline(
    gauge_model_path=os.path.join(
        MODEL_DIR,
        "best_gauge.pt"
    ),
    seg_model_path=os.path.join(
        MODEL_DIR,
        "best_seg.pt"
    ),
    pressure_max=4
)

print("PressurePipeline loaded successfully")


# --------------------------------------------------
# IMAGES
# --------------------------------------------------

EXTS = (".jpg", ".jpeg", ".png")

files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(EXTS)
])

print(f"\nFound {len(files)} images\n")


# --------------------------------------------------
# LOOP
# --------------------------------------------------

for fname in files:

    path = os.path.join(IMG_DIR, fname)

    img = cv2.imread(path)

    if img is None:
        print(f"[SKIP] {fname} -> cannot read image")
        continue

    # ----------------------------------------------
    # PIPELINE
    # ----------------------------------------------

    pipeline_pressure = None
    pipeline_error = None

    try:

        pipeline_pressure = pipeline.process(
            img,
            file_name=fname,
            visualize=False
        )

    except Exception as e:

        pipeline_error = str(e)

    # ----------------------------------------------
    # MOBILENET
    # ----------------------------------------------

    mobile_pressure = None
    mobile_scale = None
    mobile_scale_class = None
    mobile_latency = None

    try:

        (
            mobile_pressure,
            mobile_scale,
            mobile_scale_class,
            mobile_latency
        ) = mobilenet.predict(img)

    except Exception as e:

        print(
            f"[MobileNet ERR] {fname} -> {e}"
        )

    # ----------------------------------------------
    # REPORT
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print(fname)
    print("=" * 60)

    # Pipeline

    if pipeline_error is not None:

        print(
            f"Pipeline          : ERROR ({pipeline_error})"
        )

    else:

        print(
            f"Pipeline Pressure : "
            f"{pipeline_pressure:.2f} MPa"
        )

    # MobileNet

    if mobile_pressure is not None:

        print(
            f"MobileNet Pressure: "
            f"{mobile_pressure:.2f} MPa"
        )

        print(
            f"MobileNet Scale   : "
            f"{mobile_scale:.1f}"
        )

        print(
            f"MobileNet Class   : "
            f"{mobile_scale_class}"
        )

        print(
            f"MobileNet Latency : "
            f"{mobile_latency:.1f} ms"
        )

    else:

        print(
            "MobileNet         : ERROR"
        )

    # Difference

    if (
        pipeline_pressure is not None
        and mobile_pressure is not None
    ):

        delta = abs(
            pipeline_pressure -
            mobile_pressure
        )

        print(
            f"Delta             : "
            f"{delta:.2f} MPa"
        )

    # ----------------------------------------------
    # VISUALIZATION
    # ----------------------------------------------

    if mobile_pressure is not None:

        ComparisonVisualizer.show(
            file_name=fname,
            image=img,
            pipeline_pressure=pipeline_pressure,
            mobile_pressure=mobile_pressure,
            mobile_scale=mobile_scale,
            mobile_latency=mobile_latency
        )

print("\nFinished.")



# import os
# import cv2
# from  core.pipeline import PressurePipeline
# from regressor.mobilenet_v3_predictor import MobileNetV3Predictor
# from visualization.comparison_visualizer import ComparisonVisualizer

# IMG_DIR = "app/data_model/real/test/images"

# MODEL_DIR = "app/models"

# mobilenet = MobileNetV3Predictor(
#     os.path.join(MODEL_DIR, "mobilenet_v3_epoch_59.pth")
# )
# print("MobileNet loaded successfully")

# pipeline = PressurePipeline(
#     gauge_model_path=os.path.join(MODEL_DIR, "best_gauge.pt"),
#     seg_model_path=os.path.join(MODEL_DIR, "best_seg.pt"),
#     pressure_max=4
# )

# # допустимые расширения
# EXTS = (".jpg", ".jpeg", ".png")

# files = sorted([
#     f for f in os.listdir(IMG_DIR)
#     if f.lower().endswith(EXTS)
# ])

# print(f"Found {len(files)} images\n")

# for fname in files:
#     path = os.path.join(IMG_DIR, fname)
#     img = cv2.imread(path)

#     if img is None:
#         print(f"[SKIP] {fname} — cannot read image")
#         continue

#     try:
#         # pipeline_pressure = pipeline.process(img, visualize=True)
#         pipeline_pressure = pipeline.process(
#             img,
#             file_name=fname,
#             visualize=False
#         )
#         mobile_pressure, mobile_scale, mobile_scale_class, mobile_latency = mobilenet.predict(img)
#         ComparisonVisualizer.show(
#             file_name=fname,
#             image=img,
#             pipeline_pressure=pipeline_pressure,
#             mobile_pressure=mobile_pressure,
#             mobile_scale=mobile_scale,
#             mobile_latency=mobile_latency
#         )

#         # print(f"[OK] {fname} → PRESSURE = {pressure:.2f}")
#         print(
#             f"\n[OK] {fname}\n"
#             f"  Pipeline Pressure : {pipeline_pressure:.2f}\n"
#             f"  MobileNet Pressure: {mobile_pressure:.2f}\n"
#             f"  MobileNet Scale   : {mobile_scale:.1f}\n"
#             f"  MobileNet Latency : {mobile_latency:.1f} ms\n"
#         )
#     except Exception as e:
#         print(f"[ERR] {fname} → {e}")

"""
Analog Gauge Reader Pipeline
Copyright (c) 2026 Konstantin Mazurenkov
Licensed under MIT License
"""
