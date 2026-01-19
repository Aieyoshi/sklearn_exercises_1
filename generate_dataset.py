import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_train_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Features (13 columnas) ---
    # bedrooms: la mayoría entre 1 y 5, pero dejamos algunos valores altos (hasta 12)
    bedrooms = rng.poisson(lam=2.2, size=n_rows) + 1
    bedrooms = np.clip(bedrooms, 1, 12)

    # total_area correlacionada con bedrooms
    total_area = bedrooms * \
        rng.uniform(18, 35, size=n_rows) + rng.normal(45, 18, size=n_rows)
    total_area = np.clip(total_area, 20, 1200)

    # living_area y kitchen_area como proporciones del total
    living_share = rng.uniform(0.45, 0.80, size=n_rows)
    kitchen_share = rng.uniform(0.07, 0.22, size=n_rows)

    living_area = total_area * living_share
    kitchen_area = total_area * kitchen_share

    # Ajuste para que living + kitchen no exceda total_area
    too_big = (living_area + kitchen_area) > (total_area * 0.95)
    kitchen_area[too_big] = total_area[too_big] * 0.12
    living_area[too_big] = total_area[too_big] * 0.68

    # Otras features “extra” (para completar 13 columnas)
    ceiling_height = np.clip(rng.normal(2.75, 0.18, size=n_rows), 2.2, 3.6)
    floors_total = rng.integers(1, 41, size=n_rows)
    floor = np.array([rng.integers(1, ft + 1) for ft in floors_total])

    is_new_building = rng.integers(0, 2, size=n_rows)
    has_elevator = (floors_total >= 5).astype(
        int)  # más probable si hay muchos pisos
    has_parking = rng.integers(0, 2, size=n_rows)
    has_balcony = rng.integers(0, 2, size=n_rows)

    # Distancias (en metros)
    city_center_dist = rng.uniform(500, 40000, size=n_rows)
    metro_dist = rng.uniform(100, 10000, size=n_rows)

    # --- last_price (objetivo real) ---
    noise = rng.normal(0, 15000, size=n_rows)

    last_price = (
        20000
        + total_area * 950
        + bedrooms * 4500
        + is_new_building * 16000
        + has_elevator * 8000
        + has_parking * 9000
        + has_balcony * 3500
        - city_center_dist * 0.55
        - metro_dist * 0.25
        + noise
    )
    last_price = np.clip(last_price, 20000, None)

    # Ajuste para que la mediana quede ~113000 (como tu umbral)
    median_price = np.median(last_price)
    scale = 113000 / median_price
    last_price = last_price * scale

    df = pd.DataFrame({
        "bedrooms": bedrooms.astype(int),
        "total_area": total_area,
        "ceiling_height": ceiling_height,
        "floors_total": floors_total.astype(int),
        "living_area": living_area,
        "floor": floor.astype(int),
        "is_new_building": is_new_building.astype(int),
        "has_elevator": has_elevator.astype(int),
        "has_parking": has_parking.astype(int),
        "kitchen_area": kitchen_area,
        "has_balcony": has_balcony.astype(int),
        "city_center_dist": city_center_dist,
        "metro_dist": metro_dist,
        "last_price": np.round(last_price, 2),
    })

    return df


if __name__ == "__main__":
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "train_data_us.csv"
    df = generate_synthetic_train_data(n_rows=5000, seed=42)
    df.to_csv(out_path, index=False)

    print(f"✅ Dataset guardado en: {out_path}")
    print("Mediana last_price:", df["last_price"].median())
    print(df.head())
