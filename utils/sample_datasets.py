"""
Sample dataset generators for demo/playground use.
Each function returns a pd.DataFrame with realistic synthetic data.
"""

import numpy as np
import pandas as pd


def generate_sales_data():
    rng = np.random.default_rng(42)
    n = 500
    regions = ["East", "West", "North", "South"]
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    products = {
        "Electronics": ["Laptop", "Headphones", "Tablet", "Smartwatch"],
        "Clothing": ["T-Shirt", "Jacket", "Sneakers", "Jeans"],
        "Home & Garden": ["Lamp", "Rug", "Plant Pot", "Curtains"],
        "Sports": ["Yoga Mat", "Dumbbells", "Basketball", "Tennis Racket"],
        "Books": ["Novel", "Textbook", "Cookbook", "Biography"],
    }
    segments = ["Consumer", "Corporate", "Small Business"]

    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    rows = []
    for _ in range(n):
        cat = rng.choice(categories)
        prod = rng.choice(products[cat])
        qty = int(rng.integers(1, 20))
        price = round(rng.uniform(5, 500), 2)
        discount = round(rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2]), 2)
        rows.append({
            "date": rng.choice(dates),
            "region": rng.choice(regions),
            "product_category": cat,
            "product_name": prod,
            "quantity": qty,
            "unit_price": price,
            "revenue": round(qty * price * (1 - discount), 2),
            "discount": discount,
            "customer_segment": rng.choice(segments),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def generate_student_data():
    rng = np.random.default_rng(42)
    n = 300
    rows = []
    for i in range(n):
        study = round(rng.uniform(2, 40), 1)
        attendance = round(rng.uniform(50, 100), 1)
        base = study * 1.5 + attendance * 0.3
        rows.append({
            "student_id": f"S{1000 + i}",
            "gender": rng.choice(["Male", "Female"]),
            "age": int(rng.integers(17, 25)),
            "study_hours_per_week": study,
            "attendance_pct": attendance,
            "math_score": min(100, max(0, round(base + rng.normal(0, 10), 1))),
            "science_score": min(100, max(0, round(base + rng.normal(0, 12), 1))),
            "english_score": min(100, max(0, round(base + rng.normal(0, 11), 1))),
            "grade": rng.choice(["A", "B", "C", "D", "F"], p=[0.15, 0.30, 0.30, 0.15, 0.10]),
            "extracurricular": rng.choice(["Yes", "No"]),
        })
    return pd.DataFrame(rows)


def generate_weather_data():
    rng = np.random.default_rng(42)
    cities = ["New York", "Los Angeles", "Chicago", "Houston"]
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rows = []
    for date in dates:
        for city in cities:
            month = date.month
            base_temp = 10 + 15 * np.sin((month - 3) * np.pi / 6)
            rows.append({
                "date": date,
                "city": city,
                "temperature_c": round(base_temp + rng.normal(0, 5), 1),
                "humidity_pct": round(rng.uniform(30, 95), 1),
                "wind_speed_kmh": round(rng.uniform(0, 50), 1),
                "precipitation_mm": round(max(0, rng.normal(2, 5)), 1),
                "condition": rng.choice(["Sunny", "Rainy", "Cloudy", "Snowy"],
                                        p=[0.35, 0.25, 0.30, 0.10]),
            })
    return pd.DataFrame(rows)


def generate_stock_data():
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    dates = pd.date_range("2023-01-01", periods=125, freq="B")
    base_prices = {"AAPL": 150, "GOOGL": 100, "MSFT": 250, "AMZN": 95}
    rows = []
    for ticker in tickers:
        price = base_prices[ticker]
        for date in dates:
            change = rng.normal(0, 0.02)
            price *= (1 + change)
            o = round(price, 2)
            h = round(o * (1 + abs(rng.normal(0, 0.01))), 2)
            l = round(o * (1 - abs(rng.normal(0, 0.01))), 2)
            c = round(rng.uniform(l, h), 2)
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": int(rng.integers(1_000_000, 50_000_000)),
            })
    return pd.DataFrame(rows)


SAMPLE_DATASETS = {
    "Sales Data": {"fn": generate_sales_data, "desc": "Retail sales across regions and products"},
    "Student Performance": {"fn": generate_student_data, "desc": "Student scores and demographics"},
    "Weather Data": {"fn": generate_weather_data, "desc": "Daily weather observations by city"},
    "Stock Prices": {"fn": generate_stock_data, "desc": "Daily OHLCV for 4 tech stocks"},
}
