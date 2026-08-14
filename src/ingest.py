"""Clean the Ben Roshan Indian e-commerce CSVs and load them into DuckDB."""

import os
import re
import sys

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.join("data", "raw")
DB_PATH = os.getenv("DUCKDB_PATH", "data/warehouse.duckdb")

ORDERS_CSV = os.path.join(RAW_DIR, "List of Orders.csv")
ORDER_DETAILS_CSV = os.path.join(RAW_DIR, "Order Details.csv")
SALES_TARGET_CSV = os.path.join(RAW_DIR, "Sales target.csv")


def to_snake_case(col: str) -> str:
    col = col.strip().replace("-", " ")
    col = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", col)
    col = re.sub(r"\s+", "_", col.strip())
    return col.lower()


def load_orders(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df.columns = [to_snake_case(c) for c in df.columns]
    df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y")
    return df


def load_order_details(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df.columns = [to_snake_case(c) for c in df.columns]
    return df


def load_sales_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df.columns = [to_snake_case(c) for c in df.columns]
    df = df.rename(columns={"month_of_order_date": "month_raw"})
    df["month_key"] = pd.to_datetime(df["month_raw"], format="%b-%y").dt.strftime("%Y-%m")
    df = df.drop(columns=["month_raw"])
    return df


def main() -> None:
    for path in (ORDERS_CSV, ORDER_DETAILS_CSV, SALES_TARGET_CSV):
        if not os.path.exists(path):
            print(f"Missing input file: {path}", file=sys.stderr)
            sys.exit(1)

    orders = load_orders(ORDERS_CSV)
    order_items = load_order_details(ORDER_DETAILS_CSV)
    sales_targets = load_sales_targets(SALES_TARGET_CSV)

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE OR REPLACE TABLE orders AS SELECT * FROM orders")
    con.execute("CREATE OR REPLACE TABLE order_items AS SELECT * FROM order_items")
    con.execute("CREATE OR REPLACE TABLE sales_targets AS SELECT * FROM sales_targets")

    print(f"Loaded into {DB_PATH}\n")
    for table in ("orders", "order_items", "sales_targets"):
        print(f"=== {table} ===")
        con.sql(f"DESCRIBE {table}").show()
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"row count: {count}\n")

    con.close()


if __name__ == "__main__":
    main()
