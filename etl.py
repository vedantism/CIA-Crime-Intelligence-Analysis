
import pandas as pd

from warehouse import get_connection, create_warehouse, BASE_DIR

CSV_PATH = BASE_DIR / "data" / "dstrIPC_1.csv"

TOTAL_COLUMN = "TOTAL IPC CRIMES"


def run_etl():
    # Check whether the source dataset exists.
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Place dstrIPC_1.csv inside the data folder."
        )

    create_warehouse()

    # ==========================================
    # 1. EXTRACT
    # ==========================================
    print("Extracting crime data...")

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    required = [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        TOTAL_COLUMN,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns missing from CSV: {missing}"
        )

    original_rows = len(df)

    # ==========================================
    # 2. TRANSFORM
    # ==========================================
    print("Cleaning and transforming data...")

    df = df.rename(columns={
        "STATE/UT": "state",
        "DISTRICT": "district",
        "YEAR": "year",
        TOTAL_COLUMN: "total_ipc_crimes",
    })

    # Standardize geographical fields.
    df["state"] = (
        df["state"].astype("string").str.strip().str.upper()
    )

    df["district"] = (
        df["district"].astype("string").str.strip().str.upper()
    )

    # Convert year and total counts to numeric values.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df["total_ipc_crimes"] = pd.to_numeric(
        df["total_ipc_crimes"], errors="coerce"
    )

    # Remove records with missing essential fields.
    df = df.dropna(
        subset=[
            "state",
            "district",
            "year",
            "total_ipc_crimes",
        ]
    ).copy()

    df = df[
        (df["state"] != "")
        & (df["district"] != "")
        & (df["state"].str.lower() != "nan")
        & (df["district"].str.lower() != "nan")
    ].copy()

    df["year"] = df["year"].astype(int)
    df["total_ipc_crimes"] = df["total_ipc_crimes"].astype(int)

    # Identify crime-category columns.
    excluded_columns = {
        "state",
        "district",
        "year",
        "total_ipc_crimes",
    }

    crime_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    # Convert crime counts to numeric values.
    for col in crime_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing crime counts and invalid negative values.
    df[crime_columns] = df[crime_columns].fillna(0)
    df[crime_columns] = df[crime_columns].clip(lower=0).astype(int)

    # Keep valid years and nonnegative totals.
    df = df[
        df["year"].between(1900, 2100)
        & (df["total_ipc_crimes"] >= 0)
    ].copy()

    # Resolve duplicate state-district-year records.
    duplicate_mask = df.duplicated(
        subset=["state", "district", "year"],
        keep=False,
    )

    duplicate_rows = int(duplicate_mask.sum())

    if duplicate_rows:
        print(
            f"Warning: {duplicate_rows} rows belong to "
            "duplicate district-year combinations."
        )

        numeric_columns = [
            "total_ipc_crimes",
            *crime_columns,
        ]

        df = df.groupby(
            ["state", "district", "year"],
            as_index=False,
        )[numeric_columns].sum()

        print("Duplicate groups aggregated by summing their counts.")

    # Reshape from wide format to long format.
    # Each row will represent one crime category
    # for a particular district and year.
    long_df = df.melt(
        id_vars=["state", "district", "year"],
        value_vars=crime_columns,
        var_name="crime_type",
        value_name="crime_count",
    )

    print(f"Source CSV rows: {original_rows:,}")
    print(f"Valid district-year rows: {len(df):,}")
    print(f"Crime-category records to load: {len(long_df):,}")

    # ==========================================
    # 3. LOAD
    # ==========================================
    print("Loading data into the SQLite warehouse...")

    with get_connection() as conn:

        # Load the date dimension.
        conn.executemany(
            """
            INSERT OR IGNORE INTO Dim_Date (date_id, year)
            VALUES (?, ?)
            """,
            [
                (int(year), int(year))
                for year in sorted(df["year"].unique())
            ],
        )

        # Load the location dimension.
        locations = (
            df[["state", "district"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        conn.executemany(
            """
            INSERT OR IGNORE INTO Dim_Location (state, district)
            VALUES (?, ?)
            """,
            list(locations),
        )

        # Load the crime-type dimension.
        conn.executemany(
            """
            INSERT OR IGNORE INTO Dim_Crime_Type (crime_type)
            VALUES (?)
            """,
            [(name,) for name in crime_columns],
        )

        # Build lookup dictionaries for dimension keys.
        location_map = {
            (state, district): location_id
            for location_id, state, district in conn.execute(
                """
                SELECT location_id, state, district
                FROM Dim_Location
                """
            ).fetchall()
        }

        crime_map = {
            name: crime_id
            for crime_id, name in conn.execute(
                """
                SELECT crime_type_id, crime_type
                FROM Dim_Crime_Type
                """
            ).fetchall()
        }

        # Load the crime statistics fact table.
        conn.executemany(
            """
            INSERT OR REPLACE INTO Fact_Crime_Stats
            (date_id, location_id, crime_type_id, crime_count)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    int(row.year),
                    location_map[(row.state, row.district)],
                    crime_map[row.crime_type],
                    int(row.crime_count),
                )
                for row in long_df.itertuples(index=False)
            ],
        )

        # Load aggregate IPC totals separately.
        conn.executemany(
            """
            INSERT OR REPLACE INTO Fact_District_Totals
            (date_id, location_id, total_ipc_crimes)
            VALUES (?, ?, ?)
            """,
            [
                (
                    int(row.year),
                    location_map[(row.state, row.district)],
                    int(row.total_ipc_crimes),
                )
                for row in df[
                    [
                        "state",
                        "district",
                        "year",
                        "total_ipc_crimes",
                    ]
                ].itertuples(index=False)
            ],
        )

        # ==========================================
        # 4. VERIFICATION
        # ==========================================
        print("\nETL completed successfully!")
        print("\nWarehouse table counts:")

        tables = [
            "Dim_Date",
            "Dim_Location",
            "Dim_Crime_Type",
            "Fact_Crime_Stats",
            "Fact_District_Totals",
        ]

        for table in tables:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            print(f"{table}: {count:,} rows")


if __name__ == "__main__":
    run_etl()
