
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "cia_real_warehouse.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_warehouse():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS Dim_Date (
            date_id INTEGER PRIMARY KEY,
            year INTEGER UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS Dim_Location (
            location_id INTEGER PRIMARY KEY,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            UNIQUE(state, district)
        );

        CREATE TABLE IF NOT EXISTS Dim_Crime_Type (
            crime_type_id INTEGER PRIMARY KEY,
            crime_type TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS Fact_Crime_Stats (
            date_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            crime_type_id INTEGER NOT NULL,
            crime_count INTEGER NOT NULL,
            PRIMARY KEY(date_id, location_id, crime_type_id),
            FOREIGN KEY(date_id) REFERENCES Dim_Date(date_id),
            FOREIGN KEY(location_id)
                REFERENCES Dim_Location(location_id),
            FOREIGN KEY(crime_type_id)
                REFERENCES Dim_Crime_Type(crime_type_id)
        );

        CREATE TABLE IF NOT EXISTS Fact_District_Totals (
            date_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            total_ipc_crimes INTEGER NOT NULL,
            PRIMARY KEY(date_id, location_id),
            FOREIGN KEY(date_id) REFERENCES Dim_Date(date_id),
            FOREIGN KEY(location_id)
                REFERENCES Dim_Location(location_id)
        );

        CREATE INDEX IF NOT EXISTS idx_stats_location
            ON Fact_Crime_Stats(location_id);

        CREATE INDEX IF NOT EXISTS idx_stats_crime_type
            ON Fact_Crime_Stats(crime_type_id);
        """)


if __name__ == "__main__":
    create_warehouse()
    print("CIA warehouse schema updated successfully!")
