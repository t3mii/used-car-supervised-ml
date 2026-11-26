from libs import *
columns = {"price", "year","manufacturer", "model","condition", "cylinders", "fuel", "odometer","title_status", "transmission", "drive","type", "paint_color", "state", "posting_date", "VIN"}
df = pd.read_csv("vehicles.csv", usecols=columns, low_memory=False)


#cleaning
original_size = len(df)

#duplicates and NaN values
df=df.drop_duplicates()
df=df.dropna(subset=["price", "year", "odometer","manufacturer", "model"])
df = df[~(df["VIN"].notna() & df.duplicated(subset="VIN"))]

#price column was weird  
df=df[df["price"]<=100000]    
df["price"] = df["price"].astype("int64")

#make listing date readable
df["posting_date"] = pd.to_datetime(
    df["posting_date"],
    errors="coerce",
    utc=True
)
df = df.dropna(subset=["posting_date"])
#create age feature
df["age"] = df["posting_date"].dt.year - df["year"]

#drop unrealistic values
df = df[
    (df["price"] >= 1000) &
    (df["odometer"] >= 0) &
    (df["odometer"] <= 500000) &
    (df["age"] >= 0) &
    (df["age"] <= 100)
]

categorical_columns = [
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type"
]

for column in categorical_columns:
    df[column] = df[column].fillna("unknown")

