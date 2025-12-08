# 1. Load the CSV
df = pd.read_csv("NP_Adresses.csv", dtype=str)

# 2. Clean column names
df.columns = (
    df.columns.str.strip()           # remove spaces at ends
             .str.lower()            # make lowercase
             .str.replace(" ", "_")  # replace spaces with _
)

# 3. Trim whitespace in all text cells
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 4. Drop empty rows and duplicate rows
df = df.dropna(how="all")
df = df.drop_duplicates()

# 5. Save cleaned CSV
df.to_csv("NP_Adresses_CLEANED.csv", index=False)

print("Cleaned file saved as 'NP_Adresses_CLEANED.csv'")