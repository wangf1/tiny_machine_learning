import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Google Spreadsheet URL
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1B30XbPre5XyD9ItRZ6PN98xCeZe4SplpNQhEjq2ASHo/edit?gid=0#gid=0"

# Define the scope and initialize credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    'C:/Users/markw/Documents/Keys/circular-study-427417-m0-5483493804ee.json', scope)

# Authenticate and initialize client
client = gspread.authorize(creds)

spreadsheet = client.open_by_url(spreadsheet_url)

sheet = spreadsheet.sheet1


# Read data from the sheet
data = sheet.get_all_values()

# Create a pandas DataFrame
df = pd.DataFrame(data[3:], columns=data[2])

print(df)

# Months looks like this:
# months = ["July 2024", "August 2024", "September 2024", "October 2024",
#           "November 2024", "December 2024", "January 2025", "February 2025",
#           "March 2025", "April 2025", "May 2025", "June 2025", "July 2025"]

months = df.columns.tolist()[1:]


pyplot_data = []


# Learn each category of data, predicting future data for 3 months for each category
for row in df.itertuples():
    item_name = row[1]
    if not item_name:
        continue
    raw_data_by_month = row[2:]
    # data_by_month looks like this:
    # data_by_month = [80000.00, 83200.00, 86528.00, 89989.12, 93588.68, 97332.23,
    #          101225.52, 105274.54, 109485.52, 113864.94, 118419.54, 123156.32]
    data_by_month = [float(amount.replace('$', '').replace(',', '')) for amount in raw_data_by_month]

    pyplot_data.append((months, data_by_month, item_name, "o", "-"))

    # Convert months to numeric values (e.g., 1 for July 2024, 2 for August 2024, ...)
    X = np.arange(len(months)).reshape(-1, 1)  # Feature matrix
    y = np.array(data_by_month)  # Target variable

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future sales
    future_months = 3  # Number of future months to predict
    future_X = np.arange(len(months), len(months) + future_months).reshape(-1, 1)
    future_data = model.predict(future_X)
    future_months_labels = [f"Month {i + len(months) + 1}" for i in range(future_months)]

    pyplot_data.append((future_months_labels, future_data, item_name+" Prediction", "x", "--"))

# Plot the results
plt.xlabel('Month')
plt.ylabel('Financial Amount ($)')
plt.title('Financial Prediction')

for month, data, label, marker, linestyle in pyplot_data:
    plt.plot(month, data, label=label, marker=marker, linestyle=linestyle)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()