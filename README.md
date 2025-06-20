# GHG-Emissions-Prediction
A machine learning project to predict greenhouse gas emissions using supply chain emission factors dataset. Built for AICTE internship.

# US Supply Chain Emission Analysis

This project processes and analyzes greenhouse gas emission factors for U.S. industries and commodities from 2010 to 2016. It uses a Random Forest Regressor to model emission trends and extract insights, enabling data-driven policy and economic decisions.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Bonus Tip](#bonus-tip)

---

## Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/supply-chain-emissions.git
cd supply-chain-emissions


2. Install required Python packages:

bash
pip install -r requirements.txt


3. Make sure the Excel file SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx is placed in the root directory.

---

## Usage

Run the Python script:

bash
python emission_analysis.py


This will:

- Load the emission data for each year from the Excel file.
- Merge commodity and industry data.
- Clean and prepare the dataset.
- Print data samples and statistics for verification.

> Note: The script currently ends after loading and printing initial data. Further steps like model training, prediction, and visualization would be added in future phases.

---

## Features

- ✅ Reads multi-sheet Excel files for multiple years.
- ✅ Combines commodity and industry data.
- ✅ Cleans and prepares large datasets.
- ✅ Provides initial structure for emission modeling using RandomForestRegressor.

---

## Tech Stack

- Python
- Pandas – data manipulation
- NumPy – numerical operations
- Matplotlib & Seaborn – visualization
- scikit-learn – machine learning modeling
- joblib – model saving/loading

---

## Screenshots

To be added later – model performance graphs, emissions over time visualizations, etc.

---

## Contributing

Feel free to fork the repository, make changes, and submit pull requests. For significant feature requests or bug fixes, please open an issue first.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Author

Laxmi Prasanna  
[LinkedIn](https://www.linkedin.com/in/sri-laxmi-prasanna-polu-322376324?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
polusrilaxmiprasanna2005@gmail.com

---

## Bonus Tip

Create a requirements.txt file with the following contents:


pandas
numpy
seaborn
matplotlib
scikit-learn
openpyxl
joblib
