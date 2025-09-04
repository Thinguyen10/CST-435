import pandas as pd
import re

def convert_arff_to_xlsx(arff_file, xlsx_file):
    """
    Simple function to convert ARFF file to XLSX
    """
    # Read the ARFF file
    with open(arff_file, 'r') as f:
        content = f.read()
    
    # Find where data starts
    data_start = content.lower().find('@data')
    if data_start == -1:
        raise ValueError("No @data section found")
    
    # Get data section
    data_section = content[data_start + 5:].strip()
    
    # Get attribute names
    attributes = []
    for line in content[:data_start].split('\n'):
        if line.lower().startswith('@attribute'):
            # Extract attribute name (first word after @attribute)
            name = line.split()[1].strip('\'"')
            attributes.append(name)
    
    # Parse data rows
    rows = []
    for line in data_section.split('\n'):
        line = line.strip()
        if line and not line.startswith('%'):  # Skip empty lines and comments
            # Split by comma and clean up
            row = [val.strip('\'" ') for val in line.split(',')]
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=attributes)
    
    # Convert to Excel
    df.to_excel(xlsx_file, index=False)
    print(f"Converted {arff_file} to {xlsx_file}")

# Usage
arff_file = r'/Users/thinguyen/Library/CloudStorage/OneDrive-GrandCanyonUniversity/RECENT CLASSES/CST-435 Deep Learning/Projects/Perceptron/Rice_Cammeo_Osmancik.arff'  # Change this to your ARFF file name
xlsx_file = r'/Users/thinguyen/Library/CloudStorage/OneDrive-GrandCanyonUniversity/RECENT CLASSES/CST-435 Deep Learning/Projects/Perceptron/Rice_data.xlsx'    # Change this to desired output name

convert_arff_to_xlsx(arff_file, xlsx_file)