import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from time import sleep 
import concurrent.futures
import tkinter as tk
from tkinter import filedialog
import sys
import os

# --- Configuration ---
BASE_URL = "https://sametham.kite.kerala.gov.in/" 
MAX_WORKERS = 8 

# Define the final list of all identifying columns 
NEW_ID_COLUMNS = [
    'School Code', 
    'School Name', 
    'School Type', 
    'School Level', 
    'Sub District', 
    'Panchayat/Municipality/Corporation', 
    'Assembly Constituency', 
    'Revenue District', 
    'Parliament Constituency', 
    'PIN Code'
]

# Define new HSS stream and existing VHSE columns
HSS_COLUMNS = [
    'Class 11 HSS Science', 'Class 11 HSS Commerce', 'Class 11 HSS Humanities', 
    'Class 12 HSS Science', 'Class 12 HSS Commerce', 'Class 12 HSS Humanities'
]
VHSE_COLUMNS = ['Class 11 VHSS', 'Class 12 VHSS']
ALL_HSS_VHSE_COLUMNS = HSS_COLUMNS + VHSE_COLUMNS


# Split ID columns for final ordering
ID_INITIAL = ['School Code', 'School Name', 'School Type', 'School Level']
ID_FINAL = [col for col in NEW_ID_COLUMNS if col not in ID_INITIAL]

# Global constant for expected columns in detailed class tables
EXPECTED_DATA_COLUMNS = 16 

# --- Core Scraping Logic ---

def scrape_standard_data(soup, div_id):
    """
    Scrapes the detailed class-wise strength data (Classes 1-12/XI/XII) 
    from the main strength tables, made robust to capture all rows.
    """
    rows = []
    students_div = soup.find('div', {'id': div_id})
    if students_div:
        strength_table = students_div.find('table', {'class': 'table table-striped'})
        if strength_table and strength_table.find('tbody'):
            # Find ALL rows in the tbody for robustness
            all_rows = strength_table.find('tbody').find_all('tr')
            
            for row in all_rows: 
                cols = [td.text.strip() for td in row.find_all('td')]
                
                # Check 1: for the expected number of columns (16)
                # Check 2: Ensure the first column is not the 'Total' label
                if len(cols) == EXPECTED_DATA_COLUMNS and cols[0].strip().lower() != 'total': 
                    # Standardize class names 'XI'/'XII' to '11'/'12' for pivoting
                    if cols[0] == 'XI': cols[0] = '11'
                    if cols[0] == 'XII': cols[0] = '12'
                    rows.append(cols) 
    return rows


def scrape_hss_streams(soup):
    """
    Scrapes HSS student strength data, split by Science, Commerce, and Humanities 
    for Plus One (Class 11) and Plus Two (Class 12).
    """
    hss_streams = {
        'Class 11 HSS Science': '0', 'Class 11 HSS Commerce': '0', 'Class 11 HSS Humanities': '0',
        'Class 12 HSS Science': '0', 'Class 12 HSS Commerce': '0', 'Class 12 HSS Humanities': '0'
    }
    
    students_hss_div = soup.find('div', {'id': 'students-hss'})
    if not students_hss_div:
        return hss_streams

    target_table = students_hss_div.find('table', {'class': 'table table-striped'})
        
    if target_table and target_table.find('tbody'):
        # Find ALL rows in tbody for robustness
        data_rows = target_table.find('tbody').find_all('tr')
        
        for row in data_rows:
            cols = [td.text.strip() for td in row.find_all('td')]
            # Expected structure: 0: Std, 1: Science, 2: Commerce, 3: Humanities, 4: ALL
            if len(cols) < 5: continue 
            
            standard_label = cols[0].strip()
            
            if standard_label == '+ 1': # Plus One (Class 11)
                hss_streams['Class 11 HSS Science'] = re.sub(r'\D', '', cols[1]) or '0'
                hss_streams['Class 11 HSS Commerce'] = re.sub(r'\D', '', cols[2]) or '0'
                hss_streams['Class 11 HSS Humanities'] = re.sub(r'\D', '', cols[3]) or '0'
            elif standard_label == '+ 2': # Plus Two (Class 12)
                hss_streams['Class 12 HSS Science'] = re.sub(r'\D', '', cols[1]) or '0'
                hss_streams['Class 12 HSS Commerce'] = re.sub(r'\D', '', cols[2]) or '0'
                hss_streams['Class 12 HSS Humanities'] = re.sub(r'\D', '', cols[3]) or '0'
            
    return hss_streams


def fetch_and_process_school(school_code):
    """
    Fetches, processes, and returns the school data DataFrame.
    """
    url = f"{BASE_URL}{school_code}"
    
    # Define fields to scrape
    fields_to_extract = [
        ('School Name', 'School Name', 'Name Not Found'),
        ('School Type', 'School Type', 'N/A'),
        ('School Level', 'School Level', 'N/A'), 
        ('Sub District', 'Sub District', 'N/A'), 
        ('Panchayat/ Municipality/ Corporation', 'Panchayat/Municipality/Corporation', 'N/A'), 
        ('Assembly Constituency', 'Assembly Constituency', 'N/A'),
        ('Revenue District', 'Revenue District', 'N/A'),
        ('Parliament Constituency', 'Parliament Constituency', 'N/A'), 
        ('PIN Code', 'PIN Code', 'N/A'),
    ]
    
    # Standard Strength Data Columns
    STRENGTH_COLUMNS = [
        'Class', 'Malayalam_Boys', 'Malayalam_Girls', 'Malayalam_Total', 
        'English_Boys', 'English_Girls', 'English_Total',
        'Tamil_Boys', 'Tamil_Girls', 'Tamil_Total',
        'Kannada_Boys', 'Kannada_Girls', 'Kannada_Total',
        'ALL_Boys', 'ALL_Girls', 'ALL_Total'
    ]
    
    DF_COLUMNS = NEW_ID_COLUMNS + STRENGTH_COLUMNS + ['Error']
    
    additional_info = {'School Code': school_code}
    data = [] # List to hold all class data rows (long format)
    
    # Initialize HSS stream and VHSE data containers
    hss_stream_data = {}
    vhse_data = {'Class 11 VHSS': '0', 'Class 12 VHSS': '0'} 
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, 'html.parser')
        basic_info_table = soup.find('table', {'id': 'basic'})

        # Helper to find and extract data based on label
        def extract_value(table, label):
            if label == 'Panchayat/ Municipality/ Corporation':
                def is_local_body_label(tag):
                    if tag.name == 'td':
                        text_content = re.sub(r'\s+', ' ', tag.text).strip()
                        return 'Panchayat/ Municipality/ Corporation' in text_content
                    return False
                row = table.find(is_local_body_label)
            else:
                row = table.find('td', string=label)
            
            if row:
                value_element = row.find_next_sibling('td').find_next_sibling('td')
                for br in value_element.find_all('br'):
                    br.replace_with(' ')
                raw_value = value_element.text
                return re.sub(r'\s+', ' ', raw_value).strip()
            return None
        
        # --- SCRAPE ALL ID FIELDS ---
        if basic_info_table:
            for label, var_name, default in fields_to_extract:
                value = extract_value(basic_info_table, label)
                if var_name == 'School Level' and value:
                    additional_info[var_name] = "'" + value 
                else:
                    additional_info[var_name] = value if value else default
                if var_name == 'School Name' and not additional_info[var_name]:
                    additional_info[var_name] = 'Name Not Found'
        else:
            for label, var_name, default in fields_to_extract:
                additional_info[var_name] = default
            additional_info['School Name'] = 'Name Not Found'

        # ID_VALUES list 
        ID_VALUES = [additional_info.get(col, 'N/A') for col in NEW_ID_COLUMNS]

        # --- SCRAPE LP/UP/HS/HSS CLASS-WISE DATA (Classes 1-12 in long format) ---
        
        data.extend(scrape_standard_data(soup, 'students-lpuphs'))
        data.extend(scrape_standard_data(soup, 'students-hshse')) 


        # --- SCRAPE HSS STREAM DATA ---
        hss_stream_data = scrape_hss_streams(soup)


        # --- SCRAPE VHSE DATA (Unchanged) ---
        
        # 1. Check for the NEW, simplified VHSS structure (id="students-vhss") 
        students_vhss_div = soup.find('div', {'id': 'students-vhss'})
        if students_vhss_div:
            vhss_table = students_vhss_div.find('table', {'class': 'table table-striped'})
            if vhss_table and vhss_table.find('tbody'):
                vhss_rows = vhss_table.find('tbody').find_all('tr')
                # The data is expected in the SECOND row (index 1) of the tbody
                if len(vhss_rows) > 1:
                    cols = [td.text.strip() for td in vhss_rows[1].find_all('td')]
                    if len(cols) >= 2: 
                        vhse_data['Class 11 VHSS'] = re.sub(r'\D', '', cols[0]) or '0'
                        vhse_data['Class 12 VHSS'] = re.sub(r'\D', '', cols[1]) or '0'
                        
        # 2. Check for the OLD, nested VHSE structure (id="students-vhse")
        elif vhse_data['Class 11 VHSS'] == '0' and vhse_data['Class 12 VHSS'] == '0':
            students_vhse_div = soup.find('div', {'id': 'students-vhse'})
            if students_vhse_div:
                total_div = students_vhse_div.find('div', {'id': 'total'})
                vhse_table = None
                if total_div:
                    vhse_table = total_div.find('table', {'class': 'table table-striped'})
                
                if vhse_table and vhse_table.find('tbody'):
                    vhse_rows = vhse_table.find('tbody').find_all('tr') 
                    for vhse_row in vhse_rows:
                        cols = [td.text.strip() for td in vhse_row.find_all('td')]
                        if len(cols) >= 4:
                            class_label = cols[0].strip()
                            raw_strength = cols[-1].strip() 
                            total_strength = re.sub(r'\D', '', raw_strength) or '0'
                            
                            if class_label == 'I year':
                                vhse_data['Class 11 VHSS'] = total_strength
                            elif class_label == 'II year':
                                vhse_data['Class 12 VHSS'] = total_strength
        
        # --- CREATE DATAFRAME ---
        
        # Use set comprehension to remove duplicate rows (e.g., if a class appears in both sections)
        unique_data_rows = {row[0]: row for row in data}.values()
        
        # Combine HSS stream and VHSE data into a single dictionary
        hss_vhse_combined_data = {**hss_stream_data, **vhse_data}
        
        if unique_data_rows:
            # Case 1: Data was successfully scraped (Classes 1-12)
            combined_data = [ID_VALUES + row + [pd.NA] for row in unique_data_rows]
            df_long = pd.DataFrame(combined_data, columns=DF_COLUMNS)
            
            # Add HSS Stream and VHSE columns (fixed value across all rows for this school)
            for col in ALL_HSS_VHSE_COLUMNS:
                df_long[col] = hss_vhse_combined_data.get(col, '0')
            
            return df_long
        else:
            # Case 2: No class-wise strength data found (minimal output)
            error_row_data = {col: ID_VALUES[i] for i, col in enumerate(NEW_ID_COLUMNS)}
            error_row_data['Error'] = 'Warning: No strength tables found or no data rows extracted.'
            
            for col in ALL_HSS_VHSE_COLUMNS:
                 error_row_data[col] = hss_vhse_combined_data.get(col, '0')
            
            final_fail_cols = NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS + ['Error']
            
            return pd.DataFrame([{col: error_row_data.get(col) for col in final_fail_cols}])
        
    except requests.exceptions.RequestException:
        raise
    except Exception as e:
        # Catch any remaining internal error and return a minimal DataFrame
        error_row = {col: ID_VALUES[i] for i, col in enumerate(NEW_ID_COLUMNS)}
        error_row['Error'] = f'Scraping Logic Failed: {type(e).__name__}'
        
        final_fail_cols = NEW_ID_COLUMNS + ['Error']
        
        return pd.DataFrame([{col: error_row.get(col, pd.NA) for col in final_fail_cols}])


# --- Retry Wrapper Function (Unchanged) ---

def scrape_with_retry(school_code, max_retries=3, initial_delay=3):
    """Retries scraping on transient network errors (timeouts/connection issues/5XX/404)."""
    
    retry_count = 0
    while retry_count < max_retries:
        if retry_count > 0:
            delay = initial_delay * (2 ** (retry_count - 1)) 
            print(f"🔄 Retrying {school_code} (Attempt {retry_count + 1}/{max_retries}). Waiting {delay}s...", flush=True)
            sleep(delay)
            
        try:
            result_df = fetch_and_process_school(school_code)
            print(f"✅ Successfully scraped data for school code: {school_code}", flush=True)
            return result_df
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code in (404, 429) or status_code >= 500:
                print(f"⚠️ Error fetching data for {school_code}: Status {status_code}. Retrying...", flush=True)
                retry_count += 1
                continue
            else:
                print(f"🚫 Error fetching data for {school_code}: Client Error {status_code}. Skipping.", flush=True)
                return pd.DataFrame({'School Code': [school_code], 'School Name': ['N/A'], 'Error': [f'Permanent Error: {status_code}']})

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"⚠️ Error fetching data for {school_code}: Connection/Timeout Error. Retrying...", flush=True)
            retry_count += 1
            continue

        except Exception as e:
            print(f"🚫 Error fetching data for {school_code}: Unrecoverable Error: {type(e).__name__}: {e}. Logging and skipping scrape for this URL.", flush=True)
            return pd.DataFrame({'School Code': [school_code], 'School Name': ['N/A'], 'Error': [f'Scraping Logic Failed: {type(e).__name__}']})

    print(f"❌ Max retries reached for {school_code}. Skipping.", flush=True)
    return pd.DataFrame({'School Code': [school_code], 'School Name': ['N/A'], 'Error': [f'Failed after {max_retries} retries']})


# --- B. Execution and Saving (Main Logic - Finalization) ---

# 1. LIST OF CODES - Read from Excel File
tvm_school_codes = []
if __name__ == "__main__":
    
    # --- File Selection GUI ---
    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    print("📂 Please select the Excel file containing School Codes...", flush=True)
    file_path = filedialog.askopenfilename(
        title="Select School Code Excel File",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )
    
    if not file_path:
        print("❌ No file selected. Exiting.")
        sys.exit()
    
    print(f"✅ File selected: {file_path}", flush=True)

    try:
        print("Reading school codes from Excel...", flush=True)
        df = pd.read_excel(file_path) 
        
        # --- Smart Column Detection ---
        possible_cols = ['schoolcode', 'school code', 'code', 'school_code', 's_code']
        found_col = None
        
        # 1. Check strict lowercase match
        for col in df.columns:
            if str(col).lower().strip() in possible_cols:
                found_col = col
                break
        
        # 2. If found, use it. If not, ask user.
        if found_col:
            print(f"✅ Found school code column: '{found_col}'", flush=True)
            tvm_school_codes = df[found_col].astype(str).tolist()
        else:
            print("⚠️ Could not automatically find 'schoolcode' column.")
            print(f"Available columns: {list(df.columns)}")
            found_col = input("👉 Please type the exact column name for School Codes: ").strip()
            tvm_school_codes = df[found_col].astype(str).tolist()

        print(f"Total codes extracted: {len(tvm_school_codes)} schools.", flush=True)
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}.", flush=True)
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading Excel: {e}", flush=True)

    # 2. Loop and Collect Data (Concurrent with Retry)
    all_schools_data = []
    if tvm_school_codes:
        print(f"Starting concurrent scraping for {len(tvm_school_codes)} schools with {MAX_WORKERS} threads...", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_code = {executor.submit(scrape_with_retry, code): code for code in tvm_school_codes}
            
            for future in concurrent.futures.as_completed(future_to_code):
                try:
                    school_df = future.result()
                    all_schools_data.append(school_df)
                except Exception as exc:
                    print(f"❌ Unexpected Executor Error: {exc}", flush=True)

    # 3. Combine and Save (Wide Format)
    if all_schools_data:
        final_df = pd.concat(all_schools_data, ignore_index=True)
        
        # 1. Separate successful data (has class data) and error data (minimal info)
        if 'Class' in final_df.columns:
            df_successful = final_df[final_df['Class'].notna()].drop_duplicates(subset=NEW_ID_COLUMNS + ['Class']).copy()
        else:
            # Pandas KeyError Fix - Define df_successful with a 'School Code' column
            df_successful = pd.DataFrame(columns=['School Code'])
            
        df_errors = final_df[~final_df['School Code'].isin(df_successful['School Code'])].copy().drop_duplicates(subset=['School Code'])
        
        # --- Process Successful Data (Pivot) ---
        if not df_successful.empty:
            df_totals = df_successful[NEW_ID_COLUMNS + ['Class', 'ALL_Total'] + ALL_HSS_VHSE_COLUMNS].copy()
            
            try:
                # Convert class column to numeric to handle classes 1-12
                df_totals['Class'] = pd.to_numeric(df_totals['Class'], errors='coerce').astype('Int64')
                df_totals.dropna(subset=['Class'], inplace=True)
            except Exception:
                pass 

            # Pivot the long data to wide format (Class X Total)
            df_wide = df_totals.pivot_table(
                index=NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS,
                columns='Class',
                values='ALL_Total',
                aggfunc='first'
            ).reset_index()

            new_cols = {c: f'Class {c} Total' for c in df_wide.columns if isinstance(c, int)}
            df_wide.rename(columns=new_cols, inplace=True)
        else:
            df_wide = pd.DataFrame(columns=NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS)

        # --- Combine Pivoted Data with Error/Warning Schools ---
        
        error_cols_to_keep = list(set(NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS + ['Error']) & set(df_errors.columns))
        df_errors_info = df_errors[error_cols_to_keep].copy().drop_duplicates(subset=['School Code'])

        final_output_df = df_wide.merge(
            df_errors_info, 
            on='School Code', 
            how='outer',
            suffixes=('_data', '_error')
        )
        
        # Consolidate all ID/HSS/VHSE/Error fields
        for col in NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS:
            col_data = col + '_data'
            col_error = col + '_error'
            
            if col_data in final_output_df.columns and col_error in final_output_df.columns:
                final_output_df[col] = final_output_df[col_data].fillna(final_output_df[col_error])
            elif col_error in final_output_df.columns:
                 final_output_df[col] = final_output_df[col_error]
        
        if 'Error_error' in final_output_df.columns:
            final_output_df['Error'] = final_output_df['Error_error'].fillna(final_output_df.get('Error_data', pd.NA))
        else:
            final_output_df['Error'] = final_output_df.get('Error_data', pd.NA)

        cols_to_drop = [col for col in final_output_df.columns if col.endswith(('_data', '_error'))]
        final_output_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # 4. Final Cleanup and Saving
        
        class_cols = [col for col in final_output_df.columns if col.startswith('Class') and col.endswith('Total')]
        # Sort class columns numerically (Class 1, Class 2, ..., Class 12)
        class_cols.sort(key=lambda x: int(re.search(r'Class (\d+) Total', x).group(1)))
        
        # Define ALL columns that must be numerical integers
        NUMERIC_COLUMNS = class_cols + ALL_HSS_VHSE_COLUMNS
        
        # 1. Fill missing values with 0
        final_output_df[NUMERIC_COLUMNS] = final_output_df[NUMERIC_COLUMNS].fillna(0)
        
        # 2. Explicitly convert the columns to integer type
        for col in NUMERIC_COLUMNS:
            final_output_df[col] = pd.to_numeric(final_output_df[col], errors='coerce').astype(int) 
        
        # Order the final columns
        all_cols = ID_INITIAL + ['Sub District'] + [c for c in ID_FINAL if c != 'Sub District'] + class_cols + HSS_COLUMNS + VHSE_COLUMNS + ['Error']
        
        final_output_df = final_output_df.reindex(columns=all_cols)
        
        # --- SAVE DIALOG (Custom Directory Selection) ---
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        default_output_name = f'{input_filename}_Scraped_Output.xlsx'
        
        print("💾 Please choose where to save the file...", flush=True)
        save_path = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            initialfile=default_output_name
        )
        
        if save_path:
            OUTPUT_FILENAME = save_path
        else:
            print("⚠️ Save cancelled. Saving to default location to prevent data loss...", flush=True)
            OUTPUT_FILENAME = default_output_name
        
        final_output_df.to_excel(OUTPUT_FILENAME, index=False, engine='openpyxl')
        
        # Print Summary
        num_successful = len(df_successful['School Code'].unique()) if not df_successful.empty else 0
        num_total = len(final_output_df)
        
        print("\n" + "="*50, flush=True)
        print("✅ Scraping Complete!", flush=True)
        print(f"File saved to: **{OUTPUT_FILENAME}**", flush=True)
        print("Student strength data is now saved as numerical integers for compatibility with Excel formulas. 🎉", flush=True)
        print(f"Summary: {num_successful} schools provided full class data. {num_total} schools recorded in XLSX.", flush=True)
        print("="*50, flush=True)
    else:
        print("❌ No data was successfully scraped.", flush=True)