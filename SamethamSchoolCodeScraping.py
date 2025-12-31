import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from time import sleep 
import concurrent.futures
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, simpledialog
import threading
import queue
import os
import sys

# --- Configuration ---
BASE_URL = "https://sametham.kite.kerala.gov.in/" 
MAX_WORKERS = 8 

# Identifiers
NEW_ID_COLUMNS = [
    'School Code', 'School Name', 'School Type', 'School Level', 'Sub District', 
    'Panchayat/Municipality/Corporation', 'Assembly Constituency', 
    'Revenue District', 'Parliament Constituency', 'PIN Code'
]

# HSS & VHSE Columns
HSS_COLUMNS = [
    'Class 11 HSS Science', 'Class 11 HSS Commerce', 'Class 11 HSS Humanities', 
    'Class 12 HSS Science', 'Class 12 HSS Commerce', 'Class 12 HSS Humanities'
]
VHSE_COLUMNS = ['Class 11 VHSS', 'Class 12 VHSS']
ALL_HSS_VHSE_COLUMNS = HSS_COLUMNS + VHSE_COLUMNS

ID_INITIAL = ['School Code', 'School Name', 'School Type', 'School Level']
ID_FINAL = [col for col in NEW_ID_COLUMNS if col not in ID_INITIAL]
EXPECTED_DATA_COLUMNS = 16 

# --- Core Scraping Logic (Stateless) ---

def scrape_standard_data(soup, div_id):
    rows = []
    students_div = soup.find('div', {'id': div_id})
    if students_div:
        strength_table = students_div.find('table', {'class': 'table table-striped'})
        if strength_table and strength_table.find('tbody'):
            all_rows = strength_table.find('tbody').find_all('tr')
            for row in all_rows: 
                cols = [td.text.strip() for td in row.find_all('td')]
                if len(cols) == EXPECTED_DATA_COLUMNS and cols[0].strip().lower() != 'total': 
                    if cols[0] == 'XI': cols[0] = '11'
                    if cols[0] == 'XII': cols[0] = '12'
                    rows.append(cols) 
    return rows

def scrape_hss_streams(soup):
    hss_streams = {k: '0' for k in HSS_COLUMNS}
    students_hss_div = soup.find('div', {'id': 'students-hss'})
    if not students_hss_div: return hss_streams

    target_table = students_hss_div.find('table', {'class': 'table table-striped'})
    if target_table and target_table.find('tbody'):
        data_rows = target_table.find('tbody').find_all('tr')
        for row in data_rows:
            cols = [td.text.strip() for td in row.find_all('td')]
            if len(cols) < 5: continue 
            standard_label = cols[0].strip()
            if standard_label == '+ 1':
                hss_streams['Class 11 HSS Science'] = re.sub(r'\D', '', cols[1]) or '0'
                hss_streams['Class 11 HSS Commerce'] = re.sub(r'\D', '', cols[2]) or '0'
                hss_streams['Class 11 HSS Humanities'] = re.sub(r'\D', '', cols[3]) or '0'
            elif standard_label == '+ 2':
                hss_streams['Class 12 HSS Science'] = re.sub(r'\D', '', cols[1]) or '0'
                hss_streams['Class 12 HSS Commerce'] = re.sub(r'\D', '', cols[2]) or '0'
                hss_streams['Class 12 HSS Humanities'] = re.sub(r'\D', '', cols[3]) or '0'
    return hss_streams

def fetch_and_process_school(school_code):
    url = f"{BASE_URL}{school_code}"
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
    STRENGTH_COLUMNS = [
        'Class', 'Malayalam_Boys', 'Malayalam_Girls', 'Malayalam_Total', 
        'English_Boys', 'English_Girls', 'English_Total',
        'Tamil_Boys', 'Tamil_Girls', 'Tamil_Total',
        'Kannada_Boys', 'Kannada_Girls', 'Kannada_Total',
        'ALL_Boys', 'ALL_Girls', 'ALL_Total'
    ]
    DF_COLUMNS = NEW_ID_COLUMNS + STRENGTH_COLUMNS + ['Error']
    additional_info = {'School Code': school_code}
    data = []
    hss_stream_data = {}
    vhse_data = {'Class 11 VHSS': '0', 'Class 12 VHSS': '0'} 
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        basic_info_table = soup.find('table', {'id': 'basic'})

        def extract_value(table, label):
            if label == 'Panchayat/ Municipality/ Corporation':
                def is_local_body_label(tag):
                    if tag.name == 'td':
                        return 'Panchayat/ Municipality/ Corporation' in re.sub(r'\s+', ' ', tag.text).strip()
                    return False
                row = table.find(is_local_body_label)
            else:
                row = table.find('td', string=label)
            if row:
                value_element = row.find_next_sibling('td').find_next_sibling('td')
                for br in value_element.find_all('br'): br.replace_with(' ')
                return re.sub(r'\s+', ' ', value_element.text).strip()
            return None
        
        if basic_info_table:
            for label, var_name, default in fields_to_extract:
                value = extract_value(basic_info_table, label)
                additional_info[var_name] = ("'" + value) if var_name == 'School Level' and value else (value if value else default)
                if var_name == 'School Name' and not additional_info[var_name]: additional_info[var_name] = 'Name Not Found'
        else:
            for label, var_name, default in fields_to_extract: additional_info[var_name] = default
            additional_info['School Name'] = 'Name Not Found'

        ID_VALUES = [additional_info.get(col, 'N/A') for col in NEW_ID_COLUMNS]
        data.extend(scrape_standard_data(soup, 'students-lpuphs'))
        data.extend(scrape_standard_data(soup, 'students-hshse')) 
        hss_stream_data = scrape_hss_streams(soup)

        # VHSE Logic
        students_vhss_div = soup.find('div', {'id': 'students-vhss'})
        if students_vhss_div:
            vhss_table = students_vhss_div.find('table', {'class': 'table table-striped'})
            if vhss_table and vhss_table.find('tbody'):
                vhss_rows = vhss_table.find('tbody').find_all('tr')
                if len(vhss_rows) > 1:
                    cols = [td.text.strip() for td in vhss_rows[1].find_all('td')]
                    if len(cols) >= 2: 
                        vhse_data['Class 11 VHSS'] = re.sub(r'\D', '', cols[0]) or '0'
                        vhse_data['Class 12 VHSS'] = re.sub(r'\D', '', cols[1]) or '0'
        elif vhse_data['Class 11 VHSS'] == '0' and vhse_data['Class 12 VHSS'] == '0':
            students_vhse_div = soup.find('div', {'id': 'students-vhse'})
            if students_vhse_div:
                total_div = students_vhse_div.find('div', {'id': 'total'})
                vhse_table = total_div.find('table', {'class': 'table table-striped'}) if total_div else None
                if vhse_table and vhse_table.find('tbody'):
                    for vhse_row in vhse_table.find('tbody').find_all('tr'):
                        cols = [td.text.strip() for td in vhse_row.find_all('td')]
                        if len(cols) >= 4:
                            if cols[0].strip() == 'I year': vhse_data['Class 11 VHSS'] = re.sub(r'\D', '', cols[-1].strip()) or '0'
                            elif cols[0].strip() == 'II year': vhse_data['Class 12 VHSS'] = re.sub(r'\D', '', cols[-1].strip()) or '0'
        
        unique_data_rows = {row[0]: row for row in data}.values()
        hss_vhse_combined_data = {**hss_stream_data, **vhse_data}
        
        if unique_data_rows:
            combined_data = [ID_VALUES + row + [pd.NA] for row in unique_data_rows]
            df_long = pd.DataFrame(combined_data, columns=DF_COLUMNS)
            for col in ALL_HSS_VHSE_COLUMNS: df_long[col] = hss_vhse_combined_data.get(col, '0')
            return df_long
        else:
            error_row = {col: ID_VALUES[i] for i, col in enumerate(NEW_ID_COLUMNS)}
            error_row['Error'] = 'Warning: No strength tables found.'
            for col in ALL_HSS_VHSE_COLUMNS: error_row[col] = hss_vhse_combined_data.get(col, '0')
            return pd.DataFrame([{col: error_row.get(col) for col in NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS + ['Error']}])
            
    except Exception as e:
        error_row = {col: ID_VALUES[i] for i, col in enumerate(NEW_ID_COLUMNS)}
        error_row['Error'] = f'Scraping Logic Failed: {type(e).__name__}'
        return pd.DataFrame([{col: error_row.get(col, pd.NA) for col in NEW_ID_COLUMNS + ['Error']}])

def scrape_with_retry(school_code, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        if retry_count > 0: sleep(3 * (2 ** (retry_count - 1)))
        try:
            return fetch_and_process_school(school_code)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (404, 429) or e.response.status_code >= 500:
                retry_count += 1
                continue
            return pd.DataFrame({'School Code': [school_code], 'School Name': ['N/A'], 'Error': [f'Permanent Error: {e.response.status_code}']})
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            retry_count += 1
            continue
        except Exception as e:
            return pd.DataFrame({'School Code': [school_code], 'School Name': ['N/A'], 'Error': [f'Scraping Logic Failed: {type(e).__name__}']})
    return pd.DataFrame({'School Code': [school_code], 'School Name': ['N/A'], 'Error': [f'Failed after {max_retries} retries']})


# --- GUI APPLICATION CLASS ---

class ScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sametham School Scraper")
        self.root.geometry("700x550")
        
        self.file_path = None
        self.log_queue = queue.Queue()
        self.is_scraping = False

        # Styles
        style = ttk.Style()
        style.configure("TButton", padding=6, font=('Helvetica', 10))
        style.configure("TLabel", font=('Helvetica', 10))
        
        # --- UI Components ---
        
        # Frame 1: File Selection
        frame_top = ttk.LabelFrame(root, text="Step 1: Input", padding=10)
        frame_top.pack(fill="x", padx=10, pady=5)
        
        self.lbl_file = ttk.Label(frame_top, text="No file selected")
        self.lbl_file.pack(side="left", fill="x", expand=True)
        
        btn_browse = ttk.Button(frame_top, text="Browse Excel", command=self.load_file)
        btn_browse.pack(side="right")

        # Frame 2: Progress
        frame_progress = ttk.Frame(root, padding=10)
        frame_progress.pack(fill="x", padx=10)
        
        self.lbl_status = ttk.Label(frame_progress, text="Status: Idle")
        self.lbl_status.pack(anchor="w")
        
        self.progress = ttk.Progressbar(frame_progress, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=5)

        # Frame 3: Logs
        frame_log = ttk.LabelFrame(root, text="Activity Log", padding=10)
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.txt_log = scrolledtext.ScrolledText(frame_log, height=15, state='disabled', font=('Consolas', 9))
        self.txt_log.pack(fill="both", expand=True)

        # Frame 4: Actions
        frame_actions = ttk.Frame(root, padding=10)
        frame_actions.pack(fill="x", padx=10, pady=5)
        
        self.btn_start = ttk.Button(frame_actions, text="Start Scraping", command=self.start_scraping_thread, state='disabled')
        self.btn_start.pack(side="right")

        # Start Log Polling
        self.root.after(100, self.process_log_queue)

    def log(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.txt_log.config(state='normal')
            self.txt_log.insert(tk.END, msg + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state='disabled')
        self.root.after(100, self.process_log_queue)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls")])
        if path:
            self.file_path = path
            self.lbl_file.config(text=os.path.basename(path))
            self.btn_start.config(state='normal')
            self.log(f"📂 Loaded: {path}")

    def start_scraping_thread(self):
        if self.is_scraping: return
        self.is_scraping = True
        self.btn_start.config(state='disabled')
        thread = threading.Thread(target=self.run_scraping_logic)
        thread.start()

    def run_scraping_logic(self):
        try:
            self.log("⏳ Reading Excel file...")
            df = pd.read_excel(self.file_path)
            
            # Smart Column Detection
            possible_cols = ['schoolcode', 'school code', 'code', 'school_code', 's_code']
            found_col = None
            for col in df.columns:
                if str(col).lower().strip() in possible_cols:
                    found_col = col
                    break
            
            if not found_col:
                found_col = simpledialog.askstring("Column Required", "Could not auto-detect 'School Code' column.\nPlease type the column name:")
                if not found_col or found_col not in df.columns:
                    self.log("❌ Error: Invalid column name or cancelled.")
                    self.is_scraping = False
                    self.btn_start.config(state='normal')
                    return

            school_codes = df[found_col].astype(str).tolist()
            total_schools = len(school_codes)
            self.log(f"🚀 Found {total_schools} schools. Starting scraper...")
            
            # Setup Progress Bar
            self.progress['maximum'] = total_schools
            self.progress['value'] = 0

            results = []
            completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_code = {executor.submit(scrape_with_retry, code): code for code in school_codes}
                
                for future in concurrent.futures.as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        data = future.result()
                        results.append(data)
                        
                        # Logging Status
                        err = data.iloc[0].get('Error')
                        if pd.isna(err) or not err:
                            self.log(f"✅ Scraped: {code}")
                        else:
                            self.log(f"⚠️ Issue with {code}: {err}")
                            
                    except Exception as e:
                        self.log(f"❌ Critical Fail {code}: {e}")
                    
                    completed_count += 1
                    self.progress['value'] = completed_count
                    self.lbl_status.config(text=f"Status: Processing {completed_count}/{total_schools}")

            # Processing Results
            self.log("📊 Processing data table...")
            final_df = pd.concat(results, ignore_index=True)
            
            # Pivot Logic (Same as before)
            if 'Class' in final_df.columns:
                df_success = final_df[final_df['Class'].notna()].drop_duplicates(subset=NEW_ID_COLUMNS + ['Class']).copy()
            else:
                df_success = pd.DataFrame(columns=['School Code'])
            
            df_errors = final_df[~final_df['School Code'].isin(df_success['School Code'])].copy().drop_duplicates(subset=['School Code'])

            if not df_success.empty:
                df_totals = df_success[NEW_ID_COLUMNS + ['Class', 'ALL_Total'] + ALL_HSS_VHSE_COLUMNS].copy()
                df_totals['Class'] = pd.to_numeric(df_totals['Class'], errors='coerce').astype('Int64')
                df_totals.dropna(subset=['Class'], inplace=True)
                
                df_wide = df_totals.pivot_table(index=NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS, columns='Class', values='ALL_Total', aggfunc='first').reset_index()
                df_wide.rename(columns={c: f'Class {c} Total' for c in df_wide.columns if isinstance(c, int)}, inplace=True)
            else:
                df_wide = pd.DataFrame(columns=NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS)

            # Merging
            error_cols = list(set(NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS + ['Error']) & set(df_errors.columns))
            final_output = df_wide.merge(df_errors[error_cols], on='School Code', how='outer', suffixes=('_data', '_error'))

            # Coalesce
            for col in NEW_ID_COLUMNS + ALL_HSS_VHSE_COLUMNS:
                c_d, c_e = col + '_data', col + '_error'
                if c_d in final_output.columns and c_e in final_output.columns:
                    final_output[col] = final_output[c_d].fillna(final_output[c_e])
                elif c_e in final_output.columns:
                    final_output[col] = final_output[c_e]
            
            final_output['Error'] = final_output.get('Error_error', pd.NA).fillna(final_output.get('Error_data', pd.NA)) if 'Error_error' in final_output else final_output.get('Error_data', pd.NA)
            final_output.drop(columns=[c for c in final_output.columns if c.endswith(('_data', '_error'))], inplace=True, errors='ignore')

            # Sorting & Numeric Conversion
            class_cols = [c for c in final_output.columns if c.startswith('Class') and c.endswith('Total')]
            class_cols.sort(key=lambda x: int(re.search(r'Class (\d+) Total', x).group(1)))
            
            num_cols = class_cols + ALL_HSS_VHSE_COLUMNS
            final_output[num_cols] = final_output[num_cols].fillna(0)
            for col in num_cols: final_output[col] = pd.to_numeric(final_output[col], errors='coerce').astype(int)

            final_cols_ordered = ID_INITIAL + ['Sub District'] + [c for c in ID_FINAL if c != 'Sub District'] + class_cols + HSS_COLUMNS + VHSE_COLUMNS + ['Error']
            final_output = final_output.reindex(columns=final_cols_ordered)

            # Trigger Save on Main Thread
            self.root.after(0, lambda: self.save_file(final_output))

        except Exception as e:
            self.log(f"❌ Fatal Error: {e}")
            self.is_scraping = False
            self.btn_start.config(state='normal')

    def save_file(self, df):
        self.lbl_status.config(text="Status: Finished. Waiting for save...")
        default_name = "Output.xlsx"
        if self.file_path:
            default_name = os.path.splitext(os.path.basename(self.file_path))[0] + "_Scraped.xlsx"

        save_path = filedialog.asksaveasfilename(
            title="Save Output",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            initialfile=default_name,
            confirmoverwrite=True
        )
        
        if save_path:
            try:
                df.to_excel(save_path, index=False, engine='openpyxl')
                self.log(f"💾 Saved to: {save_path}")
                messagebox.showinfo("Success", "Scraping Completed and File Saved!")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file:\n{e}")
        else:
            self.log("⚠️ Save Cancelled.")
        
        self.lbl_status.config(text="Status: Done")
        self.is_scraping = False
        self.btn_start.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = ScraperApp(root)
    root.mainloop()