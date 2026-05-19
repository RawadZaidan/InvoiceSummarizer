import time
import json
import os
import sys

import invoice_extractor
import invoice_scanner

class DummyStream:
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass
    def isatty(self): return False

def run_extractor(file_path):
    start_time = time.time()
    result = None
    error = None
    
    old_stdout = sys.stdout
    sys.stdout = DummyStream()
    try:
        res_str = invoice_extractor.extract_invoice_data(file_path)
        if res_str:
            result = json.loads(res_str)
    except Exception as e:
        error = str(e)
    finally:
        sys.stdout = old_stdout
        
    end_time = time.time()
    return {"time": end_time - start_time, "result": result, "error": error}

def run_scanner(file_path):
    start_time = time.time()
    result = None
    error = None
    
    old_stdout = sys.stdout
    sys.stdout = DummyStream()
    try:
        result = invoice_scanner.process_file(file_path)
    except Exception as e:
        error = str(e)
    finally:
        sys.stdout = old_stdout
        
    end_time = time.time()
    return {"time": end_time - start_time, "result": result, "error": error}

def format_ext(res):
    if not res: return "N/A"
    return f"Type:{res.get('type','')} | Date:{res.get('date','')} | Tot:{res.get('total_amount','')} | Client:{res.get('client_name','')}"

def format_sca(res):
    if not res: return "N/A"
    tots = ", ".join([f"{x.get('amount')} {x.get('currency')}" for x in res.get("totals", [])])
    return f"Type:{res.get('document_type','')} | Date:{res.get('date','')} | Tot:{tots} | Client:{res.get('client_name','')}"

def main():
    files = ["1.pdf", "2.pdf", "3.pdf", "4.pdf"]
    
    print(f"{'File':<6} | {'Metric':<10} | {'invoice_extractor (GPT-4o)':<45} | {'invoice_scanner (Qwen VL)':<45}")
    print("-" * 115)
    
    for f in files:
        if not os.path.exists(f):
            print(f"{f} not found.")
            continue
            
        res_ext = run_extractor(f)
        res_sca = run_scanner(f)
        
        t_ext = f"{res_ext['time']:.2f}s"
        t_sca = f"{res_sca['time']:.2f}s"
        
        # Extractor cost: 1 high-res image to GPT-4o
        cost_ext = "~$0.007" 
        
        # Scanner cost depends on strategy
        strat = res_sca['result'].get('_strategy', '') if res_sca['result'] else ''
        if strat == 'native_text_llm':
            cost_sca = "~$0.0001 (Text)"
        elif strat == 'scanned_pdf_vision':
            cost_sca = "~$0.002 (Vision)"
        else:
            cost_sca = "Unknown"
            
        out_ext = format_ext(res_ext['result']) if not res_ext['error'] else f"ERROR: {res_ext['error']}"
        out_sca = format_sca(res_sca['result']) if not res_sca['error'] else f"ERROR: {res_sca['error']}"
        
        # Truncate outputs to fit table if needed
        out_ext_disp = out_ext[:43] + ".." if len(out_ext) > 45 else out_ext
        out_sca_disp = out_sca[:43] + ".." if len(out_sca) > 45 else out_sca
        
        print(f"{f:<6} | {'Time':<10} | {t_ext:<45} | {t_sca:<45}")
        print(f"{'':<6} | {'Est. Cost':<10} | {cost_ext:<45} | {cost_sca:<45}")
        print(f"{'':<6} | {'Extraction':<10} | {out_ext_disp:<45} | {out_sca_disp:<45}")
        print("-" * 115)

if __name__ == '__main__':
    main()
