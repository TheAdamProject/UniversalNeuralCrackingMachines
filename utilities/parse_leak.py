import sys
import re

"""
Given an input file that stores credentials in the format 'email:password', this script parses the data using the format supported by the functions in 'input_pipeline.py' and then writes it to disk.
"""

EMAIL_RE = "([A-Za-z0-9\.\-_]+)@([A-Za-z0-9\.\-_]+)\.([A-Za-z0-9\.\-_]+)"
MATCH_ENTRY = re.compile(f'{EMAIL_RE}:(.+)')

def parse_and_write_leak(inpath, outpath, encoding='ascii'):
    with open(outpath, 'w', encoding=encoding) as outf:
        with open(inpath, 'r', encoding=encoding, errors='ignore') as inf:
            for l in inf:
                l = l.strip()
                b = MATCH_ENTRY.match(l)
                if b:
                    username, d0, d1, password = b.groups()
                    username = username.lower()
                    d0 = d0.lower()
                    d1 = d1.lower()
                    print(f'{username}\t{d0}\t{d1}\t{password}', file=outf)
                else:
                    print(f"[WARNING]: line: '{l}' has not been parsed correctly!.")
                    
                    
if __name__ == "__main__":
    try:
        inpath = sys.argv[1]
        outpath = sys.argv[2]
        encoding = sys.argv[3]
    except:
        print("[USAGE] path_to_file_to_convert path_to_output_file file_encoding(e.g., ascii)")
        sys.exit(1)
        
    parse_and_write_leak(inpath, outpath, encoding=encoding)
    print(f"Exported in {outpath}.")
       
        
    