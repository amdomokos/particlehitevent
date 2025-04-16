# data/download.py
import os
import requests
import gzip
import shutil
from tqdm import tqdm
import concurrent.futures

def download_file(url, output_path):
    """Download a single file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def extract_gz(gz_path):
    """Extract a .gz file and remove the compressed version."""
    output_path = gz_path[:-3]  # Remove .gz extension
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)  # Remove the compressed file

def download_and_extract_data(start_num, end_num, base_url, output_dir, max_workers=4):
    """Download and extract multiple data files in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    
    def process_file(num):
        url = f"{base_url}{num}.out.gz"
        gz_path = os.path.join(output_dir, f"pixel_clusters_d{num}.out.gz")
        
        try:
            # Download the file
            download_file(url, gz_path)
            # Extract the file
            extract_gz(gz_path)
            return True
        except Exception as e:
            print(f"Error processing file {num}: {e}")
            return False

    # Create a list of file numbers to process
    file_numbers = range(start_num, end_num + 1)
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_file, file_numbers),
            total=len(file_numbers),
            desc="Processing files"
        ))
    
    # Print summary
    successful = sum(1 for r in results if r)
    print(f"\nDownload and extraction complete:")
    print(f"Successfully processed: {successful}/{len(file_numbers)} files")
    print(f"Failed: {len(file_numbers) - successful} files")

if __name__ == "__main__":
    # Example usage
    base_url = "https://cernbox.cern.ch/remote.php/dav/public-files/wJhzxMJq6gmbhE7/pixel_clusters_d"
    start_num = 16401
    end_num = 16480
    output_dir = "training_data"
    
    download_and_extract_data(start_num, end_num, base_url, output_dir)