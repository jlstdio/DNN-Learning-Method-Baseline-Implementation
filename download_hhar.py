import os
import requests
import zipfile
import pandas as pd
import warnings

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def download_and_extract_hhar(base_dir='./dataset'):
    dataset_dir = os.path.join(base_dir, 'HHAR')
    zip_path = os.path.join(base_dir, 'hhar.zip')
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. 데이터 다운로드
    if not os.path.exists(zip_path):
        print("UCI 저장소에서 HHAR 원본 ZIP 파일을 직접 다운로드합니다...")
        # HHAR 데이터셋(ID 344)의 UCI 정적 다운로드 링크
        url = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
        
        # verify=False로 SSL 인증서 검증 우회
        response = requests.get(url, verify=False, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("다운로드 완료!")
    else:
        print("이미 다운로드된 ZIP 파일이 존재합니다.")

    # 2. 압축 해제
    # HHAR 데이터셋은 최상단에 바로 CSV 파일들이 압축 풀리는 형태가 많습니다.
    # 확인을 위해 핵심 파일 중 하나인 Watch_accelerometer.csv 경로를 체크합니다.
    sample_csv_path = os.path.join(dataset_dir, 'Watch_accelerometer.csv')
    
    if not os.path.exists(sample_csv_path):
        print("압축을 해제하는 중입니다...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("압축 해제 완료!")
    else:
        print("이미 압축이 해제되어 있습니다.")
        
    return dataset_dir

# 실행
if __name__ == "__main__":
    hhar_folder = download_and_extract_hhar()
    
    # 3. 데이터 로드 테스트 (예: Watch_accelerometer.csv 파일 로드)
    # HHAR 데이터는 일반적인 쉼표(,)로 구분된 CSV 파일이며 헤더(Header)가 존재합니다.
    sample_file = os.path.join(hhar_folder, 'Watch_accelerometer.csv')
    
    if os.path.exists(sample_file):
        print(f"\n{sample_file} 파일을 읽어옵니다...")
        df = pd.read_csv(sample_file)
        
        print("\n[HHAR 데이터 로드 성공!]")
        print(f"데이터 형태 (Shape): {df.shape}")
        # Index, Arrival_Time, Creation_Time, x, y, z, User, Model, Device, gt(Ground Truth) 등의 컬럼
        print(df.head())
    else:
        print("파일을 찾을 수 없습니다. 압축 해제 경로를 확인해 주세요.")