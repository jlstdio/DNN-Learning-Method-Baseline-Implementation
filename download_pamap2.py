import os
import requests
import zipfile
import pandas as pd
import warnings

# SSL 경고 무시 (우리가 신뢰하는 UCI 사이트이므로 안전합니다)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def download_and_extract_pamap2(base_dir='./dataset'):
    dataset_dir = os.path.join(base_dir, 'PAMAP2')
    zip_path = os.path.join(base_dir, 'pamap2.zip')
    
    # 폴더 생성
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. 데이터 다운로드
    if not os.path.exists(zip_path):
        print("UCI 저장소에서 PAMAP2 원본 ZIP 파일을 직접 다운로드합니다...")
        url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
        
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
    # PAMAP2 데이터셋 구조 특성상 압축 파일 내부에 'PAMAP2_Dataset' 폴더가 있습니다.
    protocol_path = os.path.join(dataset_dir, 'PAMAP2_Dataset', 'Protocol')
    if not os.path.exists(protocol_path):
        print("압축을 해제하는 중입니다...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("압축 해제 완료!")
    else:
        print("이미 압축이 해제되어 있습니다.")
        
    return protocol_path

# 실행
if __name__ == "__main__":
    protocol_folder = download_and_extract_pamap2()
    
    # 3. 데이터 로드 테스트 (예: subject101.dat 파일 로드)
    # PAMAP2 데이터는 공백으로 구분되어 있으며, 결측치는 NaN으로 표기됩니다.
    sample_file = os.path.join(protocol_folder, 'subject101.dat')
    
    if os.path.exists(sample_file):
        print(f"\n{sample_file} 파일을 읽어옵니다...")
        # 총 54개의 컬럼 (Timestamp, ActivityID, HeartRate, 그리고 각 IMU 센서 데이터들)
        df = pd.read_csv(sample_file, sep='\s+', header=None)
        
        print("\n[데이터 로드 성공!]")
        print(f"데이터 형태 (Shape): {df.shape}")
        print(df.head())
    else:
        print("파일을 찾을 수 없습니다. 경로를 확인해 주세요.")