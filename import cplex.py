import cplex
import os

try:
    c = cplex.Cplex()
    print("CPLEX 버전:", c.get_version())
    
    # 라이센스 파일 위치 확인
    license_path = os.environ.get('ILOG_LICENSE_FILE')
    print("라이센스 파일 경로:", license_path if license_path else "환경 변수에서 찾을 수 없음")
    
    # 간단한 테스트 문제 생성
    try:
        # 2000개 이상의 변수 생성 시도 (Community Edition 제한 테스트)
        c.variables.add(names=['test' + str(i) for i in range(2000)])
        print("라이센스 상태: Full Version")
        
    except cplex.exceptions.CplexError as e:
        if "1016" in str(e):
            print("라이센스 상태: Community Edition")
        else:
            print("CPLEX 오류:", str(e))
    
    c.end()

except Exception as e:
    print("CPLEX 초기화 오류:", str(e))

# CPLEX 설치 경로 확인
print("\nCPLEX 설치 정보:")
print("Python 모듈 위치:", cplex.__file__)