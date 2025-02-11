import pandas as pd
import numpy as np
import pypsa

def analyze_network_constraints(filename):
    """네트워크 제약조건 분석"""
    print("\n=== 네트워크 제약조건 분석 ===")
    
    # 네트워크 생성
    network = pypsa.Network()
    network.import_from_csv_folder(filename)
    
    # 제약조건 정보 출력
    print("\n1. 전력 수급 제약:")
    for bus in network.buses.index:
        if 'EL' in bus:
            generators = network.generators[network.generators.bus == bus]
            loads = network.loads[network.loads.bus == bus]
            
            gen_capacity = generators.p_nom.sum()
            load_max = loads.p_set.max() if not loads.empty else 0
            
            print(f"\n버스 {bus}:")
            print(f"- 발전 용량: {gen_capacity:.2f} MW")
            print(f"- 최대 부하: {load_max:.2f} MW")
            print(f"- 발전/부하 비율: {gen_capacity/load_max:.2f}" if load_max > 0 else "- 부하 없음")
    
    print("\n2. 수소 수급 제약:")
    for bus in network.buses.index:
        if 'hydrogen' in bus:
            links = network.links[network.links.bus1 == bus]  # 전해조
            loads = network.loads[network.loads.bus == bus]
            
            h2_production = links.p_nom.sum() if not links.empty else 0
            h2_demand = loads.p_set.sum() if not loads.empty else 0
            
            print(f"\n버스 {bus}:")
            print(f"- 수소 생산 용량: {h2_production:.2f} MW")
            print(f"- 수소 수요: {h2_demand:.2f} MW")
            print(f"- 생산/수요 비율: {h2_production/h2_demand:.2f}" if h2_demand > 0 else "- 수요 없음")
    
    print("\n3. 시간별 제약 분석:")
    snapshots = network.snapshots
    print(f"총 시간 단계: {len(snapshots)}")
    
    # 발전 패턴 분석
    for gen in network.generators.index:
        if hasattr(network.generators_t, 'p_max_pu') and gen in network.generators_t.p_max_pu:
            pattern = network.generators_t.p_max_pu[gen]
            print(f"\n발전기 {gen} 패턴:")
            print(f"- 최소값: {pattern.min():.4f}")
            print(f"- 최대값: {pattern.max():.4f}")
            print(f"- 평균값: {pattern.mean():.4f}")
            print(f"- 0인 시간: {(pattern == 0).sum()}시간")

def main():
    # CSV 폴더 경로 (PyPSA가 생성한 임시 파일 위치)
    network_path = "path/to/network/files"  # 실제 경로로 수정 필요
    
    # 제약조건 분석
    analyze_network_constraints(network_path)

if __name__ == "__main__":
    main()