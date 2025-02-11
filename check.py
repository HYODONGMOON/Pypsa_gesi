import pandas as pd
import numpy as np

def read_input_data(filename):
    """엑셀 파일에서 입력 데이터 읽기"""
    try:
        with pd.ExcelFile('input_data.xlsx') as xls:
            data = {}
            for sheet in ['buses', 'generators', 'lines', 'loads', 'links', 'stores', 
                         'renewable_patterns', 'load_patterns', 'timeseries']:
                if sheet in xls.sheet_names:
                    data[sheet] = pd.read_excel(xls, sheet)
            return data
    except Exception as e:
        print(f"입력 파일 읽기 오류: {str(e)}")
        return None

def check_system_balance(input_data):
    """시스템 수급 균형 체크"""
    print("\n=== 시스템 수급 균형 분석 ===")
    
    # 1. 전력 수요-공급 체크
    print("\n[전력 수급 현황]")
    
    # 전력 수요 계산
    el_loads = input_data['loads'][input_data['loads']['name'].str.contains('EL')]
    total_el_demand = el_loads['p_set'].sum()
    print(f"전력 수요: {total_el_demand:,.0f} MW")
    
    # 재생에너지 발전량
    re_gens = input_data['generators'][
        input_data['generators']['name'].str.contains('PV') | 
        input_data['generators']['name'].str.contains('WT')
    ]
    
    print("\n재생에너지 설비:")
    for _, gen in re_gens.iterrows():
        print(f"- {gen['name']}: {gen['p_nom']}MW")
        if 'p_nom_extendable' in gen and gen['p_nom_extendable']:
            print(f"  (확장가능: {gen.get('p_nom_min', 0)}MW ~ {gen.get('p_nom_max', 'unlimited')}MW)")
    
    total_re_capacity = re_gens['p_nom'].sum()
    print(f"\n총 재생에너지 설비용량: {total_re_capacity:,.0f} MW")
    
    # 2. 수소 수요-공급 체크
    print("\n[수소 수급 현황]")
    
    # 수소 수요 계산
    h2_loads = input_data['loads'][input_data['loads']['name'].str.contains('h2')]
    total_h2_demand = h2_loads['p_set'].sum()
    print(f"수소 수요: {total_h2_demand:,.0f} MW")
    
    # 전해조 용량
    electrolyzers = input_data['links'][input_data['links']['name'].str.contains('Electrolyzer')]
    print("\n전해조 설비:")
    for _, el in electrolyzers.iterrows():
        print(f"- {el['name']}: {el['p_nom']}MW")
        if 'p_nom_extendable' in el and el['p_nom_extendable']:
            print(f"  (확장가능: {el.get('p_nom_min', 0)}MW ~ {el.get('p_nom_max', 'unlimited')}MW)")
    
    # 3. 저장소 용량 체크
    if 'stores' in input_data:
        print("\n[저장소 현황]")
        for _, store in input_data['stores'].iterrows():
            print(f"- {store['name']}: {store['e_nom']}MWh")
            if 'e_nom_extendable' in store and store['e_nom_extendable']:
                print(f"  (확장가능: {store.get('e_nom_min', 0)}MWh ~ {store.get('e_nom_max', 'unlimited')}MWh)")

def analyze_hydrogen_chain(input_data):
    """수소 생산-운송 체인 전체 분석"""
    print("\n=== 수소 체인 분석 ===")
    
    # 1. 수소 수요 확인
    h2_loads = input_data['loads'][input_data['loads']['name'].str.contains('h2')]
    print("\n1. 수소 수요:")
    total_h2_demand = 0
    for _, load in h2_loads.iterrows():
        demand = float(load['p_set'])
        total_h2_demand += demand
        print(f"{load['name']} ({load['bus']}): {demand} MW")
    print(f"총 수소 수요: {total_h2_demand} MW")
    
    # 2. 전해조 설정 확인
    electrolyzers = input_data['links'][input_data['links']['name'].str.contains('Electrolyzer')]
    print("\n2. 전해조 설정:")
    for _, el in electrolyzers.iterrows():
        print(f"\n{el['name']}:")
        print(f"- 입력 버스: {el['bus0']}")
        print(f"- 출력 버스: {el['bus1']}")
        print(f"- 기본 용량: {el['p_nom']} MW")
        print(f"- 확장 가능: {el.get('p_nom_extendable', False)}")
        if el.get('p_nom_extendable', False):
            print(f"- 최대 용량: {el.get('p_nom_max', 'unlimited')} MW")
        print(f"- 효율: {el.get('efficiency', 'N/A')}")
        print(f"- 운영 범위: {el.get('p_min_pu', 0)} ~ {el.get('p_max_pu', 1)} pu")
    
    # 3. 전력 공급 확인
    print("\n3. 전해조 전력 공급:")
    for _, el in electrolyzers.iterrows():
        power_bus = el['bus0']
        
        # 해당 버스의 발전 용량
        generators = input_data['generators'][input_data['generators']['bus'] == power_bus]
        total_gen = generators['p_nom'].sum()
        
        # 해당 버스의 전력 부하 (전해조 제외)
        loads = input_data['loads'][
            (input_data['loads']['bus'] == power_bus) & 
            (~input_data['loads']['name'].str.contains('h2'))
        ]
        total_load = loads['p_set'].sum() if not loads.empty else 0
        
        print(f"\n전해조 {el['name']}의 전력 버스 {power_bus}:")
        print(f"- 발전 용량: {total_gen} MW")
        print(f"- 기본 부하: {total_load} MW")
        print(f"- 가용 전력: {total_gen - total_load} MW")
    
    # 4. 수소 네트워크 연결성 확인
    print("\n4. 수소 네트워크 연결성:")
    h2_pipes = input_data['links'][input_data['links']['name'].str.contains('Hydrogen')]
    for _, pipe in h2_pipes.iterrows():
        print(f"\n{pipe['name']}:")
        print(f"- 연결: {pipe['bus0']} ↔ {pipe['bus1']}")
        print(f"- 용량: {pipe['p_nom']} MW")
        if pipe.get('p_min_pu') is not None:
            print(f"- 양방향 가능: {'예' if pipe['p_min_pu'] < 0 else '아니오'}")

def check_model_settings(input_data):
    """모델 설정 확인"""
    print("\n=== 모델 설정 확인 ===")
    
    # 시간 설정 확인
    if 'timeseries' in input_data:
        ts = input_data['timeseries'].iloc[0]
        print(f"\n시간 설정:")
        print(f"- 시작: {ts['start_time']}")
        print(f"- 종료: {ts['end_time']}")
        print(f"- 주기: {ts['frequency']}")

def check_electrolyzer_setup(input_data):
    """전해조 설정 상세 확인"""
    print("\n=== 전해조 설정 확인 ===")
    
    # 1. 전해조 Link 확인
    electrolyzers = input_data['links'][input_data['links']['name'].str.contains('Electrolyzer')]
    print("\n1. 전해조 기본 설정:")
    for _, el in electrolyzers.iterrows():
        print(f"\n{el['name']}:")
        required_fields = ['bus0', 'bus1', 'carrier', 'p_nom', 'efficiency']
        for field in required_fields:
            value = el.get(field, 'Missing')
            print(f"- {field}: {value}")
        
        # 확장 관련 설정
        print("- 확장 설정:")
        print(f"  * p_nom_extendable: {el.get('p_nom_extendable', 'Missing')}")
        print(f"  * p_nom_max: {el.get('p_nom_max', 'Missing')}")
        print(f"  * p_min_pu: {el.get('p_min_pu', 0)}")
        print(f"  * p_max_pu: {el.get('p_max_pu', 1)}")
    
    # 2. 버스 설정 확인
    print("\n2. 관련 버스 설정:")
    el_buses = set()
    for _, el in electrolyzers.iterrows():
        el_buses.add(el['bus0'])
        el_buses.add(el['bus1'])
    
    for bus_name in el_buses:
        bus = input_data['buses'][input_data['buses']['name'] == bus_name]
        if not bus.empty:
            print(f"\n{bus_name}:")
            print(f"- carrier: {bus.iloc[0].get('carrier', 'Missing')}")
            print(f"- v_nom: {bus.iloc[0].get('v_nom', 'Missing')}")
    
    # 3. carrier 일관성 확인
    print("\n3. Carrier 설정 확인:")
    unique_carriers = set()
    for _, el in electrolyzers.iterrows():
        carrier = el.get('carrier', 'Missing')
        unique_carriers.add(carrier)
    print(f"전해조 carriers: {unique_carriers}")
    
    # 4. 용량 제약 확인
    print("\n4. 용량 제약 분석:")
    for _, el in electrolyzers.iterrows():
        print(f"\n{el['name']} 용량 분석:")
        base_capacity = float(el['p_nom'])
        max_capacity = float(el.get('p_nom_max', float('inf')))
        efficiency = float(el.get('efficiency', 0.7))
        
        print(f"- 기본 용량: {base_capacity} MW")
        print(f"- 최대 용량: {max_capacity if max_capacity != float('inf') else 'unlimited'} MW")
        print(f"- 효율: {efficiency}")
        print(f"- 최대 수소 생산: {base_capacity * efficiency} MW (기본 용량 기준)")
        if max_capacity != float('inf'):
            print(f"- 최대 수소 생산: {max_capacity * efficiency} MW (최대 용량 기준)")

def main():
    # 입력 파일 경로
    input_file = "C:/Users/Hyodong.Moon/Desktop/HDMOON/python workplace/PyPSA-master/PyPSA-HD/input_data.xlsx"  # 실제 파일 경로로 수정하세요
    
    # 데이터 읽기
    input_data = read_input_data(input_file)
    if input_data is None:
        return
    
    # 시스템 균형 체크
    check_system_balance(input_data)
    
    # 수소 체인 분석
    analyze_hydrogen_chain(input_data)
    
    # 모델 설정 확인
    check_model_settings(input_data)
    
    # 전해조 설정 확인
    check_electrolyzer_setup(input_data)

if __name__ == "__main__":
    main()