import pypsa
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

# 상수 정의
INPUT_FILE = "C:/Users/Hyodong.Moon/Desktop/HDMOON/python workplace/PyPSA-master/PyPSA-HD/input_data.xlsx"

def read_input_data(input_file):
    """엑셀 파일에서 입력 데이터 읽기"""
    try:
        xls = pd.ExcelFile(input_file)
        input_data = {}
        
        # 필수 시트 및 컬럼 정의
        required_sheets = {
            'timeseries': ['start_time', 'end_time', 'frequency'],
            'buses': ['name', 'v_nom', 'carrier', 'x', 'y'],
            'generators': ['name', 'bus', 'carrier', 'p_nom', 'p_nom_extendable', 
                           'p_nom_min', 'p_nom_max', 'marginal_cost', 'capital_cost', 'efficiency',  
                           'committable', 'p_max_pu', 'p_min_pu', 'ramp_limit_up', 'min_up_time', 'start_up_cost', 'lifetime'
            ],
            'lines': ['name', 'bus0', 'bus1', 'carrier', 'x', 'r', 's_nom', 'length', 'v_nom'],
            'loads': ['name', 'bus', 'p_set'],
            'stores': ['name', 'bus', 'carrier', 'e_nom', 'e_nom_extendable', 'e_cyclic', 'standing_loss', 'efficiency_store', 'efficiency_dispatch', 'e_initial', 'e_nom_max'],  # 저장장치 추가
            'links': [
                'name', 'from_bus', 'to_bus', 'efficiency', 'p_nom_extendable', 'p_nom',
                'p_nom_max', 'marginal_cost', 'p_min_pu', 'p_max_pu'],
            'load_patterns': ['hour'],
            'renewable_patterns': ['hour', 'PV', 'WT'],
            'constraints': ['name', 'type', 'carrier_attribte', 'sense', 'constant']
        }
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()
            input_data[sheet_name] = df
        
        return input_data
        
    except Exception as e:
        print(f"데이터 읽기 오류: {str(e)}")
        raise

def adjust_pattern_length(pattern_values, required_length):
    """패턴 길이를 필요한 길이에 맞게 조정"""
    if len(pattern_values) == required_length:
        return pattern_values
    
    # 패턴이 더 짧은 경우
    elif len(pattern_values) < required_length:
        # 부족한 만큼 처음부터 반복
        repetitions = required_length // len(pattern_values) + 1
        extended_pattern = np.tile(pattern_values, repetitions)
        return extended_pattern[:required_length]
    
    # 패턴이 더 긴 경우
    else:
        return pattern_values[:required_length]

def normalize_pattern(pattern):
    """발전 패턴을 0~1 사이로 정규화"""
    if np.max(pattern) > 0:
        return pattern / np.max(pattern)
    return pattern

def create_network(input_data):
    try:
        network = pypsa.Network()
        
        # carriers 정의 시 hydrogen 추가
        carriers = {
            'AC': {'name': 'AC', 'co2_emissions': 0},
            'DC': {'name': 'DC', 'co2_emissions': 0},
            'electricity': {'name': 'electricity', 'co2_emissions': 0},
            'coal': {'name': 'coal', 'co2_emissions': 0.9},
            'gas': {'name': 'gas', 'co2_emissions': 0.4},
            'nuclear': {'name': 'nuclear', 'co2_emissions': 0},
            'solar': {'name': 'solar', 'co2_emissions': 0},
            'wind': {'name': 'wind', 'co2_emissions': 0},
            'hydrogen': {'name': 'hydrogen', 'co2_emissions': 0}  # 수소 추가
        }
        
        # carriers 추가
        for carrier, specs in carriers.items():
            network.add("Carrier",
                       name=specs['name'],
                       co2_emissions=specs['co2_emissions'])
        
        # 시간 설정 - snapshots 길이 확인을 위해 먼저 설정
        if 'timeseries' in input_data:
            ts = input_data['timeseries'].iloc[0]
            snapshots = pd.date_range(
                start=ts['start_time'],
                end=ts['end_time'],
                freq=ts['frequency'],
                inclusive='left'
            )
            network.set_snapshots(snapshots)
            snapshots_length = len(snapshots)
        
        # 3. 버스 추가
        if 'buses' in input_data:
            for _, bus in input_data['buses'].iterrows():
                bus_name = str(bus['name'])
                if 'EL' in bus_name:
                    carrier = 'electricity'
                elif 'hydrogen' in bus_name.lower():
                    carrier = 'hydrogen'
                else:
                    carrier = 'AC'
                
                network.add("Bus",
                          name=bus_name,
                          v_nom=float(bus['v_nom']),
                          carrier=carrier)
        
        # 재생에너지 패턴 준비 - 길이 조정 추가
        renewable_patterns = {}
        if 'renewable_patterns' in input_data:
            patterns_df = input_data['renewable_patterns']
            if 'PV' in patterns_df.columns:
                pv_pattern = normalize_pattern(patterns_df['PV'].values)
                pv_pattern = adjust_pattern_length(pv_pattern, snapshots_length)  # 길이 조정
                renewable_patterns['PV_pattern'] = pv_pattern
                print(f"PV 패턴 길이: {len(pv_pattern)}, Snapshots 길이: {snapshots_length}")
            
            if 'WT' in patterns_df.columns:
                wt_pattern = normalize_pattern(patterns_df['WT'].values)
                wt_pattern = adjust_pattern_length(wt_pattern, snapshots_length)  # 길이 조정
                renewable_patterns['WT_pattern'] = wt_pattern
                print(f"WT 패턴 길이: {len(wt_pattern)}, Snapshots 길이: {snapshots_length}")
        
        # 발전기 추가
        if 'generators' in input_data:
            for _, gen in input_data['generators'].iterrows():
                gen_name = str(gen['name'])
                params = {
                    'name': gen_name,
                    'bus': str(gen['bus']),
                    'p_nom': float(gen['p_nom'])
                }
                
                # carrier 설정
                if 'PV' in gen_name:
                    params['carrier'] = 'solar'
                elif 'WT' in gen_name:
                    params['carrier'] = 'wind'
                elif 'Coal' in gen_name:
                    params['carrier'] = 'coal'
                elif 'LNG' in gen_name:
                    params['carrier'] = 'gas'
                elif 'Nuclear' in gen_name:
                    params['carrier'] = 'nuclear'
                else:
                    params['carrier'] = 'electricity'
                
                # p_max_pu와 p_min_pu 패턴 적용
                if 'p_max_pu' in gen and pd.notna(gen['p_max_pu']):
                    pattern_name = str(gen['p_max_pu'])
                    if pattern_name in renewable_patterns:
                        network.generators_t.p_max_pu[gen_name] = renewable_patterns[pattern_name]
                        print(f"{gen_name}에 {pattern_name} 패턴이 p_max_pu로 적용되었습니다.")
                
                if 'p_min_pu' in gen and pd.notna(gen['p_min_pu']):
                    pattern_name = str(gen['p_min_pu'])
                    if pattern_name in renewable_patterns:
                        network.generators_t.p_min_pu[gen_name] = renewable_patterns[pattern_name]
                        print(f"{gen_name}에 {pattern_name} 패턴이 p_min_pu로 적용되었습니다.")
                
                # 나머지 파라미터 추가
                optional_params = ['p_nom_extendable', 'p_nom_min', 'p_nom_max', 
                                 'marginal_cost', 'capital_cost', 'efficiency']
                for param in optional_params:
                    if param in gen and pd.notna(gen[param]):
                        params[param] = float(gen[param])
                
                network.add("Generator", **params)
        
        # 5. 저장장치 추가
        storage_buses = ['A_EL_Store', 'B_EL_Store', 'A_Hydrogen_Store']
        for bus in storage_buses:
            if bus in network.buses.index:
                carrier = 'hydrogen' if 'Hydrogen' in bus else 'electricity'
                network.add("Store",
                          name=f"Storage_{bus}",
                          bus=bus,
                          carrier=carrier,
                          e_nom_extendable=True,
                          e_cyclic=True)
        
        # 6. 부하 추가
        if 'loads' in input_data:
            for _, load in input_data['loads'].iterrows():
                name = str(load['name'])
                bus_name = str(load['bus'])
                p_set = float(load['p_set'])
                
                # 부하 패턴 적용 (길이 조정)
                if 'load_patterns' in input_data and name in input_data['load_patterns'].columns:
                    original_pattern = input_data['load_patterns'][name].values
                    
                    # 패턴 길이 조정
                    if len(original_pattern) != len(snapshots):
                        print(f"부하 {name}의 패턴 길이 조정: {len(original_pattern)} → {len(snapshots)}")
                        if len(original_pattern) < len(snapshots):
                            # 패턴 반복
                            repeats = len(snapshots) // len(original_pattern) + 1
                            pattern = np.tile(original_pattern, repeats)[:len(snapshots)]
                        else:
                            # 패턴 잘라내기
                            pattern = original_pattern[:len(snapshots)]
                    else:
                        pattern = original_pattern
                    
                    # 기준 부하에 패턴 적용
                    p_set_array = np.full(len(snapshots), p_set)
                    p_set = p_set_array * pattern
                else:
                    # 패턴이 없는 경우 일정한 부하
                    p_set = np.full(len(snapshots), p_set)
                
                network.add("Load",
                          name=name,
                          bus=bus_name,
                          p_set=p_set)
        
        # Links 추가
        if 'links' in input_data:
            for _, link in input_data['links'].iterrows():
                params = {
                    'name': str(link['name']),
                    'bus0': str(link['bus0']),  # 원래 컬럼명으로 변경
                    'bus1': str(link['bus1']),  # 원래 컬럼명으로 변경
                    'p_nom': float(link['p_nom']),
                    'efficiency': float(link.get('efficiency', 1.0))  # 기본값 1.0
                }
                
                # p_nom_extendable 처리
                if 'p_nom_extendable' in link:
                    params['p_nom_extendable'] = bool(link['p_nom_extendable'])
                    
                    # p_nom_extendable이 True인 경우 min/max 처리
                    if params['p_nom_extendable']:
                        if 'p_nom_min' in link and pd.notna(link['p_nom_min']):
                            params['p_nom_min'] = float(link['p_nom_min'])
                        if 'p_nom_max' in link and pd.notna(link['p_nom_max']):
                            params['p_nom_max'] = float(link['p_nom_max'])
                
                network.add("Link", **params)
        
        # CO2 제약 추가
        if 'constraints' in input_data:
            constraints_df = input_data['constraints']
            co2_limit = constraints_df[constraints_df['name'] == 'CO2Limit']
            
            if not co2_limit.empty:
                limit_value = float(co2_limit.iloc[0]['constant'])
                network.add("GlobalConstraint",
                          "CO2Limit",
                          sense="<=",
                          constant=limit_value)
                print(f"CO2 제약이 {limit_value} tCO2로 설정되었습니다.")

        # ESS 추가 부분 수정
        if 'stores' in input_data:
            for _, store in input_data['stores'].iterrows():
                # p_nom_extendable을 e_nom_extendable로 변경
                e_nom_extendable = bool(store['p_nom_extendable']) if 'p_nom_extendable' in store else True
                
                params = {
                    'name': str(store['name']),
                    'bus': str(store['bus']),
                    'carrier': str(store['carrier']),
                    'e_nom': float(store['e_nom']) if pd.notna(store['e_nom']) else 0,
                    'e_nom_extendable': e_nom_extendable,  # 수정된 부분
                    'e_cyclic': bool(store['e_cyclic']) if 'e_cyclic' in store else True,
                    'efficiency_store': float(store['efficiency_store']) if 'efficiency_store' in store and pd.notna(store['efficiency_store']) else 0.95,
                    'efficiency_dispatch': float(store['efficiency_dispatch']) if 'efficiency_dispatch' in store and pd.notna(store['efficiency_dispatch']) else 0.95,
                    'standing_loss': float(store['standing_loss']) if 'standing_loss' in store and pd.notna(store['standing_loss']) else 0,
                    'e_initial': float(store['e_initial']) if 'e_initial' in store and pd.notna(store['e_initial']) else 0
                }
                
                if 'e_nom_max' in store and pd.notna(store['e_nom_max']):
                    params['e_nom_max'] = float(store['e_nom_max'])
                
                network.add("Store", **params)
                print(f"저장장치 {store['name']} 추가됨 (e_nom_extendable: {e_nom_extendable})")

        return network
        
    except Exception as e:
        print(f"네트워크 생성 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

def optimize_network(network):
    """네트워크 최적화"""
    if network is None:
        print("네트워크가 생성되지 않았습니다.")
        return False
    
    try:
        print("\n최적화 시작...")
        
        # CPU 코어 수 확인
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        print(f"사용 가능한 CPU 코어 수: {num_cores}")
        
        # 최적화 옵션 단순화 - 핵심 옵션만 유지
        solver_options = {
            'threads': num_cores,     # 모든 가용 코어 사용
            'lpmethod': 4,            # Barrier method
            'parallel': 1,            # 병렬 모드 활성화
            'barrier.algorithm': 3    # 대체 알고리즘
        }
        
        # 발전기 제약조건 확인 및 출력
        print("\n발전기 제약조건 확인:")
        for gen in network.generators.index:
            print(f"\n발전기: {gen}")
            print(f"p_nom: {network.generators.at[gen, 'p_nom']}")
            if hasattr(network.generators_t, 'p_min_pu') and gen in network.generators_t.p_min_pu:
                print(f"p_min_pu 설정됨: {network.generators_t.p_min_pu[gen].mean():.3f} (평균)")
            if hasattr(network.generators_t, 'p_max_pu') and gen in network.generators_t.p_max_pu:
                print(f"p_max_pu 설정됨: {network.generators_t.p_max_pu[gen].mean():.3f} (평균)")
        
        # 부하 확인
        print("\n부하 확인:")
        for load in network.loads.index:
            print(f"부하: {load}, 평균값: {network.loads_t.p_set[load].mean():.2f}")
        
        # 최적화 실행
        status = network.optimize(solver_name='cplex', 
                                solver_options=solver_options)
        
        print(f"\n최적화 상태: {status}")
        if hasattr(network, 'objective'):
            print(f"목적함수 값: {network.objective}")
        
        return True
        
    except Exception as e:
        print(f"\n최적화 중 오류 발생: {str(e)}")
        traceback.print_exc()  # 상세 오류 정보 출력
        return False

def extract_results(network):
    """주요 결과 추출"""
    
    results = {
        'generator_output': network.generators_t.p,
        'node_prices': network.buses_t.marginal_price,
        'line_flows': network.lines_t.p0,
        'total_cost': network.objective,
        'load_balance': network.buses_t.p,
        'storage_state': network.storage_units_t.state_of_charge if not network.storage_units_t.empty else None
    }
    
    return results

def save_results(network, filename=None):
    """최적화 결과를 Excel 파일로 저장"""
    try:
        if not hasattr(network, 'objective') or network.objective is None:
            print("최적화 결과가 없어 저장할 수 없습니다.")
            return False
            
        # 현재 시간을 포함한 파일명 생성
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results_({current_time}).xlsx'
            
        print("결과 저장 중...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 발전기 출력 결과
            network.generators_t.p.to_excel(writer, sheet_name='Generator_Output')
            
            # AC 선로 조류 결과
            network.lines_t.p0.to_excel(writer, sheet_name='Line_Flow')
            
            # HVDC Link 조류 결과 추가
            network.links_t.p0.to_excel(writer, sheet_name='Link_Flow')
            
            # 버스 정보
            bus_results = pd.DataFrame({
                'v_nom': network.buses.v_nom,
                'carrier': network.buses.carrier
            })
            bus_results.to_excel(writer, sheet_name='Bus_Info')
            
            # 발전기 정보
            gen_results = pd.DataFrame({
                'bus': network.generators.bus,
                'p_nom': network.generators.p_nom,
                'p_max_pu': network.generators.p_max_pu,
                'marginal_cost': network.generators.marginal_cost
            })
            gen_results.to_excel(writer, sheet_name='Generator_Info')
            
            # Link 정보 추가
            link_results = pd.DataFrame({
                'bus0': network.links.bus0,
                'bus1': network.links.bus1,
                'p_nom': network.links.p_nom,
                'efficiency': network.links.efficiency
            })
            link_results.to_excel(writer, sheet_name='Link_Info')
            
            # ESS 충방전 결과 (있는 경우에만)
            if hasattr(network, 'stores_t') and not network.stores_t.p.empty:
                network.stores_t.p.to_excel(writer, sheet_name='Storage_Power')
                network.stores_t.e.to_excel(writer, sheet_name='Storage_Energy')
            
            # ESS 정보 (있는 경우에만)
            if not network.stores.empty:
                store_results = pd.DataFrame({
                    'bus': network.stores.bus,
                    'carrier': network.stores.carrier,
                    'e_nom': network.stores.e_nom,
                    'e_cyclic': network.stores.e_cyclic
                })
                store_results.to_excel(writer, sheet_name='Storage_Info')
            
            # 시간별 부하 결과
            network.loads_t.p.to_excel(writer, sheet_name='Hourly_Loads')
            
            # 최적화 요약
            summary = pd.DataFrame({
                'Parameter': ['Total Cost', 'Status'],
                'Value': [network.objective, 'Optimal']
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"결과가 {filename}에 저장되었습니다.")
        return True
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {str(e)}")
        return False

def validate_input_data(input_data):
    """입력 데이터 유효성 검사"""
    required_sheets = ['buses', 'generators', 'lines', 'loads', 'timeseries']
    
    # 필수 시트 확인
    for sheet in required_sheets:
        if sheet not in input_data:
            raise ValueError(f"필수 시트 '{sheet}'가 없습니다.")
        if input_data[sheet].empty:
            raise ValueError(f"'{sheet}' 시트가 비어있습니다.")
    
    # 데이터 타입 확인 및 변환
    for _, row in input_data['buses'].iterrows():
        if not isinstance(row['name'], str):
            input_data['buses'].loc[_, 'name'] = str(row['name'])
    
    # timeseries 데이터 확인
    timeseries = input_data['timeseries'].iloc[0]
    if not pd.to_datetime(timeseries['start_time']):
        raise ValueError("잘못된 start_time 형식입니다.")
    if not pd.to_datetime(timeseries['end_time']):
        raise ValueError("잘못된 end_time 형식입니다.")
    if 'h' not in str(timeseries['frequency']).lower():
        raise ValueError("frequency는 'h' 형식이어야 합니다.")

    return input_data

def check_network_connections(network):
    """네트워크 연결 상태 확인"""
    print("\n=== 네트워크 연결 상태 확인 ===")
    
    # 1. 버스 연결 확인
    print("\n버스 정보:")
    for bus in network.buses.index:
        print(f"\n버스 {bus}:")
        # 연결된 발전기
        gens = network.generators[network.generators.bus == bus].index
        print(f"연결된 발전기: {list(gens)}")
        # 연결된 부하
        loads = network.loads[network.loads.bus == bus].index
        print(f"연결된 부하: {list(loads)}")
        # 연결된 저장장치
        stores = network.stores[network.stores.bus == bus].index
        print(f"연결된 저장장치: {list(stores)}")
    
    # 2. ESS 연결 및 설정 확인
    print("\nESS 상세 설정:")
    for store in network.stores.index:
        print(f"\n저장장치 {store}:")
        print(f"연결된 버스: {network.stores.at[store, 'bus']}")
        print(f"저장용량: {network.stores.at[store, 'e_nom']} MWh")
        print(f"충전효율: {network.stores.at[store, 'efficiency_store'] if 'efficiency_store' in network.stores else 1.0}")
        print(f"방전효율: {network.stores.at[store, 'efficiency_dispatch'] if 'efficiency_dispatch' in network.stores else 1.0}")
        print(f"순환 운전: {network.stores.at[store, 'e_cyclic']}")
        print(f"초기 충전상태: {network.stores.at[store, 'e_initial'] if 'e_initial' in network.stores else 0}")

    # 3. 선로 연결 확인
    print("\n선로 연결:")
    if not network.lines.empty:
        for line in network.lines.index:
            print(f"\n선로 {line}:")
            print(f"From: {network.lines.at[line, 'bus0']} To: {network.lines.at[line, 'bus1']}")
            print(f"용량: {network.lines.at[line, 's_nom']} MVA")

def check_excel_data_loading(input_data):
    """엑셀 데이터 로드 상태 확인"""
    print("\n=== 엑셀 데이터 로드 상태 확인 ===")
    
    # 1. 시트 존재 여부 확인
    print("\n로드된 시트:")
    for sheet_name in input_data.keys():
        print(f"\n[{sheet_name}] 시트:")
        print(f"행 수: {len(input_data[sheet_name])}")
        print(f"컬럼: {list(input_data[sheet_name].columns)}")
    
    # 2. stores 시트 상세 확인
    if 'stores' in input_data:
        print("\n\n=== Stores 시트 상세 데이터 ===")
        stores_df = input_data['stores']
        print("\n컬럼별 데이터 타입:")
        print(stores_df.dtypes)
        print("\n실제 데이터:")
        print(stores_df)
        
        # 필수 컬럼 확인
        required_columns = [
            'name', 'bus', 'carrier', 'e_nom', 'e_nom_extendable', 
            'e_cyclic', 'standing_loss', 'efficiency_store', 
            'efficiency_dispatch', 'e_initial', 'e_nom_max'
        ]
        missing_columns = [col for col in required_columns if col not in stores_df.columns]
        if missing_columns:
            print(f"\n누락된 필수 컬럼: {missing_columns}")
    
    # 3. links 시트 상세 확인
    if 'links' in input_data:
        print("\n\n=== Links 시트 상세 데이터 ===")
        links_df = input_data['links']
        print("\n컬럼별 데이터 타입:")
        print(links_df.dtypes)
        print("\n실제 데이터:")
        print(links_df)
        
        # 필수 컬럼 확인
        required_columns = [
            'name', 'bus0', 'bus1', 'efficiency', 'p_nom', 
            'p_nom_extendable', 'p_nom_max'
        ]
        missing_columns = [col for col in required_columns if col not in links_df.columns]
        if missing_columns:
            print(f"\n누락된 필수 컬럼: {missing_columns}")

def main():
    input_data = read_input_data(INPUT_FILE)
    if input_data is None:
        return
    
    check_excel_data_loading(input_data)  # 데이터 로드 상태 확인
    
    network = create_network(input_data)
    if optimize_network(network):
        save_results(network)

if __name__ == "__main__":
    main()