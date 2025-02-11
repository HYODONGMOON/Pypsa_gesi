import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import contextily as ctx
import traceback
import matplotlib.font_manager as fm
import platform
import pandas as pd
import numpy as np

class KoreaMapVisualizer:
    def __init__(self):
        """한국 지도 시각화를 위한 클래스 초기화"""
        self.map_data = None
        self.ax = None
        self._set_font()  # 폰트 설정
        
    def _set_font(self):
        """시스템에 맞는 한글 폰트 설정"""
        system_name = platform.system()
        
        if system_name == "Windows":
            plt.rc('font', family='Malgun Gothic')  # Windows의 경우 맑은 고딕
        elif system_name == "Darwin":
            plt.rc('font', family='AppleGothic')   # Mac의 경우 애플고딕
        else:
            print("Warning: 한글 폰트가 없을 수 있습니다.")
            
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
    def load_map_data(self, shapefile_path="data/BND_SIDO_PG.shp"):
        """행정구역 데이터 로드"""
        try:
            self.map_data = gpd.read_file(shapefile_path, encoding='cp949')
            self.map_data = self.map_data.to_crs(epsg=3857)  # Web Mercator로 변환
            return True
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            return False
    
    def plot_korea_map(self, save_path='korea_map.png', show_centroids=True):
        """한국 지도 시각화"""
        try:
            if self.map_data is None:
                print("지도 데이터가 로드되지 않았습니다.")
                return False
            
            # 데이터 컬럼 확인
            print("\n사용 가능한 컬럼명:")
            print(self.map_data.columns)
            
            # 시도 이름 컬럼 찾기
            possible_name_columns = ['SIDO_NM', 'CTP_KOR_NM', 'SIG_KOR_NM', 'CTPRVN_NM', 'NAME', 'KOR_NM']
            name_column = None
            for col in possible_name_columns:
                if col in self.map_data.columns:
                    name_column = col
                    print(f"\n시도 이름 컬럼으로 '{col}' 사용")
                    break
                
            if name_column is None:
                print("\n시도 이름 컬럼을 찾을 수 없습니다.")
                return False
            
            # 그림 크기 설정
            plt.figure(figsize=(15, 15))
            
            # 지도 그리기
            self.ax = self.map_data.plot(
                figsize=(15, 15),
                edgecolor='black',
                facecolor='none',
                linewidth=1,
                alpha=0.7
            )
            
            # 시도 이름 표시 및 중심점 저장
            centroids = {}
            for idx, row in self.map_data.iterrows():
                centroid = row.geometry.centroid
                name = row[name_column]
                
                # 시도 이름 표시
                plt.annotate(
                    text=name,
                    xy=(centroid.x, centroid.y),
                    xytext=(0, 20),  # 텍스트 위치를 위로 20포인트 이동
                    textcoords='offset points',
                    horizontalalignment='center',
                    fontsize=8
                )
                
                # 중심점 저장
                centroids[name] = (centroid.x, centroid.y)
                
                # 중심점 표시
                if show_centroids:
                    plt.plot(centroid.x, centroid.y, 'ro', markersize=5, alpha=0.7)
                    # 중심점 좌표 출력
                    print(f"{name} 중심점 좌표: ({centroid.x:.2f}, {centroid.y:.2f})")
            
            # 배경지도 추가
            ctx.add_basemap(
                self.ax,
                source=ctx.providers.CartoDB.Positron,
                zoom=7
            )
            
            # 축 제거
            self.ax.set_axis_off()
            
            # 제목 추가
            plt.title('대한민국 행정구역도 및 중심점', fontsize=15, pad=20)
            
            # 범례 추가
            if show_centroids:
                plt.plot([], [], 'ro', label='시도 중심점', markersize=5)
                plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
            
            # 지도 저장
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight', 
                       pad_inches=0.1)
            
            print(f"지도가 '{save_path}'로 저장되었습니다.")
            return centroids
            
        except Exception as e:
            print(f"지도 생성 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_province_centroids(self):
        """각 시도의 중심점 좌표 반환"""
        if self.map_data is None:
            print("지도 데이터가 로드되지 않았습니다.")
            return None
        
        name_column = 'SIG_KOR_NM' if 'SIG_KOR_NM' in self.map_data.columns else 'CTP_KOR_NM'
        
        centroids = {}
        for idx, row in self.map_data.iterrows():
            centroid = row.geometry.centroid
            centroids[row[name_column]] = {
                'x': centroid.x,
                'y': centroid.y
            }
        return centroids

    def add_transmission_lines(self, excel_path, sheet_name='GIS', save_path='korea_map_with_lines.png'):
        """엑셀 데이터를 기반으로 송전선 추가"""
        try:
            # 엑셀에서 연결 정보 읽기 (GIS 시트)
            connections_df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0)
            print("\nGIS 시트에서 읽은 연결 정보:")
            print(connections_df)
            
            # 이미 그려진 지도가 없다면 새로 그리기
            if self.ax is None:
                self.plot_korea_map(save_path=save_path)
            
            # 각 행정구역의 중심점 좌표 얻기
            centroids = {}
            for idx, row in self.map_data.iterrows():
                name = row['SIDO_NM']
                centroid = row.geometry.centroid
                centroids[name] = (centroid.x, centroid.y)
            
            # 연결선 그리기
            for i, region1 in enumerate(connections_df.index):
                for j, region2 in enumerate(connections_df.columns):
                    if i < j:  # 중복 방지
                        try:
                            num_lines = float(connections_df.loc[region1, region2])
                            if pd.notna(num_lines) and num_lines > 0:
                                # 중심점 좌표 가져오기
                                x1, y1 = centroids[region1]
                                x2, y2 = centroids[region2]
                                
                                # 송전선 수에 따라 선 스타일 변경
                                line_width = num_lines * 0.5  # 선 굵기
                                alpha = min(0.8, 0.3 + num_lines * 0.1)  # 투명도
                                
                                # 송전선 그리기
                                self.ax.plot([x1, x2], [y1, y2], 
                                           'b-',  # 파란색 실선
                                           linewidth=line_width,
                                           alpha=alpha,
                                           zorder=1)  # 지도 위에 그리기
                                
                                # 송전선 수 표시
                                mid_x = (x1 + x2) / 2
                                mid_y = (y1 + y2) / 2
                                plt.annotate(str(int(num_lines)),
                                           xy=(mid_x, mid_y),
                                           xytext=(5, 5),
                                           textcoords='offset points',
                                           fontsize=8,
                                           bbox=dict(facecolor='white', 
                                                   edgecolor='none',
                                                   alpha=0.7))
                        except KeyError as e:
                            print(f"경고: {region1}와 {region2} 사이의 연결 정보를 찾을 수 없습니다. {e}")
                        except ValueError as e:
                            print(f"경고: {region1}와 {region2} 사이의 연결 값이 올바르지 않습니다. {e}")
            
            # 범례 추가
            plt.plot([], [], 'b-', label='송전선', linewidth=1)
            plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
            
            # 지도 저장
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       pad_inches=0.1)
            
            print(f"송전선이 추가된 지도가 '{save_path}'로 저장되었습니다.")
            return True
            
        except Exception as e:
            print(f"송전선 추가 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return False

    def calculate_distances(self, save_path='distance.xlsx'):
        """행정구역 간 거리 계산 및 엑셀 파일로 저장"""
        try:
            if self.map_data is None:
                print("지도 데이터가 로드되지 않았습니다.")
                return False
            
            # 행정구역 이름과 중심점 좌표 가져오기
            regions = []
            centroids = {}
            for idx, row in self.map_data.iterrows():
                name = row['SIDO_NM']
                centroid = row.geometry.centroid
                regions.append(name)
                centroids[name] = (centroid.x, centroid.y)
            
            # 거리 행렬 생성
            distances = pd.DataFrame(index=regions, columns=regions)
            
            # 각 행정구역 쌍에 대해 거리 계산
            for i, region1 in enumerate(regions):
                for j, region2 in enumerate(regions):
                    if i == j:
                        distances.loc[region1, region2] = 0
                    else:
                        x1, y1 = centroids[region1]
                        x2, y2 = centroids[region2]
                        # 유클리드 거리 계산 (미터 단위)
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        # 킬로미터로 변환하고 소수점 둘째자리까지 반올림
                        distance_km = round(distance/1000, 2)
                        distances.loc[region1, region2] = distance_km
            
            # 엑셀 파일로 저장
            distances.to_excel(save_path)
            print(f"\n행정구역 간 거리가 '{save_path}'로 저장되었습니다.")
            
            # 거리 정보 출력
            print("\n행정구역 간 거리 (km):")
            print(distances)
            
            return distances
            
        except Exception as e:
            print(f"거리 계산 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return None

def main():
    """테스트 실행용 메인 함수"""
    visualizer = KoreaMapVisualizer()
    if visualizer.load_map_data():
        visualizer.plot_korea_map()
        # 송전선 정보 추가
        visualizer.add_transmission_lines('input_data.xlsx')
        # 거리 계산 및 저장
        visualizer.calculate_distances()

if __name__ == "__main__":
    main() 