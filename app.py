# app.py
import logging
import math
import re

import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Literal, Optional

# ----- 로거 설정 -----
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger("conductance")

# ----- FastAPI 인스턴스 -----
app = FastAPI(title="Conductance Explorer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- 지원 컴포넌트 & regime ---
COMPONENTS = ["pipe", "elbow", "reducer"]
REGIMES    = ["점성", "천이"]

# --- 컴포넌트별 매개변수 설정 ---
COMPONENT_PARAMS_CONFIG = {
    "pipe": {
        "feature_keys": ("Diameter_cm", "Length_cm"),
        "D0_key": "Diameter_cm",
    },
    "elbow": {
        "feature_keys": ("Diameter_cm", "BendAngle_deg"),
        "D0_key": "Diameter_cm",
    },
    "reducer": {
        "feature_keys": ("D1_cm", "D2_cm", "Length_cm"),
        "D0_key": "D1_cm",
    },
}

# --- 모델 & poly 로드 ---
model_dict = {}
for comp in COMPONENTS:
    model_dict[comp] = {}
    for regime in REGIMES:
        mpath = f"models/{comp}/model_{regime}.pkl"
        ppath = f"models/{comp}/poly_{regime}.pkl"
        model_dict[comp][regime] = (
            joblib.load(mpath),
            joblib.load(ppath)
        )
        logger.debug(f"Loaded {mpath}, {ppath}")

# --- 요청 스키마 ---
class PredictRequest(BaseModel):
    component: Literal["pipe", "elbow", "reducer"]
    Pressure_Torr: float
    Diameter_cm: Optional[float] = None
    Length_cm: Optional[float] = None
    BendAngle_deg: Optional[float] = None
    D1_cm: Optional[float] = None
    D2_cm: Optional[float] = None

class RangeRequest(BaseModel):
    component: Literal["pipe", "elbow", "reducer"]
    Pressure_Torr: float = None  # dummy, not used
    Diameter_cm: Optional[float] = None
    Length_cm: Optional[float] = None
    BendAngle_deg: Optional[float] = None
    D1_cm: Optional[float] = None
    D2_cm: Optional[float] = None
    start_torr: float = 0.09
    end_torr: float = 0.12
    n_points: int = 369

# --- Knudsen regime 계산 ---
def calculate_knudsen_number(P_torr: float, D_cm: float, T_K: float = 293) -> str:
    k = 1.38e-23
    d_air = 3.7e-10
    P_Pa = P_torr * 133.322
    lam = (k * T_K) / (math.sqrt(2) * math.pi * d_air**2 * P_Pa)
    Kn = lam / (D_cm / 100.0)
    regime = "점성" if Kn < 0.01 else "천이" if Kn < 0.1 else "분자"
    logger.debug(f"regime P={P_torr}, D={D_cm} → {regime}")
    return regime

# --- 단일 예측 재사용 함수 ---
def _predict_single(comp: str, P: float, **kwargs) -> float:
    if comp not in COMPONENT_PARAMS_CONFIG:
        # Pydantic Literal에 의해 이미 처리되어야 하지만, 안전장치로 추가
        raise HTTPException(status_code=400, detail=f"알 수 없는 컴포넌트 유형: {comp}")

    config = COMPONENT_PARAMS_CONFIG[comp]
    
    param_values = []
    for key in config["feature_keys"]:
        val = kwargs.get(key)
        if val is None:
            raise HTTPException(status_code=400, detail=f"컴포넌트 '{comp}'에 매개변수 '{key}'가 필요하지만 제공되지 않았거나 null입니다.")
        param_values.append(val)

    D0_val = kwargs.get(config["D0_key"])
    if D0_val is None:
        raise HTTPException(status_code=400, detail=f"컴포넌트 '{comp}'의 Knudsen 번호 계산에 매개변수 '{config['D0_key']}'가 필요하지만 제공되지 않았거나 null입니다.")

    feats = param_values + [P]
    
    regime = calculate_knudsen_number(P, D0_val)
    if regime not in model_dict[comp]:
        raise HTTPException(400, f"{comp}:{regime} 모델 없음")

    model, poly = model_dict[comp][regime]
    try:
        X_log = np.log1p(np.array([feats], dtype=np.float32))
    except TypeError as e:
        logger.error(f"컴포넌트 {comp}, 압력 P={P}, 피처 feats={feats}에 대한 피처 준비 중 타입 오류 발생: {e}")
        raise HTTPException(status_code=400, detail=f"컴포넌트 {comp}의 피처에 유효하지 않은 데이터 타입이 있습니다.")
        
    y_log = model.predict(poly.transform(X_log))[0]
    return float(np.expm1(y_log))

# --- 엔드포인트 정의 ---
@app.get(
    "/",
    summary="HTML 인터페이스 제공",
    description="Conductance Explorer의 웹 UI를 반환합니다."
)
def index():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post(
    "/api/predict",
    summary="단일 컴포넌트 컨덕턴스 예측",
    description="주어진 컴포넌트 유형, 압력 및 기하학적 매개변수에 대해 단일 컨덕턴스 값을 예측합니다."
)
def predict(req: PredictRequest):
    result = _predict_single(
        req.component,
        req.Pressure_Torr,
        Diameter_cm   = req.Diameter_cm,
        Length_cm     = req.Length_cm,
        BendAngle_deg = req.BendAngle_deg,
        D1_cm         = req.D1_cm,
        D2_cm         = req.D2_cm
    )
    logger.debug(f"Predict: component={req.component}, P={req.Pressure_Torr}, result={result}")
    return JSONResponse({"conductance": result})

@app.post(
    "/api/range",
    summary="압력 범위에 대한 컨덕턴스 예측",
    description="지정된 압력 범위에 대해 컴포넌트의 컨덕턴스 값을 여러 지점에서 계산합니다."
)
def predict_range(req: RangeRequest):
    comp = req.component
    
    predict_kwargs = {
        "Diameter_cm": req.Diameter_cm,
        "Length_cm": req.Length_cm,
        "BendAngle_deg": req.BendAngle_deg,
        "D1_cm": req.D1_cm,
        "D2_cm": req.D2_cm,
    }

    ps = np.linspace(req.start_torr, req.end_torr, req.n_points, dtype=np.float32)
    pressures, conductances = [], []

    for P_val in ps:
        try:
            conductance = _predict_single(comp, float(P_val), **predict_kwargs)
            pressures.append(float(P_val))
            conductances.append(conductance)
        except HTTPException as e:
            # "모델 없음" 오류인 경우 해당 압력 지점을 건너뜀
            if "모델 없음" in str(e.detail):
                # D0 값을 가져와서 로그를 남길 수 있지만, 여기서는 단순화
                # D0_key = COMPONENT_PARAMS_CONFIG[comp]['D0_key']
                # current_D0 = predict_kwargs.get(D0_key)
                # current_regime = calculate_knudsen_number(P_val, current_D0) if current_D0 is not None else "알 수 없음"
                logger.warning(f"압력 P={P_val}, 컴포넌트={comp} 지점 건너뜀 (레짐 모델 없음): {e.detail}")
            else:
                # 다른 HTTPException (예: RangeRequest의 필수 매개변수 누락)은 전체 요청 실패로 처리
                logger.error(f"압력 범위에 대한 컨덕턴스 계산 실패 (설정/매개변수 오류): {e.detail}")
                raise e 
            
    return JSONResponse({"pressures": pressures, "conductances": conductances})

@app.post(
    "/api/series",
    summary="CSV 파일 기반 직렬 연결 컨덕턴스 계산",
    description="""CSV 파일을 업로드하여 여러 컴포넌트(ReducerX 접두사 그룹)의 직렬 연결된 총 컨덕턴스를 계산합니다.
    
각 ReducerX 그룹은 '前 배관', '前 Elbow', 'Core Reducer' 세그먼트로 구성될 수 있습니다.
CSV 파일 형식:
- 첫 번째 열: 파라미터 이름 (예: "Chamber_ID", "Reducer1 前 배관 직경[inch]", "Reducer1 前 배관 총장[mm]" 등)
- 이후 각 열: 하나의 Chamber 또는 시스템에 대한 파라미터 값들.
- "Chamber_ID" 또는 "Chamber" 키를 사용하여 결과를 그룹화합니다.
"""
)
async def series_conductance(
    file: UploadFile = File(..., description="컨덕턴스 계산을 위한 파라미터가 포함된 CSV 파일"),
    Pressure_Torr: float = Form(..., description="모든 컴포넌트에 적용될 기준 압력 (Torr 단위)")
):
    # 1) CSV 읽기 (UTF-8 → CP949 fallback)
    try:
        raw = pd.read_csv(file.file, header=None, dtype=str)
    except UnicodeDecodeError:
        file.file.seek(0)
        raw = pd.read_csv(file.file, header=None, dtype=str, encoding='cp949')

    # 2) 빈 문자열(공백) → NaN
    raw = raw.replace(r'^\s*$', np.nan, regex=True)

    # 3) 파라미터 이름 추출 (따옴표·줄바꿈 제거 → 첫 줄만)
    params = (
        raw.iloc[:, 0]  # CSV의 첫 번째 열을 파라미터 이름으로 사용
           .astype(str)
           .str.replace(r'^"|"$', '', regex=True)
           .str.split('\n').str[0]
           .str.strip()
    )
    logger.debug(f"Params: {params.tolist()}")

    ncols = raw.shape[1]
    results = {}

    # 4) 각 열을 하나의 Chamber 블록으로 처리
    for j in range(1, ncols): # 데이터가 시작되는 열을 인덱스 1로 변경
        col = raw.iloc[:, j]
        non_null = col.dropna().shape[0]
        logger.debug(f"Column {j}: non-null rows={non_null}")
        if non_null < 3:
            logger.debug(f"Skipping column {j}: not enough data")
            continue

        # 5) 파라미터→값 매핑
        d = {}
        for i, val in enumerate(col.values):
            key = params.iloc[i]
            # "nan" 문자열이거나 "unnamed"으로 시작하는 키는 건너뜀
            if not key or key.lower() == "nan" or key.lower().startswith("unnamed"):
                continue
            if pd.isna(val):
                continue
            try:
                v = float(val)
            except:
                v = val
            d[key] = v
        logger.debug(f"Mapped dict for column {j}: {d}")

        # 6) Chamber ID
        chamber_id = d.get("Chamber_ID") or d.get("Chamber")
        if not chamber_id:
            logger.debug(f"No chamber_id in column {j}")
            continue

        # 7) ReducerX 접두사 탐색
        prefixes = {
            m.group(1)
            for k in d.keys()
            if (m := re.match(r"^(Reducer\d+)\s+前 배관 직경", k))
        }
        logger.debug(f"Prefixes in column {j}: {prefixes}")

        # 8) 각 prefix별 pipe, elbow, reducer 계산
        C_list = []
        for prefix in prefixes:
            logger.debug(f"Processing prefix {prefix} in column {j}")
            # 前 Pipe
            inch = d.get(f"{prefix} 前 배관 직경[inch]")
            mm   = d.get(f"{prefix} 前 배관 총장[mm]")
            if inch and mm:
                C_list.append(_predict_single(
                    "pipe", Pressure_Torr,
                    Diameter_cm = inch * 2.54,
                    Length_cm   = mm   / 10.0
                ))
            # 前 Elbow
            cnt     = d.get(f"{prefix} 前 Elbow 개수")
            diam_el = d.get(f"{prefix} 前 Elbow 직경")
            if cnt and diam_el:
                try: cnt_i = int(float(cnt))
                except: cnt_i = 0
                if cnt_i > 0:
                    angle_keys = [k for k in d if k.startswith(f"{prefix} 前 Elbow 각도")]
                    if len(angle_keys) == 1:
                        angles = [d[angle_keys[0]]] * cnt_i
                    else:
                        angles = [d[k] for k in angle_keys[:cnt_i] if k in d]
                    for ang in angles:
                        if pd.notna(ang):
                            C_list.append(_predict_single(
                                "elbow", Pressure_Torr,
                                Diameter_cm   = diam_el * 2.54,
                                BendAngle_deg = ang
                            ))
            # Core Reducer
            d1 = d.get(f"{prefix} 입구 직경[IN]")
            d2 = d.get(f"{prefix} 출구 직경[IN]")
            l  = d.get(f"{prefix} 길이[mm]")
            if d1 and d2 and l:
                C_list.append(_predict_single(
                    "reducer", Pressure_Torr,
                    D1_cm     = d1 * 2.54,
                    D2_cm     = d2 * 2.54,
                    Length_cm = l  / 10.0
                ))

        if not C_list:
            logger.warning(f"No segments found for chamber {chamber_id} in column {j}")
            continue

        # 9) Series 연결
        C_total = 1.0 / sum(1.0 / c for c in C_list)
        results[str(chamber_id)] = C_total

    if not results:
        raise HTTPException(400, "유효한 Chamber 데이터가 없습니다.")

    logger.debug(f"Final conductance results by chamber: {results}")
    return JSONResponse({"conductances": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
