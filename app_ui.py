# app_ui.py
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import math
import os
from typing import Literal, Dict

# --- 1. 계산 로직 클래스 ---


class ConductanceCalculator:
    """
    Pipe, Elbow, Reducer의 컨덕턴스를 머신러닝 모델을 이용해 계산하는 클래스.
    """

    def __init__(self, model_dir: str = "models"):
        self.models = self._load_all_models(model_dir)

    def _load_all_models(self, model_dir: str) -> Dict:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"모델 디렉토리 '{model_dir}'를 찾을 수 없습니다. "
                "models 폴더 안에 pipe, elbow, reducer 폴더가 있는지 확인해주세요."
            )
        model_dict = {}
        components = ["pipe", "elbow", "reducer"]
        regimes = ["점성", "천이"]
        for comp in components:
            model_dict[comp] = {}
            for regime in regimes:
                mpath = os.path.join(model_dir, comp, f"model_{regime}.pkl")
                ppath = os.path.join(model_dir, comp, f"poly_{regime}.pkl")
                model_dict[comp][regime] = {
                    "model": joblib.load(mpath),
                    "poly": joblib.load(ppath)
                }
        return model_dict

    def _calculate_knudsen_regime(self, pressure_torr: float, diameter_cm: float) -> Literal["점성", "천이"]:
        k, d_air, T_K = 1.38e-23, 3.7e-10, 293.0
        P_Pa = pressure_torr * 133.322
        lam = (k * T_K) / (math.sqrt(2) * math.pi * d_air**2 * P_Pa)
        Kn = lam / (diameter_cm / 100.0)
        return "점성" if Kn < 0.01 else "천이"

    def _predict(self, component: str, pressure_torr: float, features: list) -> float:
        d0 = features[0]  # Reducer는 D1, 나머지는 Diameter 기준
        regime = self._calculate_knudsen_regime(pressure_torr, d0)

        if regime not in self.models[component]:
            raise ValueError(f"'{regime}' 영역에 대한 '{component}' 모델이 없습니다.")

        model_info = self.models[component][regime]
        model, poly = model_info["model"], model_info["poly"]

        all_features = np.array([features + [pressure_torr]], dtype=np.float32)
        X_log = np.log1p(all_features)
        X_poly = poly.transform(X_log)
        y_log_pred = model.predict(X_poly)

        return float(np.expm1(y_log_pred)[0])

    def calculate(self, component: str, params: Dict[str, float]) -> float:
        """모든 부품 계산을 처리하는 통합 메소드"""
        pressure = params.get("Pressure_Torr", 0.1)

        if component == 'pipe':
            features = [params['Diameter_cm'], params['Length_cm']]
        elif component == 'reducer':
            features = [params['D1_cm'], params['D2_cm'], params['Length_cm']]
        elif component == 'elbow':
            features = [params['Diameter_cm'], params['BendAngle_deg']]
        else:
            raise ValueError(f"알 수 없는 부품: {component}")

        return self._predict(component, pressure, features)


# --- 2. GUI 애플리케이션 클래스 ---
class App(tk.Tk):
    def __init__(self, calculator: ConductanceCalculator):
        super().__init__()
        self.calculator = calculator

        self.title("Conductance Calculator")
        self.geometry("450x400")

        self.param_entries = {}
        self.current_component = tk.StringVar(value="pipe")

        self._create_widgets()
        self._update_ui_for_component()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Component Selection ---
        ttk.Label(main_frame, text="부품 선택:", font=(
            "", 10, "bold")).pack(fill=tk.X)
        component_menu = ttk.Combobox(main_frame, textvariable=self.current_component,
                                      values=["pipe", "reducer", "elbow"], state="readonly")
        component_menu.pack(fill=tk.X, pady=(2, 10))
        component_menu.bind("<<ComboboxSelected>>",
                            lambda e: self._update_ui_for_component())

        # --- Dynamic Parameter Frame ---
        self.param_frame = ttk.Frame(main_frame)
        self.param_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Calculation Button ---
        calc_button = ttk.Button(
            main_frame, text="계산하기", command=self.perform_calculation)
        calc_button.pack(fill=tk.X, ipady=5, pady=10)

        # --- Result Display ---
        self.result_label = ttk.Label(
            main_frame, text="결과: -", font=("", 12, "bold"), foreground="blue")
        self.result_label.pack(fill=tk.X, pady=5)

    def _update_ui_for_component(self):
        # Clear previous entries
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        component = self.current_component.get()

        # Define params for each component
        params = {
            "pipe": ["Diameter (cm)", "Length (cm)", "Pressure (Torr)"],
            "reducer": ["Diameter 1 (cm)", "Diameter 2 (cm)", "Length (cm)", "Pressure (Torr)"],
            "elbow": ["Diameter (cm)", "Angle (deg)", "Pressure (Torr)"]
        }

        # Create new entries
        for param_text in params.get(component, []):
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=4)

            label = ttk.Label(frame, text=f"{param_text}:", width=18)
            label.pack(side=tk.LEFT)

            entry = ttk.Entry(frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.param_entries[param_text] = entry

            # Set default pressure
            if "Pressure" in param_text:
                entry.insert(0, "0.1")

    def perform_calculation(self):
        try:
            component = self.current_component.get()

            # Convert UI text to calculation parameter keys
            key_map = {
                "Diameter (cm)": "Diameter_cm", "Length (cm)": "Length_cm",
                "Pressure (Torr)": "Pressure_Torr", "Diameter 1 (cm)": "D1_cm",
                "Diameter 2 (cm)": "D2_cm", "Angle (deg)": "BendAngle_deg"
            }

            params_for_calc = {}
            for ui_text, entry in self.param_entries.items():
                value = entry.get()
                if not value:
                    raise ValueError(f"'{ui_text}' 값을 입력해주세요.")
                params_for_calc[key_map[ui_text]] = float(value)

            # Perform calculation
            conductance = self.calculator.calculate(component, params_for_calc)

            # Display result
            self.result_label.config(text=f"결과: {conductance:.4f} L/s")

        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
        except Exception as e:
            messagebox.showerror("계산 오류", f"계산 중 오류가 발생했습니다:\n{e}")


# --- 3. 애플리케이션 실행 ---
if __name__ == "__main__":
    try:
        # 계산기 객체 생성 (모델 로딩)
        calculator_instance = ConductanceCalculator()

        # UI 애플리케이션 실행
        app = App(calculator_instance)
        app.mainloop()

    except FileNotFoundError as e:
        messagebox.showerror(
            "파일 오류", f"필수 파일을 찾을 수 없습니다:\n{e}\n\n'models' 폴더가 올바르게 구성되었는지 확인하세요.")
    except Exception as e:
        messagebox.showerror("실행 오류", f"프로그램 실행 중 심각한 오류가 발생했습니다:\n{e}")
