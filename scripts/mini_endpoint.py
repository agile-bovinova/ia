#!/usr/bin/env python3
"""
BoviSense AI — Mini Servidor / Endpoint de Inferencia Predictiva
================================================================
Este script implementa un endpoint HTTP local y ejecutor CLI sin
dependencias externas (usa únicamente la biblioteca estándar de Python 3).
Permite probar la IA predictiva enviando y recibiendo JSONs reales sin
necesidad de hardware IoT ni servidores complejos.

Uso:
  1. Modo prueba automática (evalúa casos predefinidos):
     python3 mini_endpoint.py --test

  2. Modo evaluación manual por CLI (evalúa un JSON puntual):
     python3 mini_endpoint.py --eval '{"temperature": 39.2, "heart_rate": 88, "temp_slope_3h": 0.55, "hr_slope_3h": 12, "temp_variability_3h": 0.28, "thi_index": 76}'

  3. Modo Servidor HTTP (Mini Endpoint local en puerto 5005):
     python3 mini_endpoint.py --serve 5005
     -> Escucha POST en http://localhost:5005/predict
"""

import sys
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_PATH = os.path.join(BASE_DIR, "models", "forest_ensemble.json")

# Cargar el ensamble de árboles en memoria una sola vez
with open(ENSEMBLE_PATH, "r", encoding="utf-8") as f:
    ENSEMBLE = json.load(f)

FEATURES = ENSEMBLE["features"]
TREES = ENSEMBLE["trees"]

def predict_single_tree(tree, x):
    node = 0
    while True:
        feat_idx = tree["feature"][node]
        if feat_idx == -2 or tree["children_left"][node] == -1:
            # Nodo hoja: normalizar conteos a probabilidades
            val = tree["value"][node]
            total = sum(val)
            return val[1] / total if total > 0 else 0.0
        
        threshold = tree["threshold"][node]
        if x[feat_idx] <= threshold:
            node = tree["children_left"][node]
        else:
            node = tree["children_right"][node]

def predict_risk(temperature, heart_rate, temp_slope_3h=0.0, hr_slope_3h=0.0, temp_variability_3h=0.10, thi_index=72.0, **kwargs):
    temp = float(temperature)
    hr = float(heart_rate)
    temp_slope_3h = float(temp_slope_3h)
    hr_slope_3h = float(hr_slope_3h)
    temp_var_3h = float(temp_variability_3h)
    thi = float(thi_index)
    
    x = [temp, hr, temp_slope_3h, hr_slope_3h, temp_var_3h, thi]
    
    # Promediar la predicción de los 100 estimadores
    votes = [predict_single_tree(t, x) for t in TREES]
    risk_prob = sum(votes) / len(votes)
    risk_percent = round(risk_prob * 100, 1)

    if risk_prob < 0.35:
        level = "Bajo"
        badge = "🟢"
        predicted_class = 0
        factor = "Parámetros biológicos estables dentro del ciclo circadiano."
        action = "Sin acción requerida. El animal se encuentra en estado saludable."
        horizon = "Estable (Sin riesgo en 24h)"
    elif risk_prob < 0.70:
        level = "Moderado"
        badge = "🟡"
        predicted_class = 0
        factor = "Leve elevación térmica o estrés ambiental (THI moderado). Monitorear tendencia."
        action = "Aumentar frecuencia de lecturas del collar y verificar sombra/hidratación en corral."
        horizon = "Precaución (Observación próximas 12-24h)"
    else:
        level = "Alto (Inminente)"
        badge = "🔴"
        predicted_class = 1
        factor = f"Aceleración térmica (+{temp_slope_3h:.2f} °C en 3h) con taquicardia ({hr:.0f} BPM). Probable incubación infecciosa."
        action = "Aislamiento preventivo inmediato, chequeo veterinario clínico y preparación de antipirético."
        horizon = "Fiebre inminente en 12 a 24 horas"

    return {
        "risk_percentage": risk_percent,
        "risk_level": level,
        "badge": badge,
        "predicted_class": predicted_class,
        "time_horizon": horizon,
        "primary_factor": factor,
        "recommended_action": action,
        "input_summary": {
            "temperature": temp,
            "heart_rate": hr,
            "temp_slope_3h": temp_slope_3h,
            "hr_slope_3h": hr_slope_3h,
            "temp_variability_3h": temp_var_3h,
            "thi_index": thi
        }
    }


class PredictHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        info = {
            "service": "BoviSense Predictive Health AI Endpoint",
            "version": "1.0.0",
            "status": "online",
            "endpoint": "POST /predict",
            "expected_json": {
                "temperature": 39.15,
                "heart_rate": 86.0,
                "temp_slope_3h": 0.52,
                "hr_slope_3h": 12.0,
                "temp_variability_3h": 0.28,
                "thi_index": 76.5
            }
        }
        self.wfile.write(json.dumps(info, indent=2, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Ruta no encontrada. Usa POST /predict"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
            temp = float(payload.get("temperature", 38.5))
            hr = float(payload.get("heart_rate", 60.0))
            ts = float(payload.get("temp_slope_3h", 0.0))
            hs = float(payload.get("hr_slope_3h", 0.0))
            tv = float(payload.get("temp_variability_3h", 0.10))
            thi = float(payload.get("thi_index", 72.0))
            bovine_id = payload.get("bovine_id", None)

            res = predict_risk(temp, hr, ts, hs, tv, thi)
            if bovine_id is not None:
                res["bovine_id"] = bovine_id

            response_data = {
                "status": "success",
                "prediction": res
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, indent=2, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            err_resp = {"status": "error", "message": f"JSON inválido o campos faltantes: {str(e)}"}
            self.wfile.write(json.dumps(err_resp, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_tests():
    print("=" * 65)
    print("EJECUCIÓN DE PRUEBAS DE INFERENCIA PREDICTIVA (JSON EVALUATION)")
    print("=" * 65)
    
    test_cases = [
        {
            "nombre": "Vaca Sana (Margarita) - Límites normales",
            "input": {"temperature": 38.5, "heart_rate": 58.0, "temp_slope_3h": 0.02, "hr_slope_3h": 0.5, "temp_variability_3h": 0.08, "thi_index": 62.0}
        },
        {
            "nombre": "Vaca en Alerta Precoz (Pinta) - Temp 39.1°C pero alza térmica acelerada",
            "input": {"temperature": 39.1, "heart_rate": 88.0, "temp_slope_3h": 0.58, "hr_slope_3h": 12.0, "temp_variability_3h": 0.30, "thi_index": 75.0}
        },
        {
            "nombre": "Vaca con Estrés por Calor Ambiental (Estrella) - Sin infección bacteriana",
            "input": {"temperature": 38.9, "heart_rate": 76.0, "temp_slope_3h": 0.18, "hr_slope_3h": 4.0, "temp_variability_3h": 0.15, "thi_index": 84.0}
        }
    ]

    for tc in test_cases:
        res = predict_risk(**tc["input"])
        print(f"\n{res['badge']} {tc['nombre']}:")
        print(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("Pruebas completadas con éxito.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5005
        server = HTTPServer(("0.0.0.0", port), PredictHandler)
        print(f"BoviSense AI Endpoint escuchando en http://localhost:{port}/predict")
        print("Presiona Ctrl+C para detener el servidor.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")
    elif len(sys.argv) > 1 and sys.argv[1] == "--eval":
        payload = json.loads(sys.argv[2])
        res = predict_risk(**payload)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        run_tests()
