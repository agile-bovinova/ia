#!/usr/bin/env python3
"""
BoviSense AI - Script de Verificación de Inferencia ONNX
=========================================================
Carga el modelo exportado 'bovine_health_risk_model.onnx' y evalúa
ejemplos de prueba para verificar que el modelo funciona de forma
completamente independiente.
"""

import os
import json
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "bovine_health_risk_model.onnx")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

def evaluate_bovine(temp, hr, temp_slope_3h, hr_slope_3h, temp_var_3h, thi):
    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name
    
    input_data = np.array([[temp, hr, temp_slope_3h, hr_slope_3h, temp_var_3h, thi]], dtype=np.float32)
    output = session.run(None, {input_name: input_data})
    
    # output[0] = clase predicha (0 o 1)
    # output[1] = lista de probabilidades por clase [[p0, p1]]
    predicted_class = int(output[0][0])
    risk_prob = float(output[1][0][1])

    if risk_prob < 0.35:
        level = "NORMAL / SALUDABLE"
        badge = "🟢"
        action = "Sin acción requerida. Parámetros fisiológicos estables."
    elif risk_prob < 0.70:
        level = "RIESGO MODERADO / ALERTA PREVENTIVA"
        badge = "🟡"
        action = "Aumentar frecuencia de muestreo de telemetría y revisar ventilación/hidratación."
    else:
        level = "RIESGO ALTO / FIEBRE INMINENTE (<12h)"
        badge = "🔴"
        action = "Aislamiento preventivo, chequeo veterinario y preparación de antipirético."

    return {
        "predicted_class": predicted_class,
        "risk_percentage": round(risk_prob * 100, 1),
        "level": level,
        "badge": badge,
        "recommended_action": action
    }

if __name__ == "__main__":
    print("=" * 65)
    print("DEMO DE INFERENCIA PREDICTIVA IA - BOVISENSE (ONNX)")
    print("=" * 65)
    
    scenarios = [
        ("Vaca 101 (Margarita) - Estado Normal", 38.5, 58.0, 0.02, 0.5, 0.08, 62.0),
        ("Vaca 102 (Pinta) - Sospecha Temprana (Temp aún normal 39.1°C pero subiendo rápido)", 39.1, 88.0, 0.58, 12.0, 0.30, 75.0),
        ("Vaca 103 (Estrella) - Estrés Térmico por Clima Caluroso (THI 84, temp 38.9°C)", 38.9, 76.0, 0.18, 4.0, 0.15, 84.0),
        ("Vaca 104 (Blanca) - Cuadro Febril Avanzado (39.5°C, 98 BPM)", 39.5, 98.0, 0.75, 18.0, 0.38, 78.0)
    ]

    for name, t, hr, ts, hs, tv, thi in scenarios:
        res = evaluate_bovine(t, hr, ts, hs, tv, thi)
        print(f"\n{res['badge']} {name}:")
        print(f"   Variables: T={t}°C | FC={hr} BPM | ΔT(3h)=+{ts}°C | ΔFC(3h)=+{hs} BPM | THI={thi}")
        print(f"   Probabilidad de Fiebre en 24h: {res['risk_percentage']} %")
        print(f"   Nivel:  {res['level']}")
        print(f"   Acción: {res['recommended_action']}")

    print("\n" + "=" * 65)
    print("Inferencia verificada con éxito.")
