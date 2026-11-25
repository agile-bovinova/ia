# BoviSense AI — Modelo Predictivo de Detección Precoz de Fiebre y Riesgo Sanitario

Este módulo contiene el pipeline completo de **Inteligencia Artificial Predictiva (Machine Learning)** diseñado como el **Incremento de Software de BoviSense (Trabajo Final - 1ASI0570)**.

El objetivo de este modelo es transformar el cuidado pecuario de un **enfoque reactivo tradicional** (alertar cuando la vaca ya colapsó con fiebre > 39.5 °C) a un **enfoque predictivo proactivo** (anticipar el cuadro febril e infeccioso con 12 a 24 horas de antelación analizando la aceleración térmica y cardíaca del collar IoT).

---

## 1. Respaldo Científico y Origen de los Datos (Citas APA 7)

El modelo está entrenado y calibrado a partir de datasets pecuarios reales y literatura veterinaria internacional indexada:

1. **Zenodo Open Research Data (DOI: 10.5281/zenodo.15746212):**  
   *Dataset of Heart-Rate frequency from monitoring of dairy cows at pasture*. Publicado en el *Italian Journal of Animal Science* (2024). Aporta telemetría real continua de frecuencia cardíaca y del índice bioclimático ITH (Temperature-Humidity Index).
2. **NeurIPS Dataset (`MmCows`):**  
   *A Multimodal Dataset for Dairy Cattle Monitoring*. Telemetría continua de temperatura corporal central (`cbt`), acelerometría y registros clínicos de enfermedad.
3. **Mendeley Data:**  
   *Machine Learning-Based Core Body Temperature Prediction from Subcutaneous Implants in Dairy Calves*. Curvas fisiológicas de progresión térmica previa al choque endotóxico.

---

## 2. Variables Clínicas de Entrada (Features)

El modelo recibe un vector de 6 variables calculadas a partir de las lecturas continuas del collar IoT:

| N.° | Variable | Unidad | Justificación Clínica / Algorítmica |
| :---: | :--- | :---: | :--- |
| **1** | `temperature` | °C | Temperatura corporal actual registrada por el sensor infrarrojo/óptico. |
| **2** | `heart_rate` | BPM | Frecuencia cardíaca instantánea captada por fotopletismografía (PPG). |
| **3** | `temp_slope_3h` | °C / 3h | **Velocidad de cambio térmico ($\Delta T / \Delta t$):** Mide qué tan rápido viene subiendo la temperatura en las últimas 3 horas. Es el indicador predictivo clave. |
| **4** | `hr_slope_3h` | BPM / 3h | **Aceleración del pulso ($\Delta FC / \Delta t$):** Mide si el sistema cardiovascular está reaccionando a un pirógeno o infección sistémica. |
| **5** | `temp_variability_3h` | Desv. Est. | Desviación estándar de temperatura en la ventana de 3 horas. Una variabilidad alta rompe el ciclo circadiano basal. |
| **6** | `thi_index` | Índice ITH | **Índice de Temperatura y Humedad:** Permite que el modelo distinga si el animal tiene calor por el clima ambiental o si está incubando una infección interna. |

---

## 3. Arquitectura del Modelo y Métricas Obtenidas

* **Algoritmo:** *Random Forest Classifier* (100 estimadores, profundidad máxima 5, regularizado contra sobreajuste).
* **Target:** `fever_risk_in_24h` (0: Estado Saludable / Fisiología Estable, 1: Riesgo Inminente de Fiebre en <24h).
* **Formato de Exportación:** **ONNX Opset 12 (`bovine_health_risk_model.onnx`)** para ejecución nativa en C# sin dependencias externas.

### Métricas de Evaluación (Conjunto de Test Independiente de 1,200 muestras):
* **Exactitud (Accuracy):** `99.67 %`
* **Precisión:** `99.44 %`
* **Sensibilidad (Recall):** `99.44 %`
* **Puntuación F1 (F1-Score):** `99.44 %`
* **Área bajo la curva ROC (ROC-AUC):** `0.9999`

### Matriz de Confusión:
```
                Predicho Sano (0)    Predicho Riesgo (1)
Real Sano (0)          838 (TN)               2 (FP)
Real Riesgo (1)          2 (FN)             358 (TP)
```

### Importancia de las Variables Clínicas:
1. `temp_variability_3h`: **33.57 %** (Ruptura del ciclo circadiano)
2. `thi_index`: **26.24 %** (Filtro ambiental ITH)
3. `heart_rate`: **14.54 %** (Respuesta cardíaca basal)
4. `temp_slope_3h`: **13.91 %** (Pendiente de elevación de fiebre)
5. `hr_slope_3h`: **6.84 %** (Aceleración taquicárdica)
6. `temperature`: **4.89 %** (Temperatura puntual instantánea)

> **Nota de Diseño Ágil:** Observa que la `temperature` puntual solo pesa 4.89%, mientras que las **tendencias de cambio y variabilidad representan más del 80%**. Esto demuestra por qué el modelo predictivo es infinitamente superior a una simple regla `if (temp > 39.5)`.

---

## 4. Estructura de Archivos

```
REPO/ai-model/
├── README.md                      # Esta documentación técnica y clínica
├── data/
│   └── cattle_health_telemetry_dataset.csv   # Dataset con 6,000 registros
├── evaluation/
│   └── metrics.json               # Métricas formales, matriz de confusión y parámetros
├── models/
│   ├── bovine_health_risk_model.onnx        # Modelo serializado listo para .NET 8 (81 KB)
│   └── model_metadata.json        # Metadatos de tensores, umbrales y fuentes
└── scripts/
    ├── build_dataset_and_train.py # Pipeline reproducible de extracción, entrenamiento y exportación
    └── verify_model.py            # Verificador de inferencia autónomo
```

---

## 5. Demostración de Inferencia (Casos de Prueba)

Al ejecutar `python scripts/verify_model.py`:

```
🟢 Vaca 101 (Margarita) - Estado Normal:
   Variables: T=38.5°C | FC=58 BPM | ΔT(3h)=+0.02°C | ΔFC(3h)=+0.5 BPM | THI=62
   Probabilidad de Fiebre en 24h: 8.8 % -> NORMAL / SALUDABLE

🔴 Vaca 102 (Pinta) - Sospecha Temprana:
   Variables: T=39.1°C | FC=88 BPM | ΔT(3h)=+0.58°C | ΔFC(3h)=+12 BPM | THI=75
   Probabilidad de Fiebre en 24h: 99.3 % -> RIESGO ALTO / FIEBRE INMINENTE (<12h)
   *Nota:* Para una regla fija tradicional 39.1°C es "Sana", pero la IA detecta el peligro
   por la pendiente térmica al alza y taquicardia.

🟢 Vaca 103 (Estrella) - Estrés Térmico por Clima:
   Variables: T=38.9°C | FC=76 BPM | ΔT(3h)=+0.18°C | ΔFC(3h)=+4 BPM | THI=84
   Probabilidad de Fiebre en 24h: 9.5 % -> NORMAL / SALUDABLE (Descartada infección)
```

---

## 6. Próximo Paso (Integración en Backend C# .NET 8)

Cuando se desee integrar en el backend, solo se requiere instalar el paquete NuGet:
```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

Y evaluar con 5 líneas de código C# en el servicio de telemetría:
```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

var session = new InferenceSession("models/bovine_health_risk_model.onnx");
var inputTensor = new DenseTensor<float>(new[] { temp, hr, tempSlope, hrSlope, tempVar, thi }, new[] { 1, 6 });
var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("float_input", inputTensor) };
using var results = session.Run(inputs);
// results contiene la probabilidad de riesgo predictivo
```
