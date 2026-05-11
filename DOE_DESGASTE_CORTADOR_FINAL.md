# DOE Final — Detección de Desgaste en Fresado CNC

## Configuración del sistema
- **Cortador**: AMSA3100HS18E, d=10mm, Z=2 insertos, carburo WC-Co
- **Material**: Aluminio
- **RPM**: 200–1000 (rango controlado para generar señal de histéresis)
- **Pitch tornillo mesa**: p=5 mm/rev

## Niveles de desgaste (abrasión manual + rugosímetro)
| Nivel | Cortadores | Método | Ra objetivo (μm) |
|-------|-----------|--------|-----------------|
| Nuevo | C1, C2 | Sin tratar | Ra < 0.4 |
| Moderado | C3, C4 | Papel diamante 400 grit ~2 min | Ra 1.0–2.5 |
| Severo | C5, C6 | Papel diamante 120 grit ~5 min | Ra 3.5–6.0 |

## Tabla de corridas (Z=2, p=5mm)
| Corrida | Cortador | N (rpm) | a (mm/d) | v_mesa (mm/min) | ω_tor (rpm) |
|---------|----------|---------|----------|-----------------|-------------|
| 1  | C1 | 200  | 0.05 | 20  | 4  |
| 2  | C1 | 200  | 0.10 | 40  | 8  |
| 3  | C1 | 600  | 0.05 | 60  | 12 |
| 4  | C1 | 600  | 0.10 | 120 | 24 |
| 5  | C1 | 1000 | 0.05 | 100 | 20 |
| 6  | C1 | 1000 | 0.10 | 200 | 40 |
| 7–12  | C2 | igual | igual | igual | igual |
| 13–18 | C3 (moderado) | igual | igual | igual | igual |
| 19–24 | C4 (moderado) | igual | igual | igual | igual |
| 25–30 | C5 (severo)   | igual | igual | igual | igual |
| 31–36 | C6 (severo)   | igual | igual | igual | igual |

**Total: 36 corridas** | Tiempo estimado: 2 días de laboratorio

## Velocidades de corte resultantes
| N (rpm) | Vc (m/min) | Régimen |
|---------|-----------|---------|
| 200  | 6.3  | Muy lento — alta fuerza, buen índice H |
| 600  | 18.8 | Medio — balance fuerza/vibración |
| 1000 | 31.4 | Rápido para este rango — referencia |

## Variables de respuesta
1. Índice H = E_loop / (W_loop × ptp(F))
2. r(F, envolvente aceleración)
3. Parámetros Bouc-Wen: α, A, β, γ
4. Error predicción CBR

## Hipótesis testable
H aumenta ≥30% de cortador nuevo (Ra<0.4) a severo (Ra>3.5), p<0.05
