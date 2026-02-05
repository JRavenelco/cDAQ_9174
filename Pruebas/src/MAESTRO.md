# 📚 DOCUMENTO MAESTRO - Tesis Doctoral UAQ
## Modelado de Histéresis en CNC mediante KAN-PINN
**Jesús | Doctorado UAQ | Diciembre 2025**

---

## 🎯 CONCLUSIÓN PRINCIPAL
> **Sistema LINEAL (α ≈ 0.81 - 1.0) - No requiere compensación de histéresis**

---

## 📊 RESULTADOS CLAVE

| Modelo | α Corte | R² | Conclusión |
|--------|---------|-----|------------|
| BW Simple | 0.811 | 0.71 | 81% lineal |
| BW Viscoso | 0.427 | 0.74 | Mejor R² |
| Duhem | 0.485 | 0.70 | Polinomial |
| BW-ENV | 0.811 | 0.71 | Con envolvente |
| **KAN-PINN** | **0.999** | 0.70 | **100% lineal** |

---

## 🔧 PARÁMETROS FRF IDENTIFICADOS
| Param | Valor | Unidad |
|-------|-------|--------|
| m | 0.107 | kg |
| k | 1424 | N/m |
| c | 3.73 | Ns/m |
| fn | ~25 | Hz |

---

## 📁 ESTRUCTURA DEL PROYECTO

```
caracterizacion_fuerza/
├── 32 scripts .py (análisis)
├── 10 archivos .txt (datos)
├── graficas_bouc_wen/ (27 gráficas)
└── presentacion_bouc_wen/ (5 LaTeX)

manim_videos/
├── presentacion_semestral_v3.py (principal)
├── video_*.py (6 videos)
├── *.html (presentaciones web)
└── ANEXO_Tecnicas_Experimentales.md
```

---

## 🧮 MODELOS DE HISTÉRESIS

### 1. BW Simple
`F = α·k·a + (1-α)·k·z`

### 2. BW Viscoso  
`F = α·k·a + (1-α)·k·z + c·da/dt`

### 3. Duhem
`F = a·z + b·z² + c·z³ + d·dz/dt`

### 4. BW-ENV
`F = α·k·E + (1-α)·k·z` (E = envolvente Hilbert)

### 5. KAN-PINN (Completo)
`F = m·a + k·x + c·v + α·k·E + (1-α)·k·z`

---

## 🎬 PRESENTACIÓN MANIM

### Secciones (presentacion_semestral_v3.py)
1. Portada/Contenido
2. Introducción/Antecedentes
3. Problema/Justificación/Hipótesis
4. Teoría: Histéresis, Bouc-Wen, CBR
5. Arquitectura KAN-PINN
6. Sistema Físico
7. Metodología FRF (9 slides)
8. Validación Bouc-Wen
9. Resultados
10. Conclusiones

---

## 🔬 TÉCNICAS EXPERIMENTALES

### Envolvente (Hilbert)
```python
from scipy.signal import hilbert
env = np.abs(hilbert(señal))
```
**Impacto**: R² de 0.01 → 0.70

### Hipótesis Two-Loop
| Loop | Tipo | Frecuencia | Modelo |
|------|------|------------|--------|
| 1 | Fricción | <10 Hz | KAN |
| 2 | Corte | 100-500 Hz | PINN |

### Arquitectura Híbrida
```
Stage 1: Air-cut → KAN → F_fric (CONGELADO)
Stage 2: Corte → PINN → F_corte = F_med - F_KAN
```

---

## 📝 PRÓXIMOS PASOS

- [ ] Graficar lazos F-x de air-cutting
- [ ] Implementar Integral Loss PINN
- [ ] Cortes en Al 6061-T6
- [ ] Separación KAN-PINN Stage 2
- [ ] Integrar CBR con aritmética modular

---

## 📚 REFERENCIAS COMPLETAS

### Papers Fundamentales (2020-2025)

1. **Cornelius et al. (2024)** - Process damping identification using Bayesian learning  
   https://www.osti.gov/pages/servlets/purl/2329603

2. **Wang, Rabczuk, Liu (2025)** - KAN para fricción en manipuladores robóticos  
   https://arxiv.org/html/2511.10079v1

3. **Li et al. (2023)** - PINN para fricción y stick-slip  
   https://arxiv.org/abs/2303.02542

4. **Liu et al. (2024)** - Kolmogorov-Arnold Networks (KAN original)  
   https://arxiv.org/abs/2404.19756

5. **Raissi et al. (2019)** - Physics-Informed Neural Networks (PINN original)  
   J. Comp. Phys. https://doi.org/10.1016/j.jcp.2018.10.045

6. **Altintas & Ber (1994)** - Process Damping fundacional  
   CIRP Annals

7. **Jung et al. (2023)** - Integral Loss PINN para datos ruidosos  
   https://www.researchgate.net/publication/381869386

### Referencias Adicionales

8. **Identificación de Process Damping en Fresado**  
   https://www.researchgate.net/publication/273147741

9. **Análisis de Process Damping**  
   https://www.researchgate.net/publication/309656504

10. **Modelado Analítico de Process Damping**  
   https://www.researchgate.net/publication/332054313

11. **KAN para Aerodinámica**  
   https://www.researchgate.net/publication/391997282

12. **Fricción en Harmonic Drives**  
   https://arxiv.org/html/2410.12685v1

13. **Grey Box para Fricción en CNC**  
   http://jmacheng.not.pl/pdf-186269-107654

14. **Dinámica No-Suave con Histéresis**  
   https://www.researchgate.net/publication/222513784

15. **Reconstrucción de Presión desde PIV**  
   https://www.researchgate.net/publication/359840956

---

## 🗂️ ARCHIVOS PRINCIPALES

| Archivo | Propósito |
|---------|-----------|
| `test_modelos_envolvente.py` | Comparación 5 modelos |
| `test_kan_pinn_simple.py` | KAN-PINN individual |
| `presentacion_semestral_v3.py` | Manim principal |
| `analisis_modelos_histeresis.tex` | LaTeX final |
| `README.md` | Documentación técnica |

---

*Generado: Diciembre 2025*
