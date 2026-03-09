# TradingBot — Estado actual del proyecto

## Resumen ejecutivo

El proyecto ha evolucionado desde una fase de investigación de señal hacia una fase de **arquitectura pre-live**.

La conclusión principal hasta este punto es:

- el sistema válido y congelado trabaja en **BTC 1H**;
- el **champion oficial** ya no es solo una señal, sino una arquitectura compuesta por:
  - **primary layer** (motor principal de alpha),
  - **secondary layer** (capa secundaria para densificar actividad),
  - **sizer v2** (gestión de tamaño diferenciada por fuente y régimen),
  - **reason codes / heartbeat / replay** para observabilidad completa;
- el motor unificado oficial actual es:
  - `scripts/30_champion_prelive_engine.py`

La rama 15m quedó como experimento y no pasó la validación con contexto real. La línea viva del proyecto es **1H**.

---

## Estado validado del sistema

### Champion oficial congelado

El script oficial actual es:

- `scripts/30_champion_prelive_engine.py`

Integra:

- **primary**: core alpha congelado
- **secondary**: capa secundaria subordinada
- **sizer_v2**: tamaño distinto para primary y secondary
- **reason codes**: explicación de por qué opera o no opera
- **replay**: validación acelerada barra a barra
- **logs persistentes**: trazas, trades simulados, equity, summary

### Resultado de verificación del unificado

El script unificado reproduce correctamente la versión buena ya validada.

#### Últimas 1000 barras

- `total_return ≈ +0.4928%`
- `n_trades = 14`
- `primary_trades = 7`
- `secondary_trades = 7`
- `avg_primary_size ≈ 0.5843`
- `avg_secondary_size = 0.15`

#### 2024-10-01 a 2024-12-31

- `total_return ≈ +5.1685%`
- `n_trades = 29`
- `primary_trades = 9`
- `secondary_trades = 20`
- `avg_primary_size ≈ 0.5947`
- `avg_secondary_size = 0.15`

Interpretación:

- `primary` sigue siendo el motor principal;
- `secondary` añade actividad sin dominar el riesgo;
- el sizing v2 deja a la secondary claramente subordinada.

---

## Aprendizajes clave acumulados

### 1. El edge real está en 1H, no en 15m

Se probó una rama 15m.

Conclusión:

- con hack de contexto parecía prometedora;
- con contexto real (`BTC + ETH + SOL 15m`) perdió edge;
- se descarta como línea principal.

### 2. El gran salto vino del candidate engine

Se confirmó que abrir el embudo inicial de forma controlada y dejar la dureza a:

- meta-model,
- EV,
- ranking,

mejoró de forma material:

- trades,
- Sharpe,
- retorno,
- estabilidad operativa.

### 3. El champion puro es selectivo y de frecuencia moderada

No es un sistema hiperactivo.

Opera cuando detecta estructura clara. Eso obliga a trabajar mucho la **observabilidad** para evitar volver al escenario antiguo de "parece roto porque no imprime nada".

### 4. La secondary layer sí aporta

La capa secundaria:

- aumenta frecuencia,
- reduce ventanas muertas,
- añade retorno agregado,
- pero gana bastante menos por trade que la primary.

Por eso no debe tener el mismo tamaño ni el mismo peso económico.

### 5. El sizing v2 mejora la arquitectura

El sizing v2 no cambia la señal; mejora la monetización del alpha:

- primary con tamaño grande,
- secondary con tamaño pequeño,
- jerarquía clara de capital.

---

## Estructura actual recomendada del repositorio

### Scripts vivos en raíz

Estos son los scripts que deben permanecer como línea principal del proyecto:

- `scripts/01_download_data.py`
- `scripts/02_build_triple_barrier_dataset.py`
- `scripts/03_train_meta_label_model_v3.py`
- `scripts/07_debug_trade_funnel.py`
- `scripts/08_backtest_meta_model_v6.py`
- `scripts/09_trade_attribution.py`
- `scripts/10_walkforward_meta_model_v6.py`
- `scripts/17_walkforward_frozen_champions.py`
- `scripts/30_champion_prelive_engine.py`

### Scripts de research / archivo

Estos scripts ya cumplieron su función investigadora y deberían quedar archivados en:

- `scripts/archive_research/`

Incluyen:

- `11_walkforward_dual_long_research.py`
- `12_walkforward_dual_activity_research.py`
- `13_walkforward_dual_refine.py`
- `14_promotion_validation.py`
- `15_walkforward_dual_short_refine.py`
- `16_walkforward_dual_long_conflict_refine.py`
- `18_walkforward_15m_experiment.py`
- `19_paper_trace_simulator.py`
- `20_live_replay_engine.py`
- `21_rolling_replay_scanner.py`
- `22_secondary_layer_shadow_replay.py`
- `23_secondary_layer_rolling_scanner.py`
- `24_trade_source_attribution.py`
- `25_secondary_layer_sizer_v2.py`

---

## Archivos / artefactos relevantes generados por la fase research

### Reports / summaries útiles

- `artifacts/reports/walkforward_frozen_champions/`
- `artifacts/reports/rolling_replay_scanner/rolling_summary.csv`
- `artifacts/reports/secondary_layer_rolling/secondary_rolling_summary.csv`
- `artifacts/reports/champion_prelive_engine/summary.json`
- `artifacts/reports/champion_prelive_engine/decision_trace.csv`
- `artifacts/reports/champion_prelive_engine/simulated_trades.csv`
- `artifacts/reports/champion_prelive_engine/equity_curve.csv`

### Modelos / scalers esperados

- `artifacts/models/meta_model_long_v3.pt`
- `artifacts/models/meta_model_short_v3.pt`
- `artifacts/scalers/meta_model_long_v3_scaler.json`
- `artifacts/scalers/meta_model_short_v3_scaler.json`

### Dataset principal vivo

- `data/processed/dataset_btc_triple_barrier_1h.csv`

---

## Qué debe verificar el próximo chat al revisar Git

Cuando se revise si lo subido a Git es correcto, hay que comprobar como mínimo:

### 1. Scripts oficiales presentes

Deben existir en `scripts/`:

- `01_download_data.py`
- `02_build_triple_barrier_dataset.py`
- `03_train_meta_label_model_v3.py`
- `07_debug_trade_funnel.py`
- `08_backtest_meta_model_v6.py`
- `09_trade_attribution.py`
- `10_walkforward_meta_model_v6.py`
- `17_walkforward_frozen_champions.py`
- `30_champion_prelive_engine.py`

### 2. Research archivado

Deben existir en `scripts/archive_research/` los scripts experimentales 11–25 ya cerrados.

### 3. No usar como operativos los scripts antiguos

Los scripts operativos ya no deben ser:

- `20_live_replay_engine.py`
- `22_secondary_layer_shadow_replay.py`
- `25_secondary_layer_sizer_v2.py`

Esos fueron absorbidos por `30_champion_prelive_engine.py`.

### 4. Dataset principal correcto

- `data/processed/dataset_btc_triple_barrier_1h.csv`

### 5. Modelos y scalers presentes

- `artifacts/models/meta_model_long_v3.pt`
- `artifacts/models/meta_model_short_v3.pt`
- `artifacts/scalers/meta_model_long_v3_scaler.json`
- `artifacts/scalers/meta_model_short_v3_scaler.json`

---

## Esquema corto de carpetas/archivos importantes

```text
tradingbot/
├── data/
│   ├── raw/
│   └── processed/
│       └── dataset_btc_triple_barrier_1h.csv
├── artifacts/
│   ├── models/
│   │   ├── meta_model_long_v3.pt
│   │   └── meta_model_short_v3.pt
│   ├── scalers/
│   │   ├── meta_model_long_v3_scaler.json
│   │   └── meta_model_short_v3_scaler.json
│   └── reports/
│       ├── walkforward_frozen_champions/
│       ├── rolling_replay_scanner/
│       ├── secondary_layer_rolling/
│       └── champion_prelive_engine/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_triple_barrier_dataset.py
│   ├── 03_train_meta_label_model_v3.py
│   ├── 07_debug_trade_funnel.py
│   ├── 08_backtest_meta_model_v6.py
│   ├── 09_trade_attribution.py
│   ├── 10_walkforward_meta_model_v6.py
│   ├── 17_walkforward_frozen_champions.py
│   ├── 30_champion_prelive_engine.py
│   └── archive_research/
│       ├── 11_walkforward_dual_long_research.py
│       ├── 12_walkforward_dual_activity_research.py
│       ├── 13_walkforward_dual_refine.py
│       ├── 14_promotion_validation.py
│       ├── 15_walkforward_dual_short_refine.py
│       ├── 16_walkforward_dual_long_conflict_refine.py
│       ├── 18_walkforward_15m_experiment.py
│       ├── 19_paper_trace_simulator.py
│       ├── 20_live_replay_engine.py
│       ├── 21_rolling_replay_scanner.py
│       ├── 22_secondary_layer_shadow_replay.py
│       ├── 23_secondary_layer_rolling_scanner.py
│       ├── 24_trade_source_attribution.py
│       └── 25_secondary_layer_sizer_v2.py
└── src/
    ├── features/
    ├── strategy/
    └── structure/
```

---

## Qué se ha hecho hasta ahora

### Research / validación completada

- limpieza de la arquitectura viva del bot
- validación del pipeline principal 1H
- corrección de bugs históricos serios
- mejora del candidate engine
- validación por backtest
- validación por walk-forward
- rolling scanner del champion
- replay engine con heartbeat y reason codes
- secondary layer en shadow mode
- attribution por fuente (`primary` vs `secondary`)
- `sizer_v2` diferenciando tamaño entre capas
- unificación del champion en `30_champion_prelive_engine.py`

### Conclusión de research

La arquitectura válida actual es:

- **primary layer** = core alpha
- **secondary layer** = densidad operativa
- **sizer_v2** = monetización disciplinada del alpha

---

## Próximo paso acordado

El siguiente chat debe continuar con:

# **paper trading / pre-live operativo**

La prioridad ya no es mejorar señal, sino convertir el motor actual en un modo paper operativo real.

### Objetivo del siguiente paso

Usar `scripts/30_champion_prelive_engine.py` como base para:

- procesar velas cerradas nuevas
- ejecutar lógica champion + secondary + sizing v2
- imprimir heartbeat por vela
- dejar trazas completas persistentes
- mantener estado entre ejecuciones
- validar comportamiento paper casi real

### Qué se deberá verificar en el próximo chat

1. que los archivos subidos a Git son los correctos;
2. que el repositorio quedó limpio;
3. que `30_champion_prelive_engine.py` es el punto de entrada oficial;
4. que la fase siguiente ya es paper/pre-live y no más research de señal.

---

## Nota final

El proyecto ya no está en fase de “buscar una idea bonita”. Está en fase de **convertir un alpha validado en una máquina ejecutable y observable**.

Ese cambio de fase es el hito principal alcanzado hasta ahora.
