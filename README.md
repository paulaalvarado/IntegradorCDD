# Proyecto Integrador – App Streamlit (Plantilla rápida)

Este repo contiene lo mínimo para cumplir la **4ª entrega** (visualización + app Streamlit).
Funciona con tu `data/df_limpio.csv` y un pipeline exportado en `models/model.joblib` (o `.pkl`).

## Estructura
```
data/df_limpio.csv
models/model.joblib        # exportado desde tu notebook (Pipeline sklearn)
streamlit_app.py
requirements.txt
README.md
```

## Paso a paso (SIN VSCode)
1. **Descargá** estos archivos desde ChatGPT: `streamlit_app.py`, `requirements.txt`, `README.md`.
2. Creá un repo en GitHub → `Add file` → `Upload files` y subí esos 3 archivos.
3. Subí **tu dataset** a `data/df_limpio.csv` (creá la carpeta `data/` en el repo).
   - En tu notebook, exportá el DataFrame limpio a CSV:
     ```python
     df.to_csv('df_limpio.csv', index=False)
     ```
     Descargalo y **renombralo** si hace falta.
4. Exportá tu **pipeline sklearn** entrenado a `models/model.joblib`:
   ```python
   import os, joblib
   os.makedirs('models', exist_ok=True)
   joblib.dump(pipe, 'models/model.joblib')
   ```
   (El `pipe` debe incluir el preprocesamiento que usaste al entrenar).
5. **Publicar en Streamlit Cloud**: Nueva app → conectá tu repo → `streamlit_app.py` como archivo principal.

### Notas
- La app es **agnóstica al dataset**: elegís X/Y/color para gráficos. Si tenés columna de tiempo (`Año`, `year`, `fecha`), habilita serie temporal.
- Para predicción, el modelo debe ser un **Pipeline** que transforme categóricas/numéricas igual que en tu notebook.
- Si en Cloud hay problemas de render, `vegafusion` ya está agregado en `requirements.txt`.
