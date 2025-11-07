
    # streamlit_app.py
    # -------------------------------------------------------------
    # App base para la 4ª entrega (Visualización e Integración)
    # - Carga dataset procesado desde data/df_limpio.csv
    # - 2–3 visualizaciones interactivas (Altair)
    # - Predicción con modelo entrenado (pipeline .joblib/.pkl)
    # - Upload de CSV para inferencia por lote
    # -------------------------------------------------------------

    import os
    import numpy as np
    import pandas as pd
    import altair as alt
    import streamlit as st

    try:
        import vegafusion  # acelera Altair en Streamlit Cloud
        import vl_convert  # exportaciones
        alt.data_transformers.enable("vegafusion")
    except Exception:
        pass

    st.set_page_config(page_title="Exploración & Modelo – Proyecto Integrador", layout="wide")

    DATA_PATH = os.getenv("DATA_PATH", "data/df_limpio.csv")
    MODEL_PATHS = [
        "models/model.joblib",
        "models/model.pkl",
        "models/model_optimizado.pkl",
        "models/random_forest_optimizado.pkl",
    ]

    @st.cache_data(show_spinner=False)
    def load_data(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # limpieza mínima
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @st.cache_resource(show_spinner=False)
    def try_load_model(paths=MODEL_PATHS):
        import joblib, pickle
        for p in paths:
            if os.path.exists(p):
                try:
                    return joblib.load(p)
                except Exception:
                    try:
                        with open(p, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        continue
        return None

    df = load_data(DATA_PATH)
    modelo = try_load_model()

    # Inferimos tipos
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    cat_cols = [c for c in df.columns if c not in num_cols]

    st.sidebar.title("Proyecto Integrador")
    page = st.sidebar.radio("Ir a…", ["Explorar datos", "Modelo", "Predecir"])

    st.sidebar.markdown("---")
    st.sidebar.caption("Usá este repo como base: data/df_limpio.csv + models/model.joblib")

    # -------------------- EXPLORAR --------------------
    if page == "Explorar datos":
        st.header("Exploración interactiva")
        st.write("Elegí variables y crea vistas interactivas con Altair.")

        with st.expander("Info del dataset"):
            st.write({"filas": df.shape[0], "columnas": df.shape[1]})
            st.dataframe(df.head(10))

        # Permitir elegir variables
        st.subheader("1) Dispersión (Scatter)")
        c1, c2, c3 = st.columns(3)
        with c1:
            xvar = st.selectbox("Eje X (numérico)", options=num_cols, index=0 if num_cols else None)
        with c2:
            yvar = st.selectbox("Eje Y (numérico)", options=num_cols, index=1 if len(num_cols) > 1 else 0)
        with c3:
            color_by = st.selectbox("Color por (categórica/opcional)", options=["(ninguno)"] + cat_cols, index=0)
        if xvar and yvar:
            brush = alt.selection_interval()
            enc_color = alt.Color(color_by + ":N") if color_by != "(ninguno)" else alt.value("steelblue")
            sc = (
                alt.Chart(df)
                .mark_circle(size=90)
                .encode(
                    x=alt.X(f"{xvar}:Q", title=xvar),
                    y=alt.Y(f"{yvar}:Q", title=yvar),
                    color=alt.condition(brush, enc_color, alt.value("lightgray")),
                    tooltip=[xvar, yvar] + ([color_by] if color_by != "(ninguno)" else []),
                )
                .add_params(brush)
                .properties(height=380)
                .interactive()
            )
            st.altair_chart(sc, use_container_width=True)

        st.subheader("2) Distribución / Boxplot")
        if cat_cols and num_cols:
            b1, b2 = st.columns(2)
            with b1:
                catv = st.selectbox("Categoría (X)", options=cat_cols)
            with b2:
                numv = st.selectbox("Variable numérica (Y)", options=num_cols, index=0)
            box = (
                alt.Chart(df)
                .mark_boxplot(extent="min-max")
                .encode(x=alt.X(f"{catv}:N"), y=alt.Y(f"{numv}:Q"), tooltip=[catv, numv])
                .properties(height=350)
            )
            st.altair_chart(box, use_container_width=True)
        else:
            st.info("Para boxplot necesitás al menos una categórica y una numérica.")

        # Time series si hay una columna de año/fecha
        time_candidates = [c for c in df.columns if str(c).lower() in ["año", "anio", "year", "fecha", "date"]]
        st.subheader("3) Serie temporal (si corresponde)")
        if time_candidates and num_cols:
            tcol = st.selectbox("Columna temporal", options=time_candidates)
            yts = st.selectbox("Variable numérica (Y)", options=num_cols, index=0, key="ts-y")
            tsdf = df.copy()
            # Intentamos convertir fecha/año
            if str(tcol).lower() in ["año", "anio", "year"]:
                xenc = alt.X(f"{tcol}:O", title=tcol)
            else:
                tsdf[tcol] = pd.to_datetime(tsdf[tcol], errors="coerce")
                xenc = alt.X(f"{tcol}:T", title=tcol)
            line = (
                alt.Chart(tsdf.dropna(subset=[tcol]))
                .mark_line(point=True)
                .encode(x=xenc, y=alt.Y(f"{yts}:Q"), tooltip=[tcol, yts])
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.caption("No se detectó columna temporal clara.")

    # -------------------- MODELO --------------------
    elif page == "Modelo":
        st.header("Modelo (pipeline scikit-learn)")
        if modelo is None:
            st.warning("No se encontró un archivo de modelo en `models`.")
            with st.expander("Cómo exporto mi pipeline + modelo (ejemplo)"):
                st.code(
                    """

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor  # o Classifier según tu caso

# Ajustá esto según tu notebook
TARGET = "TU_OBJETIVO"
num_cols = [c for c in df.select_dtypes('number').columns if c != TARGET]
cat_cols = [c for c in df.columns if c not in num_cols + [TARGET]]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([
    ("pre", pre),
    ("model", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))
])

X = df[num_cols + cat_cols]
y = df[TARGET]
pipe.fit(X, y)

import os
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/model.joblib")
print("✓ Modelo exportado en models/model.joblib")
                    """,
                    language="python",
                )
        else:
            st.success("Modelo cargado correctamente.")
            with st.expander("Evaluación rápida (holdout 80/20)"):
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

                # Permitimos al usuario elegir objetivo
                ycol = st.selectbox("Elegí columna objetivo para una evaluación rápida", options=df.columns.tolist(), index=0)
                X = df.drop(columns=[ycol])
                y = df[ycol]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                try:
                    y_pred = modelo.predict(X_test)
                    # elegimos métrica según tipo
                    if np.issubdtype(y.dtype, np.number):
                        rmse = mean_squared_error(y_test, y_pred, squared=False)
                        mae = mean_absolute_error(y_test, y_pred)
                        st.write({"RMSE": round(rmse, 4), "MAE": round(mae, 4)})
                    else:
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        st.write({"Accuracy": round(acc, 4), "F1": round(f1, 4)})
                except Exception as e:
                    st.error(f"No se pudo evaluar con este objetivo/columnas: {e}")

    # -------------------- PREDECIR --------------------
    else:
        st.header("Predicción")
        if modelo is None:
            st.warning("Subí tu pipeline exportado a la carpeta `models/` para habilitar la predicción.")
        else:
            tabs = st.tabs(["Formulario", "CSV por lote"])

            with tabs[0]:
                st.subheader("Formulario")
                inputs = {}
                # numéricas
                for c in num_cols:
                    default = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
                    val = st.number_input(c, value=default)
                    inputs[c] = val
                # categóricas
                for c in cat_cols:
                    opts = sorted([str(v) for v in df[c].dropna().unique().tolist()])
                    if not opts:
                        opts = ["(vacío)"]
                    val = st.selectbox(c, options=opts)
                    inputs[c] = val

                if st.button("Predecir"):
                    Xnew = pd.DataFrame([inputs])[list(num_cols) + list(cat_cols)]
                    try:
                        yhat = modelo.predict(Xnew)
                        st.success(f"Predicción: {yhat[0]}")
                    except Exception as e:
                        st.error(f"No se pudo predecir: {e}")

            with tabs[1]:
                st.subheader("CSV por lote")
                up = st.file_uploader("Subí un CSV con las columnas de entrada", type=["csv"])
                if up is not None:
                    try:
                        dfinf = pd.read_csv(up)
                        missing = [c for c in (list(num_cols) + list(cat_cols)) if c not in dfinf.columns]
                        if missing:
                            st.error(f"Faltan columnas requeridas para el modelo: {missing}")
                        else:
                            yhat = modelo.predict(dfinf[list(num_cols) + list(cat_cols)])
                            out = dfinf.copy()
                            out["prediccion"] = yhat
                            st.write(out.head())
                            csv = out.to_csv(index=False).encode("utf-8")
                            st.download_button("Descargar resultados", csv, file_name="predicciones.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"No se pudo procesar el archivo: {e}")

    st.markdown("---")
    st.caption("© Proyecto Integrador • Streamlit • Altair • scikit-learn")
