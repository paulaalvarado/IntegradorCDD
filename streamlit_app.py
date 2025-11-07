
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

# Opcional: acelera Altair en Cloud
try:
    import vegafusion
    import vl_convert
    alt.data_transformers.enable("vegafusion")
except Exception:
    pass

st.set_page_config(page_title="Exploración & Modelo – Proyecto Integrador", layout="wide")

DATA_PATH = os.getenv("DATA_PATH", "data/df_limpio.csv")
MODEL_PATHS = ["models/model.joblib", "models/model.pkl", "models/random_forest_optimizado.pkl"]

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_resource(show_spinner=False)
def try_load_model(paths=MODEL_PATHS):
    import joblib, pickle, traceback
    last_err = None
    for p in paths:
        if os.path.exists(p):
            # intento con joblib
            try:
                m = joblib.load(p)
                return m
            except Exception as e:
                last_err = f"{p} (joblib): {e}"
                # intento con pickle crudo
                try:
                    with open(p, "rb") as f:
                        m = pickle.load(f)
                        return m
                except Exception as e2:
                    last_err = f"{last_err} | {p} (pickle): {e2}"
    if last_err:
        st.warning(f"No pude cargar el modelo: {last_err}")
    return None

# --- DEBUG rápido del modelo (ponelo DEBAJO de modelo = try_load_model())
with st.expander("🔧 Debug modelo", expanded=False):
    st.write({"cwd": os.getcwd()})
    st.write({"models_dir_exists": os.path.exists("models")})
    st.write({"model_joblib_exists": os.path.exists("models/model.joblib")})
    if os.path.exists("models"):
        st.write({"models_dir_list": os.listdir("models")})
    # detectar si es un puntero de Git LFS
    try:
        with open("models/model.joblib", "rb") as f:
            head = f.read(200)
        is_lfs = (b"git-lfs" in head.lower()) or (b"github.com/spec/v1" in head.lower())
        st.write({"posible_archivo_LFS": bool(is_lfs)})
    except Exception as _:
        pass

df = load_data(DATA_PATH)
modelo = try_load_model()

num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
cat_cols = [c for c in df.columns if c not in num_cols]

st.sidebar.title("Proyecto Integrador")
page = st.sidebar.radio("Ir a…", ["Explorar datos", "Modelo", "Predecir"])
st.sidebar.markdown("---")
st.sidebar.caption("Usa data/df_limpio.csv + models/model.joblib")

if page == "Explorar datos":
    st.header("Exploración interactiva")
    with st.expander("Info del dataset"):
        st.write({"filas": df.shape[0], "columnas": df.shape[1]})
        st.dataframe(df.head(10))

    st.subheader("1) Dispersión (Scatter)")
    c1, c2, c3 = st.columns(3)
    xvar = c1.selectbox("X (numérico)", options=num_cols, index=0 if num_cols else None)
    yvar = c2.selectbox("Y (numérico)", options=num_cols, index=1 if len(num_cols) > 1 else 0)
    color_by = c3.selectbox("Color (opcional)", options=["(ninguno)"] + cat_cols, index=0)
    if xvar and yvar:
        brush = alt.selection_interval()
        enc_color = alt.Color(color_by + ":N") if color_by != "(ninguno)" else alt.value("steelblue")
        sc = (
            alt.Chart(df)
            .mark_circle(size=90)
            .encode(
                x=alt.X(f"{xvar}:Q"), y=alt.Y(f"{yvar}:Q"),
                color=alt.condition(brush, enc_color, alt.value("lightgray")),
                tooltip=[xvar, yvar] + ([color_by] if color_by != "(ninguno)" else []),
            )
            .add_params(brush).properties(height=380).interactive()
        )
        st.altair_chart(sc, use_container_width=True)

    st.subheader("2) Boxplot")
    if cat_cols and num_cols:
        catv = st.selectbox("Categoría (X)", options=cat_cols)
        numv = st.selectbox("Num (Y)", options=num_cols, index=0, key="box-y")
        box = (
            alt.Chart(df).mark_boxplot(extent="min-max")
            .encode(x=alt.X(f"{catv}:N"), y=alt.Y(f"{numv}:Q"), tooltip=[catv, numv])
            .properties(height=350)
        )
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("Necesitás al menos una categórica y una numérica.")

    st.subheader("3) Serie temporal (si hay año/fecha)")
    time_cols = [c for c in df.columns if str(c).lower() in ["año", "anio", "year", "fecha", "date"]]
    if time_cols and num_cols:
        tcol = st.selectbox("Columna temporal", options=time_cols)
        yts = st.selectbox("Num (Y)", options=num_cols, index=0, key="ts-y")
        tsdf = df.copy()
        if str(tcol).lower() in ["año", "anio", "year"]:
            xenc = alt.X(f"{tcol}:O")
        else:
            tsdf[tcol] = pd.to_datetime(tsdf[tcol], errors="coerce")
            xenc = alt.X(f"{tcol}:T")
        line = (
            alt.Chart(tsdf.dropna(subset=[tcol])).mark_line(point=True)
            .encode(x=xenc, y=alt.Y(f"{yts}:Q"), tooltip=[tcol, yts])
            .properties(height=350).interactive()
        )
        st.altair_chart(line, use_container_width=True)
    else:
        st.caption("No se detectó columna temporal clara.")

    # 4) Serie temporal por país (uno o varios)
    st.subheader("4) Natalidad por país (serie temporal)")

    # detectar columnas de tiempo y de país
    time_cols_2 = [c for c in df.columns if str(c).lower() in ["año", "anio", "year", "fecha", "date"]]
    country_cols = [c for c in df.columns if str(c).lower() in ["pais", "país", "country"]]

    if time_cols_2 and country_cols:
        tcol = st.selectbox("Columna temporal", options=time_cols_2, key="pais_ts_tcol")
        pcol = st.selectbox("Columna de país", options=country_cols, key="pais_ts_pcol")
        yvar = st.selectbox("Variable Y (numérica)", options=[c for c in df.select_dtypes('number').columns],
                            index=0, key="pais_ts_y")

        # lista de países
        opciones = sorted([str(v) for v in df[pcol].dropna().unique().tolist()])
        sel = st.multiselect("Elegí país(es)", opciones, default=opciones[:1])

        if sel:
            dff = df[df[pcol].isin(sel)].copy()
            # convertir fecha si no es año puro
            if str(tcol).lower() in ["fecha", "date"]:
                dff[tcol] = pd.to_datetime(dff[tcol], errors="coerce")

            # línea con color por país
            line = (
                alt.Chart(dff)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{tcol}:{'T' if str(tcol).lower() in ['fecha','date'] else 'O'}", title=tcol),
                    y=alt.Y(f"{yvar}:Q", title=yvar),
                    color=alt.Color(f"{pcol}:N", title="País"),
                    tooltip=[pcol, tcol, yvar],
                )
                .properties(height=380, title="Serie por país (color)")
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)

            # small multiples opcional
            if st.checkbox("Ver como 'small multiples' (un panel por país)"):
                facet = (
                    alt.Chart(dff)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{tcol}:{'T' if str(tcol).lower() in ['fecha','date'] else 'O'}", title=tcol),
                        y=alt.Y(f"{yvar}:Q", title=yvar),
                        facet=alt.Facet(f"{pcol}:N", columns=4),
                        tooltip=[pcol, tcol, yvar],
                    )
                    .properties(height=180)
                )
                st.altair_chart(facet, use_container_width=True)
        else:
            st.info("Seleccioná al menos un país para graficar.")
    else:
        st.info("No encontré columnas de tiempo (Año/fecha) y/o país (Pais/País/Country).")


elif page == "Modelo":
    st.header("Modelo (pipeline scikit-learn)")
    if modelo is None:
        st.warning("No se encontró un archivo de modelo en `models/`.")
    else:
        st.success("Modelo cargado correctamente.")

else:
    st.header("Predicción")
    if modelo is None:
        st.warning("Subí tu pipeline exportado a `models/` para habilitar la predicción.")
    else:
        tabs = st.tabs(["Formulario", "CSV por lote"])
        with tabs[0]:
            st.subheader("Formulario")
            inputs = {}
            for c in num_cols:
                default = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
                inputs[c] = st.number_input(c, value=default)
            for c in cat_cols:
                opts = sorted([str(v) for v in df[c].dropna().unique().tolist()]) or ["(vacío)"]
                inputs[c] = st.selectbox(c, options=opts)
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
                        st.error(f"Faltan columnas: {missing}")
                    else:
                        yhat = modelo.predict(dfinf[list(num_cols) + list(cat_cols)])
                        out = dfinf.copy(); out["prediccion"] = yhat
                        st.write(out.head())
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Descargar resultados", csv, file_name="predicciones.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"No se pudo procesar el archivo: {e}")

st.markdown("---")
st.caption("© Proyecto Integrador • Streamlit • Altair • scikit-learn")
