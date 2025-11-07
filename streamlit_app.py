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
    import joblib, pickle
    last_err = None
    for p in paths:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m
            except Exception as e:
                last_err = f"{p} (joblib): {e}"
                try:
                    with open(p, "rb") as f:
                        m = pickle.load(f)
                        return m
                except Exception as e2:
                    last_err = f"{last_err} | {p} (pickle): {e2}"
    if last_err:
        st.warning(f"No pude cargar el modelo: {last_err}")
    return None

with st.expander("Debug modelo", expanded=False):
    st.write({"cwd": os.getcwd()})
    st.write({"models_dir_exists": os.path.exists("models")})
    st.write({"model_joblib_exists": os.path.exists("models/model.joblib")})
    if os.path.exists("models"):
        st.write({"models_dir_list": os.listdir("models")})
    try:
        with open("models/model.joblib", "rb") as f:
            head = f.read(200)
        is_lfs = (b"git-lfs" in head.lower()) or (b"github.com/spec/v1" in head.lower())
        st.write({"posible_archivo_LFS": bool(is_lfs)})
    except Exception:
        pass

df = load_data(DATA_PATH)
modelo = try_load_model()

# ==== NORMALIZAR AÑO COMO ENTERO ====
YEAR_COLS = [c for c in df.columns if str(c).lower() in ["año", "anio", "year"]]
for c in YEAR_COLS:
    try:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    except Exception:
        pass

# ========== PASO 0) CONFIG SIN-LEAKAGE + HELPERS (A/B/C) ==========
TARGET = "Natalidad"

# Variables consecuencia del target -> nunca usarlas en inputs/escenarios
LEAKAGE = {
    "TasaFertilidad",
    "MortalidadInfantil",
    "MortalidadMenores5",
    "MortalidadNeonatal",
    "MortalidadMaterna",
    "PrevalenciaAnemiaEmbarazadas",
    "RatioDependenciaJovenes",
}

ALL_NUM = [c for c in df.select_dtypes("number").columns if c != TARGET]
ALL_CAT = [c for c in df.columns if c not in df.select_dtypes("number").columns]

NUM_ALLOWED = [c for c in ALL_NUM if c not in LEAKAGE]   # incluye 'Año'
CAT_ALLOWED = [c for c in ALL_CAT if c not in LEAKAGE]

from sklearn.linear_model import LinearRegression

def _trend_value(dfx, col, anio):
    """Proyecta col→año con regresión lineal; si no alcanza data, usa mediana país→global."""
    ycol = YEAR_COLS[0] if YEAR_COLS else "Año"
    if ycol not in dfx.columns or col not in dfx.columns:
        return float(df[col].median()) if col in df.columns else None
    dfx = dfx.dropna(subset=[col, ycol])
    if len(dfx) >= 3:
        X = dfx[ycol].astype(float).to_numpy().reshape(-1, 1)
        y = dfx[col].astype(float).to_numpy()
        lr = LinearRegression().fit(X, y)
        return float(lr.predict([[float(anio)]])[0])
    return float(dfx[col].median()) if len(dfx) else float(df[col].median())

def build_feature_vector(df, pais, anio):
    """
    Completa todas las features NO-leakage para alimentar el MODELO COMPLETO.
    - País y Año vienen del usuario (A).
    - Resto numéricas: tendencia por país (o global).
    - Categóricas: modo del país (o global).
    """
    X = {}
    if "Año" in df.columns:
        X["Año"] = int(anio)
    if "Pais" in df.columns:
        X["Pais"] = pais

    dfp = df[df["Pais"] == pais] if "Pais" in df.columns else df

    # Numéricas (excepto target y Año)
    for col in [c for c in NUM_ALLOWED if c not in [TARGET, "Año"]]:
        if col in df.columns:
            try:
                val = _trend_value(dfp, col, anio)
            except Exception:
                val = float(dfp[col].median()) if col in dfp.columns and len(dfp[col].dropna()) else float(df[col].median())
            if val is not None and np.isfinite(val):
                X[col] = float(val)

    # Categóricas (excepto Pais)
    for col in [c for c in CAT_ALLOWED if c not in ["Pais", TARGET]]:
        if col in dfp.columns and len(dfp[col].dropna()):
            X[col] = str(dfp[col].mode().iloc[0])
        elif col in df.columns and len(df[col].dropna()):
            X[col] = str(df[col].mode().iloc[0])
        else:
            X[col] = ""
    return pd.DataFrame([X])

def local_what_if(modelo, base_row: pd.DataFrame, df_ref: pd.DataFrame, k=10):
    """
    Importancia local aproximada: sube +1σ cada numérica (sin leakage) y mide Δ predicción.
    Devuelve top-k por |Δ|.
    """
    y0 = float(modelo.predict(base_row)[0])
    rows = []
    for col in [c for c in base_row.columns if c in NUM_ALLOWED and c not in ["Año"]]:
        if col not in df_ref.columns:
            continue
        s = float(df_ref[col].std(skipna=True)) if col in df_ref else 0.0
        rng = float(df_ref[col].max(skipna=True) - df_ref[col].min(skipna=True)) if col in df_ref else 0.0
        bump = s if s > 0 else (0.1 * rng if rng > 0 else 1.0)
        test = base_row.copy()
        test[col] = float(test[col]) + bump
        try:
            y1 = float(modelo.predict(test)[0])
            rows.append({"variable": col, "delta": y1 - y0})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["variable", "delta"]), y0
    out = pd.DataFrame(rows).sort_values(by="delta", key=np.abs, ascending=False).head(k)
    out["delta"] = out["delta"].astype(float)
    return out, y0
# ============================ FIN PASO 0 ============================

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
    time_cols_2 = [c for c in df.columns if str(c).lower() in ["año", "anio", "year", "fecha", "date"]]
    country_cols = [c for c in df.columns if str(c).lower() in ["pais", "país", "country"]]

    if time_cols_2 and country_cols:
        tcol = st.selectbox("Columna temporal", options=time_cols_2, key="pais_ts_tcol")
        pcol = st.selectbox("Columna de país", options=country_cols, key="pais_ts_pcol")
        yvar2 = st.selectbox("Variable Y (numérica)", options=[c for c in df.select_dtypes('number').columns],
                            index=0, key="pais_ts_y")

        opciones = sorted([str(v) for v in df[pcol].dropna().unique().tolist()])
        sel = st.multiselect("Elegí país(es)", opciones, default=opciones[:1])

        if sel:
            dff = df[df[pcol].isin(sel)].copy()
            if str(tcol).lower() in ["fecha", "date"]:
                dff[tcol] = pd.to_datetime(dff[tcol], errors="coerce")

            line = (
                alt.Chart(dff)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{tcol}:{'T' if str(tcol).lower() in ['fecha','date'] else 'O'}", title=tcol),
                    y=alt.Y(f"{yvar2}:Q", title=yvar2),
                    color=alt.Color(f"{pcol}:N", title="País"),
                    tooltip=[pcol, tcol, yvar2],
                )
                .properties(height=380, title="Serie por país (color)")
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)

            if st.checkbox("Ver como 'small multiples' (un panel por país)"):
                facet = (
                    alt.Chart(dff)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{tcol}:{'T' if str(tcol).lower() in ['fecha','date'] else 'O'}", title=tcol),
                        y=alt.Y(f"{yvar2}:Q", title=yvar2),
                        facet=alt.Facet(f"{pcol}:N", columns=4),
                        tooltip=[pcol, tcol, yvar2],
                    )
                    .properties(height=180)
                )
                st.altair_chart(facet, use_container_width=True)
        else:
            st.info("Seleccioná al menos un país para graficar.")
    else:
        st.info("No encontré columnas de tiempo (Año/fecha) y/o país (Pais/País/Country).")

    st.markdown("5) Real vs Predicho y Residuos")
    try:
        y_true = df["Natalidad"]
        X_all = df.drop(columns=["Natalidad"])
        y_pred = modelo.predict(X_all) if modelo is not None else np.full(len(y_true), np.nan)
        dd = pd.DataFrame({"Real": y_true, "Pred": y_pred, "Residuo": y_true - y_pred}).dropna()

        line_df = pd.DataFrame({"Real": [dd["Real"].min(), dd["Real"].max()]})
        line_df["Pred"] = line_df["Real"]
        sc = (
            alt.Chart(dd).mark_circle(size=60)
            .encode(x="Real:Q", y="Pred:Q", tooltip=["Real","Pred"])
            .properties(height=320, width=430, title=f"RandomForest: Real vs Predicho  (R² ≈ {np.corrcoef(dd['Real'], dd['Pred'])[0,1]**2:.4f})")
        )
        diag = alt.Chart(line_df).mark_line(strokeDash=[6,4]).encode(x="Real:Q", y="Pred:Q")
        left = sc + diag

        hist = (
            alt.Chart(dd).mark_bar()
            .encode(x=alt.X("Residuo:Q", bin=alt.Bin(maxbins=40)), y="count()", tooltip=["count()"])
            .properties(height=320, width=430, title=f"Distribución de Residuos (media = {dd['Residuo'].mean():.3f})")
        )
        st.altair_chart(alt.hconcat(left, hist), use_container_width=True)
    except Exception as e:
        st.caption(f"No se pudo generar el panel Real/Pred/Residuos: {e}")

    st.markdown("6) Importancias y efecto de n_estimators")
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.inspection import permutation_importance
        from sklearn.base import clone
        from sklearn.metrics import make_scorer, mean_squared_error
        y = df["Natalidad"]; X = df.drop(columns=["Natalidad"])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        r = permutation_importance(modelo, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1, scoring="r2")
        imp = pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False).head(10).reset_index()
        imp.columns = ["Variable","Importancia"]
        bars = alt.Chart(imp).mark_bar().encode(x="Importancia:Q", y=alt.Y("Variable:N", sort="-x"))

        scorer = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)))
        sample = X.sample(min(800, len(X)), random_state=42); ysample = y.loc[sample.index]
        vals = [450, 500, 550]
        rows = []
        base_est = getattr(modelo, "named_steps", {}).get("model", None)
        if base_est is None:
            st.caption("No pude detectar el estimador final del pipeline para la curva de CV.")
        else:
            from sklearn.pipeline import Pipeline
            pre = modelo.named_steps.get("pre", None)
            for n in vals:
                est = clone(base_est).set_params(n_estimators=n)
                pipe = Pipeline([('pre', pre), ('model', est)]) if pre is not None else est
                scores = cross_val_score(pipe, sample, ysample, cv=3, scoring=scorer, n_jobs=-1)
                rows.append({"n_estimators": n, "cv_rmse": -scores.mean(), "std": scores.std()})
            cvdf = pd.DataFrame(rows)
            line = alt.Chart(cvdf).mark_line(point=True).encode(x="n_estimators:Q", y="cv_rmse:Q")
            err = alt.Chart(cvdf).mark_errorbar().encode(x="n_estimators:Q", y="cv_rmse:Q", yError="std:Q")
            st.altair_chart(alt.hconcat(bars.properties(title="Top 10 features"), (line+err).properties(title="CV RMSE vs n_estimators")), use_container_width=True)
    except Exception as e:
        st.caption(f"No se pudo generar Importancias/Curva: {e}")

    st.markdown("7) Evolución temporal por región")
    if "Continente" in df.columns:
        base = (df.groupby(["Continente","Año"], as_index=False)["Natalidad"].mean()
                  .rename(columns={"Natalidad":"NatalidadProm"}))
        sel = alt.selection_point(fields=["Continente"], bind="legend")
        lineas = (
            alt.Chart(base)
            .mark_line(point=True, opacity=0.6)
            .encode(
                x="Año:O", y="NatalidadProm:Q",
                color=alt.Color("Continente:N"),
                opacity=alt.condition(sel, alt.value(1), alt.value(0.2)),
                tooltip=["Continente","Año","NatalidadProm"]
            ).add_params(sel)
            .properties(height=380)
        )
        trend = lineas.transform_regression("Año","NatalidadProm", groupby=["Continente"]).mark_line(strokeDash=[4,3], color="red")
        st.altair_chart(lineas + trend, use_container_width=True)
    else:
        st.caption("No encontré la columna 'Continente'.")

    st.markdown("8) Evolución de la natalidad (mapa)")
    try:
        from vega_datasets import data as vdata
        topo_url = vdata.world_110m.url
        if "CodigoPais" in df.columns:
            anios = sorted(df["Año"].dropna().astype(int).unique().tolist())
            a = st.slider("Año", min_value=min(anios), max_value=max(anios), value=max(anios), step=1)
            dfa = df[df["Año"]==a][["CodigoPais","Natalidad"]].dropna()
            countries = alt.topo_feature(topo_url, "countries")
            map_chart = (
                alt.Chart(countries)
                .mark_geoshape(stroke="white")
                .transform_lookup(
                    lookup="id",
                    from_=alt.LookupData(dfa, "CodigoPais", ["Natalidad"])
                )
                .encode(color=alt.Color("Natalidad:Q", title="Natalidad (x1000)"))
                .properties(height=420)
                .project("equalEarth")
            )
            st.altair_chart(map_chart, use_container_width=True)
        else:
            st.info("Falta columna 'CodigoPais' (ISO3) para dibujar el mapa.")
    except Exception as e:
        st.caption(f"No se pudo dibujar el mapa: {e}")

elif page == "Modelo":
    st.header("Modelo (pipeline scikit-learn)")
    if modelo is None:
        st.warning("No se encontró un archivo de modelo en `models/`.")
    else:
        st.success("Modelo cargado correctamente.")

        final_est = getattr(modelo, "named_steps", {}).get("model", modelo)
        st.write(f"**Estimador final:** `{final_est.__class__.__name__}`")
        with st.expander("Hiperparámetros del estimador final"):
            try:
                params = final_est.get_params()
                params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
                st.json(params)
            except Exception as e:
                st.caption(f"No pude listar parámetros: {e}")

        TARGET_DEFAULT = "Natalidad"
        target_col = st.selectbox(
            "Columna objetivo (y)",
            options=[c for c in df.columns if c != ""],
            index=(list(df.columns).index(TARGET_DEFAULT) if TARGET_DEFAULT in df.columns else 0)
        )

        if target_col not in df.columns:
            st.error("No se encuentra la columna objetivo en el dataset.")
        else:
            from sklearn.model_selection import train_test_split
            y = df[target_col]
            X = df.drop(columns=[target_col])
            is_reg = pd.api.types.is_numeric_dtype(y)

            test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.2, 0.05)
            rs = 42
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=rs, stratify=None if is_reg else y
            )
            try:
                y_hat = modelo.predict(X_te)
                if is_reg:
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    r2 = r2_score(y_te, y_hat)
                    mae = mean_absolute_error(y_te, y_hat)
                    rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))  # FIX RMSE
                    st.subheader("Métricas")
                    st.write({"R2": round(r2, 3), "MAE": round(mae, 3), "RMSE": round(rmse, 3)})
                else:
                    from sklearn.metrics import accuracy_score, f1_score
                    acc = accuracy_score(y_te, y_hat)
                    f1w = f1_score(y_te, y_hat, average="weighted")
                    st.subheader("Métricas")
                    st.write({"Accuracy": round(acc, 3), "F1 (weighted)": round(f1w, 3)})
            except Exception as e:
                st.warning(f"No pude evaluar el modelo con el dataset actual: {e}")

        with st.expander("Importancia de variables (Permutation Importance)"):
            try:
                from sklearn.inspection import permutation_importance
                ns = min(300, len(X_te))
                Xs = X_te.sample(ns, random_state=rs)
                ys = y_te.loc[Xs.index]
                scoring = "r2" if is_reg else "f1_weighted"
                r = permutation_importance(modelo, Xs, ys, n_repeats=5, random_state=rs, n_jobs=-1, scoring=scoring)
                imp = pd.Series(r.importances_mean, index=Xs.columns).sort_values(ascending=False)
                top = imp.head(15).reset_index()
                top.columns = ["variable", "importancia"]
                chart = (
                    alt.Chart(top).mark_bar()
                    .encode(x=alt.X("importancia:Q", title="Importancia media (perm.)"),
                            y=alt.Y("variable:N", sort="-x", title="Variable"))
                    .properties(height=400)
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(top)
            except Exception as e:
                st.caption(f"No pude calcular importancias: {e}")

else:
    st.header("Predicción")

    tabA, tabB, tabC = st.tabs([
        "A) Rápida (País + Año)", 
        "B) Automática (proyectar explicativas)", 
        "C) Escenarios (editar supuestos)"
    ])

    # ===== A) SOLO País + Año: usa modelo completo + autocompletar (sin leakage)
    with tabA:
        if modelo is None:
            st.warning("Subí tu pipeline exportado a `models/`.")
        else:
            paises = sorted(df["Pais"].dropna().astype(str).unique().tolist()) if "Pais" in df.columns else []
            a_min = int(pd.to_numeric(df["Año"], errors="coerce").dropna().min()) if "Año" in df.columns else 2000
            a_max = int(pd.to_numeric(df["Año"], errors="coerce").dropna().max()) if "Año" in df.columns else a_min
            c1, c2 = st.columns(2)
            pais = c1.selectbox("País", options=paises) if paises else None
            anio = c2.slider("Año", min_value=a_min, max_value=a_max+10, value=a_max, step=1)

            if st.button("Predecir (A)"):
                Xnew = build_feature_vector(df, pais, anio)
                st.caption("Vector autocompletado (no incluye variables de fuga):")
                st.dataframe(Xnew)
                try:
                    yhat = float(modelo.predict(Xnew)[0])
                    st.metric("Natalidad predicha", f"{yhat:.2f}")
                    try:
                        imp, y0 = local_what_if(modelo, Xnew, df, k=8)
                        if len(imp):
                            st.markdown("**Factores más influyentes (variación local +1σ):**")
                            chart = (
                                alt.Chart(imp).mark_bar()
                                .encode(x=alt.X("delta:Q", title="Δ pred."),
                                        y=alt.Y("variable:N", sort="-x", title="Variable"))
                                .properties(height=280)
                            )
                            st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"No se pudo predecir: {e}")

    # ===== B) Automática (informativa: mismo cálculo que A, pero explicado)
    with tabB:
        st.info("Se proyectan automáticamente las variables explicativas NO-leakage por tendencia (país → global).")
        st.caption("El resultado es equivalente a (A); esta pestaña documenta el proceso de proyección.")

    # ===== C) Escenarios: usuario edita supuestos (sin leakage), partiendo de la proyección
    with tabC:
        if modelo is None:
            st.warning("Subí tu pipeline exportado a `models/`.")
        else:
            paises = sorted(df["Pais"].dropna().astype(str).unique().tolist()) if "Pais" in df.columns else []
            a_min = int(pd.to_numeric(df["Año"], errors="coerce").dropna().min()) if "Año" in df.columns else 2000
            a_max = int(pd.to_numeric(df["Año"], errors="coerce").dropna().max()) if "Año" in df.columns else a_min
            c1, c2 = st.columns(2)
            pais = c1.selectbox("País", options=paises, key="esc_pais") if paises else None
            anio = c2.slider("Año", min_value=a_min, max_value=a_max+10, value=a_max, step=1, key="esc_anio")

            baseX = build_feature_vector(df, pais, anio)
            st.caption("Valores base (proyección):")
            st.dataframe(baseX)

            st.subheader("Editar supuestos (solo variables NO-leakage)")
            num_edits = [c for c in baseX.columns if c in NUM_ALLOWED and c not in ["Año", TARGET]]
            cat_edits = [c for c in baseX.columns if c in CAT_ALLOWED and c not in ["Pais", TARGET]]

            edits = {}
            for c in num_edits:
                default = float(baseX[c].iloc[0])
                edits[c] = st.number_input(c, value=float(default))
            for c in cat_edits:
                opts = sorted(df[c].dropna().astype(str).unique().tolist()) if c in df.columns else [str(baseX[c].iloc[0])]
                default = str(baseX[c].iloc[0]) if str(baseX[c].iloc[0]) in opts else (opts[0] if opts else "")
                edits[c] = st.selectbox(c, options=opts, index=opts.index(default) if default in opts else 0)

            if st.button("Predecir (C)"):
                Xnew = baseX.copy()
                for k, v in edits.items():
                    Xnew[k] = v
                try:
                    yhat = float(modelo.predict(Xnew)[0])
                    st.success(f"Predicción bajo tu escenario: {yhat:.2f}")
                except Exception as e:
                    st.error(f"No se pudo predecir: {e}")
