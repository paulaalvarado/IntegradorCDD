# streamlit_app.py — Entrega 4: Visualización e Integración
# ------------------------------------------------------------------
# • Lee el CSV final preprocesado del notebook (sin leakage)
# • Muestra todas las visualizaciones pedidas (Altair)
# • Carga el pipeline entrenado y permite predecir de 2 formas:
#     A) País + Año (auto-proyecta explicativas sin leakage)
#     B) Manual (usuario ingresa todas las features que usa el modelo)
# • Pensado para Streamlit Cloud + GitHub
# ------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Opcional (si está disponible en Cloud, acelera Altair):
try:
    import vegafusion  # noqa: F401
    import vl_convert  # noqa: F401
    alt.data_transformers.enable("vegafusion")
except Exception:
    pass

st.set_page_config(page_title="Proyecto Integrador – Datos & Modelo", layout="wide")

# ==== RUTAS DE DATOS/MODELO ====
DATA_PATHS = [
    "data/df_preprocesado_con_regiones.csv",  # <- tu CSV final (recomendado)
    "data/df_app.csv",                         # alias alternativo
    "data/df_limpio.csv",                      # fallback
]
MODEL_PATHS = ["models/model.joblib", "models/model.pkl"]

# ==== CARGA DE DATOS/MODELO ====
@st.cache_data(show_spinner=False)
def load_first_existing_csv(paths):
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [str(c).strip() for c in df.columns]
            return df, p
    raise FileNotFoundError("No se encontró ningún CSV en data/. Subí tu CSV preprocesado.")

@st.cache_resource(show_spinner=False)
def load_model(paths=MODEL_PATHS):
    import joblib, pickle
    last = None
    for p in paths:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e1:
                last = f"{p} (joblib): {e1}"
                try:
                    with open(p, "rb") as f:
                        return pickle.load(f)
                except Exception as e2:
                    last = f"{last} | {p} (pickle): {e2}"
    if last:
        st.warning(f"No pude cargar modelo: {last}")
    return None

df, DATA_USED = load_first_existing_csv(DATA_PATHS)
modelo = load_model()

# Normalizar Año a entero si existe
YEAR_COLS = [c for c in df.columns if str(c).lower() in ["año", "anio", "year"]]
for c in YEAR_COLS:
    try:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    except Exception:
        pass

TARGET = "Natalidad"

# Variables que NO deben ofrecerse para edición/proyección (leakage)
LEAKAGE = {
    "TasaFertilidad", "MortalidadInfantil", "MortalidadMenores5",
    "MortalidadNeonatal", "MortalidadMaterna",
    "PrevalenciaAnemiaEmbarazadas", "RatioDependenciaJovenes",
    # agregá acá cualquier derivada directa del target usada en tu notebook
}

ALL_NUM = [c for c in df.select_dtypes("number").columns if c != TARGET]
ALL_CAT = [c for c in df.columns if c not in df.select_dtypes("number").columns]

NUM_ALLOWED = [c for c in ALL_NUM if c not in LEAKAGE]   # incluye 'Año' si es numérico
CAT_ALLOWED = [c for c in ALL_CAT if c not in LEAKAGE]

# ==== HELPERS de proyección sin leakage (para tab A y escenarios) ====
from sklearn.linear_model import LinearRegression

def _trend_value(dfx, col, anio):
    """Proyecta col→año con regresión lineal (país→global)."""
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
    """Completa TODAS las features NO-leakage para el modelo final."""
    X = {}
    if "Año" in df.columns: X["Año"] = int(anio)
    if "Pais" in df.columns: X["Pais"] = pais
    dfp = df[df["Pais"] == pais] if "Pais" in df.columns else df

    for col in [c for c in NUM_ALLOWED if c not in [TARGET, "Año"] and c in df.columns]:
        try:
            val = _trend_value(dfp, col, anio)
        except Exception:
            val = float(dfp[col].median()) if col in dfp.columns and len(dfp[col].dropna()) else float(df[col].median())
        if val is not None and np.isfinite(val):
            X[col] = float(val)

    for col in [c for c in CAT_ALLOWED if c not in ["Pais", TARGET] and c in df.columns]:
        if col in dfp.columns and len(dfp[col].dropna()):
            X[col] = str(dfp[col].mode().iloc[0])
        elif len(df[col].dropna()):
            X[col] = str(df[col].mode().iloc[0])
        else:
            X[col] = ""
    return pd.DataFrame([X])

def local_what_if(modelo, base_row: pd.DataFrame, df_ref: pd.DataFrame, k=10):
    """Importancia local aproximada (+1σ a cada numérica permitida)."""
    y0 = float(modelo.predict(base_row)[0])
    rows = []
    for col in [c for c in base_row.columns if c in NUM_ALLOWED and c not in ["Año"]]:
        if col not in df_ref.columns: continue
        s = float(df_ref[col].std(skipna=True)) if col in df_ref else 0.0
        rng = float(df_ref[col].max(skipna=True) - df_ref[col].min(skipna=True)) if col in df_ref else 0.0
        bump = s if s > 0 else (0.1 * rng if rng > 0 else 1.0)
        test = base_row.copy(); test[col] = float(test[col]) + bump
        try:
            y1 = float(modelo.predict(test)[0])
            rows.append({"variable": col, "delta": y1 - y0})
        except Exception:
            pass
    out = pd.DataFrame(rows)
    if out.empty: return out, y0
    out = out.sort_values(by="delta", key=np.abs, ascending=False).head(k)
    out["delta"] = out["delta"].astype(float)
    return out, y0

# ==== LAYOUT ====
st.sidebar.title("Proyecto Integrador")
st.sidebar.caption(f"CSV: `{DATA_USED}`")
page = st.sidebar.radio("Ir a…", ["Explorar datos", "Modelo", "Predecir"])
st.sidebar.markdown("---")
st.sidebar.caption("Requiere: data/df_preprocesado_con_regiones.csv + models/model.joblib")

# ===================================================================
# EXPLORAR DATOS
# ===================================================================
if page == "Explorar datos":
    st.header("Exploración interactiva de datos")
    with st.expander("Info del dataset", expanded=False):
        st.write({"filas": int(df.shape[0]), "columnas": int(df.shape[1])})
        st.dataframe(df.head(12), use_container_width=True)

    num_cols = list(df.select_dtypes("number").columns)
    cat_cols = [c for c in df.columns if c not in num_cols]

    # 1) Dispersión
    st.subheader("1) Dispersión (Scatter)")
    c1, c2, c3 = st.columns(3)
    xvar = c1.selectbox("X (numérico)", options=num_cols, index=(num_cols.index("Año") if "Año" in num_cols else 0))
    yvar = c2.selectbox("Y (numérico)", options=num_cols, index=(num_cols.index(TARGET) if TARGET in num_cols else 0))
    color_by = c3.selectbox("Color (opcional)", options=["(ninguno)"] + cat_cols, index=cat_cols.index("Pais")+1 if "Pais" in cat_cols else 0)
    if xvar and yvar:
        brush = alt.selection_interval()
        enc_color = alt.Color(color_by + ":N") if color_by != "(ninguno)" else alt.value("steelblue")
        sc = (
            alt.Chart(df).mark_circle(size=90, opacity=0.8)
            .encode(
                x=alt.X(f"{xvar}:Q"), y=alt.Y(f"{yvar}:Q"),
                color=alt.condition(brush, enc_color, alt.value("lightgray")),
                tooltip=[xvar, yvar] + ([color_by] if color_by != "(ninguno)" else []),
            ).add_params(brush).properties(height=360).interactive()
        )
        st.altair_chart(sc, use_container_width=True)

    # 2) Boxplot
    st.subheader("2) Boxplot")
    if cat_cols and num_cols:
        catv = st.selectbox("Categoría (X)", options=cat_cols, index=(cat_cols.index("Pais") if "Pais" in cat_cols else 0))
        numv = st.selectbox("Num (Y)", options=num_cols, index=(num_cols.index(TARGET) if TARGET in num_cols else 0), key="box-y")
        box = alt.Chart(df).mark_boxplot(extent="min-max").encode(x=f"{catv}:N", y=f"{numv}:Q", tooltip=[catv, numv]).properties(height=360)
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("Necesitás al menos 1 categórica y 1 numérica.")

    # 3) Serie temporal (genérica)
    st.subheader("3) Serie temporal")
    time_cols = [c for c in df.columns if str(c).lower() in ["año", "anio", "year", "fecha", "date"]]
    if time_cols and num_cols:
        tcol = st.selectbox("Columna temporal", options=time_cols, index=(time_cols.index("Año") if "Año" in time_cols else 0))
        yts = st.selectbox("Num (Y)", options=num_cols, index=(num_cols.index(TARGET) if TARGET in num_cols else 0), key="ts-y")
        tsdf = df.copy()
        if str(tcol).lower() in ["año", "anio", "year"]:
            xenc = alt.X(f"{tcol}:O")
        else:
            tsdf[tcol] = pd.to_datetime(tsdf[tcol], errors="coerce"); xenc = alt.X(f"{tcol}:T")
        line = alt.Chart(tsdf.dropna(subset=[tcol])).mark_line(point=True).encode(x=xenc, y=f"{yts}:Q", tooltip=[tcol, yts]).properties(height=360).interactive()
        st.altair_chart(line, use_container_width=True)

    # 4) Serie por país (color/facet)
    st.subheader("4) Natalidad por país (serie)")
    country_cols = [c for c in df.columns if str(c).lower() in ["pais", "país", "country"]]
    if time_cols and country_cols:
        tcol = st.selectbox("Tiempo", options=time_cols, key="pais_ts_t")
        pcol = st.selectbox("Col. país", options=country_cols, key="pais_ts_p")
        y_S = st.selectbox("Y (num.)", options=num_cols, index=(num_cols.index(TARGET) if TARGET in num_cols else 0), key="pais_ts_y")
        opciones = sorted([str(v) for v in df[pcol].dropna().unique().tolist()])
        sel = st.multiselect("Elegí país(es)", opciones, default=opciones[:1])
        if sel:
            dff = df[df[pcol].isin(sel)].copy()
            if str(tcol).lower() in ["fecha", "date"]: dff[tcol] = pd.to_datetime(dff[tcol], errors="coerce")
            line = (alt.Chart(dff).mark_line(point=True).encode(
                x=alt.X(f"{tcol}:{'T' if str(tcol).lower() in ['fecha','date'] else 'O'}"),
                y=f"{y_S}:Q", color=f"{pcol}:N", tooltip=[pcol, tcol, y_S]
            ).properties(height=380).interactive())
            st.altair_chart(line, use_container_width=True)
            if st.checkbox("Ver como 'small multiples'"):
                facet = (alt.Chart(dff).mark_line(point=True).encode(
                    x=alt.X(f"{tcol}:{'T' if str(tcol).lower() in ['fecha','date'] else 'O'}"),
                    y=f"{y_S}:Q", facet=alt.Facet(f"{pcol}:N", columns=4), tooltip=[pcol, tcol, y_S]
                ).properties(height=180))
                st.altair_chart(facet, use_container_width=True)

    # 5) Real vs Predicho + Residuos (si hay modelo)
    st.subheader("5) Real vs Predicho y Residuos")
    try:
        if modelo is None or TARGET not in df.columns:
            st.info("Subí un modelo y asegurate de tener la columna 'Natalidad'.")
        else:
            cols_model = getattr(modelo, "feature_names_in_", None)
            X_all = df[cols_model] if cols_model is not None else df.drop(columns=[TARGET])
            y_true = df[TARGET]
            y_pred = modelo.predict(X_all)
            dd = pd.DataFrame({"Real": y_true, "Pred": y_pred, "Residuo": y_true - y_pred}).dropna()

            diag = alt.Chart(pd.DataFrame({"Real":[dd.Real.min(), dd.Real.max()], "Pred":[dd.Real.min(), dd.Real.max()]})).mark_line(strokeDash=[5,4])
            sc = alt.Chart(dd).mark_circle(size=60, opacity=0.7).encode(x="Real:Q", y="Pred:Q", tooltip=["Real","Pred"])
            left = (sc + diag).properties(height=320, width=430, title=f"Real vs Predicho (R² ≈ {np.corrcoef(dd['Real'], dd['Pred'])[0,1]**2:.3f})")

            hist = alt.Chart(dd).mark_bar().encode(x=alt.X("Residuo:Q", bin=alt.Bin(maxbins=40)), y="count()", tooltip=["count()"]).properties(
                height=320, width=430, title=f"Distribución de Residuos (media = {dd['Residuo'].mean():.2f})"
            )
            st.altair_chart(alt.hconcat(left, hist), use_container_width=True)
    except Exception as e:
        st.caption(f"No se pudo generar el panel Real/Pred/Residuos: {e}")

    # 6) Importancias (perm.) + curva n_estimators (si aplica)
    st.subheader("6) Importancias y efecto de n_estimators")
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.inspection import permutation_importance
        from sklearn.base import clone
        from sklearn.metrics import make_scorer, mean_squared_error

        cols_model = getattr(modelo, "feature_names_in_", None)
        X = df[cols_model].copy() if cols_model is not None else df.drop(columns=[TARGET]).copy()
        y = df[TARGET].copy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        r = permutation_importance(modelo, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1, scoring="r2")
        imp = pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False).head(10).reset_index()
        imp.columns = ["Variable","Importancia"]
        bars = alt.Chart(imp).mark_bar().encode(x="Importancia:Q", y=alt.Y("Variable:N", sort="-x")).properties(width=520, height=360)

        # Curva CV RMSE vs n_estimators (solo si el estimador final lo tiene)
        try:
            base_est = getattr(modelo, "named_steps", {}).get("model", modelo)
            if hasattr(base_est, "get_params") and "n_estimators" in base_est.get_params():
                from sklearn.pipeline import Pipeline
                pre = getattr(modelo, "named_steps", {}).get("pre", None)
                vals = [450, 500, 550]
                scorer = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)))
                sample = X.sample(min(800, len(X)), random_state=42); ysample = y.loc[sample.index]
                rows = []
                for n in vals:
                    est = clone(base_est).set_params(n_estimators=n)
                    pipe = Pipeline([('pre', pre), ('model', est)]) if pre is not None else est
                    scores = cross_val_score(pipe, sample, ysample, cv=3, scoring=scorer, n_jobs=-1)
                    rows.append({"n_estimators": n, "cv_rmse": -scores.mean(), "std": scores.std()})
                cvdf = pd.DataFrame(rows)
                line = alt.Chart(cvdf).mark_line(point=True).encode(x="n_estimators:Q", y="cv_rmse:Q").properties(width=520, height=360)
                err = alt.Chart(cvdf).mark_errorbar().encode(x="n_estimators:Q", y="cv_rmse:Q", yError="std:Q")
                right = line + err
            else:
                right = alt.Chart(pd.DataFrame({"info":["El estimador no tiene n_estimators"]})).mark_text().encode(text="info").properties(width=520, height=360)
        except Exception as e:
            right = alt.Chart(pd.DataFrame({"err":[str(e)[:120]]})).mark_text().encode(text="err").properties(width=520, height=360)

        st.altair_chart(alt.hconcat(bars, right), use_container_width=True)
    except Exception as e:
        st.caption(f"No se pudo generar Importancias/Curva: {e}")

    # 7) Evolución temporal por región/continente (si existe)
    st.subheader("7) Evolución temporal por región")
    if "Continente" in df.columns and "Año" in df.columns:
        base = (df.groupby(["Continente","Año"], as_index=False)[TARGET].mean().rename(columns={TARGET:"NatalidadProm"}))
        sel = alt.selection_point(fields=["Continente"], bind="legend")
        lines = (alt.Chart(base).mark_line(point=True, opacity=0.7).encode(
            x="Año:O", y="NatalidadProm:Q", color="Continente:N",
            opacity=alt.condition(sel, alt.value(1), alt.value(0.25)),
            tooltip=["Continente","Año","NatalidadProm"]
        ).add_params(sel).properties(height=380))
        trend = lines.transform_regression("Año","NatalidadProm", groupby=["Continente"]).mark_line(strokeDash=[4,3], color="red")
        st.altair_chart(lines + trend, use_container_width=True)
    else:
        st.info("No encontré la columna 'Continente' y/o 'Año'.")

    # 8) Mapa coroplético por año (si existe CodigoPais compatible)
    st.subheader("8) Evolución de la natalidad (mapa)")
    try:
        from vega_datasets import data as vdata
        topo = alt.topo_feature(vdata.world_110m.url, "countries")
        if "CodigoPais" in df.columns and "Año" in df.columns:
            anios = sorted([int(a) for a in df["Año"].dropna().unique()])
            a = st.slider("Año", min_value=min(anios), max_value=max(anios), value=max(anios), step=1, key="mapyear")
            dfa = df[df["Año"] == a][["CodigoPais", TARGET]].dropna()
            # Nota: el 'id' de world_110m es numérico; tu CodigoPais debe matchear ese id.
            mapa = (alt.Chart(topo).mark_geoshape(stroke="white").transform_lookup(
                        lookup="id", from_=alt.LookupData(dfa, "CodigoPais", [TARGET])
                    ).encode(color=alt.Color(f"{TARGET}:Q", title="Natalidad (x1000)"))
                    .properties(height=420).project("equalEarth"))
            st.altair_chart(mapa, use_container_width=True)
        else:
            st.info("Falta 'CodigoPais' (id de Natural Earth) y/o 'Año' para el mapa.")
    except Exception as e:
        st.caption(f"No se pudo dibujar el mapa: {e}")

    # 9) Histograma + KDE de natalidad y top correlaciones
    st.subheader("9) Distribución y correlaciones")
    if TARGET in df.columns:
        # Hist + KDE
        density = alt.Chart(df).transform_density(TARGET, as_=[TARGET, 'density']).mark_line()
        hist = alt.Chart(df).mark_bar(opacity=0.6).encode(x=alt.X(f"{TARGET}:Q", bin=alt.Bin(maxbins=60), title="Natalidad"), y="count()")
        mean_v = float(df[TARGET].mean()); med_v = float(df[TARGET].median())
        vmean = alt.Chart(pd.DataFrame({TARGET:[mean_v]})).mark_rule(color="red").encode(x=f"{TARGET}:Q")
        vmed = alt.Chart(pd.DataFrame({TARGET:[med_v]})).mark_rule(color="green").encode(x=f"{TARGET}:Q")
        left = (hist + density).properties(width=520, height=360, title=f"Distribución de {TARGET} (media {mean_v:.2f} / mediana {med_v:.2f})")

        # Top correlaciones con Natalidad (numéricas)
        corr = df.select_dtypes("number").corr()[TARGET].drop(labels=[TARGET]).dropna()
        topcorr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).reset_index()
        topcorr.columns = ["Variable", "Correlacion"]
        right = alt.Chart(topcorr).mark_bar().encode(
            x=alt.X("Correlacion:Q", scale=alt.Scale(domain=[-1,1])),
            y=alt.Y("Variable:N", sort="-x"),
            color=alt.condition("datum.Correlacion > 0", alt.value("#7fc97f"), alt.value("#d95f02"))
        ).properties(width=520, height=360, title="Top 15 correlaciones con Natalidad")
        st.altair_chart(alt.hconcat(left, right), use_container_width=True)

    # 10) Tendencia global + Heatmap correlación (Top 10 + target)
    st.subheader("10) Tendencia global y matriz de correlación")
    try:
        if "Año" in df.columns and TARGET in df.columns:
            glob = df.groupby("Año", as_index=False)[TARGET].mean()
            trend = (alt.Chart(glob).mark_line(point=True).encode(x="Año:O", y=f"{TARGET}:Q")
                     .properties(width=520, height=360, title=f"Evolución global de {TARGET}"))
            reg = alt.Chart(glob).transform_regression("Año", TARGET).mark_line(strokeDash=[5,4], color="red")
            left = trend + reg
        else:
            left = alt.Chart(pd.DataFrame({"msg":["Falta Año o Natalidad"]})).mark_text().encode(text="msg").properties(width=520, height=360)

        numdf = df.select_dtypes("number").copy()
        cols = [c for c in numdf.columns if c != TARGET]
        top10 = numdf[cols].corr()[TARGET].abs().sort_values(ascending=False).head(10).index.tolist()
        cm = numdf[top10 + [TARGET]].corr().round(2)
        mat = cm.reset_index().melt(id_vars="index", var_name="col2", value_name="corr").rename(columns={"index":"col1"})
        heat = (alt.Chart(mat).mark_rect().encode(
            x=alt.X("col1:N", title=""), y=alt.Y("col2:N", title=""),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1,1])),
            tooltip=["col1","col2","corr"]
        ).properties(width=520, height=360, title="Matriz de correlación (Top10 + Natalidad)"))
        st.altair_chart(alt.hconcat(left, heat), use_container_width=True)
    except Exception as e:
        st.caption(f"No se pudo generar tendencia/heatmap: {e}")

# ===================================================================
# MODELO
# ===================================================================
elif page == "Modelo":
    st.header("Modelo (pipeline scikit-learn)")
    if modelo is None:
        st.warning("No se encontró un archivo de modelo en models/.")
    else:
        st.success("Modelo cargado correctamente.")
        final_est = getattr(modelo, "named_steps", {}).get("model", modelo)
        st.write(f"**Estimador final:** `{final_est.__class__.__name__}`")
        with st.expander("Hiperparámetros"):
            try:
                params = final_est.get_params()
                params = {k:v for k,v in params.items() if isinstance(v,(int,float,str,bool,type(None)))}
                st.json(params)
            except Exception as e:
                st.caption(f"No pude listar parámetros: {e}")

        # Evaluación rápida holdout (no reentrena)
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            cols_model = getattr(modelo, "feature_names_in_", None)
            X = df[cols_model].copy() if cols_model is not None else df.drop(columns=[TARGET]).copy()
            y = df[TARGET].copy()
            test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.2, 0.05)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
            yhat = modelo.predict(Xte)
            r2 = r2_score(yte, yhat); mae = mean_absolute_error(yte, yhat); rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
            st.write({"R2": round(r2,3), "MAE": round(mae,3), "RMSE": round(rmse,3)})
        except Exception as e:
            st.caption(f"No pude evaluar con holdout: {e}")

# ===================================================================
# PREDECIR
# ===================================================================
else:
    st.header("Predicción")

    tabA, tabB = st.tabs(["A) País + Año (auto)", "B) Manual (todas las variables)"])

    # ---- A) País + Año: auto-proyección (sin leakage)
    with tabA:
        if modelo is None:
            st.warning("Subí tu pipeline exportado a models/.")
        else:
            paises = sorted(df["Pais"].dropna().astype(str).unique().tolist()) if "Pais" in df.columns else []
            if "Año" in df.columns:
                a_min = int(pd.to_numeric(df["Año"], errors="coerce").dropna().min())
                a_max = int(pd.to_numeric(df["Año"], errors="coerce").dropna().max())
            else:
                a_min = 2000; a_max = 2025
            c1, c2 = st.columns(2)
            pais = c1.selectbox("País", options=paises) if paises else None
            anio = c2.slider("Año", min_value=a_min, max_value=a_max+10, value=a_max, step=1)
            if st.button("Predecir (A)"):
                try:
                    Xnew = build_feature_vector(df, pais, anio)
                    st.caption("Vector autocompletado (no incluye variables de fuga):")
                    st.dataframe(Xnew, use_container_width=True)
                    yhat = float(modelo.predict(Xnew)[0])
                    st.metric("Natalidad predicha", f"{yhat:.2f}")
                    # factores locales
                    try:
                        imp_loc, _ = local_what_if(modelo, Xnew, df, k=8)
                        if not imp_loc.empty:
                            st.markdown("**Factores más influyentes (variación local +1σ):**")
                            ch = alt.Chart(imp_loc).mark_bar().encode(
                                x=alt.X("delta:Q", title="Δ pred"), y=alt.Y("variable:N", sort="-x")
                            ).properties(height=280)
                            st.altair_chart(ch, use_container_width=True)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"No se pudo predecir: {e}")

    # ---- B) Manual: el usuario ingresa TODAS las features que requiere el modelo
    with tabB:
        if modelo is None:
            st.warning("Subí tu pipeline exportado a models/.")
        else:
            cols_model = list(getattr(modelo, "feature_names_in_", []))
            if not cols_model:
                st.error("El modelo no expone feature_names_in_. Re-exportalo con scikit-learn ≥1.2.")
            else:
                num_model = [c for c in cols_model if c in df.select_dtypes("number").columns and c != TARGET]
                cat_model = [c for c in cols_model if c not in num_model and c in df.columns]

                inputs = {}
                if "Pais" in cols_model:
                    inputs["Pais"] = st.selectbox("Pais", options=sorted(df["Pais"].dropna().astype(str).unique().tolist()))
                if "Año" in cols_model:
                    a_min = int(pd.to_numeric(df["Año"], errors="coerce").dropna().min())
                    a_max = int(pd.to_numeric(df["Año"], errors="coerce").dropna().max())
                    inputs["Año"] = st.slider("Año", min_value=a_min, max_value=a_max+10, value=a_max, step=1)

                for c in [x for x in num_model if x not in ["Año"]]:
                    default = float(df[c].median()) if c in df.columns else 0.0
                    inputs[c] = st.number_input(c, value=default)

                for c in [x for x in cat_model if x not in ["Pais"]]:
                    opts = sorted(df[c].dropna().astype(str).unique().tolist()) if c in df.columns else ["(vacío)"]
                    default = opts[0] if opts else ""
                    inputs[c] = st.selectbox(c, options=opts, index=0)

                if st.button("Predecir (B)"):
                    try:
                        Xnew = pd.DataFrame([inputs])[cols_model]
                        yhat = float(modelo.predict(Xnew)[0])
                        st.success(f"Natalidad predicha: {yhat:.2f}")
                    except Exception as e:
                        st.error(f"No se pudo predecir: {e}")
