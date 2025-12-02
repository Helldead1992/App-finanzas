import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Finanzas App 7.0", layout="wide")
st.title("💰 Análisis Financiero: Actuals vs Forecast")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1. Carga de Datos")
    file_gl = st.file_uploader("Cargar GL (Oracle)", type=["xlsx", "csv"])
    file_forecast = st.file_uploader("Cargar Forecast (VML)", type=["xlsx", "csv"])
    
    st.divider()
    st.header("2. Filtros")
    c_period = st.container()
    c_vendor = st.container()
    c_filters = st.container()

# --- FUNCIONES DE LIMPIEZA ---

def clean_money(val):
    """Limpia moneda a float"""
    if pd.isna(val) or val == '': return 0.0
    s = str(val).replace('$', '').replace(',', '').replace(' ', '')
    if '(' in s and ')' in s: s = s.replace('(', '-').replace(')', '')
    try: return float(s)
    except: return 0.0

def clean_id(val):
    """
    Limpia IDs de forma agresiva para asegurar coincidencia.
    61220.0 (float/str) -> "61220"
    " 61220 " -> "61220"
    """
    if pd.isna(val) or val == '': return ""
    s = str(val).strip()
    try:
        f = float(s)
        if f.is_integer(): return str(int(f))
    except: pass
    if s.endswith('.0'): return s[:-2]
    return s

def clean_gl(file):
    try:
        df = pd.read_csv(file, header=12) if file.name.endswith('.csv') else pd.read_excel(file, header=12)
        df.columns = df.columns.str.strip()
        
        cols_map = {
            'Accounting Date': 'Date',
            'Vendor Name/ Customer Name': 'Vendor',
            'Game': 'Game_Code',
            'Department': 'Dept_Code',
            'Account': 'Account_Num', 
            'Account Description': 'Account_Name',
            'Location': 'Location',
            'Accounted Total': 'Amount'
        }
        
        valid = [c for c in cols_map.keys() if c in df.columns]
        df = df[valid].rename(columns=cols_map)
        df = df.dropna(subset=['Date']) 
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Period'] = df['Date'].dt.strftime('%Y-%m')
        df['Type'] = 'Actuals'
        
        # Limpieza IDs
        for col in ['Account_Num', 'Game_Code', 'Dept_Code', 'Location']:
            if col in df.columns:
                df[col] = df[col].apply(clean_id)
        
        df['Amount'] = df['Amount'].apply(clean_money)
        return df
    except Exception as e:
        st.error(f"Error GL: {e}")
        return pd.DataFrame()

def clean_forecast(file):
    try:
        df = pd.read_csv(file, header=3) if file.name.endswith('.csv') else pd.read_excel(file, header=3)
        df.columns = df.columns.str.strip()
        
        if 'Vendors Nombre' in df.columns:
            df = df.dropna(subset=['Vendors Nombre'])
            df = df[df['Vendors Nombre'].astype(str).str.lower() != 'total']
            
        month_cols = [c for c in df.columns if "-20" in str(c)]
        for col in month_cols: df[col] = df[col].apply(clean_money)

        # Mapeo explícito incluyendo PARENT ACCOUNT (Columna R aprox)
        id_vars = [
            'Vendors Nombre', 'Game_Code Código', 'Level Code Código', 
            'Cost_Account_Number Código', 
            'Cost_Account_Number Nombre', 'Studio_Location_Code Código',
            'Parent Account Nombre' # <--- NUEVO CAMPO
        ]
        
        valid_ids = [c for c in id_vars if c in df.columns]
        df_melt = df.melt(id_vars=valid_ids, value_vars=month_cols, var_name='Period_Text', value_name='Amount')
        
        rename_map = {
            'Vendors Nombre': 'Vendor',
            'Game_Code Código': 'Game_Code',
            'Level Code Código': 'Dept_Code',
            'Cost_Account_Number Código': 'Account_Num',
            'Cost_Account_Number Nombre': 'Account_Name',
            'Studio_Location_Code Código': 'Location',
            'Parent Account Nombre': 'Parent_Account' # <--- RENOMBRAMOS
        }
        df_melt = df_melt.rename(columns=rename_map)
        
        df_melt['Date'] = pd.to_datetime(df_melt['Period_Text'], format='%b-%Y', errors='coerce')
        df_melt['Period'] = df_melt['Date'].dt.strftime('%Y-%m')
        df_melt['Type'] = 'Forecast'
        
        for col in ['Account_Num', 'Game_Code', 'Dept_Code', 'Location']:
            if col in df_melt.columns:
                df_melt[col] = df_melt[col].apply(clean_id)
                
        return df_melt
    except Exception as e:
        st.error(f"Error Forecast: {e}")
        return pd.DataFrame()

# --- LÓGICA PRINCIPAL ---

if file_gl and file_forecast:
    df_act = clean_gl(file_gl)
    df_fcst = clean_forecast(file_forecast)
    
    if not df_act.empty and not df_fcst.empty:
        master = pd.concat([df_act, df_fcst], ignore_index=True)
        
        if 'Account_Num' in master.columns:
            master['Account_Num'] = master['Account_Num'].apply(clean_id)

        # --- PREPARACIÓN DE MAESTROS (NOMBRES Y PADRES) ---
        # Creamos diccionarios de referencia basados en el Forecast (VML)
        # 1. Parent Account Map
        parent_map = {}
        if not df_fcst.empty and 'Parent_Account' in df_fcst.columns:
            # Usamos dropna para ignorar filas sin parent
            pmap_df = df_fcst[['Account_Num', 'Parent_Account']].dropna().drop_duplicates()
            parent_map = pmap_df.set_index('Account_Num')['Parent_Account'].to_dict()

        # 2. Account Name Map (Prioridad GL > Forecast para nombres)
        best_names = {}
        if not df_fcst.empty:
            fcst_pairs = df_fcst[['Account_Num', 'Account_Name']].dropna().drop_duplicates()
            for _, row in fcst_pairs.iterrows():
                nid = str(row['Account_Num']).strip()
                name = str(row['Account_Name']).strip()
                if name and name.lower() != 'nan' and name != nid:
                    best_names[nid] = name 
        if not df_act.empty:
            gl_pairs = df_act[['Account_Num', 'Account_Name']].dropna().drop_duplicates()
            for _, row in gl_pairs.iterrows():
                nid = str(row['Account_Num']).strip()
                name = str(row['Account_Name']).strip()
                if name and name.lower() != 'nan' and name != nid:
                    best_names[nid] = name

        # --- APLICACIÓN DE MAESTROS ---
        def enrich_master(row):
            acc_id = str(row['Account_Num'])
            
            # Nombre
            final_name = row['Account_Name']
            if acc_id in best_names:
                final_name = best_names[acc_id]
            else:
                curr = str(row['Account_Name']).strip()
                if not curr or curr.lower() == 'nan' or curr == acc_id:
                    final_name = f"Cuenta {acc_id}"
            
            # Parent
            # Si ya tiene parent (viene del forecast), lo deja. Si no, busca en mapa. Si no, "Un-assigned"
            final_parent = "Un-assigned parent account"
            if 'Parent_Account' in row and pd.notna(row['Parent_Account']):
                final_parent = row['Parent_Account']
            elif acc_id in parent_map:
                final_parent = parent_map[acc_id]
                
            return pd.Series([final_name, final_parent])

        master[['Account_Name', 'Parent_Account']] = master.apply(enrich_master, axis=1)
        master['Vendor'] = master['Vendor'].fillna('Vendor uncategorized').replace(['nan', '', 'None'], 'Vendor uncategorized')

        # --- FILTROS ---
        with c_period:
            periods = sorted(master['Period'].dropna().unique())
            if not periods: st.stop()
            p_sel = st.selectbox("Seleccionar Mes:", periods)
            
        df_month = master[master['Period'] == p_sel].copy()
        
        sel_year = p_sel.split('-')[0]
        ytd_periods = [p for p in periods if p.startswith(sel_year) and p <= p_sel]
        df_ytd = master[(master['Period'].isin(ytd_periods)) & (master['Type'] == 'Actuals')].copy()

        with c_vendor:
            all_vends = sorted(df_month['Vendor'].astype(str).unique())
            f_vend = st.multiselect("Filtrar por Vendor:", all_vends)
            if f_vend: 
                df_month = df_month[df_month['Vendor'].isin(f_vend)]
                df_ytd = df_ytd[df_ytd['Vendor'].isin(f_vend)]
            
        with c_filters:
            c1, c2, c3 = st.columns(3)
            games = sorted(df_month['Game_Code'].astype(str).unique())
            f_game = c1.multiselect("Game Code", games)
            
            locs = sorted(df_month['Location'].astype(str).unique())
            f_loc = c2.multiselect("Location", locs)
            
            depts = sorted(df_month['Dept_Code'].astype(str).unique())
            f_dept = c3.multiselect("Department", depts)
            
            df_month['Acc_Display'] = df_month['Account_Num'].astype(str) + " - " + df_month['Account_Name'].astype(str)
            accs = sorted(df_month['Acc_Display'].unique())
            f_acc = st.multiselect("Filtrar Cuentas", accs)

        if f_game: 
            df_month = df_month[df_month['Game_Code'].isin(f_game)]
            df_ytd = df_ytd[df_ytd['Game_Code'].isin(f_game)]
        if f_loc: 
            df_month = df_month[df_month['Location'].isin(f_loc)]
            df_ytd = df_ytd[df_ytd['Location'].isin(f_loc)]
        if f_dept: 
            df_month = df_month[df_month['Dept_Code'].isin(f_dept)]
            df_ytd = df_ytd[df_ytd['Dept_Code'].isin(f_dept)]
        if f_acc: 
            df_ytd['Acc_Display'] = df_ytd['Account_Num'].astype(str) + " - " + df_ytd['Account_Name'].astype(str)
            df_month = df_month[df_month['Acc_Display'].isin(f_acc)]
            df_ytd = df_ytd[df_ytd['Acc_Display'].isin(f_acc)]

        # --- KPI ---
        act = df_month[df_month['Type']=='Actuals']['Amount'].sum()
        fcst = df_month[df_month['Type']=='Forecast']['Amount'].sum()
        var = fcst - act
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Actuals", f"${act:,.0f}")
        k2.metric("Forecast", f"${fcst:,.0f}")
        k3.metric("Variación", f"${var:,.0f}", delta=f"{var:,.0f}")
        
        st.divider()
        st.subheader(f"Análisis por Grupos (Parent Accounts): {p_sel}")
        
        # --- VISTA AGRUPADA (NIVEL 1: PARENT) ---
        # Agrupamos primero por Parent
        parents_view = df_month.groupby(['Parent_Account']).agg(
            Actuals=('Amount', lambda x: x[df_month['Type']=='Actuals'].sum()),
            Forecast=('Amount', lambda x: x[df_month['Type']=='Forecast'].sum())
        ).reset_index()
        parents_view['Variacion'] = parents_view['Forecast'] - parents_view['Actuals']
        parents_view = parents_view.sort_values('Variacion')

        def color_var(val):
            c = '#ff4b4b' if val < 0 else '#28a745'
            return f'color: {c}; font-weight: bold'

        if parents_view.empty:
            st.warning("No hay datos para mostrar.")
        else:
            st.caption("Estructura: Grupo Principal > Cuentas Individuales > Detalle")
            
            # ITERAMOS SOBRE GRUPOS (PARENTS)
            for _, p_row in parents_view.iterrows():
                parent_name = str(p_row['Parent_Account'])
                p_act = p_row['Actuals']
                p_fcst = p_row['Forecast']
                p_var = p_row['Variacion']
                
                # Etiqueta del Grupo
                p_icon = "📂"
                p_color = "green" if p_var >= 0 else "red"
                p_label = (f"{p_icon} **{parent_name}** ┃ "
                           f"Act: :blue[${p_act:,.0f}] ┃ "
                           f"Fcst: :orange[${p_fcst:,.0f}] ┃ "
                           f"Var: :{p_color}[${p_var:,.0f}]")
                
                # NIVEL 1: EXPANDER DEL GRUPO
                with st.expander(p_label, expanded=False):
                    
                    # Filtramos las cuentas de este grupo
                    subset_parent = df_month[df_month['Parent_Account'] == parent_name]
                    
                    # Agrupamos por cuentas individuales dentro del grupo
                    accounts_view = subset_parent.groupby(['Account_Num', 'Account_Name']).agg(
                        Actuals=('Amount', lambda x: x[subset_parent['Type']=='Actuals'].sum()),
                        Forecast=('Amount', lambda x: x[subset_parent['Type']=='Forecast'].sum())
                    ).reset_index()
                    accounts_view['Variacion'] = accounts_view['Forecast'] - accounts_view['Actuals']
                    accounts_view = accounts_view.sort_values('Variacion')
                    
                    # ITERAMOS SOBRE CUENTAS (ACCOUNTS)
                    for _, a_row in accounts_view.iterrows():
                        num = str(a_row['Account_Num'])
                        name = str(a_row['Account_Name'])
                        v_a = a_row['Actuals']
                        v_f = a_row['Forecast']
                        v_v = a_row['Variacion']
                        
                        icon = "🔴" if v_v < 0 else "🟢"
                        c_v = "green" if v_v >= 0 else "red"
                        
                        a_label = (f"{icon} **{name}** :gray[({num})] ┃ "
                                   f"Act: ${v_a:,.0f} ┃ "
                                   f"Fcst: ${v_f:,.0f} ┃ "
                                   f"Var: :{c_v}[${v_v:,.0f}]")
                        
                        # NIVEL 2: EXPANDER DE LA CUENTA (ANIDADO)
                        with st.expander(a_label):
                            # Gráfico YTD
                            sub_ytd = df_ytd[df_ytd['Account_Num'] == num]
                            if not sub_ytd.empty:
                                c_data = sub_ytd.groupby('Period')['Amount'].sum().reset_index()
                                chart = alt.Chart(c_data).mark_bar(color='#2E86C1').encode(
                                    x=alt.X('Period', title='Mes'),
                                    y=alt.Y('Amount', title='Actuals $'),
                                    tooltip=['Period', alt.Tooltip('Amount', format='$,.0f')]
                                ).properties(height=180) # Un poco más chico por estar anidado
                                st.altair_chart(chart, use_container_width=True)
                            
                            st.divider()
                            
                            # Tabla Vendors
                            sub_m = subset_parent[subset_parent['Account_Num'] == num]
                            piv = sub_m.pivot_table(index='Vendor', columns='Type', values='Amount', aggfunc='sum').fillna(0)
                            for c in ['Actuals', 'Forecast']: 
                                if c not in piv: piv[c] = 0
                            piv['Variacion'] = piv['Forecast'] - piv['Actuals']
                            piv = piv[['Actuals', 'Forecast', 'Variacion']].sort_values('Variacion')
                            
                            st.dataframe(piv.style.applymap(color_var, subset=['Variacion']).format("${:,.0f}"), use_container_width=True)
    else:
        st.warning("Esperando archivos...")