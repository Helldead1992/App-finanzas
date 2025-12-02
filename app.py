import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- VISUAL CONFIGURATION ---
st.set_page_config(page_title="Finance App 8.0", layout="wide")
st.title("💰 Financial Analysis: Actuals vs Forecast")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Data Upload")
    file_gl = st.file_uploader("Upload GL (Oracle)", type=["xlsx", "csv"])
    file_forecast = st.file_uploader("Upload Forecast (VML)", type=["xlsx", "csv"])
    
    st.divider()
    st.header("2. Filters")
    c_period = st.container()
    c_vendor = st.container()
    c_filters = st.container()

# --- CLEANING FUNCTIONS ---

def clean_money(val):
    """Clean currency string to float"""
    if pd.isna(val) or val == '': return 0.0
    s = str(val).replace('$', '').replace(',', '').replace(' ', '')
    if '(' in s and ')' in s: s = s.replace('(', '-').replace(')', '')
    try: return float(s)
    except: return 0.0

def clean_id(val):
    """
    Aggressive ID cleaning to ensure matching.
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
        # GL usually has header at row 13 (index 12)
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
        
        for col in ['Account_Num', 'Game_Code', 'Dept_Code', 'Location']:
            if col in df.columns:
                df[col] = df[col].apply(clean_id)
        
        df['Amount'] = df['Amount'].apply(clean_money)
        return df
    except Exception as e:
        st.error(f"GL Error: {e}")
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

        id_vars = [
            'Vendors Nombre', 'Game_Code Código', 'Level Code Código', 
            'Cost_Account_Number Código', 
            'Cost_Account_Number Nombre', 'Studio_Location_Code Código',
            'Parent Account Nombre'
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
            'Parent Account Nombre': 'Parent_Account'
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
        st.error(f"Forecast Error: {e}")
        return pd.DataFrame()

# --- MAIN LOGIC ---

if file_gl and file_forecast:
    df_act = clean_gl(file_gl)
    df_fcst = clean_forecast(file_forecast)
    
    if not df_act.empty and not df_fcst.empty:
        master = pd.concat([df_act, df_fcst], ignore_index=True)
        
        if 'Account_Num' in master.columns:
            master['Account_Num'] = master['Account_Num'].apply(clean_id)

        # --- MASTER DATA SETUP ---
        parent_map = {}
        if not df_fcst.empty and 'Parent_Account' in df_fcst.columns:
            pmap_df = df_fcst[['Account_Num', 'Parent_Account']].dropna().drop_duplicates()
            parent_map = pmap_df.set_index('Account_Num')['Parent_Account'].to_dict()

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

        def enrich_master(row):
            acc_id = str(row['Account_Num'])
            final_name = row['Account_Name']
            if acc_id in best_names:
                final_name = best_names[acc_id]
            else:
                curr = str(row['Account_Name']).strip()
                if not curr or curr.lower() == 'nan' or curr == acc_id:
                    final_name = f"Account {acc_id}"
            
            final_parent = "Un-assigned Parent Account"
            if 'Parent_Account' in row and pd.notna(row['Parent_Account']):
                final_parent = row['Parent_Account']
            elif acc_id in parent_map:
                final_parent = parent_map[acc_id]
                
            return pd.Series([final_name, final_parent])

        master[['Account_Name', 'Parent_Account']] = master.apply(enrich_master, axis=1)
        master['Vendor'] = master['Vendor'].fillna('Vendor Uncategorized').replace(['nan', '', 'None'], 'Vendor Uncategorized')

        # --- FILTERS ---
        with c_period:
            periods = sorted(master['Period'].dropna().unique())
            if not periods: st.stop()
            p_sel = st.selectbox("Select Month:", periods)
            
        df_month = master[master['Period'] == p_sel].copy()
        
        sel_year = p_sel.split('-')[0]
        ytd_periods = [p for p in periods if p.startswith(sel_year) and p <= p_sel]
        df_ytd = master[(master['Period'].isin(ytd_periods)) & (master['Type'] == 'Actuals')].copy()

        with c_vendor:
            all_vends = sorted(df_month['Vendor'].astype(str).unique())
            f_vend = st.multiselect("Filter by Vendor:", all_vends)
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
            f_acc = st.multiselect("Filter Accounts", accs)

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
        k3.metric("Variance", f"${var:,.0f}", delta=f"{var:,.0f}")
        
        st.divider()
        st.subheader(f"Group Analysis (Parent Accounts): {p_sel}")
        
        # --- GROUPED VIEW ---
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
            st.warning("No data to display.")
        else:
            st.caption("Structure: Main Group > Individual Accounts > Detail by Dimensions")
            
            for _, p_row in parents_view.iterrows():
                parent_name = str(p_row['Parent_Account'])
                p_act = p_row['Actuals']
                p_fcst = p_row['Forecast']
                p_var = p_row['Variacion']
                
                p_icon = "📂"
                p_color = "green" if p_var >= 0 else "red"
                p_label = (f"{p_icon} **{parent_name}** ┃ "
                           f"Act: :blue[${p_act:,.0f}] ┃ "
                           f"Fcst: :orange[${p_fcst:,.0f}] ┃ "
                           f"Var: :{p_color}[${p_var:,.0f}]")
                
                with st.expander(p_label, expanded=False):
                    subset_parent = df_month[df_month['Parent_Account'] == parent_name]
                    
                    accounts_view = subset_parent.groupby(['Account_Num', 'Account_Name']).agg(
                        Actuals=('Amount', lambda x: x[subset_parent['Type']=='Actuals'].sum()),
                        Forecast=('Amount', lambda x: x[subset_parent['Type']=='Forecast'].sum())
                    ).reset_index()
                    accounts_view['Variacion'] = accounts_view['Forecast'] - accounts_view['Actuals']
                    accounts_view = accounts_view.sort_values('Variacion')
                    
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
                        
                        with st.expander(a_label):
                            # --- PROFESSIONAL CHART (ALTAIR LAYERED) ---
                            sub_ytd = df_ytd[df_ytd['Account_Num'] == num]
                            if not sub_ytd.empty:
                                c_data = sub_ytd.groupby('Period')['Amount'].sum().reset_index()
                                
                                # 1. Base chart
                                base = alt.Chart(c_data).encode(
                                    x=alt.X('Period', title='Month', axis=alt.Axis(labelAngle=0)), 
                                    y=alt.Y('Amount', title='Actuals ($)'),
                                    tooltip=['Period', alt.Tooltip('Amount', format='$,.0f', title='Amount')]
                                )
                                
                                # 2. Bars
                                bars = base.mark_bar(
                                    color='#0068C9', 
                                    cornerRadiusTopLeft=4,
                                    cornerRadiusTopRight=4
                                )
                                
                                # 3. Data Labels
                                text = base.mark_text(
                                    align='center',
                                    baseline='bottom',
                                    dy=-5,
                                    fontSize=12,
                                    color='black'
                                ).encode(
                                    text=alt.Text('Amount', format='$,.0f')
                                )
                                
                                # 4. Combine
                                chart = (bars + text).properties(height=400).configure_axis(
                                    grid=False,
                                    domain=False,
                                    labelFontSize=12,
                                    titleFontSize=14
                                ).configure_view(
                                    strokeWidth=0
                                )
                                
                                st.altair_chart(chart, use_container_width=True)
                            
                            st.divider()
                            
                            tab_vend, tab_dept, tab_loc, tab_game = st.tabs(["Vendor", "Department", "Location", "Game Code"])
                            
                            sub_m = subset_parent[subset_parent['Account_Num'] == num]
                            
                            with tab_vend:
                                piv = sub_m.pivot_table(index='Vendor', columns='Type', values='Amount', aggfunc='sum').fillna(0)
                                for c in ['Actuals', 'Forecast']: 
                                    if c not in piv: piv[c] = 0
                                piv['Variacion'] = piv['Forecast'] - piv['Actuals']
                                piv = piv[['Actuals', 'Forecast', 'Variacion']].sort_values('Variacion')
                                st.dataframe(piv.style.applymap(color_var, subset=['Variacion']).format("${:,.0f}"), use_container_width=True)

                            with tab_dept:
                                piv_d = sub_m.pivot_table(index='Dept_Code', columns='Type', values='Amount', aggfunc='sum').fillna(0)
                                for c in ['Actuals', 'Forecast']: 
                                    if c not in piv_d: piv_d[c] = 0
                                piv_d['Variacion'] = piv_d['Forecast'] - piv_d['Actuals']
                                piv_d = piv_d[['Actuals', 'Forecast', 'Variacion']].sort_values('Variacion')
                                st.dataframe(piv_d.style.applymap(color_var, subset=['Variacion']).format("${:,.0f}"), use_container_width=True)

                            with tab_loc:
                                piv_l = sub_m.pivot_table(index='Location', columns='Type', values='Amount', aggfunc='sum').fillna(0)
                                for c in ['Actuals', 'Forecast']: 
                                    if c not in piv_l: piv_l[c] = 0
                                piv_l['Variacion'] = piv_l['Forecast'] - piv_l['Actuals']
                                piv_l = piv_l[['Actuals', 'Forecast', 'Variacion']].sort_values('Variacion')
                                st.dataframe(piv_l.style.applymap(color_var, subset=['Variacion']).format("${:,.0f}"), use_container_width=True)

                            with tab_game:
                                piv_g = sub_m.pivot_table(index='Game_Code', columns='Type', values='Amount', aggfunc='sum').fillna(0)
                                for c in ['Actuals', 'Forecast']: 
                                    if c not in piv_g: piv_g[c] = 0
                                piv_g['Variacion'] = piv_g['Forecast'] - piv_g['Actuals']
                                piv_g = piv_g[['Actuals', 'Forecast', 'Variacion']].sort_values('Variacion')
                                st.dataframe(piv_g.style.applymap(color_var, subset=['Variacion']).format("${:,.0f}"), use_container_width=True)
    else:
        st.warning("Waiting for files...")