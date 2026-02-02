import streamlit as st
import pandas as pd
import joblib
import os

# Premium Page Configuration
st.set_page_config(
    page_title="KANCHOP Pokemon Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #FF5555;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #333333;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title using columns for layout
col_title, col_logo = st.columns([3, 1])
with col_title:
    st.title("⚡ KANCHOP Pokemon Analytics")
    st.markdown("### Découvrez la Puissance des Données bien traitées")

# Load Data
@st.cache_data
def get_data():
    file_path = 'Pokemon Data_cleaned.csv'
    # Fallback if cleaned data doesn't exist, though it should for consistency
    if not os.path.exists(file_path):
        st.error(f"Fichier {file_path} introuvable. Veuillez exécuter train.py ou preprocessing.py d'abord.")
        return None
    return pd.read_csv(file_path)

try:
    df = get_data()
    
    if df is not None:
        # --- Sidebar Filters ---
        # --- Sidebar Filters ---
        st.sidebar.title("🔍 Recherche & Filtres")
        
        with st.sidebar.expander("Afficher / Masquer les filtres", expanded=True):
            name_search = st.text_input("Recherche par Nom", placeholder="Pikachu...")
            
            generations = st.multiselect(
                "Génération", 
                options=sorted(df['Generation'].unique()),
                default=sorted(df['Generation'].unique())
            )
            
            types = st.multiselect(
                "Type Principal",
                options=sorted(df['Type_1'].unique()),
                default=sorted(df['Type_1'].unique())
            )
            
            is_legendary = st.checkbox("Afficher les Légendaires Uniquement", False)
        
        # --- Filtering Logic ---
        filtered_df = df.copy()
        
        if name_search:
            filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False)]
            
        filtered_df = filtered_df[filtered_df['Generation'].isin(generations)]
        filtered_df = filtered_df[filtered_df['Type_1'].isin(types)]
        
        if is_legendary:
            filtered_df = filtered_df[filtered_df['isLegendary']]

        # --- Main Layout with Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Tableau de Bord", "📈 Analyse", "📝 Données Brutes", "🔮 Prédiction"])
        
        with tab1:
            st.markdown("#### Métriques Clés")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pokémon", len(filtered_df))
            avg_attack = round(filtered_df['Attack'].mean(), 1) if not filtered_df.empty else 0
            col2.metric("Attaque Moy.", avg_attack)
            avg_defense = round(filtered_df['Defense'].mean(), 1) if not filtered_df.empty else 0
            col3.metric("Défense Moy.", avg_defense)
            top_type = filtered_df['Type_1'].mode()[0] if not filtered_df.empty else "N/A"
            col4.metric("Type le plus courant", top_type)
            
            st.divider()
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Distribution des Types")
                st.caption("💡 Astuce : Scrollez pour zoomer")
                if not filtered_df.empty:
                    type_counts = filtered_df['Type_1'].value_counts()
                    st.bar_chart(type_counts, color="#FF5555")
                else:
                    st.info("Aucune donnée à afficher.")
            
            with col_chart2:
                st.subheader("Top 5 des plus puissants (Stats Totales)")
                if not filtered_df.empty:
                    top_5 = filtered_df.nlargest(5, 'Total')[['Name', 'Total']].set_index('Name')
                    st.bar_chart(top_5, horizontal=True) # default color

        with tab2:
            st.subheader("Corrélation des Stats")
            st.caption("💡 Astuce : Scrollez pour zoomer")
            stat_x = st.selectbox("Axe X", ['Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'HP'], index=0)
            stat_y = st.selectbox("Axe Y", ['Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'HP'], index=1)
            
            if not filtered_df.empty:
                st.scatter_chart(
                    filtered_df,
                    x=stat_x,
                    y=stat_y,
                    color='Type_1',
                    size='Total',
                    height=500
                )
        
        with tab3:
            st.markdown(f"**Affichage de {len(filtered_df)} enregistrements**")
            st.dataframe(filtered_df, use_container_width=True)

        with tab4:
            st.subheader("🔮 Prédire la Puissance")
            st.info("Utilise un modèle Random Forest pour estimer les stats 'Totales' basé sur les caractéristiques.")
            
            model_path = 'pokemon_model.pkl'
            
            if os.path.exists(model_path):
                # Input Form
                with st.form("prediction_form"):
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        p_type_1 = st.selectbox("Type Principal", options=sorted(df['Type_1'].unique()))
                        p_type_2 = st.selectbox("Type Secondaire", options=['None'] + sorted(df[df['Type_2'].notnull()]['Type_2'].unique()))
                        p_gen = st.slider("Génération", 1, 6, 1)
                        p_legendary = st.checkbox("Légendaire ?")
                        
                    with col_p2:
                        p_color = st.selectbox("Couleur", options=sorted(df['Color'].unique()))
                        p_body = st.selectbox("Forme du Corps", options=sorted(df['Body_Style'].unique()))
                        p_height = st.number_input("Taille (m)", min_value=0.1, value=1.0)
                        p_weight = st.number_input("Poids (kg)", min_value=0.1, value=10.0)
                        
                    submit_btn = st.form_submit_button("Prédire la Puissance Totale")
                
                if submit_btn:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'Type_1': [p_type_1],
                        'Type_2': [p_type_2],
                        'Generation': [p_gen],
                        'isLegendary': [p_legendary],
                        'Color': [p_color],
                        'Height_m': [p_height],
                        'Weight_kg': [p_weight],
                        'Body_Style': [p_body]
                    })
                    
                    try:
                        model = joblib.load(model_path)
                        prediction = model.predict(input_data)[0]
                        st.success(f"Puissance Totale Estimée : **{int(prediction)}**")
                        
                        # Contextualize
                        st.progress(min(int(prediction)/800, 1.0)) # 780 is roughly max total (Mewtwo Y)
                        if prediction > 600:
                            st.caption("Wow ! C'est un statut légendaire !")
                        elif prediction < 300:
                            st.caption("Un peu faible... peut-être a-t-il besoin d'évoluer ?")
                            
                    except Exception as e:
                        st.error(f"Échec de la prédiction : {e}")
            else:
                st.warning("Fichier modèle 'pokemon_model.pkl' introuvable. Veuillez d'abord entraîner le modèle.")

    else:
        st.error("Impossible de charger les données.")

except Exception as e:
    st.error(f"Une erreur est survenue : {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; font-style: italic; color: #555;'>
        🚀 Merci <b>Mr. Abdouraman</b> de la part d'un jeune qui voit la data différemment 💡🔥
    </div>
    """, 
    unsafe_allow_html=True
)
