import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Mobile Phone Recommender", layout="wide")

# Data Loading & Preprocessing 
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Mobile_Phones_Dataset.csv")
    
    
    cols_to_drop = ['age', 'language']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 1. Create the Goodness Score
    df['goodness_score'] = df['rating'] + df['sentiment']
    
    # 2. Create Budget Bins
    bins = [0, 180, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1500, 3000]
    labels = [f"${bins[i]} to ${bins[i+1]}" for i in range(len(bins)-1)]
    df['budget_range'] = pd.cut(df['price_usd'], bins=bins, labels=labels, include_lowest=True)
    
    # 3. PREPARE THE DATA FOR COSINE SIMILARITY
    categorical_col = ['brand', 'country', 'source'] 
    numerical_col = ['price_usd', 'rating', 'sentiment', 'battery_life_rating', 'camera_rating', 
                     'performance_rating', 'design_rating', 'display_rating']
    
    df_column_transformer = ColumnTransformer(
        transformers=[
            ("OneHotEncoding", OneHotEncoder(sparse_output=False), categorical_col),
            ("StandardScaling", StandardScaler(), numerical_col)
        ], remainder='drop'
    )
    
    df_scaled = df_column_transformer.fit_transform(df)
    df_final = pd.DataFrame(df_scaled, columns=df_column_transformer.get_feature_names_out())
    
    return df, df_final


df, df_final = load_data()

# UI Layout 
st.title("Intelligent Mobile Phone Recommender")
st.markdown("Find the perfect, highly-rated phone available in your region.")

# Create the Navigation Tabs
tab1, tab2, tab3 = st.tabs(["Recommender", "EDA Dashboard", "Cluster Insights"])

with tab1:
    # Row 1: Context Filters (Location & Platform)
    col1, col2 = st.columns(2)
    with col1:
        countries = sorted(df['country'].dropna().unique())
        selected_country = st.selectbox("Select Your Country", countries)

    with col2:
        sources = sorted(df['source'].dropna().unique())
        selected_source = st.selectbox("Select Platform", sources)

    st.divider()

    # Row 2: Search Strategy
    search_method = st.radio(
        "How would you like to search?", 
        ("Search by Budget", "Find Similar to a Specific Phone")
    )

    # Row 3: Dynamic Inputs based on the user's choice
    if search_method == "Search by Budget":
        # Sort the budget ranges mathematically so they display in the correct order
        budgets = df['budget_range'].dropna().unique().tolist()
        budgets = sorted(budgets, key=lambda x: float(str(x).split(' ')[0].replace('$', '')))
    
        selected_budget = st.selectbox("Select Your Price Range", budgets)
        st.info(f"Looking for phones priced **{selected_budget}** available in **{selected_country}** on **{selected_source}**...")
    
    
    
        # 1. Filter by Region, Source, and the User's chosen Budget
        filtered_df = df[
            (df['country'] == selected_country) & 
            (df['source'] == selected_source) & 
            (df['budget_range'] == selected_budget)
        ]
    
        if filtered_df.empty:
            st.warning("No phones found matching these exact criteria. Try changing the platform or budget!")
        else:
            # 2. Sort by Goodness Score (Rating + Sentiment) descending
            top_picks = filtered_df.sort_values(by=['goodness_score','price_usd'], ascending=[False,True])
            # top_picks = filtered_df.sort_values(by='goodness_score', ascending=False)
        

            # 3. Drop duplicates to show 5 unique phones
            top_picks = top_picks.drop_duplicates(subset=['model']).head(50)
        
            st.success("Here are your top recommendations!")
        
            # 4. Display a clean, professional table in Streamlit
            st.dataframe(
                top_picks[['brand', 'model', 'price_usd', 'rating', 'sentiment', 'goodness_score']].reset_index(drop=True),
                use_container_width=True
            )

    else:
        models = sorted(df['model'].dropna().unique())
        selected_model = st.selectbox("Select a Phone Model", models)
    
        target_data = df[df['model'] == selected_model]
    
        if target_data.empty:
            st.error("Model data not found.")
        else:
            # Grab the exact index of the user's chosen phone
            target_idx = target_data.index[0]
            predicted_budget = target_data.loc[target_idx, 'budget_range']
            st.markdown(f"**Predicted Target Tier:** `{predicted_budget}`")
            st.info(f"Finding top-rated alternatives to the **{selected_model}** in **{selected_country}** on **{selected_source}**...")
        
            # 1. HARD GATING: Filter by Country, Platform, AND Budget Tier FIRST!
            valid_indices = df[
                (df['country'] == selected_country) & 
                (df['source'] == selected_source) & 
                (df['budget_range'] == predicted_budget) &
                (df['model'] != selected_model) 
            ].index
        
            if len(valid_indices) == 0:
                st.warning(f"No direct alternatives found in {selected_country} on {selected_source} within {predicted_budget}.")
            else:
                # 2. Extract perfectly scaled features ONLY for the phones that passed the Hard Gate
                target_features = df_final.iloc[[target_idx]]
                pool_features = df_final.iloc[valid_indices]
            
                # 3. COSINE SIMILARITY: Find the closest feature matches within the strict budget pool
                similarity_scores = cosine_similarity(target_features, pool_features)[0]
            
                # 4. Attach scores to our hard-gated candidates
                recommendations = df.iloc[valid_indices].copy()
                recommendations['similarity_score'] = similarity_scores
            
                # 5. Sort by highest similarity FIRST, then highest goodness score for tie-breakers, then cheapest price
                top_alts = recommendations.sort_values(
                    by=['similarity_score', 'goodness_score', 'price_usd'], 
                    ascending=[False, False, True]
                )
            
                # 6. Drop duplicates to show 5 unique phone alternatives
                top_alts = top_alts.drop_duplicates(subset=['model']).head(50)
            
                st.success("Here are the most mathematically similar alternatives matching your exact budget!")
            
                # 7. Display the beautifully sorted results table
                st.dataframe(
                    top_alts[['brand', 'model', 'price_usd', 'rating', 'goodness_score', 'similarity_score']].reset_index(drop=True),
                    use_container_width=True
                )


with tab2:
    st.header("Exploratory Data Analysis")
    
    
    st.subheader("Product Distribution by Brand")
    brand_counts = df['brand'].value_counts().reset_index()
    fig_brand = px.bar(brand_counts, x='brand', y='count', color='brand', title="Number of Reviews per Brand", text_auto=True)
    # 2. Push the text labels to the top of the bars
    fig_brand.update_traces(textposition='outside')
    st.plotly_chart(fig_brand, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        # Requirement: Identify patterns across countries
        country_counts = df['country'].value_counts().reset_index()
        fig_country = px.pie(country_counts, names='country', values='count', title="Market Share by Country")
        st.plotly_chart(fig_country, use_container_width=True)

    with colB:
        # Requirement: Explore relationships between price and ratings
        fig_scatter = px.scatter(
            df, x='price_usd', y='rating', color='brand', 
            title="Price vs. Overall Rating", opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Requirement: Identify trends and correlations using Seaborn Heatmap
    st.subheader("Feature Correlation Heatmap")
    st.markdown("This heatmap reveals a massive business insight: **Price has almost zero correlation with overall user ratings (0.001).** Instead, Camera, Battery, and Performance ratings heavily dictate user satisfaction (0.76 correlation).")
    
    numeric_df = df[['price_usd', 'rating', 'battery_life_rating', 'camera_rating', 'performance_rating']]
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

with tab3:
    st.header("Market Segmentation Analysis")
    st.markdown("Visualizing K-Means clusters to identify distinct groups of products.")
    
    # Map the numerical cluster IDs back to clear business tiers
    cluster_mapping = {0: "Premium Segment", 1: "Mid-Range Segment", 2: "Budget Segment"}
    df['cluster_label'] = df['cluster'].map(cluster_mapping)
    
    # Requirement: Visualize clusters using plots
    cluster_stats = df.groupby('cluster_label')[['price_usd', 'rating', 'battery_life_rating']].mean().reset_index()
    
    colC, colD = st.columns(2)
    with colC:
        # 1. Format the price as currency ($.2f)
        fig_cluster_price = px.bar(
            cluster_stats, x='cluster_label', y='price_usd', 
            color='cluster_label', title="Average Price per Segment",
            text_auto='$.2f'
        )
        fig_cluster_price.update_traces(textposition='outside')
        st.plotly_chart(fig_cluster_price, use_container_width=True)
        
    with colD:
        # 2. Format the rating to 2 decimal places (.2f)
        fig_cluster_rating = px.bar(
            cluster_stats, x='cluster_label', y='rating', 
            color='cluster_label', title="Average Rating per Segment",
            text_auto='.2f'
        )
        fig_cluster_rating.update_traces(textposition='outside')
        st.plotly_chart(fig_cluster_rating, use_container_width=True)








































