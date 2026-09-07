import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


df=pd.read_csv(r"D:\sanjai\Tools\Visual Studio Code\Guvi Projects\Project 4\Dataset\Mobile Reviews Sentiment null.csv")

print(pd.set_option("display.max_columns", None))
print(pd.set_option("display.width", None))

# print(df.head())
# print(df.describe().T)

# print(df.isnull().sum())
# print(df.shape)
# print(df['source'])
# print(df['rating'].value_counts(dropna=False))

# df=df.columns.dropna()
df=df.dropna(axis=0)
# print(df.isnull().sum())
df=df.drop_duplicates()
# print(df.shape)
# print(df['rating'])

# print(df['price_local'])

# The regex r'[^\d.]' means "find anything that is NOT a digit (\d) or a decimal point (.)"
# We replace those symbols with nothing (''), and then convert the result to a float.
# df['price_local']=df['price_local'].str.replace(r'[^\d.]', "", regex=True).astype(float)

# print(df['price_local'])

# 1. First, remove any commas just in case some prices are formatted like "1,000.00"
df['price_local']=df["price_local"].str.replace(',','',regex=False)

# 2. Extract the exact numeric pattern (digits, a decimal point, and more digits)
df['price_local']=df['price_local'].str.extract(r'(\d+\.\d+|\d+)')[0].astype(float)

# print(df['price_local'])
# print(df['review_id'].isnull().sum())
# print(df['review_date'])
# print(df['customer_name'].iloc[6013])
# print(df.loc[df['review_id']==6013, 'customer_name'])
df=df[df['verified_purchase']==True]

df=df.drop(columns=['review_date', 'customer_name','review_id','exchange_rate_to_usd','price_local','currency','verified_purchase','age','language'])

df = df.reset_index(drop=True)

sentiment_mapping={'Negative': 0, "Neutral":1, "Positive":2}
df['sentiment']=df['sentiment'].map(sentiment_mapping)
# print(df.shape)
# print(df['sentiment'])

# print(df.info())
categorical_col=df.select_dtypes(include=['object','string']).columns.to_list()
# print(categorical_col)

numerical_col=df.select_dtypes(include=['int64', 'float64']).columns.to_list()
# print(numerical_col)

df_column_transformer=ColumnTransformer(
    transformers=[
        ("OneHotEncoding", OneHotEncoder(sparse_output=False), categorical_col),
        ("StandandScaling", StandardScaler(), numerical_col)
    ]
)


df_scaled=df_column_transformer.fit_transform(df)
# print(df_scaled)
new_column=df_column_transformer.get_feature_names_out()
df_final=pd.DataFrame(df_scaled, columns=new_column)
# print(df_final.head())

# print(df_final.columns.to_list())

# print(df_final['OneHotEncoding__brand_Apple'].unique())


# def Building_Model():
#     ineria_value=[]
#     for i in range(1,7):
#         model=KMeans(n_clusters=i)
#         model.fit(df_final)
#         ineria_value.append(model.inertia_)
#     print(ineria_value)
#     plt.plot(range(1,7),ineria_value)
#     plt.xlabel('Number Of Label')
#     plt.ylabel("Ineria Value")
#     plt.show()
# Building_Model()

# silhouette=[]
# for i in range(2,6):
#     model=KMeans(n_clusters=i)
#     model.fit(df_final)
#     score=silhouette_score(df_final,model.labels_)
#     silhouette.append(score)
#     # print(score)

# plt.plot(range(2,6),silhouette)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Ineria Value')
# plt.show()

model=KMeans(n_clusters=3,random_state=42)
model.fit(df_final)
# model.labels_

# print(model.labels_)

# Create a new column in your original dataframe to store the cluster IDs
df['cluster']=model.labels_

# Take a peek to verify it worked
# print(df[['brand','model','price_usd','cluster']].head())

# df.to_csv("Cleaned_Mobile_Phones_Dataset.csv")
# Calculate the average price and overall rating for each cluster
cluster_profile=df.groupby('cluster')[['price_usd','rating','sentiment','battery_life_rating']].mean()
# print(cluster_profile)

# print(df['verified_purchase'].value_counts().unique())

# def recommend_phones(target_model, top_n=5):
#     # 1. Find the cluster of the phone the user is looking at
#     # (Using .iloc[0] just in case there are multiple reviews for the same model)
#     target_cluster=df[df['model']==target_model]['cluster'].iloc[0]

#     # 2. Filter the dataset to ONLY include other phones in that exact same cluster
#     similar_phones=df[df['cluster']==target_cluster]

#     # 3. Exclude the target phone itself so we don't recommend the exact same item
#     similar_phones=similar_phones[similar_phones['model']!=target_model]

#     # 4. Sort the remaining phones by the best ratings and most helpful votes
#     top_picks=similar_phones.sort_values(by=['rating', 'helpful_votes'], ascending=[False,False])

#     # 5. Drop duplicate models (since your dataset has multiple reviews per phone)
#     top_picks = top_picks.drop_duplicates(subset=['model'])
    
#     # Return the top N recommendations
#     return top_picks[['brand', 'model', 'price_usd', 'rating', 'cluster']].head(top_n)

# # Test your new recommendation engine!
# print(recommend_phones('Realme 12 Pro'))

# def recommend_phones(target_model, top_n=5):
#     # 1. Force the system to ONLY pull from the "Premium Experience" tier
#     best_cluster = 2
    
#     # 2. Filter the dataset to ONLY include phones in that top cluster
#     similar_phones = df[df['cluster'] == best_cluster]
    
#     # 3. Exclude the target phone itself so we don't recommend the exact same item
#     similar_phones = similar_phones[similar_phones['model'] != target_model]
    
#     # 4. Sort the remaining phones by the best ratings and most helpful votes
#     top_picks = similar_phones.sort_values(by=['rating', 'helpful_votes'], ascending=[False, False])
    
#     # 5. Drop duplicate models to provide 5 unique phone suggestions
#     top_picks = top_picks.drop_duplicates(subset=['model'])
    
#     # Return the top N recommendations
#     return top_picks[['brand', 'model', 'price_usd', 'rating', 'cluster']].head(top_n)

# # Test the upgraded recommendation engine!
# print(recommend_phones('Poco X6'))


# def recommend_phones(target_model, top_n=5):
#     # 1. Force the system to ONLY pull from the "Premium Experience" tier
#     best_cluster = 2
    
#     # 2. Filter the dataset to ONLY include phones in that top cluster
#     similar_phones = df[df['cluster'] == best_cluster]
    
#     # 3. Exclude the target phone itself so we don't recommend the exact same item
#     similar_phones = similar_phones[similar_phones['model'] != target_model]
    
#     # 4. Sort the remaining phones by the best ratings and most helpful votes
#     top_picks = similar_phones.sort_values(by=['rating', 'helpful_votes'], ascending=[False, False])
    
#     # 5. Drop duplicate models to provide 5 unique phone suggestions
#     top_picks = top_picks.drop_duplicates(subset=['model'])
    
#     # Return the top N recommendations
#     return top_picks[['brand', 'model', 'price_usd', 'rating', 'cluster']].head(top_n)

# # Test the upgraded recommendation engine!
# print(recommend_phones('Poco X6'))

# print(df.shape)
# print(df_final.shape)
# def recommend_phones(target_model, top_n=5):
#     # 1. Grab the exact row index of the target phone
#     target_idx = df[df['model'] == target_model].index[0]
    
#     # 2. Identify the user's context (Country & Source)
#     target_country = df.loc[target_idx, 'country']
#     target_source = df.loc[target_idx, 'source']
    
#     # 3. Set your Premium Cluster (Based on your latest output, it is 0!)
#     best_cluster = 0 
    
#     # 4. Filter the dataframe by Region, Source, and the Premium Cluster
#     valid_indices = df[
#         (df['country'] == target_country) & 
#         (df['source'] == target_source) & 
#         (df['cluster'] == best_cluster)
#     ].index
    
#     # 5. Extract the perfectly scaled mathematical features for these specific phones
#     target_features = df_final.iloc[[target_idx]]
#     pool_features = df_final.iloc[valid_indices]
    
#     # 6. Calculate Cosine Similarity! 
#     # This prevents the $1200 iPhone from being recommended to a $300 budget user
#     similarity_scores = cosine_similarity(target_features, pool_features)[0]
    
#     # 7. Create a recommendations dataframe and attach the new similarity scores
#     recommendations = df.iloc[valid_indices].copy()
#     recommendations['similarity'] = similarity_scores
    
#     # 8. Remove the target model itself, sort by most similar, and drop duplicates
#     recommendations = recommendations[recommendations['model'] != target_model]
#     recommendations = recommendations.sort_values(by='similarity', ascending=False)
#     recommendations = recommendations.drop_duplicates(subset=['model'])
    
#     # Return the beautiful, context-aware results!
#     return recommendations[['brand', 'model', 'price_usd', 'country', 'source', 'similarity']].head(top_n)

# # Test the upgraded recommendation engine!
# print(recommend_phones('Poco X6'))

# print(df['price_usd'].min()) # 180.07
# print(df['price_usd'].max()) # 1499.75
# print(df['price_usd'].nunique()) # 27114
# 0 to 450, 451 to 950, 951 to 1500

# budget_phones=df[df['price_usd']<450]['model'].head(10)

# print(budget_phones)




# # The correct Pandas syntax for your manual clusters:
# budget_phones = df[df['price_usd'] <= 450]
# mid_range_phones = df[(df['price_usd'] > 450) & (df['price_usd'] <= 950)]
# premium_phones = df[df['price_usd'] > 950]


# # Define the exact price boundaries for your dropdown menu
# bins = [180, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1500]

# # Create string labels for the Streamlit UI
# labels = [f"${bins[i]} to ${bins[i+1]}" for i in range(len(bins)-1)]

# # Create a new column in your DataFrame to power the Streamlit dropdown
# df['budget_range'] = pd.cut(df['price_usd'], bins=bins, labels=labels, include_lowest=True)

# # Max rating is 5, max sentiment is 2. The highest possible goodness score is 7.
# df['goodness_score'] = df['rating'] + df['sentiment']



















































