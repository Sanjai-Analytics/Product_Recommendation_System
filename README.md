# Product_Recommendation_System
📱 Intelligent Mobile Phone Segmentation & Recommendation System
📖 Project Overview
The smartphone market generates massive volumes of unstructured data, leaving consumers overwhelmed and standard recommendation systems frequently suggesting out-of-budget flagship devices.

This project is a complete end-to-end data analytics and machine learning pipeline that cleans, segments, and analyzes mobile market data. It features a robust, context-aware recommendation engine that uses strict business logic (Hard Gating) combined with mathematical distance metrics (Cosine Similarity) to suggest highly rated, perfectly budgeted phone alternatives. The final solution is deployed as an interactive web application using Streamlit.

✨ Key Features
Intelligent Recommendation Engine: Utilizes Cosine Similarity on scaled hardware specifications to find the closest feature matches, strictly confined within the user's explicit geographic, platform, and budget constraints.

Custom Ranking Metric: Engineered a goodness_score by fusing user ratings and mapped sentiment to ensure the highest-quality, lowest-priced phones break any mathematical similarity ties.

Unsupervised Market Segmentation: Applied K-Means clustering (k=3) to mathematically categorize the mobile landscape into distinct Premium, Mid-Range, and Budget segments.

Interactive EDA Dashboard: Integrated Plotly and Seaborn directly into the Streamlit UI to dynamically visualize market share, brand distributions, and feature correlations.

Context-Aware Filtering: Allows users to dynamically search by strict $100 price tiers or input a specific target phone to find direct hardware competitors.

🛠️ Technology Stack
Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (K-Means, Cosine Similarity, ColumnTransformer, StandardScaler, OneHotEncoder)

Data Visualization: Plotly Express, Seaborn, Matplotlib

Web Deployment: Streamlit

⚙️ Machine Learning Pipeline
Data Cleaning & Preprocessing: Filtered ~31,000 verified mobile reviews. Removed noisy, redundant features (local currency, age, language) to prevent index misalignment and model bias.

Feature Engineering: Mapped categorical sentiment to numeric values and combined it with product ratings. Used ColumnTransformer to encode and scale mixed data types seamlessly.

Clustering & Analysis: Deployed K-Means to identify distinct groups. Generated business insights proving that hardware features (Camera, Battery) dictate user satisfaction far more than absolute price (0.001 correlation).

Similarity Modeling: Applied a hybrid recommendation approach. Sliced the dataset via hard business rules (Location, Platform, Budget) before applying Cosine Similarity to the remaining feature vectors.
