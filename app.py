import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f5f7fb;}
h1, h2, h3 {color: #123047;}
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Job Market Analysis & Recommendation System")
st.caption("Interactive job intelligence, demand forecasting and personalized recommendations")

# ---------------- DATA LOADING ----------------

@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip().str.lower()

    aliases = {
        "title": ["job_title", "position", "name"],
        "country": ["location", "job_country"],
        "published_date": ["date", "posted_date", "created_at"],
        "hourly_low": ["min_hourly_rate", "hourly_min"],
        "hourly_high": ["max_hourly_rate", "hourly_max"],
        "budget": ["project_budget", "fixed_budget"],
        "is_hourly": ["hourly", "hourly_job"],
        "link": ["url", "job_url"]
    }

    for standard, alternatives in aliases.items():
        if standard not in df.columns:
            for alternative in alternatives:
                if alternative in df.columns:
                    df.rename(columns={alternative: standard}, inplace=True)
                    break

    return df


uploaded_file = st.sidebar.file_uploader(
    "Upload job_market.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your job_market.csv dataset to get started.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()

if "title" not in df.columns:
    st.error("Dataset must contain a job title column.")
    st.stop()

# ---------------- CLEANING ----------------

df = df.drop_duplicates().copy()

if "published_date" in df.columns:
    df["published_date"] = pd.to_datetime(
        df["published_date"], errors="coerce"
    )

for col in ["hourly_low", "hourly_high", "budget"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "country" not in df.columns:
    df["country"] = "Unknown"

df["country"] = df["country"].fillna("Unknown").astype(str)

if "is_hourly" not in df.columns:
    df["is_hourly"] = False

# ---------------- CATEGORY CLASSIFICATION ----------------

def classify_category(title):
    title = str(title).lower()

    categories = {
        "Data Science": ["data scientist", "machine learning", "ml engineer"],
        "Data Analytics": ["data analyst", "analytics", "business intelligence"],
        "Software Development": ["developer", "software", "programmer"],
        "Web Development": ["web developer", "frontend", "backend", "react"],
        "AI & Deep Learning": ["artificial intelligence", "deep learning", "llm"],
        "Design": ["designer", "ui", "ux", "graphic"],
        "Marketing": ["marketing", "seo", "social media"],
        "Other": []
    }

    for category, keywords in categories.items():
        if any(word in title for word in keywords):
            return category

    return "Other"


df["category"] = df["title"].apply(classify_category)

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("🔎 Job Filters")

countries = sorted(df["country"].unique())

selected_countries = st.sidebar.multiselect(
    "Select Country",
    countries,
    default=countries
)

categories = sorted(df["category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

filtered = df[
    df["country"].isin(selected_countries) &
    df["category"].isin(selected_categories)
].copy()

if "budget" in filtered.columns:
    budget_values = filtered["budget"].dropna()

    if not budget_values.empty:
        minimum = float(budget_values.min())
        maximum = float(budget_values.max())

        if minimum < maximum:
            budget_range = st.sidebar.slider(
                "Project Budget Range",
                min_value=minimum,
                max_value=maximum,
                value=(minimum, maximum)
            )

            filtered = filtered[
                filtered["budget"].isna() |
                filtered["budget"].between(*budget_range)
            ]

# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Overview",
        "Demand & Forecasting",
        "Rates & Budgets",
        "Job Recommendations",
        "Export & Data Quality"
    ]
)

# ---------------- MARKET OVERVIEW ----------------

if page == "Market Overview":

    st.header("Market Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Jobs", f"{len(filtered):,}")
    c2.metric("Countries", filtered["country"].nunique())
    c3.metric("Job Categories", filtered["category"].nunique())

    hourly_count = (
        filtered["is_hourly"].astype(str).str.lower()
        .isin(["true", "1", "yes"]).sum()
    )

    c4.metric("Hourly Jobs", f"{hourly_count:,}")

    col1, col2 = st.columns(2)

    with col1:
        category_data = filtered["category"].value_counts().reset_index()
        category_data.columns = ["Category", "Jobs"]

        fig = px.pie(
            category_data,
            names="Category",
            values="Jobs",
            title="Jobs by Category",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_data = (
            filtered["country"].value_counts()
            .head(15).reset_index()
        )
        country_data.columns = ["Country", "Jobs"]

        fig = px.bar(
            country_data,
            x="Country",
            y="Jobs",
            title="Top 15 Countries",
            color="Jobs"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Job Listings")
    st.dataframe(filtered.head(100), use_container_width=True)

# ---------------- DEMAND FORECAST ----------------

elif page == "Demand & Forecasting":

    st.header("📈 Job Demand Trends & Forecasting")

    if "published_date" not in filtered.columns:
        st.warning("A published_date column is required for forecasting.")

    else:
        dated = filtered.dropna(subset=["published_date"]).copy()

        if dated.empty:
            st.warning("No valid publication dates found.")
        else:
            daily = (
                dated.set_index("published_date")
                .resample("D").size()
                .rename("jobs")
                .reset_index()
            )

            fig = px.line(
                daily,
                x="published_date",
                y="jobs",
                title="Daily Job Posting Trends"
            )

            st.plotly_chart(fig, use_container_width=True)

            if len(daily) >= 30:
                daily["day_number"] = np.arange(len(daily))

                X = daily[["day_number"]]
                y = daily["jobs"]

                model = RandomForestRegressor(
                    n_estimators=150,
                    random_state=42
                )

                model.fit(X, y)

                future_days = 30

                future_X = pd.DataFrame({
                    "day_number": np.arange(
                        len(daily),
                        len(daily) + future_days
                    )
                })

                predictions = model.predict(future_X)
                predictions = np.maximum(predictions, 0)

                future_dates = pd.date_range(
                    daily["published_date"].max()
                    + pd.Timedelta(days=1),
                    periods=future_days
                )

                forecast = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Jobs": predictions
                })

                st.subheader("30-Day Job Demand Forecast")

                fig = px.line(
                    forecast,
                    x="Date",
                    y="Predicted Jobs",
                    title="Predicted Job Demand"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "Download Forecast CSV",
                    forecast.to_csv(index=False),
                    "job_demand_forecast.csv",
                    "text/csv"
                )

            else:
                st.info("At least 30 days of data are required for forecasting.")

# ---------------- RATE ANALYSIS ----------------

elif page == "Rates & Budgets":

    st.header("💰 Hourly Rate & Project Budget Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if "hourly_low" in filtered.columns:
            fig = px.histogram(
                filtered,
                x="hourly_low",
                nbins=30,
                title="Minimum Hourly Rate Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "hourly_high" in filtered.columns:
            fig = px.histogram(
                filtered,
                x="hourly_high",
                nbins=30,
                title="Maximum Hourly Rate Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    if "budget" in filtered.columns:
        budget_data = filtered.dropna(subset=["budget"])

        if not budget_data.empty:
            st.metric(
                "Average Project Budget",
                f"{budget_data['budget'].mean():,.2f}"
            )

            fig = px.box(
                budget_data,
                x="category",
                y="budget",
                title="Project Budget by Category"
            )

            st.plotly_chart(fig, use_container_width=True)

# ---------------- RECOMMENDATIONS ----------------

elif page == "Job Recommendations":

    st.header("🤖 Personalized Job Recommendations")

    user_skills = st.text_input(
        "Enter your skills",
        placeholder="Python, SQL, Machine Learning, Power BI"
    )

    top_n = st.slider("Number of Recommendations", 5, 50, 10)

    if st.button("Find Matching Jobs"):

        if not user_skills.strip():
            st.warning("Please enter your skills.")
        else:
            recommendation_df = filtered.copy()

            recommendation_df["title"] = (
                recommendation_df["title"].fillna("")
            )

            text_columns = ["title", "category"]

            for col in ["description", "skills"]:
                if col in recommendation_df.columns:
                    text_columns.append(col)

            recommendation_df["job_text"] = (
                recommendation_df[text_columns]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
            )

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000
            )

            try:
                job_matrix = vectorizer.fit_transform(
                    recommendation_df["job_text"]
                )

                user_vector = vectorizer.transform([user_skills])

                similarity = cosine_similarity(
                    user_vector,
                    job_matrix
                ).flatten()

                recommendation_df["Match Score"] = (
                    similarity * 100
                ).round(2)

                recommendations = (
                    recommendation_df
                    .sort_values("Match Score", ascending=False)
                    .head(top_n)
                )

                st.success("Recommendations generated!")

                display_columns = [
                    col for col in [
                        "title", "country", "category",
                        "budget", "hourly_low",
                        "hourly_high", "Match Score", "link"
                    ]
                    if col in recommendations.columns
                ]

                st.dataframe(
                    recommendations[display_columns],
                    use_container_width=True
                )

                st.download_button(
                    "Download Recommendations",
                    recommendations.to_csv(index=False),
                    "job_recommendations.csv",
                    "text/csv"
                )

            except ValueError:
                st.warning("Not enough job information to generate recommendations.")

# ---------------- EXPORT ----------------

elif page == "Export & Data Quality":

    st.header("📥 Export & Data Quality")

    st.metric("Total Filtered Records", len(filtered))

    st.subheader("Missing Values")

    missing = filtered.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(missing, use_container_width=True)

    st.download_button(
        "Download Filtered Job Dataset",
        filtered.to_csv(index=False),
        "filtered_job_market.csv",
        "text/csv"
    )

    st.download_button(
        "Download Complete Analysis Report",
        filtered.describe(include="all").to_csv(),
        "job_market_summary.csv",
        "text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Job Market Intelligence | Streamlit")