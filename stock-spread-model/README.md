# Private Market Stock Price prediction - SpaceX Bid-Ask Spread Modeling

This repo contains a structured ML project for modeling bid-ask spread for SpaceX private market stock. In the project, I focused heavily on feature engineering (with an eye toward modeling goals/outcomes), code modularity, and developing a clear structure for a full-stack ML Engineering project which can grow with layers of sophistication, complexity - and, importantly, elegance and maintainability.

But, for now, the main goal is just to get the Jupyter notebooks working :smile:

So the core of the code is contained in `notebooks/analysis.ipynb` and the `src` folder.

The `src` folder contains separate files for:
- ETL
- feature engineering
- modeling
- (eventual) data engineering pipeline orchestrated via Dagster 
- (eventual) data + model versioning pipeline orchestrated via DVC 

I'm leaning toward just DVC for everything right now (since it does everything - Data, Model, Experiment management); or Dagster for data pipelines (because it's better for that - and for scale) and DVC for model versioning. 

The basic `templates` for other (eventual) project pluses are also included, though they are also still works in progress (WIP): 
- Docker files for packaging the project
- Poetry files for package versioning
- tests for unit testing, etc
- pre-commit hooks for linting, etc
- Streamlit app for data and model prediction visualization

**Future Areas for Improvement - Engineering**
- Streamlit app
- git pre-committ hooks
- Docker
- Poetry
- DVC
- Dagster

**Future Areas for Improvement - Data Science**
- Use all the models already in code
- Feature selection
- Hyperparameter tuning
- Whatever you said re: more data / feature engineering / modeling in the `analysis.ipynb` questions section

**Future Areas for Improvement - Public-Facing**
- Look into PyPI
- Look into supplementing with FinGPT
