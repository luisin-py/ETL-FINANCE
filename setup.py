from setuptools import setup, find_packages

setup(
    name="etl_finance",
    version="0.1.0",
    description="ETL and Dashboard for Brazilian Stocks",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "yfinance",
        "pandas",
        "streamlit",
        "plotly",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "etl-finance=etl:run_etl",
        ],
    },
)
