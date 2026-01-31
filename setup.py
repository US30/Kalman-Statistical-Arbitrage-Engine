import os

def create_structure_current_dir():
    # Define the subdirectories to create in the current folder
    folders = [
        "data",
        "notebooks",
        "src",
        "tests",
        "logs"
    ]
    
    # Define files to create with their initial content
    files = {
        "./README.md": "# Kalman-Statistical-Arbitrage-Engine\nM.Tech Project: Adaptive Pairs Trading using Kalman Filters.",
        "./requirements.txt": "numpy\npandas\nyfinance\nstatsmodels\nmatplotlib\nseaborn\npykalman\nbacktrader",
        "./main.py": "",
        "./src/__init__.py": "",
        "./src/data_loader.py": "",
        "./src/stats_tests.py": "",
        "./src/kalman.py": "",
        "./src/strategy.py": "",
    }

    print("🚀 Initializing project in the current directory...")

    # 1. Create Subdirectories
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")
        else:
            print(f"Directory already exists: {folder}")

    # 2. Create Files
    for file_path, content in files.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")
            
    # 3. Create environment.yml
    yml_content = """name: kalman-statarb
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn
  - statsmodels
  - yfinance
  - pip
  - pip:
    - pykalman
    - backtrader
    - jupyterlab
"""
    # Write environment.yml
    if not os.path.exists("./environment.yml"):
        with open("./environment.yml", "w") as f:
            f.write(yml_content)
        print("Created file: environment.yml")
    else:
        print("File already exists: environment.yml")

    print("\n✅ Project structure setup complete.")
    print("Next Step: Run 'conda env create -f environment.yml' to install dependencies.")

if __name__ == "__main__":
    create_structure_current_dir()