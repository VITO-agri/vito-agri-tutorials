# vito-agri-tutorials

Hands-on tutorials for agricultural monitoring.

## Getting started

1. Open your notebook service, such as
   [CDSE JupyterLab](https://jupyterhub.dataspace.copernicus.eu/), and create a
   notebook using an existing Python kernel (Python 3.11 or newer).

2. Run this cell with **Shift + Enter** and wait for it to finish:

   ```python
   %pip install --quiet --force-reinstall --no-deps "vito-agri-tutorials @ https://github.com/VITO-agri/vito-agri-tutorials/archive/refs/heads/main.zip"
   ```

3. See which tutorials are available, grouped by category:

   ```python
   from vito_agri_tutorials.tutorials import discover_tutorials

   discover_tutorials()
   ```

4. Choose a tutorial from the overview and save a copy:

   ```python
   from pathlib import Path
   from vito_agri_tutorials.tutorials import copy_tutorial

   storage = Path.home() / "mystorage"
   if not storage.is_dir():
       raise FileNotFoundError("Set storage to your notebook service's storage folder.")
   notebook = copy_tutorial("ndvi-anomalies", storage)
   print(notebook)
   ```

   Replace `ndvi-anomalies` with `asis`, `wapor`, or `vici` to choose another
   tutorial. Use the same storage folder for each.

5. Open the copied notebook in **mystorage > vito_agri_tutorials > drought**.
   Follow its instructions for choosing a kernel, preparing the exercise, and
   accessing the data. Some tutorials need a particular service or account.

Your existing files are never overwritten. When returning to an exercise, open
your saved notebook and follow its setup instructions again; you do not need to
copy it again.
