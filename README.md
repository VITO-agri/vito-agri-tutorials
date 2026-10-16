# vito-agri-tutorials

Hands-on tutorials for agricultural monitoring.

## Getting started

1. Open your notebook service, such as
   [CDSE JupyterLab](https://jupyterhub.dataspace.copernicus.eu/), and create a
   notebook using an existing Python kernel (Python 3.11 or newer).

2. Install the vito-agri-tutorials package by running this cell with **Shift + Enter** and wait for it to finish:

   ```python
   %pip install --quiet --force-reinstall --no-deps "vito-agri-tutorials @ https://github.com/VITO-agri/vito-agri-tutorials/archive/refs/heads/main.zip"
   ```

3. See which tutorials are available, grouped by category:

   ```python
   from vito_agri_tutorials.tutorials import discover_tutorials

   discover_tutorials()
   ```

| Category | Tutorial names | CDSE python kernel |
| --- | --- | --- |
| `drought` | `ndvi-anomalies`, `asis`, `wapor`, `vici` | Follow the notebook instructions |
| `parcel` | `parcel-preprocessing`, `parcel-phenology`, `parcel-croptype` | Geo science |
| `openeo` | `sentinel2-retrieval` | OpenEO |
| `timeseries_decomposition` | `timeseries-visualisation`, `timeseries-preprocessing`, `timeseries-decomposition`, `timeseries-change` | Geo science |


4. Choose a tutorial from the overview and save a copy to your jupyterlab working environment using the following code snippet.

   NOTE that the example below is tuned to the CDSE jupyterlab environment. You may need to change your "storage" path if you are working on another system. 

   Replace `ndvi-anomalies` with any name in the overview above to choose another
   tutorial. Use the same storage folder for each.

   ```python
   from pathlib import Path
   from vito_agri_tutorials.tutorials import copy_tutorial

   storage = Path.home() / "mystorage"
   if not storage.is_dir():
       raise FileNotFoundError("Set storage to your notebook service's storage folder.")
   notebook = copy_tutorial("ndvi-anomalies", storage)
   print(notebook)
   ```

   

5. Open the copied notebook in **mystorage > vito_agri_tutorials > category > xxx.ipynb**

   Follow the instructions in the tutorial for choosing a Python kernel and installing all its requirements. Your existing files are never overwritten. When returning to an exercise, open your saved notebook and follow its setup instructions again; you do not need to copy it again. If you want to start fresh, choose a different destination when copying the tutorial (see previous step).