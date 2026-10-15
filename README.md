# vito-agri-tutorials

Hands-on tutorials for agricultural monitoring.

## Run the NDVI anomaly tutorial on CDSE

You only need your CDSE account and a web browser. There is no need to download
or upload any files.

### 1. Open CDSE notebooks

1. Log in to [CDSE JupyterLab](https://jupyterhub.dataspace.copernicus.eu/).
2. Choose the **Small** server.
3. Create a new notebook using **OpenEO**.

### 2. Get the tutorial (first time only)

Paste this into a notebook cell and run it with **Shift + Enter**. Wait for it
to finish; the download may take a few minutes.

```python
%pip install "vito-agri-tutorials @ https://github.com/VITO-agri/vito-agri-tutorials/archive/refs/heads/main.zip"
```

Run the following in a new cell to save the tutorial in `mystorage`:

```python
from pathlib import Path
from vito_agri_tutorials.tutorials import copy_tutorial

storage = Path.home() / "mystorage"
if not storage.is_dir():
    raise FileNotFoundError(
        "Locate CDSE's existing mystorage folder and set storage to its absolute path."
    )
notebook = copy_tutorial("ndvi", storage / "vito-agri" / "ndvi")
print(notebook)
```

In the file browser, open `mystorage` > `vito-agri` > `ndvi`, then open
`1_ndvi_anomalies.ipynb`. Select **OpenEO** if asked.

If the tutorial already exists, open your saved copy instead. This step will
not overwrite your work.

### 3. Follow the tutorial

1. Run the first code cell to install what the exercise needs.
2. If prompted to restart, choose **Kernel > Restart Kernel**, then continue
   with the next cell.
3. Run the remaining cells one at a time with **Shift + Enter** and follow the
   instructions: select a point, download its NDVI data, and explore anomalies.
4. Save your notebook when finished.

Keep your notebook inside `mystorage` so that it and your results are saved
between sessions. Download a backup of any work you want to keep long-term.

### Coming back later

Open your saved tutorial in `mystorage` and select **OpenEO**. Run its first
code cell again, then follow the exercise. You do not need to get another copy.

### If something goes wrong

- **Installation shows an error:** stop and share the error with your instructor.
- **The map does not appear:** check that **OpenEO** is selected. If it still
  does not work, ask your instructor for help.
- **Asked to sign in:** follow the link below the cell, sign in with your CDSE
  account, and approve access.

## Other tutorials

The steps above currently apply only to the NDVI anomaly tutorial. For other
tutorials, follow the instructions in their notebooks.
