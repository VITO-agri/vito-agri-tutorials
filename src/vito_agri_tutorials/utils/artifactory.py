from loguru import logger
import requests
import time
from pathlib import Path

ARTIFACTORY_BASE_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/vito_agri/tutorials/"
)


def _run_request(method: str, url: str, **kwargs) -> requests.Response:
    """Run an HTTP request with retries and return the response.
    Parameters
    ----------
    method : str
        HTTP method to be used
    url : str
        URL to send the request to
    kwargs : dict
        Additional keyword arguments, may include `retries`, `wait` and `logging_msg`
    Raises
    ------
    RuntimeError
        if the command fails after all retries
    Returns
    -------
    requests.Response
        The response of the http request
    """
    retries = kwargs.pop("retries", 3)
    wait = kwargs.pop("wait", 2)
    logging_msg = kwargs.pop("logging_msg", "Request")

    for attempt in range(retries):
        try:
            logger.debug(f"{logging_msg} (Attempt {attempt + 1})")
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            logger.debug("Execution successful")
            return response
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                logger.error(f"Failed to execute request: {url}")
                raise
    raise RuntimeError(f"Failed to execute request: {url}")


def download_file(
    dstpath: Path,
    srcpath: str,
    retries=3,
    wait=2,
    overwrite=False,
) -> Path:
    """Download a file from Artifactory.
    Parameters
    ----------
    dstpath : Path
        Folder where the legend needs to be downloaded to.
    srcpath : str
        Full path to the file in Artifactory.
    retries : int, optional
        Number of retries, by default 3
    wait : int, optional
        Seconds to wait in between retries, by default 2
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False
    Returns
    -------
    Path
        Path to the downloaded file.
    """

    # Construct target path
    dstpath.mkdir(parents=True, exist_ok=True)
    filename = srcpath.split("/")[-1]
    download_file = dstpath / filename

    # Check if file already exists
    if download_file.exists() and not overwrite:
        logger.info(f"File {download_file} already exists. Skipping download.")
        return download_file

    # Construct full source path
    url = ARTIFACTORY_BASE_URL + srcpath

    response = _run_request(
        "GET",
        url,
        logging_msg=f"Downloading file: {filename}",
        retries=retries,
        wait=wait,
    )

    with open(download_file, "wb") as f:
        f.write(response.content)

    return download_file
