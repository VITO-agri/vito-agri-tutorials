"""Test the installation cell without installing packages or accessing the network."""

import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vito_agri_tutorials import __version__
from vito_agri_tutorials.tutorials import _TUTORIALS, install_tutorial


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/drought/1_ndvi_anomalies.ipynb"
)


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text())
        code = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertTrue(code[0].startswith("%pip install --quiet"))
        self.assertIn(
            "from vito_agri_tutorials.tutorials import install_tutorial", code[0]
        )
        self.assertEqual(code[1], 'install_tutorial("ndvi-anomalies")')
        self.code = code[1]

    def test_success_hides_logs_and_uses_current_interpreter(self):
        result = subprocess.CompletedProcess([], 0, stdout="Detailed pip output")
        output = io.StringIO()
        with (
            patch("subprocess.run", return_value=result) as run,
            patch("IPython.display.display") as display,
            redirect_stdout(output),
        ):
            exec(self.code, {"install_tutorial": install_tutorial})

        self.assertIn("Installation succeeded!", output.getvalue())
        self.assertIn("Restart Kernel", output.getvalue())
        self.assertNotIn(result.stdout, output.getvalue())
        display.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(
            command[-1], f"vito-agri-tutorials[ndvi-anomalies]=={__version__}"
        )
        self.assertEqual(run.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.STDOUT)

    def test_failure_shows_escaped_expandable_logs_and_stops(self):
        result = subprocess.CompletedProcess(
            [], 1, stdout="Download failed: <script>alert('error')</script>"
        )
        output = io.StringIO()
        with (
            patch("subprocess.run", return_value=result),
            patch("IPython.display.display") as display,
            redirect_stdout(output),
        ):
            with self.assertRaisesRegex(RuntimeError, "Do not continue"):
                exec(self.code, {"install_tutorial": install_tutorial})

        self.assertIn("Installation failed.", output.getvalue())
        self.assertNotIn("Installation succeeded!", output.getvalue())
        display.assert_called_once()
        details = display.call_args.args[0].data
        self.assertIn("<details>", details)
        self.assertIn("Show installation details", details)
        self.assertIn("&lt;script&gt;", details)
        self.assertNotIn("<script>", details)

    def test_unknown_tutorial_does_not_start_installation(self):
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "Available tutorials: ndvi"):
                install_tutorial("../unknown")
        run.assert_not_called()

    def test_every_tutorial_installs_its_declared_extra(self):
        for tutorial in _TUTORIALS:
            with self.subTest(tutorial=tutorial):
                result = subprocess.CompletedProcess([], 0, stdout="")
                with patch("subprocess.run", return_value=result) as run:
                    with redirect_stdout(io.StringIO()):
                        install_tutorial(tutorial)
                self.assertEqual(
                    run.call_args.args[0][-1],
                    f"vito-agri-tutorials[{tutorial}]=={__version__}",
                )
