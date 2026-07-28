from importlib import resources

from fastapi.testclient import TestClient

import calcium_imaging_dashboard
from calcium_imaging_dashboard.cell_registration_dashboard.main import app as dashboard_app
from calcium_imaging_dashboard.db_builder.main import app as builder_app


def test_package_metadata_and_frontends_are_installed():
    assert calcium_imaging_dashboard.__version__
    package = resources.files("calcium_imaging_dashboard")
    assert package.joinpath("cell_registration_dashboard", "frontend", "index.html").is_file()
    assert package.joinpath(
        "cell_registration_dashboard", "frontend", "assets", "chart.umd.min.js"
    ).is_file()
    assert package.joinpath(
        "cell_registration_dashboard", "frontend", "images", "docs-04-database-builder.png"
    ).is_file()
    assert package.joinpath("db_builder", "frontend", "index.html").is_file()


def test_installed_applications_serve_their_start_pages():
    dashboard_response = TestClient(dashboard_app).get("/")
    builder_response = TestClient(builder_app).get("/")

    assert dashboard_response.status_code == 200
    assert "Calcium Imaging Dashboard" in dashboard_response.text
    assert builder_response.status_code == 200
    assert "Calcium Imaging Database Builder" in builder_response.text
