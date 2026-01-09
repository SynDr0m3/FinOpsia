"""
FinOpsia Dashboard Web App (FastAPI)

Serves the admin dashboard for metrics, logs, forecasts, and system health.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

# App instance
app = FastAPI(
    title="FinOpsia Dashboard",
    description="Admin dashboard for FinOpsia platform",
    version="1.0.0",
)

# Mount static files (CSS, JS, images)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# Jinja2 templates setup
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
def dashboard_home(request: Request):
    """
    Dashboard home page: summary and navigation.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "FinOpsia Dashboard"}
    )


