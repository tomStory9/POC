from typing import Any
from collections import Counter

from fastmcp import FastMCP
from prefab_ui.app import PrefabApp
from prefab_ui.actions import SetState
from prefab_ui.actions.mcp import CallTool
from prefab_ui.components import (
    Badge,
    Button,
    Card,
    CardContent,
    CardHeader,
    Column,
    DataTable,
    DataTableColumn,
    Grid,
    Heading,
    H3,
    Input,
    Row,
    Select,
    Small,
    Text,
)
from prefab_ui.components.charts import PieChart
from prefab_ui.components.control_flow import If
from prefab_ui.rx import Rx, STATE, RESULT

mcp = FastMCP("HR Dashboard Demo")

# --- Mock data -----------------------------------------------------------------

EMPLOYEES: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Alice Martin",
        "department": "IT",
        "role": "System Administrator",
        "location": "Reims",
        "status": "Active",
        "seniority": 4,
    },
    {
        "id": 2,
        "name": "Benoit Lefevre",
        "department": "HR",
        "role": "HR Manager",
        "location": "Paris",
        "status": "Active",
        "seniority": 7,
    },
    {
        "id": 3,
        "name": "Claire Dubois",
        "department": "Finance",
        "role": "Controller",
        "location": "Lyon",
        "status": "Onboarding",
        "seniority": 1,
    },
    {
        "id": 4,
        "name": "David Morel",
        "department": "IT",
        "role": "Network Engineer",
        "location": "Reims",
        "status": "Active",
        "seniority": 6,
    },
    {
        "id": 5,
        "name": "Emma Petit",
        "department": "Operations",
        "role": "Operations Lead",
        "location": "Lille",
        "status": "Leave",
        "seniority": 3,
    },
]


# --- Helpers -------------------------------------------------------------------


def compute_metrics(employees: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(employees)
    active = sum(1 for e in employees if e["status"] == "Active")
    onboarding = sum(1 for e in employees if e["status"] == "Onboarding")
    leave = sum(1 for e in employees if e["status"] == "Leave")
    avg_seniority = (
        round(sum(e["seniority"] for e in employees) / total, 1) if total else 0
    )

    dept_counts = [
        {"department": dept, "count": count}
        for dept, count in Counter(e["department"] for e in employees).items()
    ]

    return {
        "total": total,
        "active": active,
        "onboarding": onboarding,
        "leave": leave,
        "avg_seniority": avg_seniority,
        "dept_counts": dept_counts,
    }


# --- Tools métier --------------------------------------------------------------


@mcp.tool()
def get_hr_data(department: str = "All") -> dict[str, Any]:
    """Return HR dashboard data."""
    if department == "All":
        filtered = EMPLOYEES[:]
    else:
        filtered = [e for e in EMPLOYEES if e["department"] == department]

    return {
        "employees": filtered,
        "metrics": compute_metrics(filtered),
        "department": department,
        "departments": sorted({e["department"] for e in EMPLOYEES}),
    }


@mcp.tool()
def add_employee(
    name: str,
    department: str,
    role: str,
    location: str,
    status: str = "Onboarding",
    seniority: int = 0,
) -> dict[str, Any]:
    """Add a mock employee and return refreshed HR data."""
    new_id = max(e["id"] for e in EMPLOYEES) + 1 if EMPLOYEES else 1
    EMPLOYEES.append(
        {
            "id": new_id,
            "name": name,
            "department": department,
            "role": role,
            "location": location,
            "status": status,
            "seniority": seniority,
        }
    )
    # On revient toujours à la vue "All"
    return get_hr_data("All")


# --- App (UI) ------------------------------------------------------------------


@mcp.tool(app=True)
def hr_dashboard() -> PrefabApp:
    """Interactive HR dashboard demo with mock data."""

    initial_data = get_hr_data("All")

    with PrefabApp(
        state={
            "data": initial_data,
            "selected": None,
            "department_filter": "All",
            "form_name": "",
            "form_department": "IT",
            "form_role": "",
            "form_location": "Reims",
            "form_status": "Onboarding",
            "form_seniority": 0,
        }
    ) as app:
        with Column(gap=4, css_class="p-6"):
            Heading("HR Dashboard")

            # Badges d’info
            with Row(gap=2):
                Badge("Mock data")
                Badge(Rx("data.metrics.total"), variant="secondary")

            # KPI cards
            with Grid(columns=5, gap=4):
                with Card():
                    with CardHeader():
                        Small("Headcount")
                    with CardContent():
                        H3(Rx("data.metrics.total"))

                with Card():
                    with CardHeader():
                        Small("Active")
                    with CardContent():
                        H3(Rx("data.metrics.active"))

                with Card():
                    with CardHeader():
                        Small("Onboarding")
                    with CardContent():
                        H3(Rx("data.metrics.onboarding"))

                with Card():
                    with CardHeader():
                        Small("Leave")
                    with CardContent():
                        H3(Rx("data.metrics.leave"))

                with Card():
                    with CardHeader():
                        Small("Avg seniority")
                    with CardContent():
                        H3(Rx("data.metrics.avg_seniority"))

            # Pie chart + filtre département
            with Grid(columns=[1, 2], gap=4):
                PieChart(
                    data=Rx("data.metrics.dept_counts"),
                    data_key="count",
                    name_key="department",
                    show_legend=True,
                )

                with Card():
                    with CardHeader():
                        H3("Department filter")
                    with CardContent():
                        # IMPORTANT : on laisse Prefab binder via `name`,
                        # et on ne met pas de `value=Rx(...)`.
                        Select(
                            name="department_filter",
                            options=["All", "Finance", "HR", "IT", "Operations"],
                            on_change=[
                                SetState("department_filter", Rx("$event")),
                                CallTool(
                                    "get_hr_data",
                                    arguments={"department": Rx("$event")},
                                    on_result=SetState("data", RESULT),
                                ),
                            ],
                        )

            # Tableau + panneau de détail + formulaire
            with Grid(columns=[2, 1], gap=4):
                # Table employees
                DataTable(
                    columns=[
                        DataTableColumn(key="name", header="Name", sortable=True),
                        DataTableColumn(
                            key="department", header="Department", sortable=True
                        ),
                        DataTableColumn(key="role", header="Role", sortable=True),
                        DataTableColumn(
                            key="location", header="Location", sortable=True
                        ),
                        DataTableColumn(key="status", header="Status", sortable=True),
                    ],
                    rows=Rx("data.employees"),
                    search=True,
                    paginated=True,
                    page_size=8,
                    on_row_click=SetState("selected", Rx("$event")),
                )

                # Panneau de droite : détail + formulaire ajout
                with Column(gap=4):
                    # Détails employé sélectionné
                    with Card():
                        with CardHeader():
                            H3("Employee details")
                        with CardContent():
                            with If(STATE.selected):
                                with Column(gap=2):
                                    Text(Rx("selected.name"))
                                    Small(Rx("selected.role"))
                                    Small(Rx("selected.department"))
                                    Small(Rx("selected.location"))
                                    Small(Rx("selected.status"))
                            with If(~STATE.selected):
                                Small("Select a row in the table.")

                    # Formulaire ajout
                    with Card():
                        with CardHeader():
                            H3("Add employee")
                        with CardContent():
                            with Column(gap=3):
                                Input(
                                    placeholder="Full name",
                                    value=STATE.form_name,
                                    on_change=SetState("form_name", Rx("$event")),
                                )
                                Select(
                                    name="form_department",
                                    options=["IT", "HR", "Finance", "Operations"],
                                    # ici on utilise le binding automatique via `name`
                                )
                                Input(
                                    placeholder="Role",
                                    value=STATE.form_role,
                                    on_change=SetState("form_role", Rx("$event")),
                                )
                                Input(
                                    placeholder="Location",
                                    value=STATE.form_location,
                                    on_change=SetState("form_location", Rx("$event")),
                                )
                                Select(
                                    name="form_status",
                                    options=["Active", "Onboarding", "Leave"],
                                )
                                Input(
                                    placeholder="Seniority (years)",
                                    value=STATE.form_seniority,
                                    on_change=SetState("form_seniority", Rx("$event")),
                                )
                                Button(
                                    "Add employee",
                                    on_click=CallTool(
                                        "add_employee",
                                        arguments={
                                            "name": STATE.form_name,
                                            "department": STATE.form_department,
                                            "role": STATE.form_role,
                                            "location": STATE.form_location,
                                            "status": STATE.form_status,
                                            "seniority": STATE.form_seniority,
                                        },
                                        on_result=[
                                            SetState("data", RESULT),
                                            SetState("selected", None),
                                            SetState("department_filter", "All"),
                                            SetState("form_name", ""),
                                            SetState("form_role", ""),
                                            SetState("form_location", "Reims"),
                                            SetState("form_status", "Onboarding"),
                                            SetState("form_seniority", 0),
                                        ],
                                    ),
                                )

    return app


if __name__ == "__main__":
    mcp.run()
