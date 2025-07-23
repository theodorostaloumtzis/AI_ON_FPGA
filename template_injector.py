import shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class TemplateInjector:
    def __init__(self, template_dir="templates"):
        self.template_dir = Path(template_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        self.templated_files = [
            "{{ project_name }}_stream.cpp",
            "{{ project_name }}_stream.h",
            "{{ project_name }}_test.cpp"
        ]
        self.static_files = [
            "build_tb.py",
            "compute_performance.py",
            "build_prj.tcl",
            "golden_preds.py"
        ]

    def inject(self, project_dir, project_name, force=False, confirm_fn=None):
        """
        Injects templated and static files into an HLS4ML project.

        Args:
            project_dir (str or Path): Path to the HLS4ML project output.
            project_name (str): Top-level function name (used for template substitution).
            force (bool): Overwrite existing files without asking.
            confirm_fn (callable): Optional confirmation function (e.g., lambda path: True)
        """
        project_dir = Path(project_dir)
        firmware_dir = project_dir / "firmware"
        firmware_dir.mkdir(parents=True, exist_ok=True)

        # Render and write Jinja2 templates
        for tmpl_name in self.templated_files:
            template = self.env.get_template(tmpl_name)
            rendered = template.render(project_name=project_name)

            output_file = tmpl_name.replace("{{ project_name }}", project_name)
            output_path = (
                project_dir / output_file if output_file.endswith("_test.cpp")
                else firmware_dir / output_file
            )

            if not output_path.exists() or force or (confirm_fn and confirm_fn(output_path)):
                with open(output_path, "w") as f:
                    f.write(rendered)
                print(f"✅ Injected: {output_path}")
            else:
                print(f"⏭️  Skipped (exists): {output_path}")

        # Copy static files
        for static in self.static_files:
            dest = project_dir / static
            if not dest.exists() or force:
                shutil.copy(self.template_dir / static, dest)
                print(f"📁 Copied: {static} → {dest}")
            else:
                print(f"⏭️  Skipped (exists): {static}")

