import shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class TemplateInjector:
    def __init__(self, template_dir="templates"):
        self.template_dir = Path(template_dir)
        self.templated_files = [
            "{{ project_name }}_stream.cpp",
            "{{ project_name }}_stream.h",
            "{{ project_name }}_test.cpp"
        ]
        self.static_files = [
            "build_tb.py",
            "compute_performance.py",
            "build_prj.tcl",
            "golden_preds.py",
            "vivado_synth.tcl",
        ]

    def inject(self, project_dir, project_name, force=False, confirm_fn=None, packed=False):
        project_dir = Path(project_dir)
        firmware_dir = project_dir / "firmware"
        firmware_dir.mkdir(parents=True, exist_ok=True)

        # choose the right templates folder
        tmpl_base = self.template_dir / "packing" if packed else self.template_dir
        env = Environment(loader=FileSystemLoader(str(tmpl_base)))

        # Render and write Jinja2 templates
        for tmpl_name in self.templated_files:
            template = env.get_template(tmpl_name)
            rendered = template.render(project_name=project_name)

            output_file = tmpl_name.replace("{{ project_name }}", project_name)
            output_path = (
                project_dir / output_file if output_file.endswith("_test.cpp")
                else firmware_dir / output_file
            )

            if not output_path.exists() or force or (confirm_fn and confirm_fn(output_path)):
                output_path.write_text(rendered)
                print(f"✅ Injected: {output_path}")
            else:
                print(f"⏭️  Skipped (exists): {output_path}")

        # Copy static files (always from the root templates folder)
        for static in self.static_files:
            src = self.template_dir / static
            dest = project_dir / static
            if not dest.exists() or force:
                shutil.copy(src, dest)
                print(f"📁 Copied: {static} → {dest}")
            else:
                print(f"⏭️  Skipped (exists): {static}")