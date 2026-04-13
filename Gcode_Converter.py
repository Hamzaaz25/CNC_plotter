import subprocess


class GcodeConverter:
    def __init__(self, SvgPath: str, GPath: str, scale: float = 0.3,
                 bed_width: int = 297, bed_height: int = 210):
        self.SvgPath = SvgPath
        self.GPath = GPath
        self.scale = scale
        self.bed_width = bed_width
        self.bed_height = bed_height
        self.outputPath = "./TestingImages/Centered.gc"

    def center_gcode(self, gcode: str):
        with open(gcode) as f:
            lines = f.readlines()

        x_vals, y_vals = [], []

        for line in lines:
            if line.startswith(('G1', 'G0')):
                parts = line.split()
                for p in parts:
                    if p.startswith('X'):
                        x_vals.append(float(p[1:]))
                    elif p.startswith('Y'):
                        y_vals.append(float(p[1:]))

        if not x_vals or not y_vals:
            return lines

        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)

        draw_width = max_x - min_x
        draw_height = max_y - min_y

        offset_x = (self.bed_width - draw_width) / 2 - min_x
        offset_y = (self.bed_height - draw_height) / 2 - min_y

        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('G1', 'G0')):
                parts = []
                for p in stripped.split():
                    if p.startswith('X'):
                        parts.append(f"X{float(p[1:]) + offset_x:.3f}")
                    elif p.startswith('Y'):
                        parts.append(f"Y{float(p[1:]) + offset_y:.3f}")
                    else:
                        parts.append(p)
                new_line = ' '.join(parts)
            else:
                new_line = stripped

            new_lines.append(new_line + '\n')

        with open(gcode, "w") as f:
            f.writelines(new_lines)
        print(f"{gcode} Centered")

    def clean_gcode_redundant_pen_moves(self, lines):
        """
        Removes redundant consecutive M03 S0 / M03 S180 commands,
        but keeps setup lines (G21, G90, G92, etc.) untouched.
        """
        cleaned = []
        last_pen_state = None

        for line in lines:
            stripped = line.strip().upper()

            if stripped.startswith(("G21", "G90", "G92")) or stripped == "":
                cleaned.append(line)
                continue

            if last_pen_state is None and stripped.startswith("M03"):
                cleaned.append(line)
                last_pen_state = "up" if "S70" in stripped else "down"
                continue

            if "M03 S180" in stripped:
                if last_pen_state != "down":
                    cleaned.append(line)
                    last_pen_state = "down"
            elif "M03 S70" in stripped:
                if last_pen_state != "up":
                    cleaned.append(line)
                    last_pen_state = "up"
            else:
                cleaned.append(line)

        return cleaned

    def merge(self, GpathWhite: str):
        with open(self.GPath, "r") as f1, open(GpathWhite, "r") as f2:
            full_lines = [line.strip() for line in f1.readlines()]
            white_lines = set(line.strip() for line in f2.readlines())

        new_lines = []
        for line in full_lines:
            if line.startswith(("G0", "G1")):
                if line.strip() in white_lines:
                    new_lines.append("M03 S70")   # pen up
                else:
                    new_lines.append("M03 S180")  # pen down
                new_lines.append(line)

        with open(self.outputPath, "w") as f:
            f.write("\n".join(new_lines))

        with open(self.outputPath) as f:
            lines = f.readlines()
        lines = self.clean_gcode_redundant_pen_moves(lines)
        with open(self.outputPath, "w") as f:
            f.writelines(lines)

    def gcodeConvert(self, svg, gcode):
        subprocess.run(
            f'vpype --config "TestingImages/myconfig.toml" read {svg} '
            f'scale {self.scale}mm {self.scale}mm linemerge linesort '
            f'gwrite -p marlin_servo "{gcode}"',
            shell=True)

    def firstConvert(self, SvgWhite, GpathWhite):
        self.gcodeConvert(self.SvgPath, self.GPath)
        self.gcodeConvert(SvgWhite, GpathWhite)

    def secondConvert(self, GpathWhite: str):
        self.merge(GpathWhite)
        self.center_gcode(self.outputPath)
        self._insert_setup_lines(self.outputPath)

    def secondConvertMulti(self, GpathWhite: str, gcode_paths):
        """Convert and insert setup lines into multiple G-code files."""
        self.merge(GpathWhite)
        self.center_gcode(self.outputPath)
        for path in gcode_paths:
            self._insert_setup_lines(path)

    def _insert_setup_lines(self, output_path: str):
        """Insert G92 and $X setup lines near the top of a G-code file."""
        with open(self.outputPath, "r") as file:
            lines = file.readlines()

        lines.insert(1, "G92 X0 Y0\n")
        lines.insert(2, "$X\n")

        with open(output_path, "w") as file:
            file.writelines(lines)
