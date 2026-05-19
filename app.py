"""
Heun's Method (Improved Euler) - Flask Web Application
PIT Project - Numerical Methods Online Calculator
"""

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def heun_method(f_expr, x0, y0, h, x_end):
    """
    Heun's Method (Improved Euler) for solving ODEs of the form dy/dx = f(x, y).

    Parameters:
        f_expr : callable  - function f(x, y)
        x0     : float     - initial x value
        y0     : float     - initial y value
        h      : float     - step size
        x_end  : float     - end x value

    Returns:
        List of dicts with x, y, k1, k2, y_pred per step
    """
    steps = []
    x = x0
    y = y0
    n = round((x_end - x0) / h)

    for i in range(n):
        k1 = f_expr(x, y)
        y_pred = y + h * k1                    # Euler predictor
        k2 = f_expr(x + h, y_pred)            # slope at predicted point
        y_new = y + (h / 2) * (k1 + k2)       # corrector (average slopes)

        steps.append({
            "step": i + 1,
            "x_i": round(x, 10),
            "y_i": round(y, 10),
            "k1": round(k1, 10),
            "y_pred": round(y_pred, 10),
            "k2": round(k2, 10),
            "y_new": round(y_new, 10),
        })

        x = round(x + h, 10)
        y = y_new

    return steps


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()

    try:
        f_str = data.get("f", "").strip()
        x0    = float(data.get("x0", 0))
        y0    = float(data.get("y0", 0))
        h     = float(data.get("h", 0.1))
        x_end = float(data.get("x_end", 1))

        # --- Safe expression parsing ---
        # Only allow safe math names
        import math
        allowed_names = {
            "x": 0, "y": 0,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
            "pi": math.pi, "e": math.e, "abs": abs,
        }

        # Validate step count
        if h <= 0:
            return jsonify({"error": "Step size h must be positive."}), 400
        if x_end <= x0:
            return jsonify({"error": "x_end must be greater than x0."}), 400
        n_steps = round((x_end - x0) / h)
        if n_steps > 500:
            return jsonify({"error": "Too many steps (max 500). Increase h or reduce x_end."}), 400
        if n_steps < 1:
            return jsonify({"error": "At least one step is required."}), 400

        # Build the function safely
        def f(x, y):
            local_vars = {**allowed_names, "x": x, "y": y}
            return eval(f_str, {"__builtins__": {}}, local_vars)  # noqa: S307

        # Test the function at the initial point
        _ = f(x0, y0)

        results = heun_method(f, x0, y0, h, x_end)
        return jsonify({"steps": results})

    except ZeroDivisionError:
        return jsonify({"error": "Division by zero encountered in f(x, y)."}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error evaluating f(x, y): {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
