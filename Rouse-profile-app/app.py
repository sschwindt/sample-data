from flask import Flask, request, jsonify, send_file
import plot_Rouse_profile_ssc as rouse
from flask_cors import CORS
import webbrowser
import os
from threading import Timer
import sys

app = Flask(__name__)
CORS(app)

# Track if browser has been opened
browser_opened = False

# Function to open browser after server starts
def open_browser():
    global browser_opened
    if not browser_opened:
        browser_opened = True
        webbrowser.open_new('http://127.0.0.1:5000')

# Generate default plot at startup
def generate_default_plot():
    U = rouse.USER_FLOW_VELOCITY
    h = rouse.USER_WATER_DEPTH
    d = rouse.USER_D50
    w_s = rouse.get_particle_settling_velocity(U, h, d, print_results=False)
    u_star = rouse.get_u_star(U, h, d)
    # Generate a single Rouse profile
    rouse.plot_rouse_profile(
        w_s=w_s,
        u_star=u_star,
        h=h,
        a=rouse.USER_REF_HEIGHT,
        c_a=rouse.USER_REF_CONC,
        show_plot=False,
        filename="ssc-Rouse-profile.png"
    )

@app.route('/')
def home():
    # Ensure default plot exists
    if not os.path.exists('ssc-Rouse-profile.png'):
        generate_default_plot()
    # Return your homepage
    return send_file('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    # Convert grain size from μm to m
    velocity = float(data['velocity'])
    depth    = float(data['depth'])
    grain_size = float(data['grain_size']) * 1e-6  # Convert μm -> m

    # Calculate parameters
    w_s = rouse.get_particle_settling_velocity(velocity, depth, grain_size, print_results=False)
    u_star = rouse.get_u_star(velocity, depth, grain_size)

    # Generate the single-profile plot for these inputs
    rouse.plot_rouse_profile(
        w_s=w_s,
        u_star=u_star,
        h=depth,
        a=rouse.USER_REF_HEIGHT,
        c_a=rouse.USER_REF_CONC,
        show_plot=False,
        filename="ssc-Rouse-profile.png"
    )
    
    # Calculate Rouse number
    beta = 0.0
    if u_star != 0.0:
        beta = w_s / (rouse.KAPPA * u_star)
    
    # Return JSON with interesting values
    return jsonify({
        'settling_velocity': w_s,
        'shear_velocity'   : u_star,
        'rouse_number'     : beta
    })

@app.route('/ssc-Rouse-profile.png')
def get_image():
    # Generate default plot if it doesn't exist
    if not os.path.exists('ssc-Rouse-profile.png'):
        generate_default_plot()
    return send_file('ssc-Rouse-profile.png', mimetype='image/png')

if __name__ == '__main__':
    # Check for index.html
    if not os.path.exists('index.html'):
        print("Warning: index.html not found in the current directory!")
    
    # Generate default plot at startup
    generate_default_plot()
    
    # Only open browser in the main process
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1.5, open_browser).start()
    
    # Run the Flask app
    app.run(debug=True)
