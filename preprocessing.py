# %%
# Initialization
import torch
import dill as pickle
import cvxpy as cp
import numpy as np
import pandas as pd
import sympy as sp
from pathlib import Path
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# System parameters - Linear volume-head relationship
max_vol_up = 588000  # m^3
max_vol_low = 588000  # m^3
max_vol = 588000  # m^3
head_min = 50  # m (gross) - when upper=0, lower=588000
head_max = 99  # m (gross) - when upper=588000, lower=0

# %%
# Read DA Prices

DA_price_hour = torch.tensor([], dtype=torch.float).to(device)
DA_price_quarter = torch.tensor([], dtype=torch.float).to(device)

# %% Linear regression on UPC boundaries (outside thepipeline)
# Linear regression on UPC boundaries
# Note: Only works for UPCs with boundaries as 👇
# --------------------------------------------------------------------->p
# -10         /       /           50|         \            \        10
#            /       /              |          \            \
#           /       /               |           \            \
#          /       /                |            \            \
#         /       /                 |             \            \
#        /       /                  |              \            \
#       /       /                   |               \            \
#      /_______/                  99|                \____________\
#                                   |
#                                  h↓

def get_UPC_bound():
    """
    Load boundary data for pump and turbine operations from an Excel file.

    Returns:
        tuple: Contains two numpy arrays:
            - boundaries_neg: Array of [head, min value, max value] for pump operation,
            - boundaries_pos: Array of [head, min value, max value] for turbine operation.
    """
    # Designated file path & load data
    current_dir = Path(__file__).parent
    UPC_file_path = current_dir / 'Data/UPCs/Mod_Francis_joint.xlsx'
    UPC_df = pd.read_excel(UPC_file_path, sheet_name='Flow rate')

    # Extracting boundaries from DataFrame
    h_values = UPC_df.iloc[:, 0]
    p_values = np.linspace(-10, 10, UPC_df.shape[1] - 1)
    boundaries_neg = []
    boundaries_pos = []

    # Determine non-NaN boundaries for each h
    for i, row in UPC_df.iterrows():
        non_nan_indices = np.where(~pd.isna(row[1:]))[0]
        if (non_nan_indices.size > 0) and (non_nan_indices[0] < len(p_values) // 2):
            min_p_neg = p_values[non_nan_indices[0]]
            max_p_neg = p_values[non_nan_indices[non_nan_indices < (len(p_values) // 2)][-1]]
            min_p_pos = p_values[non_nan_indices[non_nan_indices > (len(p_values) // 2)][0]]
            max_p_pos = p_values[non_nan_indices[-1]]
            boundaries_neg.append((h_values[i], min_p_neg, max_p_neg))
            boundaries_pos.append((h_values[i], min_p_pos, max_p_pos))

    boundaries_neg = np.array(boundaries_neg)
    boundaries_pos = np.array(boundaries_pos)
    return boundaries_neg, boundaries_pos

def LR_UPC_bound(boundaries_neg, boundaries_pos):
    """
    Perform linear regression to find coefficients for the minimum and maximum
    power boundaries for both negative (pump mode) and positive (turbine mode) power values.

    Args:
        boundaries_neg (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for positive power mode.

    Returns:
        tuple: 
        - h_fit (numpy.ndarray): Array of the range of head values used for fitting.
        - p_neg_min_fit, p_neg_max_fit (tuple): Coefficients of the fitted line (slope, intercept)
            for the minimum and maximum negative power boundaries.
        - p_pos_min_fit, p_pos_max_fit (tuple): Coefficients of the fitted line (slope, intercept)
            for the minimum and maximum positive power boundaries.
    """

    # Perform linear fitting
    h_fit = np.array([min(boundaries_neg[:, 0]), max(boundaries_neg[:, 0])])
    p_neg_min_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 1], 1)
    p_neg_max_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 2], 1)
    p_pos_min_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 1], 1)
    p_pos_max_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 2], 1)

    return h_fit, p_neg_min_fit, p_neg_max_fit, p_pos_min_fit, p_pos_max_fit

def plot_UPC_boundaries(boundaries_neg, boundaries_pos, h_fit, p_neg_min_fit, p_neg_max_fit, p_pos_min_fit, p_pos_max_fit):
    """
    Plot both actual and fitted data regions based on boundary data for pump and turbine operational modes.

    Args:
        boundaries_neg (numpy.ndarray): Array containing actual boundary data for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing actual boundary data for positive power mode.
        h_fit (numpy.ndarray): Array of the range of head values used for plotting the fitted lines.
        p_neg_min_fit (tuple): Coefficients (slope, intercept) of the fitted line for the minimum negative power boundary.
        p_neg_max_fit (tuple): Coefficients (slope, intercept) of the fitted line for the maximum negative power boundary.
        p_pos_min_fit (tuple): Coefficients (slope, intercept) of the fitted line for the minimum positive power boundary.
        p_pos_max_fit (tuple): Coefficients (slope, intercept) of the fitted line for the maximum positive power boundary.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate fitted regions using the linear coefficients
    neg_min_line = np.polyval(p_neg_min_fit, h_fit)
    neg_max_line = np.polyval(p_neg_max_fit, h_fit)
    pos_min_line = np.polyval(p_pos_min_fit, h_fit)
    pos_max_line = np.polyval(p_pos_max_fit, h_fit)
    
    # Plotting actual data regions
    for i in range(len(boundaries_neg) - 1):
        # Ensure only one legend entry for actual regions
        ax.fill_betweenx([boundaries_neg[i][0], boundaries_neg[i+1][0]],
                         [boundaries_neg[i][1], boundaries_neg[i+1][1]],
                         [boundaries_neg[i][2], boundaries_neg[i+1][2]], 
                         color='blue', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p<0)' if i == 0 else "")
        ax.fill_betweenx([boundaries_pos[i][0], boundaries_pos[i+1][0]],
                         [boundaries_pos[i][1], boundaries_pos[i+1][1]],
                         [boundaries_pos[i][2], boundaries_pos[i+1][2]], 
                         color='red', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p>0)' if i == 0 else "")

    # Draw fitted regions with borders
    ax.fill(np.append(neg_min_line, neg_max_line[::-1]), np.append(h_fit, h_fit[::-1]),
            color='cyan', alpha=0.5, edgecolor='blue', linewidth=2, label='Fitted Region (p<0)')
    ax.fill(np.append(pos_min_line, pos_max_line[::-1]), np.append(h_fit, h_fit[::-1]),
            color='orange', alpha=0.5, edgecolor='red', linewidth=2, label='Fitted Region (p>0)')

    # Adding legends with specific settings
    legend = ax.legend(frameon=True, framealpha=1, edgecolor='black')
    for legobj in legend.legend_handles:
        legobj.set_linewidth(1.3)  # Set uniform line width for fitted region markers

    # Adding labels and titles
    ax.set_xlabel('Power p')
    ax.set_ylabel('Head h')
    ax.set_title('Actual and Fitted Data Regions in p-h Space')

    # Show plot
    plt.show()

# If this script is the main entry point, execute the plot function
if __name__ == '__main__':
    boundaries_neg, boundaries_pos = get_UPC_bound()
    h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit = LR_UPC_bound(boundaries_neg, boundaries_pos)
    plot_UPC_boundaries(boundaries_neg, boundaries_pos, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit)

# neg_min_line = np.polyval(neg_min_fit, h_fit)

# %% Linear regression(high poly for simulation) on UPC boundaries (outside thepipeline)
# Linear regression on UPC boundaries
# Note: Only works for UPCs with boundaries as 👇
# --------------------------------------------------------------------->p
# -10         /       /           50|         \            \        10
#            /       /              |          \            \
#           /       /               |           \            \
#          /       /                |            \            \
#         /       /                 |             \            \
#        /       /                  |              \            \
#       /       /                   |               \            \
#      /_______/                  99|                \____________\
#                                   |
#                                  h↓

def poly_LR_UPC_bound(boundaries_neg, boundaries_pos, degree=4):
    """
    Perform polynomial regression to find coefficients for the minimum and maximum
    power boundaries for both negative (pump mode) and positive (turbine mode) power values.

    Args:
        boundaries_neg (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for positive power mode.
        degree (int): Degree of the polynomial model to be fitted.

    Returns:
        tuple: 
        - h_poly_fit (numpy.ndarray): Array of the range of head values used for fitting.
        - poly_neg_min_fit, poly_neg_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum negative power boundaries.
        - poly_pos_min_fit, poly_pos_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum positive power boundaries.
    """

    h_poly_fit = np.linspace(min(boundaries_neg[:, 0]), max(boundaries_neg[:, 0]), 100)
    poly_neg_min_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 1], degree)
    poly_neg_max_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 2], degree)
    poly_pos_min_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 1], degree)
    poly_pos_max_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 2], degree)

    poly_neg_min_fit = torch.tensor(poly_neg_min_fit, dtype=torch.float32, device=device)
    poly_neg_max_fit = torch.tensor(poly_neg_max_fit, dtype=torch.float32, device=device)
    poly_pos_min_fit = torch.tensor(poly_pos_min_fit, dtype=torch.float32, device=device)
    poly_pos_max_fit = torch.tensor(poly_pos_max_fit, dtype=torch.float32, device=device)

    return h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit

def LR_UPC_bound(boundaries_neg, boundaries_pos, degree=5):
    """
    Perform polynomial regression to find coefficients for the minimum and maximum
    power boundaries for both negative (pump mode) and positive (turbine mode) power values.

    Args:
        boundaries_neg (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing head and corresponding
                                        minimum and maximum power values for positive power mode.
        degree (int): Degree of the polynomial model to be fitted.

    Returns:
        tuple: 
        - h_fit (numpy.ndarray): Array of the range of head values used for fitting.
        - p_neg_min_fit, p_neg_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum negative power boundaries.
        - p_pos_min_fit, p_pos_max_fit (tuple): Coefficients of the fitted polynomial
            for the minimum and maximum positive power boundaries.
    """

    # Perform polynomial fitting
    h_fit = np.linspace(min(boundaries_neg[:, 0]), max(boundaries_neg[:, 0]), 100)
    p_neg_min_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 1], degree)
    p_neg_max_fit = np.polyfit(boundaries_neg[:, 0], boundaries_neg[:, 2], degree)
    p_pos_min_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 1], degree)
    p_pos_max_fit = np.polyfit(boundaries_pos[:, 0], boundaries_pos[:, 2], degree)

    return h_fit, p_neg_min_fit, p_neg_max_fit, p_pos_min_fit, p_pos_max_fit

def plot_poly_UPC_boundaries(boundaries_neg, boundaries_pos, h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit):
    """
    Plot both actual and fitted data regions based on boundary data for pump and turbine operational modes using polynomial fits.

    Args:
        boundaries_neg (numpy.ndarray): Array containing actual boundary data for negative power mode.
        boundaries_pos (numpy.ndarray): Array containing actual boundary data for positive power mode.
        h_poly_fit (numpy.ndarray): Array of the range of head values used for plotting the fitted polynomials.
        poly_neg_min_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the minimum negative power boundary.
        poly_neg_max_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the maximum negative power boundary.
        poly_pos_min_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the minimum positive power boundary.
        poly_pos_max_fit (tuple): Coefficients (slope, intercept) of the fitted polynomial for the maximum positive power boundary.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate fitted regions using the polynomial coefficients
    neg_min_line = np.polyval(poly_neg_min_fit.cpu().numpy(), h_poly_fit)
    neg_max_line = np.polyval(poly_neg_max_fit.cpu().numpy(), h_poly_fit)
    pos_min_line = np.polyval(poly_pos_min_fit.cpu().numpy(), h_poly_fit)
    pos_max_line = np.polyval(poly_pos_max_fit.cpu().numpy(), h_poly_fit)
    
    # Plotting actual data regions
    for i in range(len(boundaries_neg) - 1):
        ax.fill_betweenx([boundaries_neg[i][0], boundaries_neg[i+1][0]],
                         [boundaries_neg[i][1], boundaries_neg[i+1][1]],
                         [boundaries_neg[i][2], boundaries_neg[i+1][2]], 
                         color='blue', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p<0)' if i == 0 else "")
        ax.fill_betweenx([boundaries_pos[i][0], boundaries_pos[i+1][0]],
                         [boundaries_pos[i][1], boundaries_pos[i+1][1]],
                         [boundaries_pos[i][2], boundaries_pos[i+1][2]], 
                         color='red', alpha=0.3, edgecolor='none', 
                         label='Actual Region (p>0)' if i == 0 else "")

    # Draw fitted regions with borders
    ax.fill(np.append(neg_min_line, neg_max_line[::-1]), np.append(h_poly_fit, h_poly_fit[::-1]),
            color='cyan', alpha=0.5, edgecolor='blue', linewidth=2, label='Fitted Region (p<0)')
    ax.fill(np.append(pos_min_line, pos_max_line[::-1]), np.append(h_poly_fit, h_poly_fit[::-1]),
            color='orange', alpha=0.5, edgecolor='red', linewidth=2, label='Fitted Region (p>0)')

    # Set grid with major and minor ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='major', linestyle='-', linewidth=0.75)

    # Mark p=0 axis line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    legend = ax.legend(frameon=True, framealpha=1, edgecolor='black')
    ax.set_xlabel('Power p')
    ax.set_ylabel('Head h')
    ax.set_title('Polynomial Fit of UPC Boundaries in p-h Space')

    plt.show()

# Load the UPC boundary data
boundaries_neg, boundaries_pos = get_UPC_bound()

# Perform polynomial regression to find the boundary fits
h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit = poly_LR_UPC_bound(boundaries_neg, boundaries_pos)

# Convert numpy arrays to torch tensors
poly_neg_min_fit = torch.tensor(poly_neg_min_fit, dtype=torch.float32, device=device)
poly_neg_max_fit = torch.tensor(poly_neg_max_fit, dtype=torch.float32, device=device)
poly_pos_min_fit = torch.tensor(poly_pos_min_fit, dtype=torch.float32, device=device)
poly_pos_max_fit = torch.tensor(poly_pos_max_fit, dtype=torch.float32, device=device)

def neg_min(h, coefficients=poly_neg_min_fit):
    """p >= neg_min(h), in pump mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def neg_max(h, coefficients=poly_neg_max_fit):
    """p <= neg_max(h), in pump mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def pos_min(h, coefficients=poly_pos_min_fit):
    """p >= pos_min(h), in turbine mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

def pos_max(h, coefficients=poly_pos_max_fit):
    """p <= pos_max(h), in turbine mode"""
    result = coefficients[0]
    for c in coefficients[1:]:
        result = result * h + c
    return result

# If this script is the main entry point, execute the plot function
if __name__ == '__main__':

    # Example: Calculate boundary powers for a specific head value
    head_example = 80  # Example head value
    min_neg_power = neg_min(head_example)
    max_neg_power = neg_max(head_example)
    min_pos_power = pos_min(head_example)
    max_pos_power = pos_max(head_example)

    print(f"Head: {head_example}")
    print(f"Minimum Negative Power: {min_neg_power}")
    print(f"Maximum Negative Power: {max_neg_power}")
    print(f"Minimum Positive Power: {min_pos_power}")
    print(f"Maximum Positive Power: {max_pos_power}")

    # Plot the results using the polynomial fitted boundaries
    plot_poly_UPC_boundaries(boundaries_neg, boundaries_pos, h_poly_fit, poly_neg_min_fit, poly_neg_max_fit, poly_pos_min_fit, poly_pos_max_fit)

# neg_min_line = np.polyval(neg_min_fit, h_fit)

# %% Fit UPC data (outside pipeline)
# Fit UPC data (outside pipeline)

def prepare_and_fit_model(file_path, poly_degree=5):
    # Load data from the specified file
    data = pd.read_excel(file_path)
    x_values = np.array(data.columns[1:], dtype=float)  # x is Power
    y_values = np.array(data.iloc[:, 0], dtype=float)   # y is Head
    X, Y = np.meshgrid(x_values, y_values)
    z_flat = data.iloc[:, 1:].values.flatten()
    
    # Filter valid data points
    valid_indices = ~np.isnan(z_flat)
    x_valid = X.flatten()[valid_indices]
    y_valid = Y.flatten()[valid_indices]
    z_valid = z_flat[valid_indices]
    
    # Fit model
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    XY_valid = np.vstack([x_valid, y_valid]).T
    model = make_pipeline(poly, LinearRegression())
    model.fit(XY_valid, z_valid)

    # Additional data for formulas
    coefs = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    feature_names = poly.get_feature_names_out(['p', 'h'])

    # Predictions for calculating R^2 and other statistics
    z_pred = model.predict(XY_valid)
    sst = np.sum((z_valid - np.mean(z_valid)) ** 2)
    ssr = np.sum((z_valid - z_pred) ** 2)
    r2 = 1 - (ssr / sst)
    n = len(z_valid)
    p = poly.n_output_features_
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    reduced_chi_squared = ssr / (n - p)
    
    return model, x_valid, y_valid, z_valid, r2, adjusted_r2, reduced_chi_squared, coefs, intercept, feature_names

def plot_3d_surface_interactive(x_valid, y_valid, z_valid, model, title):
    # Create mesh grid for the surface
    x_surf, y_surf = np.meshgrid(np.linspace(x_valid.min(), x_valid.max(), 50),
                                 np.linspace(y_valid.min(), y_valid.max(), 50))
    xy_surf = np.vstack([x_surf.ravel(), y_surf.ravel()]).T
    z_surf = model.predict(xy_surf).reshape(x_surf.shape)

    # Determine the range of z values for plotting
    z_min = z_valid.min()
    z_max = z_valid.max()

    # Create the interactive figure
    fig = go.Figure(data=[
        go.Scatter3d(x=x_valid, y=y_valid, z=z_valid, mode='markers', name='Original Data', 
                     marker=dict(size=1, color=z_valid,colorscale='Plasma',cmin=z_min,cmax=z_max)),
        go.Surface(x=x_surf, y=y_surf, z=z_surf, name='Fitted Surface', 
                   colorscale='Viridis', cmin=z_min, cmax=z_max,opacity=0.7)
    ])
    
    # Update layout with dynamic z-axis range and title
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Power (p)',
            yaxis_title='Head (h)',
            zaxis_title='Flow (q)',
            zaxis=dict(range=[z_min-1, z_max+1])  # Set the z-axis range based on actual data
        )
    )

    fig.show()

def print_model_formula(coefs, intercept, feature_names):
    terms = [f"{coef:.2f}*{name}" for coef, name in zip(coefs, feature_names)]
    formula = " + ".join(terms)
    formula = f"q = {intercept:.2f} + {formula}"
    return formula

# Perform polynomial linear regression on UPC(Outside pipeline)
if __name__ == '__main__':
    # Fit and plot models for both pump and turbine
    results_pump = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    results_turbine = prepare_and_fit_model('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    
    # Plot the fitted models
    plot_3d_surface_interactive(*results_pump[1:4], results_pump[0], 'Pump Model')
    plot_3d_surface_interactive(*results_turbine[1:4], results_turbine[0], 'Turbine Model')

    # Print statistical measures and formulas
    print("Pump Model - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_pump[4], results_pump[5], results_pump[6]))
    print("Pump Model Formula:")
    print(print_model_formula(results_pump[7], results_pump[8], results_pump[9]))
    
    print("Turbine Model - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_turbine[4], results_turbine[5], results_turbine[6]))
    print("Turbine Model Formula:")
    print(print_model_formula(results_turbine[7], results_turbine[8], results_turbine[9]))

    # Convert coefficients and intercepts to PyTorch tensors
    coefs_tur = torch.tensor(results_turbine[7], dtype=torch.float32, device=device)
    intercept_tur = torch.tensor(results_turbine[8], dtype=torch.float32, device=device)
    coefs_pump = torch.tensor(results_pump[7], dtype=torch.float32, device=device)
    intercept_pump = torch.tensor(results_pump[8], dtype=torch.float32, device=device)


# ✨Predict by manually generated features (Used in the pipeline)

def UPC_poly_tur(p, h, coefs=coefs_tur, intercept=intercept_tur, poly_degree=5):
    """
    Polynomial prediction of flow rate for turbine mode.

    Args:
        p: torch tensor of power values of any shape
        h: torch tensor of head values (same shape as p)
        coefs: turbine model polynomial coefficients
        intercept: turbine model intercept
        poly_degree: degree of polynomial features

    Returns:
        torch tensor of predicted flow values (same shape as inputs)
    """
    # Ensure inputs are torch tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32, device=device)
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.float32, device=device)

    # Generate power matrix
    max_power = poly_degree + 1

    # Precompute all possible power combinations (p^a * h^b, a + b <= poly_degree)
    powers = torch.arange(max_power, device=p.device)
    p_pows = torch.pow(p.unsqueeze(-1), powers)  # [..., max_power]
    h_pows = torch.pow(h.unsqueeze(-1), powers)  # [..., max_power]

    # Generate polynomial feature matrix
    terms = []
    for total_degree in range(1, poly_degree + 1):
        for a in range(total_degree, -1, -1):
            b = total_degree - a
            if b <= total_degree and a + b <= poly_degree:
                terms.append(p_pows[..., a] * h_pows[..., b])

    features = torch.stack(terms, dim=-1)  # [..., num_features]

    # Calculate turbine flow rate
    q = torch.einsum('...f,f->...', features, coefs) + intercept

    # Zero out predictions for non-turbine mode (p ≤ 0)
    mask_tur = (p > 0)
    return torch.where(mask_tur, q, torch.zeros_like(p).to(p.device))

def UPC_poly_pump(p, h, coefs=coefs_pump, intercept=intercept_pump, poly_degree=5):
    """
    Polynomial prediction of flow rate for pump mode.

    Args:
        p: torch tensor of power values of any shape
        h: torch tensor of head values (same shape as p)
        coefs: pump model polynomial coefficients
        intercept: pump model intercept
        poly_degree: degree of polynomial features

    Returns:
        torch tensor of predicted flow values (same shape as inputs)
    """
    # Ensure inputs are torch tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32, device=device)
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.float32, device=device)

    # Generate power matrix
    max_power = poly_degree + 1

    # Precompute all possible power combinations (p^a * h^b, a + b <= poly_degree)
    powers = torch.arange(max_power, device=p.device)
    p_pows = torch.pow(p.unsqueeze(-1), powers)  # [..., max_power]
    h_pows = torch.pow(h.unsqueeze(-1), powers)  # [..., max_power]

    # Generate polynomial feature matrix
    terms = []
    for total_degree in range(1, poly_degree + 1):
        for a in range(total_degree, -1, -1):
            b = total_degree - a
            if b <= total_degree and a + b <= poly_degree:
                terms.append(p_pows[..., a] * h_pows[..., b])

    features = torch.stack(terms, dim=-1)  # [..., num_features]

    # Calculate pump flow rate
    q = torch.einsum('...f,f->...', features, coefs) + intercept

    # Zero out predictions for non-pump mode (p ≥ 0)
    mask_pump = (p < 0)
    return torch.where(mask_pump, q, torch.zeros_like(p).to(p.device))

# Test the functions
if __name__ == '__main__':
    # Define example inputs
    p_example = torch.tensor([-5.89, 0.0, 5.89], dtype=torch.float32, device=device)
    h_example = torch.tensor([78, 67, 91], dtype=torch.float32, device=device)
    # Expected outputs [-8.984, 0.0, 7.906]

    # Test split functions
    q_tur = UPC_poly_tur(p_example, h_example)
    q_pump = UPC_poly_pump(p_example, h_example)
    print(f"Turbine flow (q) for p={p_example.tolist()}, h={h_example.tolist()}:\n{q_tur.tolist()}")
    print(f"Pump flow (q) for p={p_example.tolist()}, h={h_example.tolist()}:\n{q_pump.tolist()}")

# %% Simple linear volume-head relationship
# Based on system specifications:
# - Max capacity (both reservoirs): 588,000 m³
# - Min capacity (both reservoirs): 0 m³
# - Min head (gross): 50m (upper at 0, lower at 588,000)
# - Max head (gross): 99m (upper at 588,000, lower at 0)
#
# Linear relationship: h = 50 + (v_up / 588000) * (99 - 50)
# where v_up is upper reservoir volume and v_low = 588000 - v_up

def v_up_to_h(v_up):
    """
    Convert upper reservoir volume to gross head.
    Linear relationship: h = 50 + (v_up / max_vol) * (99 - 50)

    Args:
        v_up: Upper reservoir volume (m³), can be scalar or tensor

    Returns:
        Gross head (m)
    """
    if isinstance(v_up, torch.Tensor):
        return 50.0 + (v_up / max_vol_up) * 49.0
    else:
        v_up = torch.tensor(v_up, dtype=torch.float32, device=device)
        return (50.0 + (v_up / max_vol_up) * 49.0).item() if v_up.numel() == 1 else 50.0 + (v_up / max_vol_up) * 49.0

def h_to_v_up(head):
    """
    Convert gross head to upper reservoir volume.
    Inverse of v_up_to_h: v_up = (h - 50) / 49 * max_vol

    Args:
        head: Gross head (m), can be scalar or tensor

    Returns:
        Upper reservoir volume (m³)
    """
    if isinstance(head, torch.Tensor):
        return (head - 50.0) / 49.0 * max_vol_up
    else:
        head = torch.tensor(head, dtype=torch.float32, device=device)
        result = (head - 50.0) / 49.0 * max_vol_up
        return result.item() if head.numel() == 1 else result

def v_low_to_h(v_low):
    """
    Convert lower reservoir volume to gross head.
    Since v_low = max_vol - v_up, we have h = 50 + ((max_vol - v_low) / max_vol) * 49

    Args:
        v_low: Lower reservoir volume (m³), can be scalar or tensor

    Returns:
        Gross head (m)
    """
    v_up = max_vol_up - v_low if not isinstance(v_low, torch.Tensor) else max_vol_up - v_low
    return v_up_to_h(v_up)

def h_to_v_low(head):
    """
    Convert gross head to lower reservoir volume.
    Since v_low = max_vol - v_up, we calculate v_up first then subtract from max_vol

    Args:
        head: Gross head (m), can be scalar or tensor

    Returns:
        Lower reservoir volume (m³)
    """
    v_up = h_to_v_up(head)
    if isinstance(v_up, torch.Tensor):
        return max_vol_low - v_up
    else:
        return max_vol_low - v_up

# Test the functions
if __name__ == '__main__':
    print("Testing linear volume-head relationships:")
    print(f"h at v_up=588000 (max): {v_up_to_h(588000)} (should be 99)")
    print(f"h at v_up=0 (min): {v_up_to_h(0)} (should be 50)")
    print(f"v_up at h=99 (max): {h_to_v_up(99)} (should be 588000)")
    print(f"v_up at h=50 (min): {h_to_v_up(50)} (should be 0)")
    print(f"v_low at h=99 (max head): {h_to_v_low(99)} (should be 0)")
    print(f"v_low at h=50 (min head): {h_to_v_low(50)} (should be 588000)")
    print(f"h at v_low=0: {v_low_to_h(0)} (should be 99)")
    print(f"h at v_low=588000: {v_low_to_h(588000)} (should be 50)")

# %% Nonlinear volume-head relationship
# Based on conical upper reservoir and spherical-cap lower reservoir geometry.
# Parameters hardcoded from portfolio_UPHES.xlsx to avoid runtime dependency.

_pi = np.pi
_r = 58.779987115799145       # upper reservoir bottom radius (m)
_m = 1.8                      # upper reservoir slope coefficient
_h_dead_up = 72.39371953002411
_h_normal_up = 99.0
_height_up = _h_normal_up - _h_dead_up
_R = 11.196859765012055       # underground mine pit radius (m)
_n = 100                      # number of mine pits
_h_dead_low = 0.0
_h_normal_low = 22.39371953002411
_height_low = _h_normal_low - _h_dead_low

def _head_to_vol_up(head):
    """Upper reservoir: head level -> volume (conical geometry)."""
    height = head - _h_dead_up
    r_curr = _r + _m * height
    return (1.0 / 3.0) * _pi * height * (r_curr**2 + r_curr * _r + _r**2)

def _head_to_vol_low(head):
    """Lower reservoir: head level -> volume (spherical-cap geometry)."""
    height = head - _h_dead_low
    return _n * _pi * height**2 * (3.0 * _R - height) / 3.0

def _vol_to_head_up(volume):
    """Upper reservoir: volume -> head level (cubic solve via np.roots)."""
    # m^2*h^3 + 3*m*r*h^2 + 3*r^2*h - 3*V/pi = 0
    coeffs = [_m**2, 3.0*_m*_r, 3.0*_r**2, -3.0*volume/_pi]
    roots = np.roots(coeffs)
    upper_bound = _h_normal_up - _h_dead_up
    for root in roots:
        if np.isreal(root):
            val = float(np.real(root))
            if -1e-9 <= val <= upper_bound + 1e-9:
                return max(0.0, val) + _h_dead_up
    raise ValueError(f"No valid root for upper volume {volume}")

def _vol_to_head_low(volume):
    """Lower reservoir: volume -> head level (cubic solve via np.roots)."""
    # (pi/3)*h^3 - pi*R*h^2 + V/n = 0
    coeffs = [_pi/3.0, -_pi*_R, 0.0, volume/_n]
    roots = np.roots(coeffs)
    upper_bound = _h_normal_low - _h_dead_low
    for root in roots:
        if np.isreal(root):
            val = float(np.real(root))
            if -1e-9 <= val <= upper_bound + 1e-9:
                return max(0.0, val) + _h_dead_low
    raise ValueError(f"No valid root for lower volume {volume}")

def _gross_head_from_v_low(v_low):
    """Compute gross head (h_up - h_low) given v_low, with v_up + v_low = max_vol."""
    v_up = max_vol - v_low
    h_up = _vol_to_head_up(v_up)
    h_low = _vol_to_head_low(v_low)
    return h_up - h_low

# Generate sample points and fit 5th-degree polynomials
print("Fitting nonlinear v-h polynomials (100k samples)...")
_nl_v_low_samples = np.linspace(0, max_vol, 100000)
_nl_h_samples = np.array([_gross_head_from_v_low(v) for v in _nl_v_low_samples])

# h -> v_low polynomial (5th degree)
_nl_h_v_coefficients = np.polyfit(_nl_h_samples, _nl_v_low_samples, 5)
nl_h_v_coeffs = torch.tensor(_nl_h_v_coefficients, dtype=torch.float32, device=device)

# v_low -> h polynomial (5th degree)
_nl_v_low_h_coefficients = np.polyfit(_nl_v_low_samples, _nl_h_samples, 5)
nl_v_low_h_coeffs = torch.tensor(_nl_v_low_h_coefficients, dtype=torch.float32, device=device)

def nl_h_to_v_low(head, coeffs=nl_h_v_coeffs):
    """Nonlinear h -> v_low using 5th-degree polynomial (Horner's method, torch-compatible)."""
    if not isinstance(head, torch.Tensor):
        head = torch.tensor(head, dtype=torch.float32, device=coeffs.device)
    result = coeffs[0]
    for i in range(1, len(coeffs)):
        result = result * head + coeffs[i]
    return result

def nl_v_low_to_h(v_low, coeffs=nl_v_low_h_coeffs):
    """Nonlinear v_low -> h using 5th-degree polynomial (Horner's method, torch-compatible)."""
    if not isinstance(v_low, torch.Tensor):
        v_low = torch.tensor(v_low, dtype=torch.float32, device=coeffs.device)
    result = coeffs[0]
    for i in range(1, len(coeffs)):
        result = result * v_low + coeffs[i]
    return result

print("Nonlinear v-h polynomial fitting complete.")

# Validation
if __name__ == '__main__':
    print("\nTesting nonlinear volume-head relationships:")
    _test_h = 77.0
    _test_v = nl_h_to_v_low(torch.tensor(_test_h, device=device))
    _test_h_back = nl_v_low_to_h(_test_v)
    print(f"  h={_test_h} -> v_low={_test_v.item():.2f} -> h={_test_h_back.item():.4f} (round-trip)")
    print(f"  nl_v_low_to_h(0) = {nl_v_low_to_h(torch.tensor(0.0, device=device)).item():.4f} (should be ~99)")
    print(f"  nl_v_low_to_h(588000) = {nl_v_low_to_h(torch.tensor(588000.0, device=device)).item():.4f} (should be ~50)")
    print(f"  nl_h_to_v_low(99) = {nl_h_to_v_low(torch.tensor(99.0, device=device)).item():.2f} (should be ~0)")
    print(f"  nl_h_to_v_low(50) = {nl_h_to_v_low(torch.tensor(50.0, device=device)).item():.2f} (should be ~588000)")

# %% Fit UPC data (linear)
# Fit UPC data (linear)

def prepare_and_fit_model_linear(file_path):
    return prepare_and_fit_model(file_path, poly_degree=1)



# Perform linear regression on UPC (Outside pipeline)
if __name__ == '__main__':
    # Fit and plot models for both pump and turbine
    results_pump_linear = prepare_and_fit_model_linear('./Data/UPCs/temp/Mod_Francis_pump_temp.xlsx')
    results_turbine_linear = prepare_and_fit_model_linear('./Data/UPCs/temp/Mod_Francis_turbine_temp.xlsx')
    
    # Plot the fitted models
    plot_3d_surface_interactive(*results_pump_linear[1:4], results_pump_linear[0], 'Pump Model (Linear)')
    plot_3d_surface_interactive(*results_turbine_linear[1:4], results_turbine_linear[0], 'Turbine Model (Linear)')

    # Print statistical measures and formulas
    print("Pump Model (Linear) - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_pump_linear[4], results_pump_linear[5], results_pump_linear[6]))
    print("Pump Model Formula (Linear):")
    print(print_model_formula(results_pump_linear[7], results_pump_linear[8], results_pump_linear[9]))
    
    print("Turbine Model (Linear) - R^2: {:f}, Adjusted R^2: {:f}, Reduced Chi-Squared: {:f}".format(results_turbine_linear[4], results_turbine_linear[5], results_turbine_linear[6]))
    print("Turbine Model Formula (Linear):")
    print(print_model_formula(results_turbine_linear[7], results_turbine_linear[8], results_turbine_linear[9]))

    # Convert coefficients and intercepts to PyTorch tensors
    coefs_tur_linear = torch.tensor(results_turbine_linear[7], dtype=torch.float32, device=device)
    intercept_tur_linear = torch.tensor(results_turbine_linear[8], dtype=torch.float32, device=device)
    coefs_pump_linear = torch.tensor(results_pump_linear[7], dtype=torch.float32, device=device)
    intercept_pump_linear = torch.tensor(results_pump_linear[8], dtype=torch.float32, device=device)

    coefs_tur_lin = results_turbine_linear[7]
    intercept_tur_lin = results_turbine_linear[8]
    coefs_pump_lin = results_pump_linear[7]
    intercept_pump_lin = results_pump_linear[8]

# Create 2 functions UPC_linear_tur and UPC_linear_pump
def UPC_linear_tur(p, h, coefs=coefs_tur_linear, intercept=intercept_tur_linear):
    """
    Linear prediction of flow rate for turbine mode.
    """
    # Create feature matrix [p, h]
    features = torch.stack([p, h], dim=-1)

    # Compute linear prediction q = c_p*p + c_h*h + intercept
    q = torch.einsum('...d,d->...', features, coefs) + intercept

    # Zero out predictions for non-turbine mode (p ≤ 0)
    mask_tur = (p > 0)
    return torch.where(mask_tur, q, torch.zeros_like(q).to(device))

def UPC_linear_pump(p, h, coefs=coefs_pump_linear, intercept=intercept_pump_linear):
    """
    Linear prediction of flow rate for pump mode.
    """
    # Create feature matrix [p, h]
    features = torch.stack([p, h], dim=-1)

    # Compute linear prediction q = c_p*p + c_h*h + intercept
    q = torch.einsum('...d,d->...', features, coefs) + intercept

    # Zero out predictions for non-pump mode (p ≥ 0)
    mask_pump = (p < 0)
    return torch.where(mask_pump, q, torch.zeros_like(q).to(device))

if __name__ == '__main__':
    # test the functions
    p_example = torch.tensor([-5.89, 0.0, 5.89], dtype=torch.float32, device=device)
    h_example = torch.tensor([78, 67, 91], dtype=torch.float32, device=device)
    q_predicted_tur = UPC_linear_tur(p_example, h_example)
    q_predicted_pump = UPC_linear_pump(p_example, h_example)
    print(f"Predicted flow (q) for p={p_example.tolist()}, h={h_example.tolist()} (Turbine):\n{q_predicted_tur.tolist()}")
    print(f"Predicted flow (q) for p={p_example.tolist()}, h={h_example.tolist()} (Pump):\n{q_predicted_pump.tolist()}")


# %%
# Save preprocessing functions and variables
with open('preprocess.pkl', 'wb') as f:
    pickle.dump((
        # Volume-head conversion functions
        v_up_to_h, h_to_v_up, v_low_to_h, h_to_v_low,
        # UPC linear model
        coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin,
        UPC_linear_tur, UPC_linear_pump,
        # UPC polynomial model
        UPC_poly_tur, UPC_poly_pump,
        # UPC boundary functions
        neg_min, neg_max, pos_min, pos_max,
        # UPC boundary fitting coefficients
        h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit,
        # Price data
        DA_price_hour, DA_price_quarter,
        # Utility functions
        prepare_and_fit_model, get_UPC_bound, LR_UPC_bound,
        # System parameters
        max_vol_up, max_vol_low, max_vol, head_min, head_max,
        # Nonlinear volume-head functions and coefficients
        nl_h_to_v_low, nl_v_low_to_h, nl_h_v_coeffs, nl_v_low_h_coeffs
    ), f)

print('---------------Preprocessing Completed---------------')
# %%
