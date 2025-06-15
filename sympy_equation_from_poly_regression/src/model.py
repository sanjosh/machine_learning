import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sympy import symbols, Add, Pow, Mul, Symbol
from scipy.stats import norm

ranges = {
    'body_weight': (60, 80),
    'height': (5, 10),
    'start_time': (1, 168),
    'destination_code': (1, 60),
    'time_to_arrival': (30, 60)
}
var_names = list(ranges.keys())

# correlation matrix (symmetric, positive definite)
# Strong dependence between body_weight and time_to_arrival (0.9)
correlation_matrix = np.array([
    [1.0, 0.2,  0.2,  0.1,  0.9],  # body_weight
    [0.2, 1.0,  0.3,  0.2,  0.2],  # height
    [0.2, 0.3,  1.0,  0.2,  0.2],  # start_time
    [0.1, 0.2,  0.2,  1.0,  0.2],  # destination_code
    [0.9, 0.2,  0.2,  0.2,  1.0],  # time_to_arrival
])

n_samples = 1000
mean = np.zeros(5)
z = np.random.multivariate_normal(mean, correlation_matrix, size=n_samples)

u = norm.cdf(z)  # shape (1000, 5), all values in [0, 1]

# copula = GaussianCopula(k_dim=5)
# copula.corr = correlation_matrix
#
# n_samples = 1000
# copula_dist = CopulaDistribution(copula=copula, marginals=[None]*5)
# u = copula_dist.random(n_samples)  # u is in [0, 1] for each variable

def scale(u_col, min_val, max_val):
    return min_val + u_col * (max_val - min_val)

data = {}
for i, var in enumerate(var_names):
    min_val, max_val = ranges[var]
    data[var] = scale(u[:, i], min_val, max_val)

df = pd.DataFrame(data)

# Check correlation between body_weight and time_to_arrival
print("Empirical correlation (body_weight vs time_to_arrival):",
      df['body_weight'].corr(df['time_to_arrival']))

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)


feature_cols = ['body_weight', 'height', 'start_time', 'destination_code']
target_col = 'time_to_arrival'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

X_val = val_df[feature_cols].values
y_val = val_df[target_col].values

# Polynomial feature transformation
degree = 2  # change if required
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

X_train_poly_const = sm.add_constant(X_train_poly)
ols_model = sm.OLS(y_train, X_train_poly_const).fit()

print(ols_model.summary())

# Evaluate on validation set
X_val_poly_const = sm.add_constant(X_val_poly)
y_val_pred = ols_model.predict(X_val_poly_const)

# R² on validation set
r2_val = r2_score(y_val, y_val_pred)
print(f"\nValidation R²: {r2_val:.4f}")

# percent of predictions within ±2 minutes
errors = np.abs(y_val_pred - y_val)
within_threshold = np.mean(errors <= 2) * 100
print(f"Predictions within ±2 min: {within_threshold:.2f}%")

poly_feature_names = poly.get_feature_names_out(feature_cols)

terms = []
for name, coef in zip(poly_feature_names, ols_model.params[1:]):
    print(f"coefficient {name}: {coef:.3f}")
    terms.append(f"{coef:.3f}*{name}")
equation = f"{ols_model.params[0]:.3f} + " + " + ".join(terms)
print("f(x) =", equation)

# Define symbols for each original variable
symbol_map = {
    'body_weight': symbols('body_weight'),
    'height': symbols('height'),
    'start_time': symbols('start_time'),
    'destination_code': symbols('destination_code')
}

def parse_feature_term(term: str):
    factors = term.split(' ')
    expr_parts = []
    for f in factors:
        # Handle powers like body_weight^2
        if '^' in f:
            base, exp = f.split('^')
            expr_parts.append(Pow(symbol_map[base], int(exp)))
        else:
            expr_parts.append(symbol_map[f])
    return Mul(*expr_parts)

# Get polynomial feature names
feature_names = poly.get_feature_names_out(['body_weight', 'height', 'start_time', 'destination_code'])
intercept = ols_model.params[0]
coeffs = ols_model.params[1:]

# Map each feature name to a sympy expression
equation = intercept
for name, coef in zip(feature_names, coeffs):
    equation += coef * parse_feature_term(name)

print("Nonlinear model equation:")
print(equation.simplify())

body_weight, height, start_time, destination_code = symbols('body_weight height start_time destination_code')

sample_values = {
    body_weight: 72,
    height: 5.9,
    start_time: 45,
    destination_code: 12
}

result = equation.evalf(subs=sample_values)
print(f"Predicted time_to_arrival: {result:.2f} minutes")